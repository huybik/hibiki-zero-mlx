import CoreML
import Foundation

/// A phone student's LM in Core ML, on the Neural Engine by default: the models
/// `inference/tools/export_lm_coreml.py` writes from `lm_step.py`. The audio comes from the
/// parallel heads (`lm-heads.mlmodelc`) or a `depformer_parallel` model's depformer
/// (`lm-depformer.mlmodelc`, one call for all slices with in-graph top-k Gumbel-max sampling).
///
///     host:   x = text_emb[text] + sum of audio_embs[q][audio q]     (lm-embeddings.bin, fp16)
///     trunk:  x, cos, sin, mask, slot -> hidden, text_logits        (KV cache: Core ML state)
///     host:   text token sampled
///     heads:  hidden, text_emb[sampled text] -> audio_logits [dep_q, card]; sampled
///   or:
///     host:   emb = slice 0's table [the sampled text]  (lm-depformer-embeddings.bin);
///             noise = temp * Gumbel(0, 1) [dep_q, card]
///     depformer:  hidden, emb, noise -> tokens [dep_q]
///
/// The KV cache is a ring of `context` slots: frame t writes slot t mod context and sees the
/// slots written so far, as `lm_step.Inputs.at` computes. A frame runs on a serial queue, so
/// `start` returns at once and the caller's codec work overlaps it.
final class NeuralLM {
    let cfg: HibikiConfig
    private let trunk: MLModel
    /// The parallel heads; nil for a depformer_parallel model, which has `depformer`.
    private let heads: MLModel?
    /// A depformer_parallel model's depformer (`lm-depformer.mlmodelc`); nil for the heads.
    private let depformer: MLModel?
    /// The depformer's input table (slice 0's) [text_card + 1, d], memory-mapped, and its row of
    /// the sampled text, the depformer's `emb`.
    private let depformerTable: Data?
    private let depformerEmb: MLMultiArray
    /// Gumbel noise [dep_q, card] fp16 for `depformer` (zeros when temp == 0), filled from a
    /// 65,536-entry quantile LUT (128 KB, pre-scaled by `temp`) using 4 lookups per 64-bit RNG step.
    private let depformerNoise: MLMultiArray?
    let gumbelLUT: [Float16]
    /// The KV cache; nil only while `reset` replaces it, so two never coexist (172 MB each).
    private var state: MLState?
    /// The input tables, memory-mapped: text_emb [text_card + 1, dim], then each audio_embs
    /// [card + 1, dim].
    let tables: Data
    private let textSampler: Sampler
    private let audioSampler: Sampler
    var rng = SplitMix64(seed: UInt64.random(in: 0...UInt64.max))
    private let queue = DispatchQueue(label: "hibiki.neural-lm", qos: .userInteractive)
    /// The trunk's inputs, rewritten each frame; `mask` changes only while the ring fills.
    let x: MLMultiArray
    let cos: MLMultiArray
    let sin: MLMultiArray
    let mask: MLMultiArray
    let slot: MLMultiArray
    private let text: MLMultiArray
    private(set) var frame = 0

    init(directory: URL, cfg: HibikiConfig, computeUnits: MLComputeUnits) throws {
        self.cfg = cfg
        let file = directory.modelFile
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        trunk = try MLModel(contentsOf: file("lm-trunk.mlmodelc"), configuration: config)
        let isDepformer = cfg.audioHeads == "depformer_parallel"
        heads = isDepformer ? nil : try MLModel(contentsOf: file("lm-heads.mlmodelc"), configuration: config)
        depformer = isDepformer
            ? try MLModel(contentsOf: file("lm-depformer.mlmodelc"), configuration: config) : nil
        depformerTable = isDepformer
            ? try Data(contentsOf: file("lm-depformer-embeddings.bin"), options: .alwaysMapped) : nil
        state = trunk.makeState()
        tables = try Data(contentsOf: file("lm-embeddings.bin"), options: .alwaysMapped)
        let rows = cfg.textCard + 1 + cfg.nQ * (cfg.card + 1)
        guard tables.count == rows * cfg.dim * MemoryLayout<Float16>.size else {
            throw HibikiError.unsupported("lm-embeddings.bin of \(tables.count) bytes")
        }
        let d = cfg.depformerDim
        if let depformerTable {
            guard depformerTable.count == (cfg.textCard + 1) * d * MemoryLayout<Float16>.size else {
                throw HibikiError.unsupported("lm-depformer-embeddings.bin of \(depformerTable.count) bytes")
            }
        }
        let sampling = cfg.lmGenConfig
        if isDepformer, sampling.temp > 0, let baked = cfg.lmCoreMLDepformerTopK, baked != sampling.topK {
            throw HibikiError.unsupported("lm-depformer top_k \(baked) for lm_gen_config top_k \(sampling.topK)")
        }
        let half = cfg.dim / cfg.numHeads / 2
        var inputs: [(MLModel, String, [Int])] = [
            (trunk, "x", [1, cfg.dim]), (trunk, "cos", [1, half]), (trunk, "sin", [1, half]),
            (trunk, "mask", [1, cfg.context]), (trunk, "slot", [1]),
        ]
        var outputs: [(MLModel, String, [Int])] = [
            (trunk, "hidden", [1, cfg.dim]), (trunk, "text_logits", [1, cfg.textCard]),
        ]
        if let heads {
            inputs += [(heads, "hidden", [1, cfg.dim]), (heads, "text", [1, cfg.dim])]
            outputs += [(heads, "audio_logits", [cfg.depQ, cfg.card])]
        }
        if let depformer {
            inputs += [
                (depformer, "hidden", [1, cfg.dim]), (depformer, "emb", [1, d]),
                (depformer, "noise", [cfg.depQ, cfg.card]),
            ]
            outputs += [(depformer, "tokens", [cfg.depQ])]
        }
        for (model, name, shape) in inputs + outputs {
            let description = model.modelDescription
            let feature = description.inputDescriptionsByName[name]
                ?? description.outputDescriptionsByName[name]
            let actual = feature?.multiArrayConstraint?.shape.map(\.intValue)
            guard actual == shape else {
                throw HibikiError.unsupported("Core ML LM \(name) \(actual ?? []) for \(shape)")
            }
        }
        let array = { (shape: [Int], type: MLMultiArrayDataType) in
            try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
        }
        x = try array([1, cfg.dim], .float16)
        cos = try array([1, half], .float16)
        sin = try array([1, half], .float16)
        mask = try array([1, cfg.context], .float16)
        slot = try array([1], .int32)
        text = try array([1, cfg.dim], .float16)
        depformerEmb = try array([1, d], .float16)
        if isDepformer {
            let noise = try array([cfg.depQ, cfg.card], .float16)
            memset(noise.dataPointer, 0, noise.count * MemoryLayout<Float16>.size)
            depformerNoise = noise
            gumbelLUT = sampling.temp > 0 ? NeuralLM.gumbelTable(temp: sampling.temp) : []
        } else {
            depformerNoise = nil
            gumbelLUT = []
        }
        textSampler = Sampler(temp: sampling.tempText, topK: sampling.topKText)
        audioSampler = Sampler(temp: sampling.temp, topK: sampling.topK)
        closeMask()
    }

    /// Empties the KV cache for a new session.
    func reset() {
        queue.sync {
            frame = 0
            state = nil
            state = trunk.makeState()
            closeMask()
        }
    }

    /// Masks every cache slot: a new session has written none.
    private func closeMask() {
        let m = mask.dataPointer.bindMemory(to: Float16.self, capacity: mask.count)
        for j in 0..<mask.count { m[j] = -1e4 }
    }

    func seed(_ value: UInt64) {
        queue.sync { rng = SplitMix64(seed: value) }
    }

    /// Starts the step of a frame from the previous text token and the 32 delayed audio tokens;
    /// the closure it returns waits for the step's tokens.
    func start(text: Int, audio: [Int]) -> () throws -> LMStep {
        let done = DispatchSemaphore(value: 0)
        var result: Result<LMStep, Error>!
        queue.async {
            // Core ML returns autoreleased objects, which a queue thread keeps until it drains.
            result = Result { try autoreleasepool { try self.step(text: text, audio: audio) } }
            done.signal()
        }
        return {
            done.wait()
            return try result.get()
        }
    }

    private func step(text textIn: Int, audio audioIn: [Int]) throws -> LMStep {
        precondition(audioIn.count == cfg.nQ)
        var busy = LMBusy()
        var clock = DispatchTime.now()
        func lap() -> Double {
            let now = DispatchTime.now()
            defer { clock = now }
            return seconds(clock, now)
        }
        fillTrunkInputs(text: textIn, audio: audioIn)
        busy.host += lap()
        let out = try trunk.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "x": x, "cos": cos, "sin": sin, "mask": mask, "slot": slot,
            ]), using: state!)
        busy.trunk += lap()
        let hidden = out.featureValue(for: "hidden")!.multiArrayValue!
        let textToken = textSampler(rows(out.featureValue(for: "text_logits")!.multiArrayValue!)[0], using: &rng)
        var audioTokens: [Int] = []
        if let heads {
            copyRow(textToken, of: tables, width: cfg.dim, into: text)
            busy.host += lap()
            let audioOut = try heads.prediction(
                from: MLDictionaryFeatureProvider(dictionary: ["hidden": hidden, "text": text]))
            busy.heads += lap()
            let logits = rows(audioOut.featureValue(for: "audio_logits")!.multiArrayValue!)
            precondition(logits.count == cfg.depQ && logits[0].count == cfg.card)
            audioTokens = logits.map { audioSampler($0, using: &rng) }
        }
        if let depformer, let depformerNoise {
            copyRow(textToken, of: depformerTable!, width: cfg.depformerDim, into: depformerEmb)
            if !gumbelLUT.isEmpty { fillGumbelNoise(into: depformerNoise) }
            busy.host += lap()
            let depOut = try depformer.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "hidden": hidden, "emb": depformerEmb, "noise": depformerNoise,
                ]))
            busy.heads += lap()
            let tokArray = depOut.featureValue(for: "tokens")!.multiArrayValue!
            audioTokens = readTokens(tokArray, count: cfg.depQ)
        }
        precondition(audioTokens.count == cfg.depQ)
        busy.host += lap()
        frame += 1
        return LMStep(text: textToken, audio: audioTokens, busy: busy)
    }
}
