import Accelerate
import CoreML
import Foundation

/// A phone student's LM in Core ML, on the Neural Engine by default: the models
/// `inference/tools/export_lm_coreml.py` writes from `lm_step.py`. The audio comes from the
/// parallel heads (two calls a frame) or from a depformer_parallel model's slices (one call each).
///
///     host:   x = text_emb[text] + sum of audio_embs[q][audio q]     (lm-embeddings.bin, fp16)
///     trunk:  x, cos, sin, mask, slot -> hidden, text_logits        (KV cache: Core ML state)
///     host:   text token sampled
///     heads:  hidden, text_emb[sampled text] -> audio_logits [dep_q, card]; sampled
///   or, slice k = 0 ..< slices:
///     host:   emb = slice k's table [the token before: the text, else codebook k - 1]
///                                                              (lm-depformer-embeddings.bin)
///     slice:  hidden, emb, keys and values of slices < k -> logits, key, value
///     host:   codebook k sampled (the last slice: the last parallel_codebooks)
///
/// The KV cache is a ring of `context` slots: frame t writes slot t mod context and sees the
/// slots written so far, as `lm_step.Inputs.at` computes. A frame runs on a serial queue, so
/// `start` returns at once and the caller's codec work overlaps it.
final class NeuralLM: FrameLM {
    let cfg: HibikiConfig
    private let trunk: MLModel
    /// The parallel heads; nil for a depformer_parallel model, which has `slices`.
    private let heads: MLModel?
    private let slices: [MLModel]
    /// The slices' input tables, memory-mapped: slice 0's [text_card + 1, depformer dim], then
    /// each later slice's [card + 1, depformer dim].
    private let sliceTables: Data?
    /// Slice k's inputs: its table row, and (k > 0) the keys and values of the frame's slices
    /// before it, [layers, heads, k, head_dim], at `past[k - 1]`.
    private let sliceEmb: MLMultiArray
    private let past: [(keys: MLMultiArray, values: MLMultiArray)]
    /// The KV cache; nil only while `reset` replaces it, so two never coexist (172 MB each).
    private var state: MLState?
    /// The input tables, memory-mapped: text_emb [text_card + 1, dim], then each audio_embs
    /// [card + 1, dim].
    private let tables: Data
    private let textSampler: Sampler
    private let audioSampler: Sampler
    private var rng = SplitMix64(seed: UInt64.random(in: 0...UInt64.max))
    private let queue = DispatchQueue(label: "hibiki.neural-lm", qos: .userInteractive)
    /// The trunk's inputs, rewritten each frame; `mask` changes only while the ring fills.
    private let x: MLMultiArray
    private let cos: MLMultiArray
    private let sin: MLMultiArray
    private let mask: MLMultiArray
    private let slot: MLMultiArray
    private let text: MLMultiArray
    private var frame = 0

    init(directory: URL, cfg: HibikiConfig, computeUnits: MLComputeUnits) throws {
        self.cfg = cfg
        let file = { (name: String) throws -> URL in
            let url = directory.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw HibikiError.missingFile(url)
            }
            return url
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        trunk = try MLModel(contentsOf: file("lm-trunk.mlmodelc"), configuration: config)
        let depformer = cfg.audioHeads == "depformer_parallel"
        heads = depformer ? nil : try MLModel(contentsOf: file("lm-heads.mlmodelc"), configuration: config)
        slices = depformer
            ? try (0..<cfg.depformerSlices).map {
                try MLModel(contentsOf: file("lm-depformer-\($0).mlmodelc"), configuration: config)
            }
            : []
        sliceTables = depformer
            ? try Data(contentsOf: file("lm-depformer-embeddings.bin"), options: .alwaysMapped) : nil
        state = trunk.makeState()
        tables = try Data(contentsOf: file("lm-embeddings.bin"), options: .alwaysMapped)
        let rows = cfg.textCard + 1 + cfg.nQ * (cfg.card + 1)
        guard tables.count == rows * cfg.dim * MemoryLayout<Float16>.size else {
            throw HibikiError.unsupported("lm-embeddings.bin of \(tables.count) bytes")
        }
        let d = cfg.depformerDim, headDim = cfg.depformerDim / cfg.depformerNumHeads
        if let sliceTables {
            let rows = cfg.textCard + 1 + (slices.count - 1) * (cfg.card + 1)
            guard sliceTables.count == rows * d * MemoryLayout<Float16>.size else {
                throw HibikiError.unsupported("lm-depformer-embeddings.bin of \(sliceTables.count) bytes")
            }
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
        for (k, slice) in slices.enumerated() {
            let pastShape = [cfg.depformerNumLayers, cfg.depformerNumHeads, k, headDim]
            let keyShape = [cfg.depformerNumLayers, cfg.depformerNumHeads, 1, headDim]
            inputs += [(slice, "hidden", [1, cfg.dim]), (slice, "emb", [1, d])]
            if k > 0 { inputs += [(slice, "keys", pastShape), (slice, "values", pastShape)] }
            let codebooks = k == slices.count - 1 ? cfg.depQ - k : 1
            outputs += [(slice, "logits", [codebooks, cfg.card]), (slice, "key", keyShape), (slice, "value", keyShape)]
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
        sliceEmb = try array([1, d], .float16)
        past = try (1..<max(slices.count, 1)).map { k in
            let shape = [cfg.depformerNumLayers, cfg.depformerNumHeads, k, headDim]
            return (try array(shape, .float16), try array(shape, .float16))
        }
        let sampling = cfg.lmGenConfig
        textSampler = Sampler(temp: sampling.tempText, topK: sampling.topKText)
        audioSampler = Sampler(temp: sampling.temp, topK: sampling.topK)
        closeMask()
    }

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
        for (k, slice) in slices.enumerated() {
            // The token before codebook k: the text, else codebook k - 1, in slice k's table.
            let previous = audioTokens.last ?? textToken
            let row = k == 0 ? previous : cfg.textCard + 1 + (k - 1) * (cfg.card + 1) + previous
            copyRow(row, of: sliceTables!, width: cfg.depformerDim, into: sliceEmb)
            var features: [String: Any] = ["hidden": hidden, "emb": sliceEmb]
            if k > 0 { features["keys"] = past[k - 1].keys; features["values"] = past[k - 1].values }
            busy.host += lap()
            let sliceOut = try slice.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
            busy.heads += lap()
            for later in (k + 1)..<slices.count {
                place(sliceOut.featureValue(for: "key")!.multiArrayValue!, at: k, in: past[later - 1].keys)
                place(sliceOut.featureValue(for: "value")!.multiArrayValue!, at: k, in: past[later - 1].values)
            }
            audioTokens += rows(sliceOut.featureValue(for: "logits")!.multiArrayValue!).map {
                audioSampler($0, using: &rng)
            }
        }
        precondition(audioTokens.count == cfg.depQ)
        busy.host += lap()
        frame += 1
        return LMStep(text: textToken, audio: audioTokens, busy: busy)
    }

    /// A slice's key or value [layers, heads, 1, head_dim] (by its strides) into position `k` of
    /// a later slice's past [layers, heads, n, head_dim], which this class allocated contiguous.
    private func place(_ source: MLMultiArray, at k: Int, in past: MLMultiArray) {
        let shape = past.shape.map(\.intValue)
        let (layers, heads, n, width) = (shape[0], shape[1], shape[2], shape[3])
        let strides = source.strides.map(\.intValue)
        let dst = past.dataPointer.bindMemory(to: Float16.self, capacity: past.count)
        source.withUnsafeBufferPointer(ofType: Float16.self) { src in
            for l in 0..<layers {
                for h in 0..<heads {
                    let from = src.baseAddress! + l * strides[0] + h * strides[1]
                    let to = dst + ((l * heads + h) * n + k) * width
                    for i in 0..<width { to[i] = from[i * strides[3]] }
                }
            }
        }
    }

    /// The summed input embeddings, the RoPE tables of this frame's position, its cache slot,
    /// and the slot it writes unmasked while the ring fills.
    private func fillTrunkInputs(text textIn: Int, audio: [Int]) {
        let dim = cfg.dim
        precondition(textIn >= 0 && textIn <= cfg.textCard, "text token \(textIn)")
        var rowsIn = [textIn]
        for (q, token) in audio.enumerated() {
            precondition(token >= 0 && token <= cfg.card, "audio token \(token)")
            rowsIn.append(cfg.textCard + 1 + q * (cfg.card + 1) + token)
        }
        // Summed in fp32 (Accelerate), as fp16 rows from the table.
        var sum = [Float](repeating: 0, count: dim)
        var row = [Float](repeating: 0, count: dim)
        tables.withUnsafeBytes { raw in
            for index in rowsIn {
                row.withUnsafeMutableBytes { dst in
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + index * dim * 2),
                        height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
                    var out = vImage_Buffer(
                        data: dst.baseAddress!, height: 1, width: vImagePixelCount(dim),
                        rowBytes: dim * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &out, 0)
                }
                sum.withUnsafeMutableBufferPointer { s in
                    vDSP_vadd(s.baseAddress!, 1, row, 1, s.baseAddress!, 1, vDSP_Length(dim))
                }
            }
        }
        sum.withUnsafeMutableBytes { src in
            var input = vImage_Buffer(
                data: src.baseAddress!, height: 1, width: vImagePixelCount(dim), rowBytes: dim * 4)
            var output = vImage_Buffer(
                data: x.dataPointer, height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
            vImageConvert_PlanarFtoPlanar16F(&input, &output, 0)
        }

        let half = cos.count
        let c = cos.dataPointer.bindMemory(to: Float16.self, capacity: half)
        let s = sin.dataPointer.bindMemory(to: Float16.self, capacity: half)
        for i in 0..<half {
            // Positions grow without bound; the angle is computed in Double.
            let angle = Double(frame) * exp(-log(cfg.maxPeriod) * Double(i) / Double(half))
            c[i] = Float16(Foundation.cos(angle))
            s[i] = Float16(Foundation.sin(angle))
        }
        slot.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(frame % cfg.context)
        if frame < cfg.context {
            mask.dataPointer.bindMemory(to: Float16.self, capacity: mask.count)[frame] = 0
        }
    }

    /// Row `row` of an fp16 `table` of `width` columns into `array` [1, width].
    private func copyRow(_ row: Int, of table: Data, width: Int, into array: MLMultiArray) {
        let dst = array.dataPointer.bindMemory(to: Float16.self, capacity: width)
        table.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float16.self).baseAddress! + row * width
            dst.update(from: src, count: width)
        }
    }

    /// A 2-D fp16 output's rows as Float, by its strides (Core ML may pad rows).
    private func rows(_ array: MLMultiArray) -> [[Float]] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        precondition(shape.count == 2 && array.dataType == .float16)
        return array.withUnsafeBufferPointer(ofType: Float16.self) { values in
            (0..<shape[0]).map { r in
                [Float](unsafeUninitializedCapacity: shape[1]) { row, count in
                    let base = values.baseAddress! + r * strides[0]
                    for c in 0..<shape[1] { row[c] = Float(base[c * strides[1]]) }
                    count = shape[1]
                }
            }
        }
    }
}
