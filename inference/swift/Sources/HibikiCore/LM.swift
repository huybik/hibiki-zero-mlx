// Hibiki-Zero multistream LM: the main transformer over text + 32 audio codebooks, then a
// depformer that samples the 16 target codebooks of one frame. Follows the Python reference in
// inference/moshi-mlx/moshi_mlx/models/lm.py.

import Foundation
import MLX
import MLXNN

public enum HibikiError: Error, CustomStringConvertible {
    case unsupported(String)
    case missingFile(URL)
    case badTokenizer(String)

    public var description: String {
        switch self {
        case .unsupported(let what): "unsupported model config: \(what)"
        case .missingFile(let url): "missing \(url.path)"
        case .badTokenizer(let what): "bad tokenizer: \(what)"
        }
    }
}

/// The subset of a Hibiki-Zero `config.json` this runtime reads.
public struct HibikiConfig: Decodable {
    public let card: Int
    public let nQ: Int
    public let depQ: Int
    public let delays: [Int]
    public let dim: Int
    public let textCard: Int
    public let numHeads: Int
    public let numLayers: Int
    public let hiddenScale: Double
    public let context: Int
    public let maxPeriod: Double
    public let positionalEmbedding: String
    public let kvRepeat: Int
    public let norm: String
    public let gating: String
    public let depformerDim: Int
    public let depformerNumHeads: Int
    public let depformerNumLayers: Int
    public let depformerDimFeedforward: Int?
    public let depformerPosEmb: String
    public let depformerMultiLinear: Bool
    public let depformerWeightsPerStep: Bool
    public let depformerWeightsPerStepSchedule: [Int]?
    public let depformerNorm: String?
    public let demuxSecondStream: Bool
    public let crossAttention: Bool
    public let layerScale: Double?
    public let weightDtype: String?
    public let moshiName: String?
    public let mimiName: String?
    public let tokenizerName: String?

    enum CodingKeys: String, CodingKey {
        case card, dim, context, norm, gating, delays
        case nQ = "n_q"
        case depQ = "dep_q"
        case textCard = "text_card"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case hiddenScale = "hidden_scale"
        case maxPeriod = "max_period"
        case positionalEmbedding = "positional_embedding"
        case kvRepeat = "kv_repeat"
        case depformerDim = "depformer_dim"
        case depformerNumHeads = "depformer_num_heads"
        case depformerNumLayers = "depformer_num_layers"
        case depformerDimFeedforward = "depformer_dim_feedforward"
        case depformerPosEmb = "depformer_pos_emb"
        case depformerMultiLinear = "depformer_multi_linear"
        case depformerWeightsPerStep = "depformer_weights_per_step"
        case depformerWeightsPerStepSchedule = "depformer_weights_per_step_schedule"
        case depformerNorm = "depformer_norm"
        case demuxSecondStream = "demux_second_stream"
        case crossAttention = "cross_attention"
        case layerScale = "layer_scale"
        case weightDtype = "weight_dtype"
        case moshiName = "moshi_name"
        case mimiName = "mimi_name"
        case tokenizerName = "tokenizer_name"
    }

    /// Fails on any setting this runtime does not implement, instead of running a wrong model.
    func validate() throws {
        if gating != "silu" { throw HibikiError.unsupported("gating \(gating)") }
        if !norm.hasPrefix("rms_norm") { throw HibikiError.unsupported("norm \(norm)") }
        if !["rope", "rope_concat"].contains(positionalEmbedding) {
            throw HibikiError.unsupported("positional_embedding \(positionalEmbedding)")
        }
        if depformerPosEmb != "none" {
            throw HibikiError.unsupported("depformer_pos_emb \(depformerPosEmb)")
        }
        if !depformerMultiLinear || !depformerWeightsPerStep {
            throw HibikiError.unsupported("a depformer without per-step weights")
        }
        if demuxSecondStream { throw HibikiError.unsupported("demux_second_stream") }
        if crossAttention { throw HibikiError.unsupported("cross_attention") }
        if layerScale != nil { throw HibikiError.unsupported("layer_scale") }
        if delays.count != nQ + 1 { throw HibikiError.unsupported("\(delays.count) delays") }
        if let schedule = depformerWeightsPerStepSchedule, schedule.count != depQ {
            throw HibikiError.unsupported("a weights-per-step schedule of \(schedule.count)")
        }
    }

    var audioDelays: [Int] { Array(delays.dropFirst()) }  // delays[0] is the text stream's
    var audioPaddingToken: Int { card }  // audio vocabulary is card + 1
    var textInitToken: Int { textCard }  // text input vocabulary is text_card + 1
    /// Which depformer weight set each codebook step uses.
    var depformerSchedule: [Int] { depformerWeightsPerStepSchedule ?? Array(0..<depQ) }

    var transformer: TransformerConfig {
        TransformerConfig(
            dModel: dim, numHeads: numHeads, numLayers: numLayers, causal: true,
            normFirst: true, biasFF: false, biasAttn: false, layerScale: nil,
            positionalEmbedding: positionalEmbedding == "rope_concat" ? .ropeConcat : .rope,
            useConvBias: true, gating: true, norm: .rmsNorm, context: context,
            maxPeriod: Int(maxPeriod), maxSeqLen: 4096, kvRepeat: kvRepeat,
            dimFeedForward: Int((hiddenScale * Double(dim)).rounded()), convLayout: false)
    }

    var depformer: TransformerConfig {
        TransformerConfig(
            dModel: depformerDim, numHeads: depformerNumHeads, numLayers: depformerNumLayers,
            causal: true, normFirst: true, biasFF: false, biasAttn: false, layerScale: nil,
            positionalEmbedding: .none, useConvBias: true, gating: true, norm: .rmsNorm,
            context: depQ, maxPeriod: 8, maxSeqLen: 4096, kvRepeat: 1,
            dimFeedForward: depformerDimFeedforward
                ?? Int((hiddenScale * Double(depformerDim)).rounded()),
            convLayout: false)
    }
}

/// One depformer weight set: the input projection from the main transformer and the
/// transformer itself. With a weights-per-step schedule several codebook steps share one.
final class DepformerCore: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "transformer") var transformer: Transformer

    init(mainDim: Int, cfg: TransformerConfig) {
        _linearIn.wrappedValue = Linear(mainDim, cfg.dModel, bias: false)
        _transformer.wrappedValue = Transformer(cfg)
    }
}

/// The per-codebook parts of a depformer step: input embedding, output norm and head.
final class DepformerSlice: Module {
    @ModuleInfo(key: "emb") var emb: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm?
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(inVocab: Int, outVocab: Int, dim: Int, outputNorm: Bool) {
        _emb.wrappedValue = Embedding(embeddingCount: inVocab, dimensions: dim)
        _norm.wrappedValue = outputNorm ? LayerNorm(dimensions: dim, eps: 1e-5) : nil
        _linearOut.wrappedValue = Linear(dim, outVocab, bias: false)
    }
}

final class Depformer: Module {
    let schedule: [Int]
    private let steps: [@Sendable ([MLXArray]) -> [MLXArray]]
    @ModuleInfo(key: "cores") var cores: [DepformerCore]
    @ModuleInfo(key: "slices") var slices: [DepformerSlice]

    init(_ cfg: HibikiConfig) {
        let dcfg = cfg.depformer
        let schedule = cfg.depformerSchedule
        let cores = (0...(schedule.max() ?? 0)).map { _ in
            DepformerCore(mainDim: cfg.dim, cfg: dcfg)
        }
        let slices = (0..<cfg.depQ).map { idx in
            DepformerSlice(
                inVocab: idx == 0 ? cfg.textCard + 1 : cfg.card + 1, outVocab: cfg.card,
                dim: dcfg.dModel, outputNorm: cfg.depformerNorm != nil)
        }
        self.schedule = schedule
        steps = slices.enumerated().map { idx, slice in
            compileDepformerStep(core: cores[schedule[idx]], slice: slice)
        }
        _cores.wrappedValue = cores
        _slices.wrappedValue = slices
    }

    /// Samples one token per target codebook, each step conditioned on the previous one.
    func sample(_ mainOut: MLXArray, textToken: MLXArray, sampler: Sampler) -> MLXArray {
        var last = textToken
        var kv: [MLXArray] = []
        var tokens: [MLXArray] = []
        for step in steps {
            let out = step([mainOut, last] + kv)
            kv = Array(out.dropFirst())
            last = sampler(out[0]).reshaped(1, 1)
            tokens.append(last)
        }
        return concatenated(tokens, axis: 1)  // [1, depQ]
    }

    /// Maps checkpoint keys (one full copy per codebook step) onto the shared cores and drops
    /// the duplicate copies.
    func remap(_ key: String) -> String? {
        let prefix = "depformer.slices."
        guard key.hasPrefix(prefix) else { return key }
        let rest = key.dropFirst(prefix.count)
        guard let dot = rest.firstIndex(of: "."), let idx = Int(rest[..<dot]) else { return key }
        let field = rest[rest.index(after: dot)...]
        guard field.hasPrefix("linear_in.") || field.hasPrefix("transformer.") else { return key }
        let core = schedule[idx]
        if schedule.firstIndex(of: core) != idx { return nil }
        return "depformer.cores.\(core).\(field)"
    }
}

public final class HibikiLM: Module {
    public let cfg: HibikiConfig
    private let decoder: DecodeTransformer
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "depformer") var depformer: Depformer
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [Embedding]

    init(_ cfg: HibikiConfig) {
        self.cfg = cfg
        let tcfg = cfg.transformer
        _transformer.wrappedValue = Transformer(tcfg)
        _depformer.wrappedValue = Depformer(cfg)
        _textEmb.wrappedValue = Embedding(embeddingCount: cfg.textCard + 1, dimensions: cfg.dim)
        _outNorm.wrappedValue = RMSNorm(dimensions: cfg.dim, eps: 1e-8)
        _textLinear.wrappedValue = Linear(cfg.dim, cfg.textCard, bias: false)
        _audioEmbs.wrappedValue = (0..<cfg.nQ).map { _ in
            Embedding(embeddingCount: cfg.card + 1, dimensions: cfg.dim)
        }
        decoder = DecodeTransformer(_transformer.wrappedValue)
    }

    /// Loads `config.json` and the q4 or BF16 weights of a staged model directory.
    public static func load(directory: URL) throws -> HibikiLM {
        let cfg = try JSONDecoder().decode(
            HibikiConfig.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        try cfg.validate()
        let bf16 = cfg.weightDtype == "bfloat16"
        let name = cfg.moshiName.flatMap {
            $0.hasSuffix(bf16 ? ".bf16.safetensors" : ".q4.safetensors") ? $0 : nil
        }
        guard let name = name ?? (bf16 ? nil : "hibiki.q4.safetensors") else {
            throw HibikiError.unsupported("bfloat16 without a moshi_name ending in .bf16.safetensors")
        }
        let url = directory.appending(path: name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw HibikiError.missingFile(url)
        }
        let model = HibikiLM(cfg)
        if !bf16 {
            quantize(model: model, groupSize: 32, bits: 4)
        }
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            if let key = model.depformer.remap(key) { weights[key] = value }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    public func resetCache() {
        decoder.reset()
    }

    /// One frame: the main transformer over the previous text token and the 32 delayed audio
    /// tokens, then the text token and the 16 target codebooks.
    func step(
        text: Int, audio: [Int], textSampler: Sampler, audioSampler: Sampler
    ) -> (text: MLXArray, audio: MLXArray) {
        var xs = textEmb(MLXArray([Int32(text)]).reshaped(1, 1))
        for (token, emb) in zip(audio, audioEmbs) {
            xs = xs + emb(MLXArray([Int32(token)]).reshaped(1, 1))
        }
        let mainOut = outNorm(decoder(xs))
        let textToken = textSampler(textLinear(mainOut[0..., -1, 0...])).reshaped(1, 1)
        let audioTokens = depformer.sample(mainOut, textToken: textToken, sampler: audioSampler)
        return (textToken, audioTokens)
    }
}
