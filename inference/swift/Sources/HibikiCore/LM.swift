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
    case badAudio(String)
    case badDirection(String)

    public var description: String {
        switch self {
        case .unsupported(let what): "unsupported model config: \(what)"
        case .missingFile(let url): "missing \(url.path)"
        case .badTokenizer(let what): "bad tokenizer: \(what)"
        case .badAudio(let what): "bad audio: \(what)"
        case .badDirection(let what): "bad direction: \(what)"
        }
    }
}

/// The two-way model's condition (`training/finetune/direction.py`): moshi's `lut` conditioner
/// `direction`, a table of one 16-wide entry per direction (`<source>-<target>`, such as `vi-en`)
/// and its projection to the model's width, summed by the fuser into every frame's input.
public struct DirectionTable: Decodable {
    public let nBins: Int
    public let dim: Int
    public let tokenizer: String
    /// The directions, in the table's row order.
    public let possibleValues: [String]

    enum CodingKeys: String, CodingKey {
        case dim, tokenizer
        case nBins = "n_bins"
        case possibleValues = "possible_values"
    }

    static let name = "direction"
}

/// The subset of a Hibiki-Zero `config.json` this runtime reads, with the defaults of
/// `moshi_mlx.models.LmConfig.from_config_dict` for keys it treats as optional.
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
    public let depformerDim: Int
    public let depformerNumHeads: Int
    public let depformerNumLayers: Int
    public let depformerDimFeedforward: Int?
    public let depformerPosEmb: String
    public let depformerWeightsPerStepSchedule: [Int]?
    /// Python: `depformer_output_layer_norm`, else whether `depformer_norm` is present.
    public let depformerOutputNorm: Bool
    /// The MLX weights file.
    public let moshiName: String
    public let mimiName: String?
    public let tokenizerName: String
    /// How quantized weights are quantized; nil for bf16 weights.
    public let quantization: Quantization?
    public let lmGenConfig: LmGenConfig
    /// The direction condition's table; nil for a model without it, which hears the direction.
    public let directionTable: DirectionTable?
    /// The directions the model is conditioned on; empty without the condition.
    public var directions: [String] { directionTable?.possibleValues ?? [] }
    // Read only to refuse models this runtime would run wrongly.
    let gating: String?
    let norm: String?
    let depformerMultiLinear: Bool?
    let depformerWeightsPerStep: Bool?
    let demuxSecondStream: Bool
    let crossAttention: Bool
    let layerScale: Double?
    let conditioners: [String: Conditioner]
    let fuser: Fuser?

    struct Conditioner: Decodable {
        let type: String
        let lut: DirectionTable?
    }

    /// Where the fuser puts each condition: summed into the input, prepended, or cross-attended.
    struct Fuser: Decodable {
        let sum: [String]
        let prepend: [String]?
        let cross: [String]?
    }

    enum CodingKeys: String, CodingKey {
        case card, dim, context, norm, gating, delays, conditioners, fuser
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
        case depformerOutputLayerNorm = "depformer_output_layer_norm"
        case demuxSecondStream = "demux_second_stream"
        case crossAttention = "cross_attention"
        case layerScale = "layer_scale"
        case moshiName = "moshi_name"
        case mimiName = "mimi_name"
        case tokenizerName = "tokenizer_name"
        case quantization
        case lmGenConfig = "lm_gen_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        card = try c.decode(Int.self, forKey: .card)
        nQ = try c.decode(Int.self, forKey: .nQ)
        depQ = try c.decode(Int.self, forKey: .depQ)
        delays = try c.decode([Int].self, forKey: .delays)
        dim = try c.decode(Int.self, forKey: .dim)
        textCard = try c.decode(Int.self, forKey: .textCard)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        hiddenScale = try c.decodeIfPresent(Double.self, forKey: .hiddenScale) ?? 4
        context = try c.decode(Int.self, forKey: .context)
        maxPeriod = try c.decode(Double.self, forKey: .maxPeriod)
        positionalEmbedding = try c.decode(String.self, forKey: .positionalEmbedding)
        kvRepeat = try c.decodeIfPresent(Int.self, forKey: .kvRepeat) ?? 1
        depformerDim = try c.decode(Int.self, forKey: .depformerDim)
        depformerNumHeads = try c.decode(Int.self, forKey: .depformerNumHeads)
        depformerNumLayers = try c.decode(Int.self, forKey: .depformerNumLayers)
        // Python: `depformer_dim_feedforward or scaled`, so 0 means unset too.
        depformerDimFeedforward = try c.decodeIfPresent(
            Int.self, forKey: .depformerDimFeedforward
        ).flatMap { $0 > 0 ? $0 : nil }
        depformerPosEmb = try c.decode(String.self, forKey: .depformerPosEmb)
        depformerWeightsPerStepSchedule = try c.decodeIfPresent(
            [Int].self, forKey: .depformerWeightsPerStepSchedule)
        depformerOutputNorm =
            try c.decodeIfPresent(Bool.self, forKey: .depformerOutputLayerNorm)
            ?? c.contains(.depformerNorm)
        moshiName = try c.decode(String.self, forKey: .moshiName)
        mimiName = try c.decodeIfPresent(String.self, forKey: .mimiName)
        tokenizerName = try c.decode(String.self, forKey: .tokenizerName)
        quantization = try c.decodeIfPresent(Quantization.self, forKey: .quantization)
        lmGenConfig = try c.decode(LmGenConfig.self, forKey: .lmGenConfig)
        gating = try c.decodeIfPresent(String.self, forKey: .gating)
        norm = try c.decodeIfPresent(String.self, forKey: .norm)
        depformerMultiLinear = try c.decodeIfPresent(Bool.self, forKey: .depformerMultiLinear)
        depformerWeightsPerStep = try c.decodeIfPresent(Bool.self, forKey: .depformerWeightsPerStep)
        demuxSecondStream = try c.decodeIfPresent(Bool.self, forKey: .demuxSecondStream) ?? false
        crossAttention = try c.decodeIfPresent(Bool.self, forKey: .crossAttention) ?? false
        layerScale = try c.decodeIfPresent(Double.self, forKey: .layerScale)
        conditioners = try c.decodeIfPresent([String: Conditioner].self, forKey: .conditioners) ?? [:]
        fuser = try c.decodeIfPresent(Fuser.self, forKey: .fuser)
        directionTable = conditioners[DirectionTable.name]?.lut
    }

    /// The `config.json` of a staged model directory, checked by `validate`.
    public static func load(directory: URL) throws -> HibikiConfig {
        let url = directory.appending(path: "config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw HibikiError.missingFile(url)  // stage the model with convert_mlx.py
        }
        let cfg = try JSONDecoder().decode(HibikiConfig.self, from: Data(contentsOf: url))
        try cfg.validate()
        return cfg
    }

    /// Fails on any setting this runtime does not implement, instead of running a wrong model.
    func validate() throws {
        if let gating, gating != "silu" { throw HibikiError.unsupported("gating \(gating)") }
        if let norm, !norm.hasPrefix("rms_norm") { throw HibikiError.unsupported("norm \(norm)") }
        if !["rope", "rope_concat"].contains(positionalEmbedding) {
            throw HibikiError.unsupported("positional_embedding \(positionalEmbedding)")
        }
        if depformerPosEmb != "none" {
            throw HibikiError.unsupported("depformer_pos_emb \(depformerPosEmb)")
        }
        if depformerMultiLinear == false || depformerWeightsPerStep == false {
            throw HibikiError.unsupported("a depformer without per-step weights")
        }
        if demuxSecondStream { throw HibikiError.unsupported("demux_second_stream") }
        if crossAttention { throw HibikiError.unsupported("cross_attention") }
        if layerScale != nil { throw HibikiError.unsupported("layer_scale") }
        // The only condition this runtime runs is the direction table, summed into the input.
        if !conditioners.isEmpty {
            let name = DirectionTable.name
            guard Set(conditioners.keys) == [name], conditioners[name]?.type == "lut",
                let table = directionTable, table.tokenizer == "noop",
                table.nBins >= table.possibleValues.count, let fuser, fuser.sum == [name],
                (fuser.prepend ?? []).isEmpty, (fuser.cross ?? []).isEmpty
            else {
                throw HibikiError.unsupported(
                    "conditioners \(conditioners.keys.sorted()) summed \(fuser?.sum ?? []), "
                        + "prepended \(fuser?.prepend ?? []), cross-attended "
                        + "\(fuser?.cross ?? []): only the \(name) table summed into the input runs")
            }
        }
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
            dModel: dim, numHeads: numHeads, numLayers: numLayers, biasFF: false, biasAttn: false, layerScale: nil,
            positionalEmbedding: positionalEmbedding == "rope_concat" ? .ropeConcat : .rope,
            gating: true, norm: .rmsNorm, context: context, maxPeriod: Float(maxPeriod),
            kvRepeat: kvRepeat,
            dimFeedForward: Int((hiddenScale * Double(dim)).rounded()))
    }

    var depformer: TransformerConfig {
        TransformerConfig(
            dModel: depformerDim, numHeads: depformerNumHeads, numLayers: depformerNumLayers,
            biasFF: false, biasAttn: false, layerScale: nil, positionalEmbedding: .none,
            gating: true, norm: .rmsNorm, context: depQ, maxPeriod: 8, kvRepeat: 1,
            dimFeedForward: depformerDimFeedforward
                ?? Int((hiddenScale * Double(depformerDim)).rounded()))
    }
}

/// `quantization` of a model's `config.json` (`convert_mlx.py --bits` writes it): the group size
/// and bits of every Linear and Embedding, and an entry of its own for each layer quantized
/// otherwise (its group size and bits, or `false` to keep it unquantized), by its module path in
/// the checkpoint.
public struct Quantization: Decodable {
    public struct Spec: Decodable {
        public let groupSize: Int
        public let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    public let base: Spec
    public let layers: [String: Spec]
    /// Layers kept unquantized (`false`), such as the direction table's.
    public let unquantized: Set<String>

    public init(from decoder: Decoder) throws {
        base = try Spec(from: decoder)
        let c = try decoder.container(keyedBy: Key.self)
        var layers: [String: Spec] = [:]
        var unquantized: Set<String> = []
        for key in c.allKeys where !["group_size", "bits"].contains(key.stringValue) {
            if let quantized = try? c.decode(Bool.self, forKey: key) {
                guard !quantized else {
                    throw DecodingError.dataCorruptedError(
                        forKey: key, in: c, debugDescription: "a layer's entry is a spec or false")
                }
                unquantized.insert(key.stringValue)
            } else {
                layers[key.stringValue] = try c.decode(Spec.self, forKey: key)
            }
        }
        self.layers = layers
        self.unquantized = unquantized
    }

    /// How the layer at `path` is quantized; nil to keep it unquantized.
    func spec(_ path: String) -> Spec? { unquantized.contains(path) ? nil : layers[path] ?? base }
}

/// `lm_gen_config` of a model's `config.json`: how its text and audio tokens are sampled, as
/// training's evaluation and upstream's own inference sample them.
public struct LmGenConfig: Decodable {
    public let temp: Float
    public let tempText: Float
    public let topK: Int
    public let topKText: Int

    enum CodingKeys: String, CodingKey {
        case temp
        case tempText = "temp_text"
        case topK = "top_k"
        case topKText = "top_k_text"
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
                dim: dcfg.dModel, outputNorm: cfg.depformerOutputNorm)
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

    /// The checkpoint's path of a module under `depformer`: a core's modules are saved under
    /// the first slice that uses it, as `remap` reads them.
    func checkpointPath(_ path: String) -> String {
        let prefix = "depformer.cores."
        guard path.hasPrefix(prefix) else { return path }
        let rest = path.dropFirst(prefix.count)
        guard let dot = rest.firstIndex(of: "."), let core = Int(rest[..<dot]),
            let idx = schedule.firstIndex(of: core)
        else { return path }
        return "depformer.slices.\(idx)\(rest[dot...])"
    }
}

final class HibikiLM: Module {
    let cfg: HibikiConfig
    private let decoder: DecodeTransformer
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "depformer") var depformer: Depformer
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm
    @ModuleInfo(key: "text_linear") var textLinear: Linear
    @ModuleInfo(key: "audio_embs") var audioEmbs: [Embedding]
    /// What each direction adds to every frame's input, [1, 1, dim] (`directionConditions`);
    /// empty for a model without the direction condition. Not a parameter (the underscore).
    private var _conditions: [String: MLXArray] = [:]

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

    /// Loads `config.json` and the weights it names (`moshi_name`) of a staged model directory,
    /// quantized as its `quantization` says (bf16 without one), strictly.
    static func load(directory: URL) throws -> HibikiLM {
        let cfg = try HibikiConfig.load(directory: directory)
        let url = directory.appending(path: cfg.moshiName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw HibikiError.missingFile(url)
        }
        let model = HibikiLM(cfg)
        if let quantization = cfg.quantization {
            // Every Linear and Embedding, as pipeline.py `quantize`; the record's paths are the
            // checkpoint's, which name a depformer core by the first slice using it.
            quantize(model: model) { path, module in
                guard module is Quantizable,
                    let spec = quantization.spec(model.depformer.checkpointPath(path))
                else { return nil }
                return (spec.groupSize, spec.bits, .affine)
            }
        }
        var weights: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            if let key = model.depformer.remap(key) { weights[key] = value }
        }
        model._conditions = try directionConditions(cfg, from: &weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        eval(Array(model._conditions.values))
        return model
    }

    /// What each direction adds to every frame's input, computed once from the direction
    /// table's weights as moshi's LUT conditioner computes it: the projection of the
    /// direction's entry, `output_proj · embed[i]`. Takes the table's tensors out of `weights`,
    /// as no module holds them; its padding row and vector (`learnt_padding`), used only for a
    /// row without a direction, are never read.
    private static func directionConditions(
        _ cfg: HibikiConfig, from weights: inout [String: MLXArray]
    ) throws -> [String: MLXArray] {
        guard let table = cfg.directionTable else { return [:] }
        let prefix = "condition_provider.conditioners.\(DirectionTable.name)."
        func take(_ name: String, shape: [Int]) throws -> MLXArray {
            guard let array = weights.removeValue(forKey: prefix + name), array.shape == shape
            else {
                throw HibikiError.unsupported("weights without \(prefix + name) of shape \(shape)")
            }
            return array
        }
        let embed = try take("embed.weight", shape: [table.nBins + 1, table.dim])
        let projection = try take("output_proj.weight", shape: [cfg.dim, table.dim])
        _ = try take("learnt_padding", shape: [1, 1, cfg.dim])
        var conditions: [String: MLXArray] = [:]
        for (row, direction) in table.possibleValues.enumerated() {
            conditions[direction] = matmul(embed[row ..< row + 1], projection.T)
                .reshaped(1, 1, cfg.dim)
        }
        return conditions
    }

    /// What translating in `direction` adds to every frame's input. A model with the direction
    /// condition requires one of its directions; a model without it refuses any.
    func condition(_ direction: String?) throws -> MLXArray? {
        guard !_conditions.isEmpty else {
            if let direction {
                throw HibikiError.badDirection(
                    "this model has no direction condition (it hears the direction): "
                        + "drop the direction \(direction)")
            }
            return nil
        }
        guard let direction, let condition = _conditions[direction] else {
            throw HibikiError.badDirection(
                "this model is conditioned on the direction: give one of "
                    + cfg.directions.joined(separator: ", ")
                    + (direction.map { ", not \($0)" } ?? ""))
        }
        return condition
    }

    func resetCache() {
        decoder.reset()
    }

    /// One frame: the main transformer over the previous text token, the 32 delayed audio
    /// tokens and the direction's `condition` (`condition(_:)`), then the text token and the 16
    /// target codebooks.
    func step(
        text: Int, audio: [Int], condition: MLXArray?, textSampler: Sampler,
        audioSampler: Sampler
    ) -> (text: MLXArray, audio: MLXArray) {
        var xs = textEmb(MLXArray([Int32(text)]).reshaped(1, 1))
        for (token, emb) in zip(audio, audioEmbs) {
            xs = xs + emb(MLXArray([Int32(token)]).reshaped(1, 1))
        }
        if let condition { xs = xs + condition }
        let mainOut = outNorm(decoder(xs))
        let textToken = textSampler(textLinear(mainOut[0..., -1, 0...])).reshaped(1, 1)
        let audioTokens = depformer.sample(mainOut, textToken: textToken, sampler: audioSampler)
        return (textToken, audioTokens)
    }
}
