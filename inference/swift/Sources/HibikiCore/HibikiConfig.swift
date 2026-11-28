// A model directory's `config.json` as this runtime reads it: the shapes of the Core ML LM that
// `inference/tools/export_lm_coreml.py` wrote beside it (`NeuralLM`), its delays, its sampling
// and its files.

import Foundation

public enum HibikiError: Error, CustomStringConvertible {
    case unsupported(String)
    case missingFile(URL)
    case badTokenizer(String)
    case badAudio(String)

    public var description: String {
        switch self {
        case .unsupported(let what): "unsupported model config: \(what)"
        case .missingFile(let url): "missing \(url.path)"
        case .badTokenizer(let what): "bad tokenizer: \(what)"
        case .badAudio(let what): "bad audio: \(what)"
        }
    }
}

/// The subset of a Hibiki-Zero `config.json` this runtime reads. The LM is the Core ML models
/// `export_lm_coreml.py` writes (`lm_coreml`), which refuses any setting `lm_step.py` does not
/// implement.
public struct HibikiConfig: Decodable {
    public let card: Int
    public let nQ: Int
    public let depQ: Int
    public let delays: [Int]
    public let dim: Int
    public let textCard: Int
    public let numHeads: Int
    public let context: Int
    public let maxPeriod: Double
    public let depformerDim: Int
    public let mimiName: String?
    public let tokenizerName: String
    public let lmGenConfig: LmGenConfig
    /// "parallel" for the parallel-heads student, whose 16 codebooks come from linear heads in
    /// the depformer's place; "depformer_parallel" for a depformer whose last slice emits the
    /// last codebooks at once.
    public let audioHeads: String?
    /// Top-k baked into `lm-depformer.mlmodelc` (nil for full-vocab sampling).
    public let lmCoreMLDepformerTopK: Int?
    /// Whether `export_lm_coreml.py` wrote this directory (`lm_coreml`).
    let lmCoreML: Bool

    enum CodingKeys: String, CodingKey {
        case card, dim, context, delays
        case nQ = "n_q"
        case depQ = "dep_q"
        case textCard = "text_card"
        case numHeads = "num_heads"
        case maxPeriod = "max_period"
        case depformerDim = "depformer_dim"
        case mimiName = "mimi_name"
        case tokenizerName = "tokenizer_name"
        case lmGenConfig = "lm_gen_config"
        case audioHeads = "audio_heads"
        case lmCoreML = "lm_coreml"
    }

    private struct LMCoreMLRecord: Decodable {
        let depformerTopK: Int?
        enum CodingKeys: String, CodingKey {
            case depformerTopK = "depformer_top_k"
        }
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
        context = try c.decode(Int.self, forKey: .context)
        maxPeriod = try c.decode(Double.self, forKey: .maxPeriod)
        depformerDim = try c.decode(Int.self, forKey: .depformerDim)
        mimiName = try c.decodeIfPresent(String.self, forKey: .mimiName)
        tokenizerName = try c.decode(String.self, forKey: .tokenizerName)
        lmGenConfig = try c.decode(LmGenConfig.self, forKey: .lmGenConfig)
        audioHeads = try c.decodeIfPresent(String.self, forKey: .audioHeads)
        lmCoreML = c.contains(.lmCoreML)
        lmCoreMLDepformerTopK = (try? c.decodeIfPresent(LMCoreMLRecord.self, forKey: .lmCoreML))?.depformerTopK
    }

    /// A model directory's `config.json`, checked by `validate`.
    static func load(directory: URL) throws -> HibikiConfig {
        let url = try directory.modelFile("config.json")  // export the model with export_lm_coreml.py
        let cfg = try JSONDecoder().decode(HibikiConfig.self, from: Data(contentsOf: url))
        try cfg.validate()
        return cfg
    }

    /// Fails on a directory this runtime cannot run, instead of running a wrong model.
    func validate() throws {
        guard lmCoreML else {
            throw HibikiError.unsupported("no lm_coreml: export the LM with export_lm_coreml.py")
        }
        guard let audioHeads, ["parallel", "depformer_parallel"].contains(audioHeads) else {
            throw HibikiError.unsupported("audio_heads \(audioHeads ?? "missing")")
        }
        if delays.count != nQ + 1 { throw HibikiError.unsupported("\(delays.count) delays") }
    }

    var audioDelays: [Int] { Array(delays.dropFirst()) }  // delays[0] is the text stream's
    var audioPaddingToken: Int { card }  // audio vocabulary is card + 1
    var textInitToken: Int { textCard }  // text input vocabulary is text_card + 1
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

extension URL {
    /// The file `name` of this model directory, which must be there.
    func modelFile(_ name: String) throws -> URL {
        let url = appending(path: name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw HibikiError.missingFile(url)
        }
        return url
    }
}
