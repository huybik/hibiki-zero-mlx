import Foundation
import MLX

/// Streaming translation: 80 ms of source audio in, a text token and 80 ms of translated
/// audio out, one frame at a time, as `inference/hibiki_mlx/pipeline.py` (sequentially: Mimi
/// runs on the GPU between LM steps instead of on CPU threads).
public final class Translator {
    public struct Frame {
        public let textToken: Int
        /// The text this frame adds, nil for special tokens.
        public let text: String?
        /// 1920 samples at 24 kHz once the model has produced audio, else nil.
        public let pcm: [Float]?
        public let timing: Timing
    }

    /// Seconds spent in each stage of one frame.
    public struct Timing {
        public var encode = 0.0
        public var main = 0.0
        public var depformer = 0.0
        public var decode = 0.0
        public var total: Double { encode + main + depformer + decode }

        public init() {}
    }

    public let config: HibikiConfig
    public let vocabulary: Vocabulary
    let mimi: Mimi
    let generator: Generator

    /// Upstream sampling (`pipeline.py`): text at 0.4 with top-k 25, audio at 0.8 with top-k 250.
    public init(
        directory: URL,
        textSampler: Sampler = Sampler(temp: 0.4, topK: 25),
        audioSampler: Sampler = Sampler(temp: 0.8, topK: 250)
    ) throws {
        let lm = try HibikiLM.load(directory: directory)
        config = lm.cfg
        let file = { (name: String) throws -> URL in
            let url = directory.appending(path: name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw HibikiError.missingFile(url)
            }
            return url
        }
        vocabulary = try Vocabulary(
            url: file(config.tokenizerName ?? "tokenizer_spm_48k_multi6_2.model"))
        if vocabulary.pieces.count != config.textCard {
            throw HibikiError.badTokenizer(
                "\(vocabulary.pieces.count) pieces for text_card \(config.textCard)")
        }
        mimi = try Mimi.load(
            url: file(config.mimiName ?? "mimi-pytorch-e351c8d8@125.safetensors"),
            numCodebooks: max(config.depQ, config.nQ - config.depQ))
        generator = Generator(lm, textSampler: textSampler, audioSampler: audioSampler)
        warmup()
    }

    /// Runs one frame through every stage so kernels are built before timing starts.
    private func warmup() {
        _ = step([Float](repeating: 0, count: Audio.frameSize))
        reset()
        Memory.clearCache()  // return load and warmup buffers instead of keeping them cached
    }

    /// Starts a new session: clears the codec and the LM state.
    public func reset() {
        mimi.resetState()
        generator.reset()
    }

    public func step(_ pcm: [Float]) -> Frame {
        precondition(pcm.count == Audio.frameSize)
        var timing = Timing()
        var start = DispatchTime.now()
        let codes = mimi.encodeStep(StreamArray(MLXArray(pcm).reshaped(1, 1, Audio.frameSize)))
        guard let codes = codes.asArray(), codes.dim(2) == 1 else {
            fatalError("Mimi produced \(codes.shape ?? []) codes for one frame")
        }
        let source = codes[0, 0..<(config.nQ - config.depQ), 0].asArray(Int32.self).map(Int.init)
        timing.encode = seconds(start, DispatchTime.now())

        let out = generator.step(source: source)
        timing.main = out.main
        timing.depformer = out.depformer

        var samples: [Float]? = nil
        if let target = out.target {
            start = DispatchTime.now()
            let tokens = MLXArray(target.map(Int32.init)).reshaped(1, config.depQ, 1)
            if let decoded = mimi.decodeStep(StreamArray(tokens)).asArray() {
                samples = decoded.reshaped(-1).asArray(Float.self)
            }
            timing.decode = seconds(start, DispatchTime.now())
        }
        return Frame(
            textToken: out.text, text: vocabulary.text(out.text), pcm: samples, timing: timing)
    }
}
