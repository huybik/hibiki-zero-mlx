import Foundation
import MLX

/// Streaming translation: 80 ms of source audio in, a text token and 80 ms of translated
/// audio out, one frame at a time, as `inference/hibiki_mlx/pipeline.py`.
///
/// Mimi runs on the CPU while the GPU runs the LM, as the Python runtime's codec threads do.
/// LM step t reads source frames before t, so frame t is encoded while step t runs, and the
/// audio of step t - 1 is decoded then too:
///
///     GPU:  [LM step t ·····································]
///     CPU:  [start] [decode step t-1] [encode frame t] [wait]  → text t, audio of step t-1
///
/// One thread drives both devices: mlx-swift holds a global lock while a thread evaluates, so
/// MLX work on two threads would take turns.
public final class Translator {
    public struct Frame {
        public let textToken: Int
        /// The text this frame adds, nil for special tokens.
        public let text: String?
        /// 1920 samples at 24 kHz, decoded from the previous step's audio tokens, once the
        /// model has produced audio, else nil.
        public let pcm: [Float]?
        public let timing: Timing
    }

    /// Seconds the driving thread spent in each part of one step, in order; they sum to the
    /// step's time. The LM runs on the GPU from `lmStart` to the end of `lmWait`.
    public struct Timing {
        /// Building the LM graph and handing it to the GPU.
        public var lmStart = 0.0
        /// Mimi decoding the previous step's audio tokens on the CPU.
        public var decode = 0.0
        /// Mimi encoding this frame on the CPU.
        public var encode = 0.0
        /// Waiting for the LM after the CPU work.
        public var lmWait = 0.0
        public var total: Double { lmStart + decode + encode + lmWait }

        public init() {}

        public static func += (lhs: inout Timing, rhs: Timing) {
            lhs.lmStart += rhs.lmStart
            lhs.decode += rhs.decode
            lhs.encode += rhs.encode
            lhs.lmWait += rhs.lmWait
        }
    }

    public let config: HibikiConfig
    public let vocabulary: Vocabulary
    let mimi: Mimi
    let generator: Generator
    /// The previous step's complete target codes, decoded during this step.
    private var target: [Int]?

    /// Upstream sampling (`pipeline.py`): text at 0.4 with top-k 25, audio at 0.8 with top-k 250.
    /// `cacheLimit` caps MLX's process-wide cache of freed buffers, which counts toward an iOS
    /// app's memory limit.
    public init(
        directory: URL,
        textSampler: Sampler = Sampler(temp: 0.4, topK: 25),
        audioSampler: Sampler = Sampler(temp: 0.8, topK: 250),
        cacheLimit: Int? = nil
    ) throws {
        // MLX closes a Metal command buffer every 20 ops or 40 MB on phones (40–50 on Macs)
        // and blocks `asyncEval` while more than 10 are in flight, so the LM step (hundreds of
        // ops over 0.6–2 GB of weights) would run almost entirely inside `asyncEval`, before
        // the CPU work instead of beside it. MLX reads these when it creates the Metal device,
        // so the translator must come before any other MLX GPU work in the process.
        setenv("MLX_MAX_OPS_PER_BUFFER", "400", 1)
        setenv("MLX_MAX_MB_PER_BUFFER", "1000000", 1)
        if let cacheLimit { Memory.cacheLimit = cacheLimit }
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

    /// Runs silent frames until every stage, Mimi decode included, has run once, so kernels
    /// are built and weights read before timing starts.
    private func warmup() {
        while step([Float](repeating: 0, count: Audio.frameSize)).pcm == nil {}
        reset()
        Memory.clearCache()  // return load and warmup buffers instead of keeping them cached
    }

    /// Starts a new session: clears the codec and the LM state.
    public func reset() {
        target = nil
        mimi.resetState()
        generator.reset()
    }

    public func step(_ pcm: [Float]) -> Frame {
        precondition(pcm.count == Audio.frameSize)
        var timing = Timing()
        var clock = DispatchTime.now()
        func lap() -> Double {
            let now = DispatchTime.now()
            defer { clock = now }
            return seconds(clock, now)
        }

        let pending = generator.start()
        timing.lmStart = lap()
        let samples = target.flatMap(decode)
        timing.decode = lap()
        let source = encode(pcm)
        timing.encode = lap()
        let out = generator.finish(pending, source: source)
        timing.lmWait = lap()
        target = out.target
        return Frame(
            textToken: out.text, text: vocabulary.text(out.text), pcm: samples, timing: timing)
    }

    /// Frame t's source codes, encoded on the CPU.
    private func encode(_ pcm: [Float]) -> [Int] {
        Device.withDefaultDevice(.cpu) {
            let codes = mimi.encodeStep(
                StreamArray(MLXArray(pcm).reshaped(1, 1, Audio.frameSize)))
            guard let codes = codes.asArray(), codes.dim(2) == 1 else {
                fatalError("Mimi produced \(codes.shape ?? []) codes for one frame")
            }
            return codes[0, 0..<(config.nQ - config.depQ), 0].asArray(Int32.self).map(Int.init)
        }
    }

    /// 80 ms of audio from one frame of target codes, decoded on the CPU.
    private func decode(_ target: [Int]) -> [Float]? {
        Device.withDefaultDevice(.cpu) {
            let tokens = MLXArray(target.map(Int32.init)).reshaped(1, config.depQ, 1)
            return mimi.decodeStep(StreamArray(tokens)).asArray()?.reshaped(-1).asArray(Float.self)
        }
    }
}
