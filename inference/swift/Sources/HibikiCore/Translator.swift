import Foundation
import MLX

/// Streaming translation: source audio in, in whole 80 ms frames, and per frame a text token
/// and 80 ms of translated audio out, as `inference/hibiki_mlx/pipeline.py`. The source goes
/// through the session's `Audio.Leveller` before Mimi. Tokens are sampled as the model's
/// `config.json` says (`lm_gen_config`). A file's input ends with `endInput`; a live one never
/// does.
///
/// The LM runs on the GPU while Mimi encodes on the CPU and decodes on the Neural Engine
/// (`NeuralDecoder`), as the Python runtime's codec threads overlap the LM. A step takes k
/// frames: the LM runs k steps one after another, Mimi encodes the k frames in one call, and the
/// decoder turns each finished LM step's audio tokens into audio. LM step t reads source frames
/// before t, so the chunk is encoded beside the chunk's first LM step and the audio decoded
/// beside its last:
///
///     k = 1   GPU:  [LM t ·································]
///             CPU:  [decode t-1] [encode t] [wait]              → text t, audio t-1
///     k = 2   GPU:  [LM t ···················] [LM t+1 ··················]
///             CPU:  [encode t, t+1]   [wait]   [decode t-1, t]   [wait]   → text t, t+1; audio t-1, t
///
/// ("decode" is the driving thread waiting on the Neural Engine.) A frame is a few rows per op,
/// so the encoder's cost per call grows much slower than its length: k = 2 cuts its time per
/// frame by about a third, for up to (2k - 2) × 80 ms more latency (the chunk's first frame
/// waits for the rest, and the step may take k × 80 ms).
///
/// One thread drives both devices: mlx-swift holds a global lock while a thread evaluates, so
/// MLX work on two threads would take turns.
public final class Translator {
    public struct Output {
        /// One text token per input frame.
        public let textTokens: [Int]
        /// The text they complete: a character spread over byte-fallback pieces comes out with
        /// its last byte, in this step or a later one; special tokens add none.
        public let text: String
        /// Translated audio at 24 kHz decoded during this step, 1920 samples for each LM step
        /// that finished since the last decode (this step's last one comes with the next step);
        /// empty until the model has produced audio.
        public let pcm: [Float]
        public let timing: Timing
    }

    /// Seconds the driving thread spent in each part of one step, summed over its LM steps;
    /// they add up to the step's time. The LM runs on the GPU from each `lmStart` to the end of
    /// the `lmWait` after it.
    public struct Timing {
        /// Building the LM graphs and handing them to the GPU.
        public var lmStart = 0.0
        /// Mimi decoding the finished LM steps' audio tokens on the Neural Engine.
        public var decode = 0.0
        /// The level stage and Mimi encoding the input frames on the CPU.
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
    let decoder: NeuralDecoder
    let generator: Generator
    /// Complete target codes of finished LM steps, one frame each, not yet decoded.
    private var targets: [[Int]] = []
    private var leveller = Audio.Leveller()
    private var textDecoder: TextDecoder

    /// `cacheLimit` caps MLX's process-wide cache of freed buffers, which counts toward an iOS
    /// app's memory limit.
    public init(directory: URL, cacheLimit: Int? = nil) throws {
        // MLX closes a Metal command buffer every 20 ops or 40 MB on phones (40–50 on Macs)
        // and blocks `asyncEval` while more than 10 are in flight, so the LM step (hundreds of
        // ops over 0.6–2 GB of weights) would run almost entirely inside `asyncEval`, before
        // the CPU work instead of beside it. MLX reads these once, when it creates the Metal
        // device, which any array or `Memory` setting does (`MLXRandom.seed` too), so create
        // the translator before anything else touches MLX. Values already in the environment
        // win, for tuning.
        setenv("MLX_MAX_OPS_PER_BUFFER", "400", 0)
        setenv("MLX_MAX_MB_PER_BUFFER", "1000000", 0)
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
        vocabulary = try Vocabulary(url: file(config.tokenizerName))
        if vocabulary.pieces.count != config.textCard {
            throw HibikiError.badTokenizer(
                "\(vocabulary.pieces.count) pieces for text_card \(config.textCard)")
        }
        textDecoder = TextDecoder(vocabulary)
        mimi = try Mimi.load(
            url: file(config.mimiName ?? "mimi-pytorch-e351c8d8@125.safetensors"),
            numCodebooks: config.nQ - config.depQ)
        decoder = try NeuralDecoder(url: file("mimi-decoder.mlmodelc"))
        if decoder.codebooks != config.depQ {
            throw HibikiError.unsupported(
                "a \(decoder.codebooks)-codebook Mimi decoder for dep_q \(config.depQ)")
        }
        let sampling = config.lmGenConfig
        generator = Generator(
            lm, textSampler: Sampler(temp: sampling.tempText, topK: sampling.topKText),
            audioSampler: Sampler(temp: sampling.temp, topK: sampling.topK))
        try warmup()
    }

    /// Runs silent frames until every stage, Mimi decode included, has run once, so kernels
    /// are built and weights read before timing starts.
    private func warmup() throws {
        while try step([Float](repeating: 0, count: Audio.frameSize)).pcm.isEmpty {}
        reset()
        Memory.clearCache()  // return load and warmup buffers instead of keeping them cached
    }

    /// Starts a new session: clears the level stage, the codec, the LM state and the text,
    /// dropping audio and text not yet returned (`flush` first to keep them).
    public func reset() {
        targets = []
        leveller = Audio.Leveller()
        textDecoder = TextDecoder(vocabulary)
        mimi.resetState()
        decoder.reset()
        generator.reset()
    }

    /// Translates k whole frames (k × 1920 samples): k LM steps, one Mimi encode of the k
    /// frames, and the decoding of every LM step finished before the last. Bad audio
    /// (`HibikiError.badAudio`) throws with the session as it was; after any other error the
    /// session is broken: `reset` before the next step.
    public func step(_ pcm: [Float]) throws -> Output {
        try step(pcm, endingInput: false)
    }

    /// Ends a file's input as training's evaluation and upstream's offline inference do: one
    /// frame whose source codes are all the source EOS (`card`), while Mimi codes a silent
    /// frame there, as a whole-file encode codes the padding. Feed silence after it.
    public func endInput() throws -> Output {
        try step([Float](repeating: 0, count: Audio.frameSize), endingInput: true)
    }

    private func step(_ pcm: [Float], endingInput: Bool) throws -> Output {
        let frames = pcm.count / Audio.frameSize
        precondition(frames > 0 && pcm.count == frames * Audio.frameSize)
        var timing = Timing()
        var clock = DispatchTime.now()
        func lap() -> Double {
            let now = DispatchTime.now()
            defer { clock = now }
            return seconds(clock, now)
        }
        // Levelled before any state moves, on a copy kept only if it succeeds: bad audio
        // throws with the session as it was.
        var next = leveller
        let source = try next(pcm)
        leveller = next
        timing.encode += lap()

        var sources: [[Int]] = []
        var audio: [Float] = []
        var tokens: [Int] = []
        for i in 0..<frames {
            let pending = generator.start()
            timing.lmStart += lap()
            if i == frames - 1 {
                audio = try decodeTargets()
                timing.decode += lap()
            }
            if i == 0 {
                sources = encode(source)
                timing.encode += lap()
            }
            let codes = endingInput
                ? Array(repeating: config.card, count: config.nQ - config.depQ) : sources[i]
            let out = generator.finish(pending, source: codes)
            timing.lmWait += lap()
            tokens.append(out.text)
            if let target = out.target { targets.append(target) }
        }
        return Output(textTokens: tokens, text: textDecoder(tokens), pcm: audio, timing: timing)
    }

    /// What the session still holds when the input ends: the audio of the finished LM steps
    /// that no step has returned yet, and the text of a character the model left unfinished
    /// (U+FFFD); both empty if there is none. If it throws, `reset` before the next step.
    public func flush() throws -> (pcm: [Float], text: String) {
        (try decodeTargets(), textDecoder.flush())
    }

    /// Decodes the audio of the finished LM steps that no step has returned yet.
    private func decodeTargets() throws -> [Float] {
        var pcm: [Float] = []
        while let target = targets.first {
            pcm += try decoder.decode(target)
            targets.removeFirst()
        }
        return pcm
    }

    /// Source codes of each frame, encoded on the CPU in one call.
    private func encode(_ pcm: [Float]) -> [[Int]] {
        let frames = pcm.count / Audio.frameSize
        return Device.withDefaultDevice(.cpu) {
            let codes = mimi.encodeStep(StreamArray(MLXArray(pcm).reshaped(1, 1, pcm.count)))
            guard let codes = codes.asArray(), codes.dim(2) == frames else {
                fatalError("Mimi produced \(codes.shape ?? []) codes for \(frames) frames")
            }
            let sourceQ = config.nQ - config.depQ
            let flat = codes[0, 0..<sourceQ].transposed().asArray(Int32.self).map(Int.init)
            return (0..<frames).map { Array(flat[($0 * sourceQ)..<(($0 + 1) * sourceQ)]) }
        }
    }
}
