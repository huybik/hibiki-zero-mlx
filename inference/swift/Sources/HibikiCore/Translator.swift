import CoreML
import Foundation
import MLX

/// Streaming translation: source audio in, one 80 ms frame per step, and per frame a text token
/// and 80 ms of translated audio out, as `inference/hibiki_mlx/pipeline.py`. The source goes
/// through the session's `Audio.Leveller` before Mimi. Tokens are sampled as the model's
/// `config.json` says (`lm_gen_config`). A file's input ends with `endInput`; a live one never
/// does.
///
/// The LM is the Core ML LM `export_lm_coreml.py` writes (`NeuralLM`), on its own thread and on
/// the Neural Engine unless told otherwise; beside it Mimi encodes on the GPU, which the LM
/// leaves idle, and decodes on the Neural Engine (`NeuralDecoder`). LM step t reads source frames
/// before t, so frame t is encoded beside it and the audio of step t - 1 decoded:
///
///     LM:   [LM t ··························································]
///     GPU:            [encode t ··············]
///     CPU:  [start LM t] [submit encode t] [decode t-1] [encode wait] [LM wait]
///                                                         → text t, audio t-1
///
/// ("decode" is the driving thread waiting on the Neural Engine.) The encode is submitted first:
/// the LM holds the Neural Engine at the start of its step, and the decoder would queue behind
/// it while the GPU sat idle. Its codes are collected after the decode (`encodeWait`).
public final class Translator {
    public struct Output {
        /// The frame's text token.
        public let textToken: Int
        /// The text it completes: a character spread over byte-fallback pieces comes out with
        /// its last byte, in this step or a later one; special tokens add none.
        public let text: String
        /// Translated audio at 24 kHz decoded during this step, 1920 samples for each LM step
        /// that finished since the last decode (this step's comes with the next step); empty
        /// until the model has produced audio.
        public let pcm: [Float]
        public let timing: Timing
    }

    /// Seconds the driving thread spent in each part of a step; they add up to the step's time.
    /// The LM runs on its own thread from `lmStart` to the end of `lmWait`.
    public struct Timing {
        /// Queueing the LM's frame on its thread.
        public var lmStart = 0.0
        /// Mimi decoding the finished LM steps' audio tokens on the Neural Engine.
        public var decode = 0.0
        /// The level stage, and building and submitting Mimi's encode to the GPU.
        public var encode = 0.0
        /// Waiting for the GPU's encode after the decode.
        public var encodeWait = 0.0
        /// Waiting for the LM after the rest.
        public var lmWait = 0.0
        public var total: Double { lmStart + decode + encode + encodeWait + lmWait }
        /// The LM's own time on its thread, overlapping the parts above: its main transformer,
        /// its audio head, and the host's embedding sums and sampling.
        public var lmTrunk = 0.0
        public var lmHeads = 0.0
        public var lmHost = 0.0

        public init() {}

        public static func += (lhs: inout Timing, rhs: Timing) {
            lhs.lmStart += rhs.lmStart
            lhs.decode += rhs.decode
            lhs.encode += rhs.encode
            lhs.encodeWait += rhs.encodeWait
            lhs.lmWait += rhs.lmWait
            lhs.lmTrunk += rhs.lmTrunk
            lhs.lmHeads += rhs.lmHeads
            lhs.lmHost += rhs.lmHost
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

    /// Where the LM may run, by the names the CLI's and the app's `--lm-units` take.
    public static let lmComputeUnitNames: [String: MLComputeUnits] = [
        "ane": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly, "all": .all,
    ]

    /// `cacheLimit` caps MLX's process-wide cache of freed buffers, which counts toward an iOS
    /// app's memory limit; `lmComputeUnits` is where the LM runs.
    public init(
        directory: URL, cacheLimit: Int? = nil,
        lmComputeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws {
        // MLX closes a Metal command buffer every 20 ops or 40 MB on phones (40–50 on Macs)
        // and blocks `asyncEval` while more than 10 are in flight, so GPU work of hundreds of
        // ops would run largely inside `asyncEval`, before the CPU work instead of beside it.
        // MLX reads these once, when it creates the Metal device, which any array or `Memory`
        // setting does, so create the translator before anything else touches MLX. Values
        // already in the environment win, for tuning.
        setenv("MLX_MAX_OPS_PER_BUFFER", "400", 0)
        setenv("MLX_MAX_MB_PER_BUFFER", "1000000", 0)
        if let cacheLimit { Memory.cacheLimit = cacheLimit }
        config = try HibikiConfig.load(directory: directory)
        let lm = try NeuralLM(directory: directory, cfg: config, computeUnits: lmComputeUnits)
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
        // The decoder may read only the first of the LM's codebooks: the slices student's 8–15,
        // emitted together by its last slice, add noise to its voice (slices-phone.md#voice).
        decoder = try NeuralDecoder(url: file("mimi-decoder.mlmodelc"))
        if decoder.codebooks > config.depQ {
            throw HibikiError.unsupported(
                "a \(decoder.codebooks)-codebook Mimi decoder for dep_q \(config.depQ)")
        }
        generator = Generator(lm)
        try warmup()
    }

    /// Runs silent frames until every stage, Mimi decode included, has run once, so kernels
    /// are built and weights read before timing starts.
    private func warmup() throws {
        while try step([Float](repeating: 0, count: Audio.frameSize)).pcm.isEmpty {}
        reset()
        Memory.clearCache()  // return load and warmup buffers instead of keeping them cached
    }

    /// Seeds the LM's sampling.
    public func seed(_ value: UInt64) {
        generator.lm.seed(value)
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

    /// Translates one whole frame (1920 samples): its LM step, its Mimi encode, and the decoding
    /// of every LM step finished before it. Bad audio (`HibikiError.badAudio`) throws with the
    /// session as it was; after any other error the session is broken: `reset` before the next
    /// step.
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
        precondition(pcm.count == Audio.frameSize, "a step takes one frame")
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

        let pending = generator.start()
        timing.lmStart += lap()
        let encoding = startEncode(source)
        timing.encode += lap()
        let audio = try decodeTargets()
        timing.decode += lap()
        let encoded = sourceCodes(encoding)
        timing.encodeWait += lap()
        let codes = endingInput
            ? Array(repeating: config.card, count: config.nQ - config.depQ) : encoded
        let out = try generator.finish(pending, source: codes)
        timing.lmWait += lap()
        timing.lmTrunk += out.busy.trunk
        timing.lmHeads += out.busy.heads
        timing.lmHost += out.busy.host
        if let target = out.target { targets.append(target) }
        return Output(
            textToken: out.text, text: textDecoder([out.text]), pcm: audio, timing: timing)
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
            pcm += try decoder.decode(Array(target.prefix(decoder.codebooks)))
            targets.removeFirst()
        }
        return pcm
    }

    /// Submits Mimi's encode of the frame; it runs on the GPU while the driving thread goes on.
    private func startEncode(_ pcm: [Float]) -> MLXArray {
        let codes = mimi.encodeStep(pcm)[0].asType(.int32)
        asyncEval(codes)
        return codes
    }

    /// The frame's source codes, once `startEncode`'s work is done.
    private func sourceCodes(_ codes: MLXArray) -> [Int] {
        guard codes.shape == [config.nQ - config.depQ, 1] else {
            fatalError("Mimi produced \(codes.shape) codes for one frame")
        }
        return codes.asArray(Int32.self).map(Int.init)
    }
}
