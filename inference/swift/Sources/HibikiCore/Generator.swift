import Foundation
import MLX

/// The delayed-stream state of one translation session, as `moshi_mlx.models.LmGen`.
///
/// Frame t feeds the model the previous text token and, for each of the 32 audio codebooks,
/// the token of frame t - 1 - delay. Source codes enter at their own frame; generated target
/// codes are stored at frame t - delay. Only the last `maxDelay + 2` frames are ever read or
/// written, so they live in a ring and a session can run indefinitely. Every frame's input also
/// gets the direction's `condition`, for a model with the direction condition.
final class Generator {
    private static let ungenerated = -2

    let lm: HibikiLM
    let condition: MLXArray?
    let textSampler: Sampler
    let audioSampler: Sampler
    private let delays: [Int]
    private let maxDelay: Int
    private var ring: [[Int]]
    private(set) var frame = 0

    init(_ lm: HibikiLM, condition: MLXArray?, textSampler: Sampler, audioSampler: Sampler) {
        self.lm = lm
        self.condition = condition
        self.textSampler = textSampler
        self.audioSampler = audioSampler
        delays = lm.cfg.audioDelays
        maxDelay = delays.max() ?? 0
        ring = Array(
            repeating: Array(repeating: Self.ungenerated, count: 1 + lm.cfg.nQ),
            count: maxDelay + 2)
    }

    func reset() {
        frame = 0
        for i in ring.indices { ring[i] = ring[i].map { _ in Self.ungenerated } }
        lm.resetCache()
    }

    private func slot(_ frame: Int) -> Int { frame % ring.count }

    private func read(_ frame: Int, _ field: Int) -> Int {
        let value = ring[slot(frame)][field]
        precondition(value != Self.ungenerated, "frame \(frame) field \(field) not generated")
        return value
    }

    /// LM step `frame` running on the GPU, from `start` until `finish` reads it back.
    struct Pending {
        let frame: Int
        let text: MLXArray
        let audio: MLXArray
    }

    /// Builds LM step t (t = `frame`) from the tokens of frames before t and starts it on the
    /// GPU. Frame t's own source codes are first read by step t + 1, so the caller can encode
    /// them while this runs.
    func start() -> Pending {
        let cfg = lm.cfg
        let t = frame
        ring[slot(t)] = ring[slot(t)].map { _ in Self.ungenerated }
        let textIn = t == 0 ? cfg.textInitToken : read(t - 1, 0)
        let audioIn = delays.enumerated().map { cb, delay in
            t - 1 - delay >= 0 ? read(t - 1 - delay, 1 + cb) : cfg.audioPaddingToken
        }
        let (text, audio) = lm.step(
            text: textIn, audio: audioIn, condition: condition, textSampler: textSampler,
            audioSampler: audioSampler)
        asyncEval(text, audio)
        return Pending(frame: t, text: text, audio: audio)
    }

    /// Stores frame t's source codes, waits for step t and stores its tokens. Returns the text
    /// token and the target codes now complete (frame t - maxDelay; nil while warming up or
    /// when any is padding).
    func finish(_ pending: Pending, source: [Int]) -> (text: Int, target: [Int]?) {
        let cfg = lm.cfg
        let t = frame
        precondition(pending.frame == t, "finishing step \(pending.frame) at frame \(t)")
        precondition(source.count == cfg.nQ - cfg.depQ)
        for (j, code) in source.enumerated() { ring[slot(t)][1 + cfg.depQ + j] = code }

        let text = Int(pending.text.item(Int32.self))
        let audio = pending.audio.asArray(Int32.self).map(Int.init)
        ring[slot(t)][0] = text
        for cb in 0..<cfg.depQ where t - delays[cb] >= 0 {
            ring[slot(t - delays[cb])][1 + cb] = audio[cb]
        }
        frame += 1

        var target: [Int]? = nil
        let done = t - maxDelay
        if done >= 0 {
            let codes = (1...cfg.depQ).map { read(done, $0) }
            if !codes.contains(cfg.audioPaddingToken) { target = codes }
        }
        return (text, target)
    }
}

func seconds(_ start: DispatchTime, _ end: DispatchTime) -> Double {
    Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
}
