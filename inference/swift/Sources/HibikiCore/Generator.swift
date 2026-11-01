import Foundation
import MLX

/// The delayed-stream state of one translation session, as `moshi_mlx.models.LmGen`.
///
/// Frame t feeds the model the previous text token and, for each of the 32 audio codebooks,
/// the token of frame t - 1 - delay. Source codes enter at their own frame; generated target
/// codes are stored at frame t - delay. Only the last `maxDelay + 2` frames are ever read or
/// written, so they live in a ring and a session can run indefinitely.
public final class Generator {
    private static let ungenerated = -2

    let lm: HibikiLM
    let textSampler: Sampler
    let audioSampler: Sampler
    private let delays: [Int]
    private let maxDelay: Int
    private var ring: [[Int]]
    public private(set) var frame = 0

    public init(_ lm: HibikiLM, textSampler: Sampler, audioSampler: Sampler) {
        self.lm = lm
        self.textSampler = textSampler
        self.audioSampler = audioSampler
        delays = lm.cfg.audioDelays
        maxDelay = delays.max() ?? 0
        ring = Array(
            repeating: Array(repeating: Self.ungenerated, count: 1 + lm.cfg.nQ),
            count: maxDelay + 2)
    }

    public func reset() {
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

    /// Feeds one frame of source codes. Returns the text token, the target codes now complete
    /// (frame - maxDelay; nil while warming up or when any is padding) and stage times.
    public func step(source: [Int]) -> (
        text: Int, target: [Int]?, main: Double, depformer: Double
    ) {
        let cfg = lm.cfg
        precondition(source.count == cfg.nQ - cfg.depQ)
        let t = frame
        ring[slot(t)] = ring[slot(t)].map { _ in Self.ungenerated }
        for (j, code) in source.enumerated() { ring[slot(t)][1 + cfg.depQ + j] = code }

        let textIn = t == 0 ? cfg.textInitToken : read(t - 1, 0)
        let audioIn = delays.enumerated().map { cb, delay in
            t - 1 - delay >= 0 ? read(t - 1 - delay, 1 + cb) : cfg.audioPaddingToken
        }
        let start = DispatchTime.now()
        let (textToken, audioTokens) = lm.step(
            text: textIn, audio: audioIn, textSampler: textSampler, audioSampler: audioSampler)
        let text = Int(textToken.item(Int32.self))
        let mainDone = DispatchTime.now()
        let audio = audioTokens.asArray(Int32.self).map(Int.init)
        let end = DispatchTime.now()

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
        return (text, target, seconds(start, mainDone), seconds(mainDone, end))
    }
}

func seconds(_ start: DispatchTime, _ end: DispatchTime) -> Double {
    Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
}
