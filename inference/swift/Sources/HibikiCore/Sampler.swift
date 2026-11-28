import Foundation

/// Top-k sampling at a temperature, as `hibiki_mlx.lm.sampling.Sampler(temp:, top_k:)`, on
/// logits in host memory (the Core ML LM's). A temperature of 0 takes the argmax.
public struct Sampler {
    public let temp: Float
    public let topK: Int

    public init(temp: Float, topK: Int) {
        precondition(temp == 0 || topK > 0, "sampling needs a top-k")
        self.temp = temp
        self.topK = topK
    }

    /// Exactly the `topK` largest at the temperature (of logits tied with the k-th, the first),
    /// the rest dropped.
    func callAsFunction(_ logits: [Float], using rng: inout some RandomNumberGenerator) -> Int {
        var best = 0
        for i in logits.indices where logits[i] > logits[best] { best = i }
        if temp == 0 { return best }
        var threshold = -Float.infinity
        // Logits equal to the threshold that may still be kept: ties with the k-th largest.
        var ties = Int.max
        if topK < logits.count {
            var scratch = logits
            threshold = Self.largest(topK, of: &scratch)
            ties = topK - logits.lazy.filter { $0 > threshold }.count
        }
        let top = logits[best]
        var kept: [(index: Int, weight: Float)] = []
        kept.reserveCapacity(topK)
        var total: Float = 0
        for i in logits.indices where logits[i] >= threshold {
            if logits[i] == threshold {
                if ties == 0 { continue }
                ties -= 1
            }
            let weight = Foundation.exp((logits[i] - top) / temp)
            kept.append((i, weight))
            total += weight
        }
        var u = Float.random(in: 0..<1, using: &rng) * total
        for (index, weight) in kept {
            u -= weight
            if u < 0 { return index }
        }
        return best  // rounding left u at or above zero
    }

    /// The k-th largest of `xs` (k from 1), which it reorders: quickselect.
    static func largest(_ k: Int, of xs: inout [Float]) -> Float {
        var (lo, hi, target) = (0, xs.count - 1, k - 1)
        while lo < hi {
            let pivot = xs[(lo + hi) / 2]
            var (i, j) = (lo, hi)
            while i <= j {
                while xs[i] > pivot { i += 1 }
                while xs[j] < pivot { j -= 1 }
                if i <= j {
                    xs.swapAt(i, j)
                    i += 1
                    j -= 1
                }
            }
            if target <= j {
                hi = j
            } else if target >= i {
                lo = i
            } else {
                break
            }
        }
        return xs[target]
    }
}

/// SplitMix64: a small seedable generator for the CPU sampling.
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
