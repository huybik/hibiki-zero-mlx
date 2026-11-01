import MLX
import MLXRandom

/// Top-k sampling at a temperature, as `moshi_mlx.utils.sampling.Sampler(top_k:, temp:)`.
/// A temperature of 0 takes the argmax.
public struct Sampler {
    public let temp: Float
    public let topK: Int

    public init(temp: Float, topK: Int) {
        self.temp = temp
        self.topK = topK
    }

    /// Logits `[1, V]` to one int32 token `[1]`.
    func callAsFunction(_ logits: MLXArray) -> MLXArray {
        if temp == 0 {
            return argMax(logits, axis: -1).asType(.int32)
        }
        var scaled = logits * (1 / temp)
        if topK > 0 && topK < logits.dim(-1) {
            let dropped = argPartition(-scaled, kth: topK - 1, axis: -1)[0..., topK...]
            scaled = putAlong(
                scaled, dropped, values: MLXArray(-Float.infinity).asType(scaled.dtype), axis: -1)
        }
        return categorical(scaled, axis: -1).asType(.int32)
    }
}
