// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX

/// Mimi's transformer cache: the keys and values of the last `context - 1` steps. Each query of
/// a call, whatever its length, attends to exactly the `context` steps ending at itself, as
/// Mimi was trained and as `training/cache/mimi_mlx.py` encodes whole clips, so chunked and
/// frame-by-frame calls give the same codes. (A ring of `context` steps, as moshi-swift keeps,
/// overwrites the oldest keys of a call's first queries, more so the longer the call.)
///
/// The held steps fill buffers of `context - 1` slots from the first call, zeros where no step
/// was written yet, which `mask` hides; so every call of t steps sees the same shapes, and Mimi
/// can compile its GPU encode over them.
final class WindowKVCache {
    let context: Int
    /// Steps written since the last reset.
    var offset = 0
    /// `[B, heads, context - 1, headDim]`: the held steps, the oldest first; nil before the first
    /// update.
    var keys: MLXArray?
    var values: MLXArray?

    init(context: Int) {
        self.context = context
    }

    /// Appends `[B, heads, t, headDim]` keys and values; returns the held slots followed by the
    /// new steps, which `mask` describes.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let held = context - 1
        let empty = { (x: MLXArray) in
            MLXArray.zeros([x.dim(0), x.dim(1), held, x.dim(3)], dtype: x.dtype)
        }
        let allKeys = concatenated([self.keys ?? empty(keys), keys], axis: 2)
        let allValues = concatenated([self.values ?? empty(values), values], axis: 2)
        self.keys = allKeys[0..., 0..., keys.dim(2)...]
        self.values = allValues[0..., 0..., values.dim(2)...]
        offset += keys.dim(2)
        return (allKeys, allValues)
    }

    func reset() {
        offset = 0
        keys = nil
        values = nil
    }

    /// The additive mask `[t, context - 1 + t]` of the next `update` of `t` steps: query i (step
    /// offset + i) sees key j (step offset − (context − 1) + j) when 0 ≤ (context − 1) + i − j <
    /// context and that step was written.
    func mask(steps t: Int, dtype: DType) -> MLXArray {
        let held = context - 1
        var mask = [Float](repeating: -1e9, count: t * (held + t))
        for i in 0..<t {
            for j in max(held - offset, i)...(held + i) {
                mask[i * (held + t) + j] = 0
            }
        }
        return MLXArray(mask, [t, held + t]).asType(dtype)
    }
}
