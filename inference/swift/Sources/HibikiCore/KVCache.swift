// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX

/// Mimi's transformer cache: the keys and values of the last `context - 1` steps. Each query of
/// a call, whatever its length, attends to exactly the `context` steps ending at itself, as
/// Mimi was trained and as `training/prep/mimi_mlx.py` encodes whole clips, so chunked and
/// frame-by-frame calls give the same codes. (A ring of `context` steps, as moshi-swift keeps,
/// overwrites the oldest keys of a call's first queries, more so the longer the call.)
final class WindowKVCache {
    let context: Int
    private(set) var offset = 0
    private var keys: MLXArray?
    private var values: MLXArray?

    init(context: Int) {
        self.context = context
    }

    /// Appends `[B, heads, t, headDim]` keys and values; returns the kept steps followed by the
    /// new ones, which `mask` describes.
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let allKeys = self.keys.map { concatenated([$0, keys], axis: 2) } ?? keys
        let allValues = self.values.map { concatenated([$0, values], axis: 2) } ?? values
        let keep = min(allKeys.dim(2), context - 1)
        self.keys = allKeys[0..., 0..., (allKeys.dim(2) - keep)...]
        self.values = allValues[0..., 0..., (allValues.dim(2) - keep)...]
        offset += keys.dim(2)
        return (allKeys, allValues)
    }

    func reset() {
        offset = 0
        keys = nil
        values = nil
    }

    /// The additive mask of the next `update` of `t` steps: query i (the kept steps come first,
    /// so it is key `kept + i`) sees key j when 0 ≤ kept + i − j < context.
    func mask(steps t: Int, dtype: DType) -> MLXArray {
        let kept = min(offset, context - 1)
        let delta =
            MLXArray(Int32(kept)..<Int32(kept + t))[0..., .newAxis]
            - MLXArray(Int32(0)..<Int32(kept + t))[.newAxis]
        let visible = (delta .>= 0) .&& (delta .< context)
        return MLX.where(visible, MLXArray(Float32(0)), MLXArray(Float32(-1e9))).asType(dtype)
    }
}
