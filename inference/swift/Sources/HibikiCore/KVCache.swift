// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Parts of this file came from:
// https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/LLM/KVCache.swift
// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Interface for Key/Value cache for LLMs.
///
/// See ``LLMModel/newCache(parameters:)-47tyu``
public protocol KVCache: Evaluatable {

    /// get the current offset
    var offset: Int { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
    func createAttentionMask(h: MLXArray) -> MLXArray?
}

class RotatingKVCache: KVCache, Evaluatable {
    let keys: MLXArray
    let values: MLXArray
    let maxSize: Int
    var offset: Int = 0

    init(bSize: Int, numHeads: Int, maxSize: Int, headDim: Int, dtype: DType) {
        self.keys = MLXArray.zeros([bSize, numHeads, maxSize, headDim], dtype: dtype)
        self.values = MLXArray.zeros([bSize, numHeads, maxSize, headDim], dtype: dtype)
        self.maxSize = maxSize
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let t = keys.dim(2)
        if t > self.maxSize {
            fatalError("query to update with shape \(keys.shape) larger than maxSize \(maxSize)")
        }
        let currentOffset = self.offset % self.maxSize
        let tMax = min(self.maxSize, currentOffset + t)
        self.keys[0..., 0..., currentOffset..<tMax] = keys[0..., 0..., 0..<(tMax - currentOffset)]
        self.values[0..., 0..., currentOffset..<tMax] =
            values[0..., 0..., 0..<(tMax - currentOffset)]
        let leftToCopy = t - tMax + currentOffset
        if 0 < leftToCopy {
            self.keys[0..., 0..., 0..<leftToCopy] = keys[0..., 0..., (tMax - currentOffset)...]
            self.values[0..., 0..., 0..<leftToCopy] = values[0..., 0..., (tMax - currentOffset)...]
        }
        self.offset += t
        return (self.keys, self.values)
    }

    func reset() {
        offset = 0
    }

    func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        let finalOffset = self.offset + t
        let finalOffsetMod = finalOffset % self.maxSize
        // As a default we use finalOffset + 1 so that these slices cannot be seen.
        var rinds = Array(repeating: Int32(finalOffset + 1), count: self.maxSize)
        for i in 0..<finalOffsetMod {
            rinds[i] = Int32(finalOffset + i - finalOffsetMod)
        }
        if finalOffsetMod != finalOffset {
            for i in finalOffsetMod..<rinds.count {
                rinds[i] = Int32(finalOffset + i - finalOffsetMod - rinds.count)
            }
        }
        let linds = MLXArray(Int32(self.offset)..<Int32(self.offset + t))
        let mask = linds[0..., .newAxis] .< MLXArray(rinds)[.newAxis]
        let res = (mask * Float32(-1e9)).asType(h.dtype)
        return res
    }
}
