// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

class StreamArray {
    let inner: MLXArray?

    init(_ x: MLXArray? = nil) {
        self.inner = x
    }

    func dim(_ dim: Int) -> Int {
        self.inner?.dim(dim) ?? 0
    }

    func cat2(_ rhs: StreamArray, axis: Int) -> StreamArray {
        switch (self.inner, rhs.inner) {
        case (.none, .none): StreamArray()
        case (.some(let lhs), .none): StreamArray(lhs)
        case (.none, .some(let rhs)): StreamArray(rhs)
        case (.some(let lhs), .some(let rhs)): StreamArray(concatenated([lhs, rhs], axis: axis))
        }
    }

    var shape: [Int]? {
        self.inner?.shape
    }

    func asArray() -> MLXArray? {
        self.inner
    }

    func narrow(_ offset: Int, _ len: Int, axis: Int) -> StreamArray {
        if let inner = self.inner {
            let totalLen = inner.dim(axis)
            if len <= 0 {
                return StreamArray()
            } else {
                // Not sure if there exists something closer to pytorch narrow
                let t = inner.split(indices: [offset, min(totalLen, offset + len)], axis: axis)
                return StreamArray(t[1])
            }
        } else {
            return StreamArray()
        }
    }

    func split(lhsLen: Int, axis: Int) -> (StreamArray, StreamArray) {
        if let t = self.inner {
            let len = t.dim(axis)
            let lhsLen = min(len, lhsLen)
            if lhsLen == 0 {
                return (StreamArray(), StreamArray(t))
            } else if lhsLen == len {
                return (StreamArray(t), StreamArray())
            } else {
                let split = t.split(indices: [lhsLen], axis: axis)
                return (StreamArray(split[0]), StreamArray(split[1]))
            }
        } else {
            return (StreamArray(), StreamArray())
        }
    }

    /// ELU written out: MLXNN's is a compiled function, and iOS cannot compile CPU kernels.
    func elu() -> StreamArray {
        self.map { which($0 .> 0, $0, exp($0) - 1) }
    }

    func map(_ f: (MLXArray) -> MLXArray) -> StreamArray {
        switch self.inner {
        case .none: StreamArray()
        case .some(let x): StreamArray(f(x))
        }
    }
}

protocol StreamingLayer {
    func resetState()
    func step(_ x: StreamArray) -> StreamArray
}

/// Adds two streams that may arrive with different lengths, keeping the unmatched tail.
final class StreamingAdd {
    var prevLHS = StreamArray()
    var prevRHS = StreamArray()
    let axis: Int

    init(axis: Int) {
        self.axis = axis
    }

    func resetState() {
        self.prevLHS = StreamArray()
        self.prevRHS = StreamArray()
    }

    func step(_ lhs: StreamArray, _ rhs: StreamArray) -> StreamArray {
        let lhs = self.prevLHS.cat2(lhs, axis: self.axis)
        let rhs = self.prevRHS.cat2(rhs, axis: self.axis)
        let commonLen = min(lhs.dim(self.axis), rhs.dim(self.axis))
        let (lhs_, prevLHS) = lhs.split(lhsLen: commonLen, axis: self.axis)
        let (rhs_, prevRHS) = rhs.split(lhsLen: commonLen, axis: self.axis)
        self.prevLHS = prevLHS
        self.prevRHS = prevRHS
        switch (lhs_.inner, rhs_.inner) {
        case (.some(let l), .some(let r)): return StreamArray(l + r)
        case (.none, .none): return StreamArray()
        case _: fatalError("internal error")
        }
    }
}
