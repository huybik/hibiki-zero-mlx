// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN
import MLXRandom

// Conv1d + dilation
class Conv1d: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?
    let padding: Int
    let groups: Int
    let stride: Int
    let dilation: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        let scale = (1 / Float(inputChannels * kernelSize)).squareRoot()

        self._weight.wrappedValue = uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels / groups])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // MLX uses NLC whereas pytorch/candle use NCL
        var y = conv1d(
            x.swappedAxes(-1, -2), weight, stride: stride, padding: padding, dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        y = y.swappedAxes(-1, -2)
        return y
    }
}

/// ConvTranspose1d with kernel = 2 × stride, the only shape Mimi uses, as one matmul and an
/// overlap-add. MLX's convTransposed1d has no groups, and on the CPU it runs a naive loop
/// (35–65 ms per Mimi decoder layer on an M4 Pro, against 0.3 ms this way).
///
/// `weight` is `[in, kernel, out / groups]` (PyTorch's `[in, out / groups, kernel]` with the
/// last two axes swapped), so it reshapes for free to the `[in, kernel × out]` matmul operand.
class ConvTransposed1d: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?
    let stride: Int
    let depthwise: Bool

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        precondition(kernelSize == 2 * stride, "only kernel = 2 × stride is implemented")
        depthwise = groups == inputChannels && groups == outputChannels
        precondition(groups == 1 || depthwise, "only dense or depthwise transposed convs")
        let scale = (1 / Float(inputChannels * kernelSize)).squareRoot()
        self._weight.wrappedValue = uniform(
            low: -scale, high: scale, [inputChannels, kernelSize, outputChannels / groups])
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
        self.stride = stride
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(2))
        let k = 2 * stride
        // Each input step's contribution to its 2 × stride output samples: [B, T, K, out].
        let y =
            depthwise
            ? x.swappedAxes(-1, -2)[0..., 0..., .newAxis, 0...]
                * weight.reshaped(-1, k).T[.newAxis, .newAxis]
            : matmul(x.swappedAxes(-1, -2), weight.reshaped(weight.dim(0), -1))
                .reshaped(b, t, k, -1)
        // Overlap-add: the first half of step i's kernel and the second half of step i - 1's
        // fill output block i, for T + 1 blocks of `stride` samples.
        let halves = y.reshaped(b, t, 2, stride, -1)
        let pad = MLXArray.zeros([b, 1, stride, halves.dim(-1)], dtype: y.dtype)
        var out =
            (concatenated([halves[0..., 0..., 0], pad], axis: 1)
            + concatenated([pad, halves[0..., 0..., 1]], axis: 1))
            .reshaped(b, (t + 1) * stride, -1)
        if let bias {
            out = out + bias
        }
        return out.swappedAxes(-1, -2)
    }
}

// weight-norm is handled externally when generating the weights file so this class is just
// a wrapper around Conv1d
class NormConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        inC: Int, outC: Int, kSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize, stride: stride,
            padding: padding,
            groups: groups, dilation: dilation, bias: bias)
    }
}

class NormConvTranspose1d: Module, UnaryLayer {
    @ModuleInfo(key: "convtr") var convtr: ConvTransposed1d

    init(
        inC: Int, outC: Int, kSize: Int,
        stride: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._convtr.wrappedValue = ConvTransposed1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize, stride: stride,
            groups: groups, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.convtr(x)
    }
}

class StreamableConv1d: Module, StreamingLayer {
    let padMode: PadMode
    let kSize: Int
    var leftPadApplied: Bool = false
    var statePrevXs: StreamArray = StreamArray()
    @ModuleInfo(key: "conv") var conv: NormConv1d

    init(
        inC: Int, outC: Int, kSize: Int, stride: Int, dilation: Int, groups: Int, bias: Bool,
        padMode: PadMode
    ) {
        self.padMode = padMode
        self.kSize = kSize
        self._conv.wrappedValue = NormConv1d(
            inC: inC, outC: outC, kSize: kSize, stride: stride, groups: groups, dilation: dilation,
            bias: bias)
    }

    func resetState() {
        statePrevXs = StreamArray()
        self.leftPadApplied = false
    }

    func step(_ x: StreamArray) -> StreamArray {
        if var inner = x.inner {
            let stride = self.conv.conv.stride
            let dilation = self.conv.conv.dilation
            if !self.leftPadApplied {
                self.leftPadApplied = true
                let kSize = (self.kSize - 1) * dilation + 1
                let paddingTotal = kSize - stride
                let z = IntOrPair.init((0, 0))
                let widths = [z, z, IntOrPair((paddingTotal, 0))]
                inner = padded(inner, widths: widths, mode: self.padMode)
            }
            let kernel = (self.kSize - 1) * dilation + 1
            var x = StreamArray(inner)
            x = self.statePrevXs.cat2(x, axis: -1)
            let seqLen = x.dim(-1)
            let numFrames = max(seqLen + stride - kernel, 0) / stride
            if numFrames > 0 {
                let offset = numFrames * stride
                self.statePrevXs = x.narrow(offset, seqLen - offset, axis: -1)
                let inL = (numFrames - 1) * stride + kernel
                x = x.narrow(0, inL, axis: -1)
                if let x = x.inner {
                    return StreamArray(self.conv.conv(x))
                } else {
                    return StreamArray()
                }
            } else {
                self.statePrevXs = x
                return StreamArray()
            }
        } else {
            return StreamArray()
        }
    }
}

class StreamableConvTranspose1d: Module, StreamingLayer {
    let kSize: Int
    var statePrevYs: StreamArray = StreamArray()
    @ModuleInfo(key: "convtr") var convtr: NormConvTranspose1d

    init(
        inC: Int, outC: Int, kSize: Int, stride: Int, groups: Int, bias: Bool
    ) {
        self.kSize = kSize
        self._convtr.wrappedValue = NormConvTranspose1d(
            inC: inC, outC: outC, kSize: kSize, stride: stride, groups: groups, bias: bias)
    }

    func resetState() {
        statePrevYs = StreamArray()
    }

    func step(_ x: StreamArray) -> StreamArray {
        if let x = x.inner {
            let stride = self.convtr.convtr.stride
            var ys = self.convtr(x)
            let ot = ys.dim(-1)
            if var prevYs = self.statePrevYs.inner {
                let pt = prevYs.dim(-1)
                if let bias = self.convtr.convtr.bias {
                    prevYs = prevYs - bias[.newAxis, 0..., .newAxis]
                }
                let ys1 = ys[.ellipsis, 0..<pt] + prevYs
                let ys2 = ys[.ellipsis, pt...]
                ys = concatenated([ys1, ys2], axis: -1)
            }
            let invalidSteps = self.kSize - stride
            let (ys_, prevYs) = StreamArray(ys).split(lhsLen: ot - invalidSteps, axis: -1)
            self.statePrevYs = prevYs
            return ys_
        } else {
            return StreamArray()
        }
    }
}

class ConvDownsample1d: Module, StreamingLayer {
    @ModuleInfo(key: "conv") var conv: StreamableConv1d

    init(stride: Int, dim: Int) {
        self._conv.wrappedValue = StreamableConv1d(
            inC: dim, outC: dim, kSize: 2 * stride, stride: stride, dilation: 1, groups: 1,
            bias: false, padMode: .edge)
    }

    func resetState() {
        self.conv.resetState()
    }

    func step(_ x: StreamArray) -> StreamArray {
        self.conv.step(x)
    }
}

class ConvTrUpsample1d: Module, StreamingLayer {
    @ModuleInfo(key: "convtr") var convtr: StreamableConvTranspose1d

    init(stride: Int, dim: Int) {
        self._convtr.wrappedValue = StreamableConvTranspose1d(
            inC: dim, outC: dim, kSize: 2 * stride, stride: stride, groups: dim, bias: false)
    }

    func resetState() {
        self.convtr.resetState()
    }

    func step(_ x: StreamArray) -> StreamArray {
        self.convtr.step(x)
    }
}
