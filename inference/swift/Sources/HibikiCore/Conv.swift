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
