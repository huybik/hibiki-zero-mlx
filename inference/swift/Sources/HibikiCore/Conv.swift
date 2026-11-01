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

// ConvTranspose1d + groups
class ConvTransposed1d: Module, UnaryLayer {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?
    let padding: Int
    let stride: Int
    let groups: Int
    let inC: Int
    let outC: Int
    let kSize: Int
    var _expandedWeight: MLXArray
    var expandedGroups: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        let scale = (1 / Float(inputChannels * kernelSize)).squareRoot()

        self._weight.wrappedValue = uniform(
            low: -scale, high: scale, [outputChannels / groups, kernelSize, inputChannels])
        let weight = self._weight.wrappedValue
        self._bias.wrappedValue = bias ? MLXArray.zeros([outputChannels]) : nil
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.inC = inputChannels
        self.outC = outputChannels
        self.kSize = kernelSize
        (self._expandedWeight, self.expandedGroups) = (weight, groups)
        super.init()
        expand()
    }

    override func update(
        parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        expand()
        return self
    }

    /// Depthwise transposed convs (groups == channels) run as one dense conv over a
    /// block-diagonal weight; MLX's convTransposed1d has no groups.
    private func expand() {
        if groups == inC && groups == outC {
            let eye = repeated(
                eye(outC).asType(weight.dtype).reshaped([outC, 1, outC]), count: kSize, axis: 1)
            _expandedWeight = repeated(weight, count: groups, axis: 0) * eye
            expandedGroups = 1
        } else if groups > 1 {
            fatalError("groups are not supported in ConvTranspose1d, \(groups), \(inC), \(outC)")
        } else {
            _expandedWeight = weight
            expandedGroups = groups
        }
    }

    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Groups are not supported in convTransposed1d as of 0.18.1 so we hack our way around it.
        var y = convTransposed1d(
            x.swappedAxes(-1, -2), _expandedWeight, stride: stride, padding: padding,
            groups: expandedGroups
        )
        if let bias {
            y = y + bias
        }
        return y.swappedAxes(-1, -2)
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
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._convtr.wrappedValue = ConvTransposed1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize, stride: stride,
            padding: padding, groups: groups, bias: bias)
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
