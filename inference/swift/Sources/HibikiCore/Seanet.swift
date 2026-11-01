// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

struct SeanetConfig {
    var dimension: Int
    var channels: Int
    var nFilters: Int
    var nResidualLayers: Int
    var ratios: [Int]
    // var activation: String: hardcoded to Elu(1) for now
    // var norm: Norm
    var kernelSize: Int
    var residualKernelSize: Int
    var lastKernelSize: Int
    var dilationBase: Int
    var padMode: PadMode
    var compress: Int
    // var disableNormOuterBlocks: Int
    // var finalActivation: String?: hardcoded to None for now

    static func v0_1() -> SeanetConfig {
        SeanetConfig(
            dimension: 512, channels: 1, nFilters: 64, nResidualLayers: 1,
            ratios: [8, 6, 5, 4], kernelSize: 7, residualKernelSize: 3, lastKernelSize: 3,
            dilationBase: 2, padMode: .constant, compress: 2)
    }
}

class SeanetResnetBlock: Module, StreamingLayer {
    let skipOp = StreamingAdd(axis: -1)
    @ModuleInfo(key: "block") var block: [StreamableConv1d]

    init(_ cfg: SeanetConfig, dim: Int, kSizesAndDilations: [(Int, Int)]) {
        let hidden = dim / cfg.compress
        self._block.wrappedValue = kSizesAndDilations.enumerated().map { i, kd in
            StreamableConv1d(
                inC: i == 0 ? dim : hidden, outC: i == kSizesAndDilations.count - 1 ? dim : hidden,
                kSize: kd.0, stride: 1, dilation: kd.1, groups: 1, bias: true,
                padMode: cfg.padMode)
        }
    }

    func resetState() {
        self.skipOp.resetState()
        for b in self.block {
            b.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        let residual = x
        var x = x
        for b in self.block {
            x = b.step(x.elu())
        }
        return self.skipOp.step(x, residual)
    }
}

class EncoderLayer: Module, StreamingLayer {
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]
    @ModuleInfo(key: "downsample") var downsample: StreamableConv1d

    init(_ cfg: SeanetConfig, ratio: Int, mult: Int) {
        var residuals: [SeanetResnetBlock] = []
        var dilation: Int = 1
        for _ in 0..<cfg.nResidualLayers {
            let b = SeanetResnetBlock(
                cfg, dim: mult * cfg.nFilters,
                kSizesAndDilations: [(cfg.residualKernelSize, dilation), (1, 1)])
            residuals.append(b)
            dilation *= cfg.dilationBase
        }
        let downsample = StreamableConv1d(
            inC: mult * cfg.nFilters, outC: mult * cfg.nFilters * 2, kSize: ratio * 2,
            stride: ratio, dilation: 1, groups: 1,
            bias: true, padMode: cfg.padMode
        )
        self._residuals.wrappedValue = residuals
        self._downsample.wrappedValue = downsample
    }

    func resetState() {
        for r in self.residuals {
            r.resetState()
        }
        self.downsample.resetState()
    }

    func step(_ x: StreamArray) -> StreamArray {
        var x = x
        for r in self.residuals {
            x = r.step(x)
        }
        return self.downsample.step(x.elu())
    }
}

class SeanetEncoder: Module, StreamingLayer {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
        var mult = 1
        let initConv1d = StreamableConv1d(
            inC: cfg.channels, outC: mult * cfg.nFilters, kSize: cfg.kernelSize, stride: 1,
            dilation: 1, groups: 1, bias: true, padMode: cfg.padMode)
        var layers: [EncoderLayer] = []
        for ratio in cfg.ratios.reversed() {
            let layer = EncoderLayer(cfg, ratio: ratio, mult: mult)
            layers.append(layer)
            mult *= 2
        }
        let finalConv1d = StreamableConv1d(
            inC: mult * cfg.nFilters, outC: cfg.dimension, kSize: cfg.lastKernelSize, stride: 1,
            dilation: 1, groups: 1, bias: true, padMode: cfg.padMode)
        self._initConv1d.wrappedValue = initConv1d
        self._layers.wrappedValue = layers
        self._finalConv1d.wrappedValue = finalConv1d
    }

    func resetState() {
        self.initConv1d.resetState()
        self.finalConv1d.resetState()
        for l in self.layers {
            l.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        var x = self.initConv1d.step(x)
        for layer in self.layers {
            x = layer.step(x)
        }
        return self.finalConv1d.step(x.elu())
    }
}

class DecoderLayer: Module, StreamingLayer {
    @ModuleInfo(key: "upsample") var upsample: StreamableConvTranspose1d
    @ModuleInfo(key: "residuals") var residuals: [SeanetResnetBlock]

    init(_ cfg: SeanetConfig, ratio: Int, mult: Int) {
        var residuals: [SeanetResnetBlock] = []
        var dilation = 1
        for _ in 0..<cfg.nResidualLayers {
            let b = SeanetResnetBlock(
                cfg, dim: mult * cfg.nFilters / 2,
                kSizesAndDilations: [(cfg.residualKernelSize, dilation), (1, 1)])
            residuals.append(b)
            dilation *= cfg.dilationBase
        }

        let upsample = StreamableConvTranspose1d(
            inC: mult * cfg.nFilters, outC: mult * cfg.nFilters / 2, kSize: ratio * 2,
            stride: ratio, groups: 1, bias: true
        )
        self._upsample.wrappedValue = upsample
        self._residuals.wrappedValue = residuals
    }

    func resetState() {
        self.upsample.resetState()
        for r in self.residuals {
            r.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        var x = self.upsample.step(x.elu())
        for r in self.residuals {
            x = r.step(x)
        }
        return x
    }
}

class SeanetDecoder: Module, StreamingLayer {
    @ModuleInfo(key: "init_conv1d") var initConv1d: StreamableConv1d
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "final_conv1d") var finalConv1d: StreamableConv1d

    init(_ cfg: SeanetConfig) {
        var layers: [DecoderLayer] = []
        var mult = 1 << cfg.ratios.count
        let initConv1d = StreamableConv1d(
            inC: cfg.dimension, outC: mult * cfg.nFilters, kSize: cfg.kernelSize, stride: 1,
            dilation: 1, groups: 1, bias: true, padMode: cfg.padMode)
        for ratio in cfg.ratios {
            let l = DecoderLayer(cfg, ratio: ratio, mult: mult)
            layers.append(l)
            mult /= 2
        }
        let finalConv1d = StreamableConv1d(
            inC: cfg.nFilters, outC: cfg.channels, kSize: cfg.lastKernelSize, stride: 1,
            dilation: 1, groups: 1, bias: true, padMode: cfg.padMode)
        self._initConv1d.wrappedValue = initConv1d
        self._layers.wrappedValue = layers
        self._finalConv1d.wrappedValue = finalConv1d
    }

    func resetState() {
        self.initConv1d.resetState()
        self.finalConv1d.resetState()
        for l in self.layers {
            l.resetState()
        }
    }

    func step(_ x: StreamArray) -> StreamArray {
        var x = self.initConv1d.step(x)
        for layer in self.layers {
            x = layer.step(x)
        }
        return self.finalConv1d.step(x.elu())
    }
}
