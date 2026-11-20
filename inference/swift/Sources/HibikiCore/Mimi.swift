// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX
import MLXNN

struct MimiConfig {
    var channels: Int
    var sampleRate: Float
    var frameRate: Float
    var renormalize: Bool
    // var resampleMethod: String
    var seanet: SeanetConfig
    var transformer: TransformerConfig
    var quantizerNQ: Int
    var quantizerBins: Int
    var quantizerDim: Int

    static func mimi_2024_07(numCodebooks: Int = 16) -> MimiConfig {
        let seanet = SeanetConfig.v0_1()
        let transformer = TransformerConfig(
            dModel: seanet.dimension,
            numHeads: 8,
            numLayers: 8,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: .rope,
            gating: false,
            norm: .layerNorm,
            context: 250,
            maxPeriod: 10000,
            kvRepeat: 1,
            dimFeedForward: 2048
        )
        return MimiConfig(
            channels: 1,
            sampleRate: 24000, frameRate: 12.5, renormalize: true, seanet: seanet,
            transformer: transformer, quantizerNQ: numCodebooks, quantizerBins: 2048,
            quantizerDim: 256)
    }
}

/// Mimi's streaming encoder, in MLX, on `device` (the CPU or the GPU). The decoder runs on the
/// Neural Engine (`NeuralDecoder`); the encoder stays here because fp16 on the Neural Engine
/// changes its codes (codebook 0 on 94% of frames, codebooks 8–15 on ~17%).
class Mimi: Module {
    let cfg: MimiConfig
    let device: Device
    let encoderCache: [WindowKVCache]
    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: ConvDownsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    init(_ cfg: MimiConfig, device: Device) {
        let dim = cfg.seanet.dimension
        self.cfg = cfg
        self.device = device
        let encoderFrameRate = cfg.sampleRate / Float(cfg.seanet.ratios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / cfg.frameRate)
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, nQ: cfg.quantizerNQ, bins: cfg.quantizerBins)
        precondition(cfg.transformer.dModel == dim, "Mimi's transformer works at the codec width")
        self._encoderTransformer.wrappedValue = ProjectedTransformer(cfg.transformer)
        self._downsample.wrappedValue = ConvDownsample1d(stride: downsampleStride, dim: dim)
        self.encoderCache = self._encoderTransformer.wrappedValue.makeCache()
    }

    /// The GPU's encode compiled over the stream's state (`streamState`), traced once per chunk
    /// size: each call replays the traced graph, its elementwise ops fused, instead of building
    /// every op in Swift, which costs the driving thread less and leaves fewer freed buffers in
    /// MLX's cache.
    private var _compiledEncode: (([MLXArray]) -> [MLXArray])?

    /// Encodes the next whole frames into codes `[1, codebooks, frames]`, with `device` as the
    /// default device (so ops pick its kernels, `onCPU`), the input included: on the other
    /// device's stream it would wait behind that device's work (the MLX LM on the GPU). On the
    /// GPU every call after a session's first runs compiled; the CPU cannot compile on iOS.
    func encodeStep(_ samples: [Float]) -> MLXArray {
        Device.withDefaultDevice(device) {
            let pcm = MLXArray(samples).reshaped(1, 1, samples.count)
            let steps = pcm.dim(-1) / cfg.seanet.ratios.reduce(1, *)
            let cache = encoderCache[0]
            let offset = MLXArray(Int32(cache.offset))
            let mask = cache.mask(steps: steps, dtype: pcm.dtype)
            // A session's first call pads each conv's input and creates the state; it runs eagerly.
            guard device.deviceType == .gpu, cache.offset > 0 else {
                return encodeFrames(pcm, offset: offset, mask: mask)
            }
            let encode =
                _compiledEncode
                ?? compile { [unowned self] args in
                    // args: pcm, offset, mask, then the state; returns codes, then the state.
                    self.streamState = Array(args[3...])
                    return [self.encodeFrames(args[0], offset: args[1], mask: args[2])]
                        + self.streamState
                }
            _compiledEncode = encode
            let offsets = encoderCache.map(\.offset)
            let out = encode([pcm, offset, mask] + streamState)
            streamState = Array(out[1...])
            // Tracing ran the caches' updates once; a replay runs none.
            for (c, o) in zip(encoderCache, offsets) { c.offset = o + steps }
            return out[0]
        }
    }

    private func encodeFrames(_ pcm: MLXArray, offset: MLXArray, mask: MLXArray) -> MLXArray {
        var x = self.encoder.step(StreamArray(pcm))
        x = x.map {
            self.encoderTransformer($0, offset: offset, mask: mask, cache: self.encoderCache)
        }
        x = self.downsample.step(x)
        guard let codes = x.map(self.quantizer.encode).asArray() else {
            fatalError("Mimi produced no codes for \(pcm.shape)")
        }
        return codes
    }

    /// The arrays that carry the stream from one call to the next: each conv's held input (none
    /// for 1-wide kernels) and each transformer layer's keys and values. Their shapes stay the
    /// same from a session's first call on.
    private var streamState: [MLXArray] {
        get {
            convs.compactMap(\.statePrevXs.inner)
                + encoderCache.flatMap { [$0.keys!, $0.values!] }
        }
        set {
            var state = newValue[...]
            for conv in convs where conv.statePrevXs.inner != nil {
                conv.statePrevXs = StreamArray(state.removeFirst())
            }
            for c in encoderCache {
                c.keys = state.removeFirst()
                c.values = state.removeFirst()
            }
            precondition(state.isEmpty, "Mimi's stream state changed shape")
        }
    }

    /// Every streaming conv, in a fixed order.
    private var convs: [StreamableConv1d] {
        encoder.convs + [downsample.conv]
    }

    func resetState() {
        self.encoder.resetState()
        self.downsample.resetState()
        self.encoderCache.forEach { c in c.reset() }
    }
}

extension Mimi {
    /// Loads Mimi's encoder from Kyutai's PyTorch checkpoint
    /// (`mimi-pytorch-e351c8d8@125.safetensors`), mapping its keys and conv layouts onto this
    /// module tree and skipping the decoder's weights, to run on `device`.
    static func load(url: URL, numCodebooks: Int, device: Device) throws -> Mimi {
        let model = Mimi(MimiConfig.mimi_2024_07(numCodebooks: numCodebooks), device: device)
        let cpu = device.deviceType == .cpu
        var weights: [String: MLXArray] = [:]
        for (var key, var weight) in try loadArrays(url: url) {
            if ["decoder.", "decoder_transformer.", "upsample."].contains(where: key.hasPrefix)
                || key.hasSuffix(".output_proj.weight")
            {
                continue
            }
            // The checkpoint holds 32 codebooks (1 first + 31 rest); keep the ones in use.
            let rest = "quantizer.rvq_rest.vq.layers."
            if key.hasPrefix(rest),
                let layer = Int(key.dropFirst(rest.count).prefix { $0 != "." }),
                layer >= numCodebooks - 1
            {
                continue
            }
            key.replace("encoder.model.", with: "encoder.")
            key.replace(".in_proj_weight", with: ".in_proj.weight")
            key.replace(".linear1.weight", with: ".gating.linear1.weight")
            key.replace(".linear2.weight", with: ".gating.linear2.weight")
            for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
                key.replace(
                    "encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
                key.replace(
                    "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
            }
            key.replace("encoder.0.", with: "encoder.init_conv1d.")
            key.replace("encoder.14.", with: "encoder.final_conv1d.")
            key.replace(".block.1.", with: ".block.0.")
            key.replace(".block.3.", with: ".block.1.")
            // PyTorch convs are (out, in, kernel); MLX wants (out, kernel, in).
            if key.hasSuffix(".conv.weight") || key.hasSuffix(".input_proj.weight") {
                weight = weight.swappedAxes(-1, -2)
            }
            // Transformer linears are (out, in); stored column-major, `Linear`'s x · wᵀ hands
            // the CPU's sgemm a row-major (in, out) operand, which it streams ~7x faster than
            // a transposed one at Mimi's two-row inputs. The GPU's gemv wants them as stored.
            if cpu && key.contains("_transformer.") && weight.ndim == 2 {
                weight = contiguous(weight.T).T
            }
            weights[key] = weight
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
