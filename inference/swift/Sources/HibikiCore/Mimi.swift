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

class Mimi: Module {
    let cfg: MimiConfig
    let encoderCache: [WindowKVCache]
    let decoderCache: [WindowKVCache]
    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "decoder") var decoder: SeanetDecoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: ConvDownsample1d
    @ModuleInfo(key: "upsample") var upsample: ConvTrUpsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    init(_ cfg: MimiConfig) {
        let dim = cfg.seanet.dimension
        self.cfg = cfg
        let encoderFrameRate = cfg.sampleRate / Float(cfg.seanet.ratios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / cfg.frameRate)
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._decoder.wrappedValue = SeanetDecoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, outputDim: dim, nQ: cfg.quantizerNQ,
            bins: cfg.quantizerBins)
        precondition(cfg.transformer.dModel == dim, "Mimi's transformer works at the codec width")
        self._encoderTransformer.wrappedValue = ProjectedTransformer(cfg.transformer)
        self._decoderTransformer.wrappedValue = ProjectedTransformer(cfg.transformer)
        self._downsample.wrappedValue = ConvDownsample1d(stride: downsampleStride, dim: dim)
        self._upsample.wrappedValue = ConvTrUpsample1d(stride: downsampleStride, dim: dim)
        self.encoderCache = self._encoderTransformer.wrappedValue.makeCache()
        self.decoderCache = self._decoderTransformer.wrappedValue.makeCache()
    }

    func encodeStep(_ x: StreamArray) -> StreamArray {
        var x = self.encoder.step(x)
        x = x.map { self.encoderTransformer($0, cache: self.encoderCache) }
        x = self.downsample.step(x)
        let codes = x.map(self.quantizer.encode)
        return codes
    }

    func decodeStep(_ codes: StreamArray) -> StreamArray {
        var emb = codes.map { self.quantizer.decode($0) }
        emb = self.upsample.step(emb)
        let out = emb.map { self.decoderTransformer($0, cache: self.decoderCache) }
        return self.decoder.step(out)
    }

    func resetState() {
        self.encoder.resetState()
        self.decoder.resetState()
        self.downsample.resetState()
        self.upsample.resetState()
        self.encoderCache.forEach { c in c.reset() }
        self.decoderCache.forEach { c in c.reset() }
    }
}

extension Mimi {
    /// Loads Mimi from Kyutai's PyTorch checkpoint (`mimi-pytorch-e351c8d8@125.safetensors`),
    /// mapping its keys and conv layouts onto this module tree.
    static func load(url: URL, numCodebooks: Int) throws -> Mimi {
        let model = Mimi(MimiConfig.mimi_2024_07(numCodebooks: numCodebooks))
        var weights: [String: MLXArray] = [:]
        for (var key, var weight) in try loadArrays(url: url) {
            // The checkpoint holds 32 codebooks (1 first + 31 rest); keep the ones in use.
            let rest = "quantizer.rvq_rest.vq.layers."
            if key.hasPrefix(rest),
                let layer = Int(key.dropFirst(rest.count).prefix { $0 != "." }),
                layer >= numCodebooks - 1
            {
                continue
            }
            key.replace("encoder.model.", with: "encoder.")
            key.replace("decoder.model.", with: "decoder.")
            key.replace(".in_proj_weight", with: ".in_proj.weight")
            key.replace(".linear1.weight", with: ".gating.linear1.weight")
            key.replace(".linear2.weight", with: ".gating.linear2.weight")
            for (layerIdx, decoderIdx) in [2, 5, 8, 11].enumerated() {
                key.replace("decoder.\(decoderIdx).", with: "decoder.layers.\(layerIdx).upsample.")
                key.replace(
                    "decoder.\(decoderIdx + 1).", with: "decoder.layers.\(layerIdx).residuals.0.")
            }
            for (layerIdx, encoderIdx) in [1, 4, 7, 10].enumerated() {
                key.replace(
                    "encoder.\(encoderIdx).", with: "encoder.layers.\(layerIdx).residuals.0.")
                key.replace(
                    "encoder.\(encoderIdx + 2).", with: "encoder.layers.\(layerIdx).downsample.")
            }
            key.replace("decoder.0.", with: "decoder.init_conv1d.")
            key.replace("decoder.14.", with: "decoder.final_conv1d.")
            key.replace("encoder.0.", with: "encoder.init_conv1d.")
            key.replace("encoder.14.", with: "encoder.final_conv1d.")
            key.replace(".block.1.", with: ".block.0.")
            key.replace(".block.3.", with: ".block.1.")
            // PyTorch convs are (out, in, kernel); MLX wants (out, kernel, in).
            if key.hasSuffix(".conv.weight") || key.hasSuffix(".output_proj.weight")
                || key.hasSuffix(".input_proj.weight")
            {
                weight = weight.swappedAxes(-1, -2)
            }
            // PyTorch transposed convs are (in, out, kernel); `ConvTransposed1d` wants
            // (in, kernel, out) in memory, to reshape to its matmul operand without a copy.
            if key.hasSuffix(".convtr.weight") {
                weight = contiguous(weight.swappedAxes(1, 2))
            }
            // Transformer linears are (out, in); stored column-major, `Linear`'s x · wᵀ hands
            // the CPU's sgemm a row-major (in, out) operand, which it streams ~7x faster than
            // a transposed one at Mimi's two-row inputs.
            if key.contains("_transformer.") && weight.ndim == 2 {
                weight = contiguous(weight.T).T
            }
            weights[key] = weight
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
