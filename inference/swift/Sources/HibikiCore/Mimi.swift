// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation
import MLX
import MLXNN
import MLXRandom

public struct MimiConfig {
    public var channels: Int
    public var sampleRate: Float
    public var frameRate: Float
    public var renormalize: Bool
    // public var resampleMethod: String
    public var seanet: SeanetConfig
    public var transformer: TransformerConfig
    public var quantizerNQ: Int
    public var quantizerBins: Int
    public var quantizerDim: Int

    public static func mimi_2024_07(numCodebooks: Int = 16) -> MimiConfig {
        let seanet = SeanetConfig.v0_1()
        let transformer = TransformerConfig(
            dModel: seanet.dimension,
            numHeads: 8,
            numLayers: 8,
            causal: true,
            normFirst: true,
            biasFF: false,
            biasAttn: false,
            layerScale: 0.01,
            positionalEmbedding: .rope,
            useConvBias: true,
            gating: false,
            norm: .layerNorm,
            context: 250,
            maxPeriod: 10000,
            maxSeqLen: 8192,
            kvRepeat: 1,
            dimFeedForward: 2048,
            convLayout: true
        )
        return MimiConfig(
            channels: 1,
            sampleRate: 24000, frameRate: 12.5, renormalize: true, seanet: seanet,
            transformer: transformer, quantizerNQ: numCodebooks, quantizerBins: 2048,
            quantizerDim: 256)
    }
}

public class Mimi: Module {
    let cfg: MimiConfig
    let encoderCache: [KVCache]
    let decoderCache: [KVCache]
    @ModuleInfo(key: "encoder") var encoder: SeanetEncoder
    @ModuleInfo(key: "decoder") var decoder: SeanetDecoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: ProjectedTransformer
    @ModuleInfo(key: "downsample") var downsample: ConvDownsample1d
    @ModuleInfo(key: "upsample") var upsample: ConvTrUpsample1d
    @ModuleInfo(key: "quantizer") var quantizer: SplitResidualVectorQuantizer

    public init(_ cfg: MimiConfig, bSize: Int) {
        let dim = cfg.seanet.dimension
        self.cfg = cfg
        let encoderFrameRate = cfg.sampleRate / Float(cfg.seanet.ratios.reduce(1, *))
        let downsampleStride = Int(encoderFrameRate / cfg.frameRate)
        self._encoder.wrappedValue = SeanetEncoder(cfg.seanet)
        self._decoder.wrappedValue = SeanetDecoder(cfg.seanet)
        self._quantizer.wrappedValue = SplitResidualVectorQuantizer(
            dim: cfg.quantizerDim, inputDim: dim, outputDim: dim, nQ: cfg.quantizerNQ,
            bins: cfg.quantizerBins)
        self._encoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
        self._decoderTransformer.wrappedValue = ProjectedTransformer(
            cfg.transformer, inputDim: dim, outputDims: [dim])
        self._downsample.wrappedValue = ConvDownsample1d(
            stride: downsampleStride, dim: dim, causal: true)
        self._upsample.wrappedValue = ConvTrUpsample1d(
            stride: downsampleStride, dim: dim, causal: true)
        self.encoderCache = self._encoderTransformer.wrappedValue.makeCache(bSize: bSize)
        self.decoderCache = self._decoderTransformer.wrappedValue.makeCache(bSize: bSize)
    }

    public func warmup() {
        let pcm = MLXArray.zeros([1, 1, 1920 * 4])
        let codes = self.encode(pcm)
        let pcmOut = self.decode(codes)
        eval(pcmOut)
    }

    public func encode(_ x: MLXArray) -> MLXArray {
        self.encoder.resetState()
        self.encoderCache.forEach { c in c.reset() }
        var x = self.encoder(x)
        x = self.encoderTransformer(x, cache: self.encoderCache)[0]
        x = self.downsample(x)
        let codes = self.quantizer.encode(x)
        return codes
    }

    public func encodeStep(_ x: StreamArray) -> StreamArray {
        var x = self.encoder.step(x)
        x = x.map { self.encoderTransformer($0, cache: self.encoderCache)[0] }
        x = self.downsample.step(x)
        let codes = x.map(self.quantizer.encode)
        return codes
    }

    public func decode(_ codes: MLXArray) -> MLXArray {
        self.decoder.resetState()
        self.decoderCache.forEach { c in c.reset() }
        let emb = self.quantizer.decode(codes)
        let embUp = self.upsample(emb)
        let outs = self.decoderTransformer(embUp, cache: self.decoderCache)
        return self.decoder(outs[0])
    }

    public func decodeStep(_ codes: StreamArray) -> StreamArray {
        var emb = codes.map { self.quantizer.decode($0) }
        emb = self.upsample.step(emb)
        let out = emb.map { self.decoderTransformer($0, cache: self.decoderCache)[0] }
        return self.decoder.step(out)
    }

    public func resetState() {
        self.encoder.resetState()
        self.decoder.resetState()
        self.encoderCache.forEach { c in c.reset() }
        self.decoderCache.forEach { c in c.reset() }
    }
}

extension Mimi {
    /// Loads Mimi from Kyutai's PyTorch checkpoint (`mimi-pytorch-e351c8d8@125.safetensors`),
    /// mapping its keys and conv layouts onto this module tree.
    public static func load(url: URL, numCodebooks: Int) throws -> Mimi {
        let model = Mimi(MimiConfig.mimi_2024_07(numCodebooks: numCodebooks), bSize: 1)
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
            // PyTorch transposed convs are (in, out, kernel).
            if key.hasSuffix(".convtr.weight") {
                weight = weight.transposed(axes: [1, 2, 0])
            }
            weights[key] = weight
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
