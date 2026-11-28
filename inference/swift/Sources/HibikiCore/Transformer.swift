// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

/// Mimi's transformer: LayerNorm, interleaved rotary embedding (MLX `traditional: true`), a
/// GELU feed-forward without gating, and layer scale.
struct TransformerConfig {
    var dModel: Int
    var numHeads: Int
    var numLayers: Int
    var biasFF: Bool
    var biasAttn: Bool
    var layerScale: Float
    var context: Int
    var maxPeriod: Float
    var kvRepeat: Int
    var dimFeedForward: Int

    func headDim() -> Int {
        self.dModel / self.numHeads
    }
}

private class MlpNoGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ cfg: TransformerConfig) {
        self._linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedForward, bias: cfg.biasFF)
        self._linear2.wrappedValue = Linear(cfg.dimFeedForward, cfg.dModel, bias: cfg.biasFF)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(x)))
    }
}

final class Attention: Module {
    let cfg: TransformerConfig
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.scale = 1.0 / Float(cfg.headDim()).squareRoot()
        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * cfg.dModel / cfg.numHeads
        self._inProj.wrappedValue = Linear(cfg.dModel, outDim, bias: cfg.biasAttn)
        self._outProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn)
        self.rope = RoPE(dimensions: cfg.headDim(), traditional: true, base: cfg.maxPeriod)
    }

    /// Attention over `cache` and `x`, whose first step is at position `offset` (an array, so a
    /// compiled caller can pass it in). q has numHeads heads, k and v numHeads / kvRepeat.
    func callAsFunction(
        _ x: MLXArray, offset: MLXArray, mask: MLXArray, cache: WindowKVCache
    ) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        let (heads, headDim) = (cfg.numHeads, cfg.headDim())
        let kvHeads = heads / cfg.kvRepeat
        let qkv = inProj(x)
        var q = qkv[.ellipsis, ..<(heads * headDim)]
            .reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
        var k = qkv[.ellipsis, (heads * headDim)..<((heads + kvHeads) * headDim)]
            .reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = qkv[.ellipsis, ((heads + kvHeads) * headDim)...]
            .reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        (k, v) = cache.update(keys: k, values: v)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: self.scale, mask: mask)
        return outProj(attn.transposed(0, 2, 1, 3).reshaped(B, T, cfg.dModel))
    }
}

class LayerScale: Module, UnaryLayer {
    @ModuleInfo(key: "scale") var scale: MLXArray

    init(_ dModel: Int, initValue: Float) {
        self._scale.wrappedValue = ones([dModel]) * initValue
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * self.scale
    }
}

final class TransformerLayer: Module {
    @ModuleInfo(key: "gating") var gating: UnaryLayer
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "layer_scale_1") var layerScale1: LayerScale
    @ModuleInfo(key: "layer_scale_2") var layerScale2: LayerScale
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: TransformerConfig) {
        self._gating.wrappedValue = MlpNoGating(cfg)
        self._norm1.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._norm2.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        self._selfAttn.wrappedValue = Attention(cfg)
        self._layerScale1.wrappedValue = LayerScale(cfg.dModel, initValue: cfg.layerScale)
        self._layerScale2.wrappedValue = LayerScale(cfg.dModel, initValue: cfg.layerScale)
    }

    func callAsFunction(
        _ x: MLXArray, offset: MLXArray, mask: MLXArray, cache: WindowKVCache
    ) -> MLXArray {
        let x = x + layerScale1(selfAttn(norm1(x), offset: offset, mask: mask, cache: cache))
        return x + layerScale2(gating(norm2(x)))
    }
}

class Transformer: Module {
    let cfg: TransformerConfig
    let layers: [TransformerLayer]

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numLayers).map { _ in TransformerLayer(cfg) }
    }

    /// The layers over `x` `[B, t, dModel]`, whose first step is at position `offset`, with the
    /// caches' `mask` for t steps.
    func callAsFunction(
        _ x: MLXArray, offset: MLXArray, mask: MLXArray, cache: [WindowKVCache]
    ) -> MLXArray {
        var x = x
        for (layer, c) in zip(self.layers, cache) {
            x = layer(x, offset: offset, mask: mask, cache: c)
        }
        return x
    }

    func makeCache() -> [WindowKVCache] {
        (0..<cfg.numLayers).map { _ in WindowKVCache(context: cfg.context) }
    }
}

/// Mimi's transformer over conv-layout features `[B, C, T]` (C = dModel, so no projections).
final class ProjectedTransformer: Module {
    @ModuleInfo(key: "transformer") var transformer: Transformer

    init(_ cfg: TransformerConfig) {
        self._transformer.wrappedValue = Transformer(cfg)
    }

    func callAsFunction(
        _ x: MLXArray, offset: MLXArray, mask: MLXArray, cache: [WindowKVCache]
    ) -> MLXArray {
        self.transformer(x.swappedAxes(1, 2), offset: offset, mask: mask, cache: cache)
            .swappedAxes(1, 2)
    }

    func makeCache() -> [WindowKVCache] {
        self.transformer.makeCache()
    }
}
