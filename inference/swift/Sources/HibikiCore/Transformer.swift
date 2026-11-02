// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

enum Norm {
    case layerNorm
    case rmsNorm
}

enum PositionalEmbedding {
    case none
    /// Interleaved rotary embedding (MLX `traditional: true`), as in Mimi.
    case rope
    /// Rotary embedding over concatenated halves (MLX `traditional: false`), as in Hibiki-Zero.
    case ropeConcat
}

struct TransformerConfig {
    var dModel: Int
    var numHeads: Int
    var numLayers: Int
    var biasFF: Bool
    var biasAttn: Bool
    var layerScale: Float?
    var positionalEmbedding: PositionalEmbedding
    var gating: Bool
    var norm: Norm
    var context: Int
    var maxPeriod: Float
    var kvRepeat: Int
    var dimFeedForward: Int

    func headDim() -> Int {
        self.dModel / self.numHeads
    }
}

final class MlpGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linear_in: Linear
    @ModuleInfo(key: "linear_out") var linear_out: Linear

    init(_ cfg: TransformerConfig) {
        let hidden =
            cfg.dimFeedForward == 4 * cfg.dModel ? 11 * cfg.dModel / 4 : 2 * cfg.dimFeedForward / 3
        self._linear_in.wrappedValue = Linear(cfg.dModel, 2 * hidden, bias: cfg.biasFF)
        self._linear_out.wrappedValue = Linear(hidden, cfg.dModel, bias: cfg.biasFF)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = linear_in(x)
        let (B, T) = (x.dim(0), x.dim(1))
        let x_reshaped = x.reshaped(B, T, 2, -1)
        return linear_out(silu(x_reshaped[0..., 0..., 0]) * x_reshaped[0..., 0..., 1])
    }
}

private class MlpNoGating: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ cfg: TransformerConfig) {
        self._linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedForward, bias: cfg.biasFF)
        self._linear2.wrappedValue = Linear(cfg.dimFeedForward, cfg.dModel, bias: cfg.biasFF)
    }

    /// The tanh GELU written out: MLXNN's `geluApproximate` is a compiled function, and iOS
    /// cannot compile CPU kernels.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = linear1(x)
        let inner = (2 / Float.pi).squareRoot() * (h + 0.044715 * h * h * h)
        return linear2(0.5 * h * (1 + tanh(inner)))
    }
}

final class Attention: Module {
    let cfg: TransformerConfig
    let scale: Float
    let rope: RoPE?

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.scale = 1.0 / Float(cfg.headDim()).squareRoot()
        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * cfg.dModel / cfg.numHeads
        self._inProj.wrappedValue = Linear(cfg.dModel, outDim, bias: cfg.biasAttn)
        self._outProj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn)
        self.rope =
            switch cfg.positionalEmbedding {
            case .none: nil
            case .rope:
                RoPE(dimensions: cfg.headDim(), traditional: true, base: cfg.maxPeriod)
            case .ropeConcat:
                RoPE(dimensions: cfg.headDim(), traditional: false, base: cfg.maxPeriod)
            }
    }

    /// Grouped-query projections before RoPE: q has numHeads heads, k and v
    /// numHeads / kvRepeat, each `[B, heads, T, headDim]`.
    func projections(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, T) = (x.dim(0), x.dim(1))
        let (heads, headDim) = (cfg.numHeads, cfg.headDim())
        let kvHeads = heads / cfg.kvRepeat
        let qkv = inProj(x)
        let q = qkv[.ellipsis, ..<(heads * headDim)]
            .reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
        let k = qkv[.ellipsis, (heads * headDim)..<((heads + kvHeads) * headDim)]
            .reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = qkv[.ellipsis, ((heads + kvHeads) * headDim)...]
            .reshaped(B, T, kvHeads, headDim).transposed(0, 2, 1, 3)
        return (q, k, v)
    }

    /// The output projection of an attention result `[B, heads, T, headDim]`.
    func output(_ attn: MLXArray) -> MLXArray {
        outProj(attn.transposed(0, 2, 1, 3).reshaped(attn.dim(0), attn.dim(2), cfg.dModel))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: WindowKVCache) -> MLXArray {
        var (q, k, v) = projections(x)
        if let rope {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        }
        (k, v) = cache.update(keys: k, values: v)
        return output(
            MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: self.scale, mask: mask))
    }
}

/// LayerNorm composed from mean and variance, for Mimi on the CPU, where MLXFast.layerNorm's
/// fallback takes ~8x as long. Only Mimi's transformers use `Norm.layerNorm`.
final class ComposedLayerNorm: LayerNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let centered = x - x.mean(axis: -1, keepDims: true)
        var y = centered * rsqrt((centered * centered).mean(axis: -1, keepDims: true) + eps)
        if let weight { y = y * weight }
        if let bias { y = y + bias }
        return y
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
    @ModuleInfo(key: "norm1") var norm1: UnaryLayer
    @ModuleInfo(key: "norm2") var norm2: UnaryLayer
    @ModuleInfo(key: "layer_scale_1") var layerScale1: LayerScale?
    @ModuleInfo(key: "layer_scale_2") var layerScale2: LayerScale?
    @ModuleInfo(key: "self_attn") var selfAttn: Attention

    init(_ cfg: TransformerConfig) {
        self._gating.wrappedValue = cfg.gating ? MlpGating(cfg) : MlpNoGating(cfg)
        self._norm1.wrappedValue =
            switch cfg.norm {
            case .layerNorm: ComposedLayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
            }
        self._norm2.wrappedValue =
            switch cfg.norm {
            case .layerNorm: ComposedLayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
            }
        self._selfAttn.wrappedValue = Attention(cfg)

        if let scale = cfg.layerScale {
            self._layerScale1.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
            self._layerScale2.wrappedValue = LayerScale(cfg.dModel, initValue: scale)
        } else {
            self._layerScale1.wrappedValue = nil
            self._layerScale2.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: WindowKVCache) -> MLXArray {
        residualFeedForward(x, attended: selfAttn(norm1(x), mask: mask, cache: cache))
    }

    /// Adds the projected attention output to the residual, then the feed-forward block.
    func residualFeedForward(_ x: MLXArray, attended: MLXArray) -> MLXArray {
        var x = x + (layerScale1?.callAsFunction(attended) ?? attended)
        let ff = gating(norm2(x))
        x = x + (layerScale2?.callAsFunction(ff) ?? ff)
        return x
    }
}

class Transformer: Module {
    let cfg: TransformerConfig
    let layers: [TransformerLayer]

    init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numLayers).map { _ in TransformerLayer(cfg) }
    }

    func callAsFunction(_ x: MLXArray, cache: [WindowKVCache]) -> MLXArray {
        var x = x
        let mask = cache.first?.mask(steps: x.dim(1), dtype: x.dtype)
        for (layer, c) in zip(self.layers, cache) {
            x = layer(x, mask: mask, cache: c)
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

    func callAsFunction(_ x: MLXArray, cache: [WindowKVCache]) -> MLXArray {
        self.transformer(x.swappedAxes(1, 2), cache: cache).swappedAxes(1, 2)
    }

    func makeCache() -> [WindowKVCache] {
        self.transformer.makeCache()
    }
}
