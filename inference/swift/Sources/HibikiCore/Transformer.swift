// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

public enum Norm {
    case layerNorm
    case rmsNorm
}

public enum PositionalEmbedding {
    case none
    /// Interleaved rotary embedding (MLX `traditional: true`), as in Mimi.
    case rope
    /// Rotary embedding over concatenated halves (MLX `traditional: false`), as in Hibiki-Zero.
    case ropeConcat
}

public struct TransformerConfig {
    public var dModel: Int
    public var numHeads: Int
    public var numLayers: Int
    public var causal: Bool
    public var normFirst: Bool
    public var biasFF: Bool
    public var biasAttn: Bool
    public var layerScale: Float?
    public var positionalEmbedding: PositionalEmbedding
    public var useConvBias: Bool
    public var gating: Bool
    public var norm: Norm
    public var context: Int
    public var maxPeriod: Int
    public var maxSeqLen: Int
    public var kvRepeat: Int
    public var dimFeedForward: Int
    public var convLayout: Bool

    public func headDim() -> Int {
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(geluApproximate(linear1(x)))
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
                RoPE(dimensions: cfg.headDim(), traditional: true, base: Float(cfg.maxPeriod))
            case .ropeConcat:
                RoPE(dimensions: cfg.headDim(), traditional: false, base: Float(cfg.maxPeriod))
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

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let T = x.dim(1)
        var (q, k, v) = projections(x)
        if let rope {
            let offset = cache?.offset ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let kLen = k.dim(2)
        let kTargetLen = T + min(self.cfg.context, kLen - T)
        if kTargetLen < kLen {
            let offset = kLen - kTargetLen
            k = k[0..., 0..., offset...]
            v = v[0..., 0..., offset...]
        }

        var mask = mask
        if let m = mask {
            let maskLen = m.dim(-1)
            if k.dim(2) < maskLen {
                let offset = maskLen - k.dim(2)
                mask = m[0..., offset...]
            }
        }
        return output(
            MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: self.scale, mask: mask))
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
            case .layerNorm:
                LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
            case .rmsNorm: RMSNorm(dimensions: cfg.dModel, eps: 1e-8)
            }
        self._norm2.wrappedValue =
            switch cfg.norm {
            case .layerNorm:
                LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
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

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache) -> MLXArray {
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

public class Transformer: Module {
    let cfg: TransformerConfig
    let layers: [TransformerLayer]

    public init(_ cfg: TransformerConfig) {
        self.cfg = cfg
        self.layers = (0..<cfg.numLayers).map { _ in TransformerLayer(cfg) }
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> MLXArray {
        var x = x
        let mask = cache.first?.createAttentionMask(h: x)
        for (layer, c) in zip(self.layers, cache) {
            x = layer(x, mask: mask, cache: c)
        }
        return x
    }

    public func makeCache(bSize: Int) -> [KVCache] {
        let kvHeads = cfg.numHeads / cfg.kvRepeat
        let dtype = self.layers.first!.selfAttn.inProj.weight.dtype
        return (0..<cfg.numLayers).map { _ in
            RotatingKVCache(
                bSize: bSize, numHeads: kvHeads, maxSize: cfg.context, headDim: cfg.headDim(),
                dtype: dtype)
        }
    }
}

public class ProjectedTransformer: Module {
    let convLayout: Bool
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "input_proj") var inputProj: Linear?
    @ModuleInfo(key: "output_proj") var outputProjs: [Linear?]

    init(_ cfg: TransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        self._transformer.wrappedValue = Transformer(cfg)
        if inputDim == cfg.dModel {
            self._inputProj.wrappedValue = nil
        } else {
            self._inputProj.wrappedValue = Linear(inputDim, cfg.dModel, bias: false)
        }
        var outputProjs: [Linear?] = []
        for outputDim in outputDims {
            let outputProj =
                outputDim != cfg.dModel ? Linear(cfg.dModel, outputDim, bias: false) : nil
            outputProjs.append(outputProj)
        }
        self._outputProjs.wrappedValue = outputProjs
    }

    public func callAsFunction(_ x: MLXArray, cache: [KVCache]) -> [MLXArray] {
        var x = x
        if self.convLayout {
            x = x.swappedAxes(1, 2)
        }
        if let inputProj = self.inputProj {
            x = inputProj(x)
        }
        x = self.transformer(x, cache: cache)
        var outs: [MLXArray] = []
        for outputProj in self.outputProjs {
            var out = outputProj?(x) ?? x
            if self.convLayout {
                out = out.swappedAxes(1, 2)
            }
            outs.append(out)
        }
        return outs
    }

    public func makeCache(bSize: Int) -> [KVCache] {
        self.transformer.makeCache(bSize: bSize)
    }
}
