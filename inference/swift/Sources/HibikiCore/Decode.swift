// Compiled one-token decode for the LM, as the fast paths in moshi_mlx: every compiled region
// sees fixed shapes, so each compiles once.
//
//   main transformer, per layer:  [compiled pre] ─► cache write + attention ─► [compiled post]
//   depformer, per codebook step: [compiled step over the whole slice, KV passed in and out]

import MLX
import MLXNN

/// Keys and values of one main-transformer layer for one-token steps: a ring of `context`
/// slots. Once RoPE is applied, attention does not depend on slot order, so a full ring needs
/// no mask and a partial one is read as a prefix.
final class DecodeCache {
    let context: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0

    init(context: Int) {
        self.context = context
    }

    func reset() {
        offset = 0
    }

    func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if keys == nil {
            keys = MLXArray.zeros([1, k.dim(1), context, k.dim(3)], dtype: k.dtype)
            values = MLXArray.zeros([1, v.dim(1), context, v.dim(3)], dtype: v.dtype)
        }
        let slot = offset % context
        keys![0..., 0..., slot..<(slot + 1)] = k
        values![0..., 0..., slot..<(slot + 1)] = v
        offset += 1
        if offset < context {
            return (keys![0..., 0..., ..<offset], values![0..., 0..., ..<offset])
        }
        return (keys!, values!)
    }
}

/// The main transformer for one-token steps.
final class DecodeTransformer {
    private let layers: [TransformerLayer]
    private let pre: [@Sendable ([MLXArray]) -> [MLXArray]]
    private let post: [@Sendable ([MLXArray]) -> [MLXArray]]
    private let caches: [DecodeCache]

    init(_ transformer: Transformer) {
        layers = transformer.layers
        caches = layers.map { _ in DecodeCache(context: transformer.cfg.context) }
        // norm1, projections and RoPE at the offset passed as an array.
        pre = layers.map { layer in
            compile { args in
                let attn = layer.selfAttn
                var (q, k, v) = attn.projections(layer.norm1(args[0]))
                if let rope = attn.rope {
                    q = rope(q, offset: args[1])
                    k = rope(k, offset: args[1])
                }
                return [q, k, v]
            }
        }
        // Output projection, residual and feed-forward.
        post = layers.map { layer in
            compile { args in
                [layer.residualFeedForward(args[0], attended: layer.selfAttn.output(args[1]))]
            }
        }
    }

    func reset() {
        caches.forEach { $0.reset() }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(1) == 1, "decode steps take one token")
        let offset = MLXArray(Int32(caches[0].offset))
        var x = x
        for (idx, layer) in layers.enumerated() {
            let qkv = pre[idx]([x, offset])
            let (k, v) = caches[idx].update(qkv[1], qkv[2])
            let attn = MLXFast.scaledDotProductAttention(
                queries: qkv[0], keys: k, values: v, scale: layer.selfAttn.scale, mask: nil)
            x = post[idx]([x, attn])[0]
        }
        return x
    }
}

/// One depformer codebook step compiled per slice: the slice's input, its core's layers with
/// the keys and values of the previous steps passed in and returned, the output norm and head.
func compileDepformerStep(core: DepformerCore, slice: DepformerSlice)
    -> @Sendable ([MLXArray]) -> [MLXArray]
{
    compile { args in
        // args: main transformer output, previous token, then k, v per layer (none at step 0).
        var xs = core.linearIn(args[0]) + slice.emb(args[1])
        var kv: [MLXArray] = []
        for (idx, layer) in core.transformer.layers.enumerated() {
            var (q, k, v) = layer.selfAttn.projections(layer.norm1(xs))
            if args.count > 2 {
                k = concatenated([args[2 + 2 * idx], k], axis: 2)
                v = concatenated([args[3 + 2 * idx], v], axis: 2)
            }
            let attn = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: layer.selfAttn.scale, mask: nil)
            xs = layer.residualFeedForward(xs, attended: layer.selfAttn.output(attn))
            kv += [k, v]
        }
        if let norm = slice.norm { xs = norm(xs) }
        return [slice.linearOut(xs)[0..., -1, 0...]] + kv
    }
}
