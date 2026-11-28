"""An ONNX graph run as PyTorch ops on the GPU, compiled by Inductor if asked: VieNeu-TTS v3 Turbo's denoiser
and speaker encoder in engine.Voicer ([tranche-tts-scaleup](../../../docs/agent-memory/tranche-tts-scaleup.md)).

onnx2torch's modules read shapes and axes out of tensors as they run, which Dynamo cannot trace. Here the first run
at each input shape computes every shape value in NumPy and records the tensor ops as closures over those values;
later calls at that shape replay the ops (through torch.compile when `compile`, one graph per input shape).

Covers the ops those two graphs use; any other op, or a form of one this does not handle, raises.
"""
from __future__ import annotations

import functools
import operator

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import helper, numpy_helper

NUMPY_DTYPES = {1: np.float32, 6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64}
TORCH_DTYPES = {1: torch.float32, 6: torch.int32, 7: torch.int64, 9: torch.bool, 10: torch.float16}


def _scalar_or_const(v, like):
    """A static operand beside a tensor: a Python scalar when it has one element, else a constant tensor."""
    v = np.asarray(v)
    if v.size == 1:
        return v.reshape(-1)[0].item()
    return torch.as_tensor(v, dtype=like.dtype, device=like.device)


def _binary(op, int_op=None):
    def plan(at, a, b):
        (sa, va), (sb, vb) = a, b
        if sa and sb:
            ints = int_op is not None and np.asarray(va).dtype.kind in "iu" and np.asarray(vb).dtype.kind in "iu"
            return "static", np.asarray((int_op if ints else op)(va, vb))
        if sa:
            c = _scalar_or_const(va, vb)
            return "step", (lambda y, c=c: op(c, y)), [1]
        if sb:
            c = _scalar_or_const(vb, va)
            return "step", (lambda x, c=c: op(x, c)), [0]
        return "step", op, [0, 1]
    return plan


def _unary(tensor_fn, numpy_fn=None):
    def plan(at, x):
        static, v = x
        if static:
            if numpy_fn is None:
                raise NotImplementedError("a static input to a tensor-only op")
            return "static", numpy_fn(v)
        return "step", tensor_fn, [0]
    return plan


def _static_only(name, fn):
    def plan(at, *args):
        if not all(a is None or a[0] for a in args):
            raise NotImplementedError(f"{name} on a tensor")
        return "static", np.asarray(fn(at, *[None if a is None else a[1] for a in args]))
    return plan


def _shape(at, x):
    shape = np.array(x[1].shape, dtype=np.int64)
    return "static", shape[at.get("start", 0): at.get("end", None)]


def _axes(axes, rank):
    return sorted(int(a) % rank for a in np.asarray(axes).reshape(-1))


def _unsqueeze(at, data, axes):
    ax = _axes(axes[1], np.ndim(data[1]) + np.asarray(axes[1]).size)
    if data[0]:
        v = data[1]
        for a in ax:
            v = np.expand_dims(v, a)
        return "static", v

    def unsqueeze(x, ax=tuple(ax)):
        for a in ax:
            x = x.unsqueeze(a)
        return x
    return "step", unsqueeze, [0]


def _squeeze(at, data, axes):
    ax = tuple(_axes(axes[1], np.ndim(data[1])))
    if data[0]:
        return "static", np.squeeze(data[1], axis=ax)
    return "step", (lambda x, ax=ax: x.squeeze(ax)), [0]


def _concat(at, *xs):
    axis = at["axis"]
    if all(s for s, _ in xs):
        return "static", np.concatenate([v for _, v in xs], axis=axis)
    if any(s for s, _ in xs):
        raise NotImplementedError("Concat of tensors and static values")
    return "step", (lambda *ts, axis=axis: torch.cat(ts, axis)), list(range(len(xs)))


def _cast(at, x):
    if x[0]:
        return "static", x[1].astype(NUMPY_DTYPES[at["to"]])
    return "step", (lambda t, dt=TORCH_DTYPES[at["to"]]: t.to(dt)), [0]


def _constant_of_shape(at, shape):
    value = numpy_helper.to_array(at["value"]).reshape(-1)[0] if "value" in at else np.float32(0)
    return "static", np.full(shape[1].astype(np.int64), value, dtype=np.asarray(value).dtype)


def _reshape(at, data, shape):
    if at.get("allowzero", 0):
        raise NotImplementedError("Reshape with allowzero")
    src = np.shape(data[1])
    target = tuple(int(src[i]) if int(s) == 0 else int(s) for i, s in enumerate(shape[1]))
    if data[0]:
        return "static", data[1].reshape(target)
    return "step", (lambda x, target=target: x.reshape(target)), [0]


def _expand(at, data, shape):
    if not shape[0]:
        raise NotImplementedError("Expand to a computed shape")
    full = tuple(np.broadcast_shapes(np.shape(data[1]), tuple(int(s) for s in shape[1])))
    if data[0]:
        return "static", np.broadcast_to(data[1], full).copy()
    return "step", (lambda x, full=full: x.expand(full)), [0]


def _slices(shape, starts, ends, axes, steps):
    """ONNX Slice's clamping as Python slices."""
    rank = len(shape)
    axes = list(range(len(starts))) if axes is None else [int(a) % rank for a in axes]
    steps = [1] * len(starts) if steps is None else [int(s) for s in steps]
    sl = [slice(None)] * rank
    for a, b, e, st in zip(axes, starts, ends, steps):
        n = int(shape[a])
        b, e = int(b), int(e)
        b, e = (b + n if b < 0 else b), (e + n if e < 0 else e)
        if st > 0:
            sl[a] = slice(max(0, min(n, b)), max(0, min(n, e)), st)
        elif st < 0:
            b, e = max(0, min(n - 1, b)), max(-1, min(n - 1, e))
            sl[a] = slice(b, e if e >= 0 else None, st)
        else:
            raise ValueError("Slice step 0")
    return tuple(sl)


def _slice(at, data, starts, ends, axes=None, steps=None):
    sl = _slices(np.shape(data[1]), starts[1], ends[1], None if axes is None else axes[1],
                 None if steps is None else steps[1])
    if data[0]:
        return "static", data[1][sl]
    if any(s.step is not None and s.step < 0 for s in sl):
        raise NotImplementedError("Slice of a tensor with a negative step")
    return "step", (lambda x, sl=sl: x[sl]), [0]


def _transpose(at, x):
    perm = tuple(at["perm"]) if "perm" in at else tuple(reversed(range(np.ndim(x[1]))))
    if x[0]:
        return "static", np.transpose(x[1], perm)
    return "step", (lambda t, perm=perm: t.permute(perm)), [0]


def _reduce_mean(at, x, axes=None):
    ax = at.get("axes") if axes is None else np.asarray(axes[1]).reshape(-1).tolist()
    keep = bool(at.get("keepdims", 1))
    if x[0]:
        return "static", np.mean(x[1], axis=None if ax is None else tuple(ax), keepdims=keep)
    dims = tuple(range(x[1].dim())) if ax is None else tuple(int(a) for a in ax)
    return "step", (lambda t, dims=dims, keep=keep: t.mean(dims, keepdim=keep)), [0]


def _pad(at, data, pads, value=None):
    if at.get("mode", b"constant") not in (b"constant", "constant") or data[0]:
        raise NotImplementedError("Pad other than constant, of a tensor")
    p = [int(v) for v in pads[1]]
    r = len(p) // 2
    torch_pads = []
    for d in reversed(range(r)):
        torch_pads += [p[d], p[d + r]]
    cv = 0.0 if value is None else float(np.asarray(value[1]).reshape(-1)[0])
    return "step", (lambda x, tp=tuple(torch_pads), cv=cv: F.pad(x, tp, value=cv)), [0]


def _same_pads(at, r):
    pads = list(at.get("pads", [0] * 2 * r))
    if pads[:r] != pads[r:] or at.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
        raise NotImplementedError("asymmetric or automatic padding")
    return tuple(pads[:r])


def _conv(at, x, w, b=None):
    r = np.ndim(w[1]) - 2
    conv = {1: F.conv1d, 2: F.conv2d}[r]
    kw = dict(stride=tuple(at.get("strides", [1] * r)), padding=_same_pads(at, r),
              dilation=tuple(at.get("dilations", [1] * r)), groups=at.get("group", 1))
    if b is None:
        return "step", (lambda x_, w_, kw=kw: conv(x_, w_, None, **kw)), [0, 1]
    return "step", (lambda x_, w_, b_, kw=kw: conv(x_, w_, b_, **kw)), [0, 1, 2]


def _avg_pool(at, x):
    k = tuple(at["kernel_shape"])
    pool = {1: F.avg_pool1d, 2: F.avg_pool2d}[len(k)]
    kw = dict(kernel_size=k, stride=tuple(at.get("strides", [1] * len(k))), padding=_same_pads(at, len(k)),
              ceil_mode=bool(at.get("ceil_mode", 0)), count_include_pad=bool(at.get("count_include_pad", 0)))
    return "step", (lambda t, kw=kw: pool(t, **kw)), [0]


def _batch_norm(at, x, scale, bias, mean, var):
    if at.get("training_mode", 0):
        raise NotImplementedError("BatchNormalization in training mode")
    eps = float(at.get("epsilon", 1e-5))
    return "step", (lambda x_, s_, b_, m_, v_, eps=eps: F.batch_norm(x_, m_, v_, s_, b_, False, 0.0, eps)), [0, 1, 2, 3, 4]


def instance_norm(x, scale, bias, eps):
    """F.instance_norm from fp32 sums reduced one axis at a time: Inductor gives F.instance_norm's reduction one
    program per (batch, channel), each over a whole 841 x 841 plane (the denoiser's are 57% of its time). This is
    11% faster on the denoiser, 51.5 dB from its fp32 eager graph either way."""
    xf = x.float()
    dims = tuple(range(2, x.ndim))
    n = 1
    for d in dims:
        n *= x.shape[d]
    s1, s2 = xf, xf.square()
    for d in reversed(dims):
        s1, s2 = s1.sum(d, keepdim=True), s2.sum(d, keepdim=True)
    mean = s1 / n
    var = (s2 / n - mean.square()).clamp(min=0)
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return ((xf - mean) * torch.rsqrt(var + eps) * scale.float().view(shape) + bias.float().view(shape)).to(x.dtype)


def _instance_norm(at, x, scale, bias):
    eps = float(at.get("epsilon", 1e-5))
    return "step", (lambda x_, s_, b_, eps=eps: instance_norm(x_, s_, b_, eps)), [0, 1, 2]


def _resize(at, x, roi=None, scales=None, sizes=None):
    if (at.get("mode"), at.get("coordinate_transformation_mode"), at.get("nearest_mode")) != (b"nearest", b"asymmetric", b"floor"):
        raise NotImplementedError("Resize other than nearest, asymmetric, floor")
    if sizes is not None and np.asarray(sizes[1]).size:
        size = tuple(int(s) for s in sizes[1][2:])
        return "step", (lambda t, size=size: F.interpolate(t, size=size, mode="nearest")), [0]
    factor = tuple(float(s) for s in scales[1][2:])
    return "step", (lambda t, factor=factor: F.interpolate(t, scale_factor=factor, mode="nearest")), [0]


def _split(at, x, split=None):
    if split is None:
        raise NotImplementedError("Split without sizes")
    sizes = [int(s) for s in split[1]]
    return "step", (lambda t, sizes=sizes, axis=at.get("axis", 0): torch.split(t, sizes, axis)), [0]


PLAN = {
    "Add": _binary(operator.add), "Sub": _binary(operator.sub), "Mul": _binary(operator.mul),
    "Div": _binary(operator.truediv, np.floor_divide), "Pow": _binary(operator.pow), "Mod": _binary(np.mod),
    "Equal": _static_only("Equal", lambda at, a, b: np.equal(a, b)),
    "Where": _static_only("Where", lambda at, c, a, b: np.where(c, a, b)),
    "Gather": _static_only("Gather", lambda at, d, i: np.take(d, i, axis=at.get("axis", 0))),
    "ReduceProd": _static_only("ReduceProd", lambda at, x, axes=None: np.prod(
        x, axis=None if at.get("axes", axes) is None else tuple(np.asarray(at.get("axes", axes)).reshape(-1)),
        keepdims=bool(at.get("keepdims", 1)))),
    "Erf": _unary(torch.erf), "Relu": _unary(torch.relu), "Sigmoid": _unary(torch.sigmoid),
    "Tanh": _unary(torch.tanh), "Sqrt": _unary(torch.sqrt, np.sqrt),
    "Shape": _shape, "Unsqueeze": _unsqueeze, "Squeeze": _squeeze, "Concat": _concat, "Cast": _cast,
    "ConstantOfShape": _constant_of_shape, "Reshape": _reshape, "Expand": _expand, "Slice": _slice,
    "Transpose": _transpose, "ReduceMean": _reduce_mean, "Pad": _pad, "Conv": _conv, "AveragePool": _avg_pool,
    "BatchNormalization": _batch_norm, "InstanceNormalization": _instance_norm, "Resize": _resize, "Split": _split,
}


class OnnxGraph:
    """`graph(*inputs)`: the graph's outputs, a tuple. With `compile`, each input shape's ops run through
    torch.compile in `mode`; the first call at a shape plans and compiles."""

    def __init__(self, path: str, device, dtype=torch.float32, compile: bool = False, mode: str = "default"):
        g = onnx.load(path).graph
        self.in_names = [i.name for i in g.input]
        self.out_names = [o.name for o in g.output]
        self.device, self.dtype = device, dtype
        self.weights = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()).to(device, dtype)
                        for i in g.initializer}
        self.consts, self.nodes = {}, []
        for n in g.node:
            at = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            if n.op_type == "Constant":
                if "value" not in at:
                    raise NotImplementedError(f"a Constant without a tensor value: {sorted(at)}")
                self.consts[n.output[0]] = numpy_helper.to_array(at["value"])
            elif n.op_type not in PLAN:
                raise NotImplementedError(f"ONNX op {n.op_type}")
            else:
                self.nodes.append((n.op_type, list(n.input), list(n.output), at))
        self.compile, self.mode = compile, mode
        if compile:  # every shape's plan is the same code (run) over other values: each one is a recompile
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)
        self.plans: dict = {}

    @torch.no_grad()
    def plan(self, *xs) -> list:
        """One eager run: the tensor ops, each a closure over its static values, with the names it reads and
        writes and those no later op reads (dropped after it)."""
        tens = dict(zip(self.in_names, xs))
        tens.update(self.weights)
        stat = dict(self.consts)
        last = {}
        for k, (_, ins, _, _) in enumerate(self.nodes):
            for n in ins:
                last[n] = k
        steps = []
        for k, (op, ins, outs, at) in enumerate(self.nodes):
            args = [None if not i else ((True, stat[i]) if i in stat else (False, tens[i])) for i in ins]
            kind, *rest = PLAN[op](at, *args)
            if kind == "static":
                stat[outs[0]] = np.asarray(rest[0])
                continue
            fn, idx = rest
            names = [ins[j] for j in idx]
            for n in names:  # a static value a tensor op reads (InstanceNormalization's scale): a constant
                if n not in tens:
                    v = stat[n]
                    self.weights[n] = tens[n] = torch.tensor(
                        v, device=self.device, dtype=self.dtype if v.dtype.kind == "f" else torch.int64)
            res = fn(*[tens[n] for n in names])
            tens.update({outs[0]: res} if len(outs) == 1 else zip(outs, res))
            steps.append((fn, names, outs))
            for n in set(ins):
                if last.get(n) == k and n in tens and n not in self.out_names and n not in self.weights:
                    del tens[n]
        last = {}
        for k, (_, names, _) in enumerate(steps):
            for n in names:
                last[n] = k
        keep = set(self.out_names) | set(self.weights)
        return [(fn, names, outs, [n for n in set(names) if last[n] == k and n not in keep])
                for k, (fn, names, outs) in enumerate(steps)]

    def run(self, steps: list, *xs) -> tuple:
        env = dict(zip(self.in_names, xs))
        env.update(self.weights)
        for fn, names, outs, done in steps:
            res = fn(*[env[n] for n in names])
            env.update({outs[0]: res} if len(outs) == 1 else zip(outs, res))
            for n in done:
                del env[n]
        return tuple(env[o] for o in self.out_names)

    def __call__(self, *xs) -> tuple:
        key = tuple(tuple(x.shape) for x in xs) + (xs[0].dtype,)
        if key not in self.plans:
            fn = functools.partial(self.run, self.plan(*xs))
            self.plans[key] = torch.compile(fn, dynamic=False, mode=self.mode) if self.compile else fn
        with torch.no_grad():
            return self.plans[key](*xs)
