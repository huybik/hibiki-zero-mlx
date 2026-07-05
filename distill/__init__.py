"""Track B — parallel codebook head via self-distillation (distill_plan.md).

Scaffold: dump_teacher (frozen teacher logits), parallel_head (delay-pattern
parallel head), train_head (KL+CE distill on head params only). See
reports/parallel_head_smoke.md.
"""
from .parallel_head import ParallelHead, ParallelHeadConfig, build_head, warm_start

__all__ = ["ParallelHead", "ParallelHeadConfig", "build_head", "warm_start"]
