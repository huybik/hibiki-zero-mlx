"""Hub commits shared by the EN-VI dataset's publisher (`en_vi_dataset.publish`) and the cache
publisher (`cache.publish`): their size and their retries."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from training.paths import log

RETRIES = 5  # a failed commit is retried after 1, 2, 4, 8 and 16 minutes, then stops the run
RETRY_WAIT = 60
# 2-7 min of upload, all a stop loses; ~20-30 commits an hour, under the Hub's commit quota (~100
# an hour). Seeding a repository with bytes Xet already holds ran at ~110 an hour, none refused.
COMMIT_BYTES = 4_000_000_000
T = TypeVar("T")


def retried(what: str, fn: Callable[[], T]) -> T:
    for attempt in range(RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == RETRIES:
                raise
            wait = RETRY_WAIT * 2**attempt
            log(f"{what} failed ({type(e).__name__}: {e}); retrying in {wait // 60} min")
            time.sleep(wait)
