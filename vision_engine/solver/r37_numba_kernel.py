"""Experimental *batch* Numba score kernel, separate from live decision code.

No fastmath, no randomness; exact integer parity checks required before use.
"""
from __future__ import annotations
import numpy as np
from vision_engine.scoring import LUCKY_MASKS, NORMAL_MASKS
from vision_engine.constants import JEWEL_SCORE, LUCKY_SCORE, NORMAL_SCORE

LUCKY = np.array(list(LUCKY_MASKS.values()), dtype=np.int64)
NORMAL = np.array(list(NORMAL_MASKS.values()), dtype=np.int64)

def score_batch_python(masks):
    """Uncached scalar score reference; useful as an apples-to-apples baseline."""
    out = np.empty(len(masks), dtype=np.int32)
    for i, mask in enumerate(masks):
        val = int(mask)
        clears = 0
        lucky = normal = 0
        for lm in LUCKY:
            lm = int(lm)
            if val & lm == lm:
                lucky += 1; clears |= lm
        for nm in NORMAL:
            nm = int(nm)
            if val & nm == nm:
                normal += 1; clears |= nm
        out[i] = lucky * LUCKY_SCORE + normal * NORMAL_SCORE + (val & ~clears).bit_count() * JEWEL_SCORE
    return out

def compiled_kernel():
    from numba import njit

    @njit(cache=True)
    def _kernel(masks, lucky_masks, normal_masks):
        out = np.empty(len(masks), dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            clears = 0
            lucky = 0
            normal = 0
            for j in range(len(lucky_masks)):
                lm = lucky_masks[j]
                if mask & lm == lm:
                    lucky += 1
                    clears |= lm
            for j in range(len(normal_masks)):
                nm = normal_masks[j]
                if mask & nm == nm:
                    normal += 1
                    clears |= nm
            leftover = mask & ~clears
            pop = 0
            while leftover:
                leftover &= leftover - 1
                pop += 1
            out[i] = lucky * LUCKY_SCORE + normal * NORMAL_SCORE + pop * JEWEL_SCORE
        return out
    return _kernel
