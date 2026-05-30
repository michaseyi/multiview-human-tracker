from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OffsetEstimate:
    tau: int             # integer frame offset that maximises ncc
    peak_value: float    # ncc at tau (close to +1 for good sync)
    curve: np.ndarray    # ncc(tau) for tau in [-max_lag, +max_lag]
    lags: np.ndarray     # tau values matching curve


def normalised_xcorr(s1: np.ndarray, s2: np.ndarray, max_lag: int) -> OffsetEstimate:
    """Pearson correlation between s1 and s2 shifted by tau, for integer tau in [-max_lag, +max_lag].

    Sign convention: if s2 lags s1 by k frames (event happens k frames
    later in s2), tau* = +k. Equivalently, s1(t) ~= s2(t + tau*).
    """
    if len(s1) != len(s2):
        raise ValueError(f"signals must have equal length, got {len(s1)} vs {len(s2)}")
    n = len(s1)
    if max_lag >= n:
        raise ValueError(f"max_lag ({max_lag}) must be < signal length ({n})")

    s1 = (s1 - s1.mean()) / (s1.std() + 1e-12)
    s2 = (s2 - s2.mean()) / (s2.std() + 1e-12)

    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    curve = np.zeros(len(lags), dtype=np.float64)
    for i, tau in enumerate(lags):
        if tau >= 0:
            a, b = s1[: n - tau], s2[tau:]
        else:
            a, b = s1[-tau:], s2[: n + tau]
        curve[i] = float((a * b).mean())

    best = int(np.argmax(curve))
    return OffsetEstimate(
        tau=int(lags[best]),
        peak_value=float(curve[best]),
        curve=curve,
        lags=lags,
    )
