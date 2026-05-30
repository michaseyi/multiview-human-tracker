from __future__ import annotations

import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection
from multiview_tracker.sync.actor import high_conf_centroid


def centroid_y_per_frame(
    detections: list[PoseDetection],
    n_frames: int,
) -> np.ndarray:
    """Per-frame centroid-y signal; NaN where no detection is available."""
    sig = np.full(n_frames, np.nan, dtype=np.float64)
    for d in detections:
        c = high_conf_centroid(d)
        if c is None:
            continue
        if 0 <= d.frame_idx < n_frames:
            sig[d.frame_idx] = float(c[1])
    return sig


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN runs; extrapolate edges with nearest value."""
    x = x.copy()
    n = len(x)
    nans = np.isnan(x)
    if not nans.any():
        return x
    if nans.all():
        return np.zeros_like(x)
    idx = np.arange(n)
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving-average smoothing."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def velocity(x: np.ndarray) -> np.ndarray:
    """First difference; length unchanged, first sample = 0."""
    v = np.zeros_like(x)
    v[1:] = np.diff(x)
    return v


def build_motion_signal(
    detections: list[PoseDetection],
    n_frames: int,
    smooth_window: int = 5,
) -> np.ndarray:
    """centroid_y_per_frame -> fill_nans -> smooth -> velocity."""
    raw = centroid_y_per_frame(detections, n_frames)
    raw = fill_nans(raw)
    raw = smooth(raw, smooth_window)
    return velocity(raw)
