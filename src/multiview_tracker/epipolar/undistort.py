"""Undistort/redistort utilities for working in ideal pinhole pixel space.

F is only valid under pinhole projection, but the lenses have substantial
Brown-Conrady distortion (k1 ~ -0.43). Pixels are undistorted before any
epipolar math and re-distorted back to raw coordinates only for drawing.
"""
from __future__ import annotations

import cv2
import numpy as np


def undistort_points(pts: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Map raw pixel coords to undistorted pixel coords (same K).

    Uses cv2.undistortPointsIter with tight termination (100 iters, eps=1e-9);
    the default 5-iteration cv2.undistortPoints silently returns non-converged
    values near image corners (300+ px round-trip errors observed).

    pts: (N, 2) in raw pixel space.
    Returns: (N, 2) in undistorted pixel space (K's pixel units).
    """
    if len(pts) == 0:
        return pts.copy()
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-9)
    u = cv2.undistortPointsIter(
        p, K, D, R=np.eye(3, dtype=np.float64), P=K, criteria=criteria,
    ).reshape(-1, 2)
    return u.astype(np.float32)


def distort_points(pts_u: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Map undistorted pixel coords to raw (distorted) pixel coords.

    Closed-form Brown-Conrady forward distortion: stable even at extreme positions
    where the cv2.projectPoints trick diverges.

    pts_u: (N, 2) in undistorted pixel space.
    Returns: (N, 2) in raw pixel space.
    """
    if len(pts_u) == 0:
        return pts_u.copy()
    pts_u = np.asarray(pts_u, dtype=np.float64).reshape(-1, 2)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    D = np.asarray(D, dtype=np.float64).flatten()
    k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
    k3 = D[4] if len(D) >= 5 else 0.0

    # normalised undistorted coords
    x = (pts_u[:, 0] - cx) / fx
    y = (pts_u[:, 1] - cy) / fy
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    out = np.empty_like(pts_u)
    out[:, 0] = fx * x_d + cx
    out[:, 1] = fy * y_d + cy
    return out.astype(np.float32)
