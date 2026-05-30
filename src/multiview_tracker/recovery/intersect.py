from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IntersectionResult:
    point: np.ndarray            # (2,) recovered point in image_a pixel coords
    method: str                  # "cross_product" or "least_squares"
    sin_angle: float             # smallest |sin(angle)| across line pairs
    parallel: bool               # True if rejected as near-parallel


def epipolar_line_in_a(F_ab: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Return the epipolar line in image A for a point in image B.

    Convention x_b^T F_ab x_a = 0, so the line in A is F_ab^T x_b.
    """
    pb_h = np.array([point_b[0], point_b[1], 1.0])
    return F_ab.T @ pb_h


def line_line_intersect(l1: np.ndarray, l2: np.ndarray) -> tuple[np.ndarray, float]:
    """Cross-product intersection of two homogeneous lines.

    Returns (intersection_xy, |sin(angle)|). |sin(angle)| approaches 0 for parallel
    lines; callers should fall back to least-squares in that case.
    """
    p_h = np.cross(l1, l2)
    n1 = np.array([l1[0], l1[1]])
    n2 = np.array([l2[0], l2[1]])
    sin_angle = abs(float(np.cross(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12))
    if abs(p_h[2]) < 1e-12:
        return np.array([np.nan, np.nan]), sin_angle
    return np.array([p_h[0] / p_h[2], p_h[1] / p_h[2]]), sin_angle


def least_squares_intersection(lines: np.ndarray) -> np.ndarray:
    """Point minimising sum of squared point-line distances.

    Each line (a, b, c) is normalised so a^2 + b^2 = 1, then minimise Σ(a x + b y + c)^2.
    """
    if len(lines) < 2:
        raise ValueError("need >= 2 lines")
    n = np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12
    lines = lines / n
    A = lines[:, :2]                 # (N, 2)
    rhs = -lines[:, 2]               # (N,)
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return sol


def recover_from_views_undist(
    support_views: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    primary_K: np.ndarray,
    primary_D: np.ndarray,
    *,
    parallel_threshold: float = 0.05,
) -> "IntersectionResult":
    """Undistort-first recovery, returning a point in raw primary pixel coords.

    Each support view is (K_b, D_b, F_ab_undist, point_b_raw):
      - K_b, D_b: support camera intrinsics for undistorting point_b
      - F_ab_undist: fundamental matrix fit in undistorted pixel space
      - point_b_raw: support point in raw (distorted) pixel coords

    Undistort each support point, intersect epipolar lines in the primary's
    undistorted space, then re-distort to raw primary pixel coords.
    """
    from multiview_tracker.epipolar.undistort import distort_points, undistort_points

    if len(support_views) < 2:
        raise ValueError("need >= 2 supporting views")

    lines = []
    for K_b, D_b, F_ab, point_b_raw in support_views:
        pb_raw = np.asarray(point_b_raw, dtype=np.float32).reshape(1, 2)
        pb_u = undistort_points(pb_raw, K_b, D_b)[0]
        lines.append(epipolar_line_in_a(F_ab, pb_u))
    lines = np.stack(lines)

    # parallel check
    sin_min = 1.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            n1, n2 = lines[i, :2], lines[j, :2]
            s = abs(float(np.cross(n1, n2)) /
                    (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12))
            sin_min = min(sin_min, s)
    parallel = sin_min < parallel_threshold

    if len(lines) == 2 and not parallel:
        point_u, _ = line_line_intersect(lines[0], lines[1])
        method = "cross_product"
    else:
        point_u = least_squares_intersection(lines)
        method = "least_squares"

    if np.isnan(point_u).any():
        return IntersectionResult(point=point_u, method=method,
                                  sin_angle=sin_min, parallel=parallel)
    # re-distort to raw primary pixel space
    point_raw = distort_points(
        np.asarray(point_u, dtype=np.float32).reshape(1, 2),
        primary_K, primary_D,
    )[0]
    return IntersectionResult(point=point_raw, method=method,
                              sin_angle=sin_min, parallel=parallel)


def recover_from_views(
    point_in_views: list[tuple[np.ndarray, np.ndarray]],
    *,
    parallel_threshold: float = 0.05,
) -> IntersectionResult:
    """Recover a missing point in image_a from N supporting (F_ab, point_b) detections.

    For N = 2 uses cross-product intersection with a parallel check; for N > 2
    or near-parallel lines falls back to least squares.
    """
    if len(point_in_views) < 2:
        raise ValueError("need >= 2 supporting views")

    lines = np.stack([epipolar_line_in_a(F, p) for F, p in point_in_views])

    # min sin(angle) across all pairs measures how parallel the worst pair is
    sin_min = 1.0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            n1, n2 = lines[i, :2], lines[j, :2]
            s = abs(float(np.cross(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-12))
            sin_min = min(sin_min, s)

    parallel = sin_min < parallel_threshold

    if len(lines) == 2 and not parallel:
        point, _ = line_line_intersect(lines[0], lines[1])
        return IntersectionResult(point=point, method="cross_product",
                                  sin_angle=sin_min, parallel=False)

    # fall back to least squares (also used for >2 supporting views)
    point = least_squares_intersection(lines)
    return IntersectionResult(point=point, method="least_squares",
                              sin_angle=sin_min, parallel=parallel)
