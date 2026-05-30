from __future__ import annotations

import numpy as np

from multiview_tracker.epipolar.opencv_solvers import symmetric_epipolar_distance


def hartley_normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to centroid 0 and scale so mean distance from origin is sqrt(2).

    Returns (T, pts_normalized): pts_normalized is the rescaled (N, 2) array,
    T is the 3x3 homogeneous transform with [pts_n; 1] = T [pts; 1].
    """
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    mean_dist = float(np.linalg.norm(centered, axis=1).mean())
    if mean_dist < 1e-12:
        raise ValueError("degenerate points: all coincide")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0,   -scale * centroid[0]],
        [0.0,   scale, -scale * centroid[1]],
        [0.0,   0.0,    1.0],
    ])
    return T, centered * scale


def build_constraint_matrix(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Stack rows of A f = 0 from x_b^T F x_a = 0.

    Each correspondence ((u, v), (u', v')) yields one row [u'u, u'v, u', v'u, v'v, v', u, v, 1].
    """
    u, v = pts_a[:, 0], pts_a[:, 1]
    up, vp = pts_b[:, 0], pts_b[:, 1]
    return np.column_stack([
        up * u, up * v, up,
        vp * u, vp * v, vp,
        u, v, np.ones_like(u),
    ])


def enforce_rank_two(F: np.ndarray) -> np.ndarray:
    """Project F onto rank-2 by zeroing the smallest singular value."""
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    return U @ np.diag(S) @ Vt


def eight_point(pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Normalized 8-point algorithm: Hartley normalize, SVD-solve A f = 0, rank-2, denormalize."""
    if len(pts_a) < 8 or len(pts_b) < 8:
        raise ValueError(f"need >= 8 correspondences, got {len(pts_a)}")
    if len(pts_a) != len(pts_b):
        raise ValueError(f"size mismatch: {len(pts_a)} vs {len(pts_b)}")

    T_a, pts_a_n = hartley_normalize(pts_a)
    T_b, pts_b_n = hartley_normalize(pts_b)

    A = build_constraint_matrix(pts_a_n, pts_b_n)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    F_n = Vt[-1].reshape(3, 3)

    F_n = enforce_rank_two(F_n)
    return T_b.T @ F_n @ T_a


def _sampson_residuals(F: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Signed Sampson residuals: smooth approximation to symmetric epipolar distance, suitable for LM."""
    pa = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pb = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    lines_b = pa @ F.T
    lines_a = pb @ F
    num = np.einsum("ij,ij->i", pb, lines_b)  # algebraic x_b^T F x_a
    den = np.sqrt(
        lines_b[:, 0] ** 2 + lines_b[:, 1] ** 2
        + lines_a[:, 0] ** 2 + lines_a[:, 1] ** 2
        + 1e-12
    )
    return num / den


def refine_F_lm(F_init: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray) -> np.ndarray:
    """Levenberg-Marquardt refinement of F via Sampson residuals, then rank-2 re-enforcement."""
    from scipy.optimize import least_squares

    def residuals(f_vec: np.ndarray) -> np.ndarray:
        return _sampson_residuals(f_vec.reshape(3, 3), pts_a, pts_b)

    result = least_squares(
        residuals, F_init.ravel(),
        method="lm", max_nfev=200, xtol=1e-10, ftol=1e-10,
    )
    return enforce_rank_two(result.x.reshape(3, 3))


def eight_point_ransac(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    *,
    threshold_px: float = 1.0,
    n_iters: int = 1000,
    seed: int | None = 0,
    refine: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC wrapper around the 8-point solver.

    Returns (F, inlier_mask). After the inlier set is fixed, F is re-fit linearly
    on all inliers and (if ``refine`` is True) further refined by LM.
    """
    rng = np.random.default_rng(seed)
    n = len(pts_a)
    if n < 8:
        raise ValueError(f"need >= 8 correspondences, got {n}")

    best_n_inliers = -1
    best_mask: np.ndarray | None = None
    best_F: np.ndarray | None = None

    for _ in range(n_iters):
        idx = rng.choice(n, size=8, replace=False)
        try:
            F_try = eight_point(pts_a[idx], pts_b[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        d = symmetric_epipolar_distance(F_try, pts_a, pts_b)
        mask = d < threshold_px
        n_in = int(mask.sum())
        if n_in > best_n_inliers:
            best_n_inliers = n_in
            best_mask = mask
            best_F = F_try

    if best_F is None or best_n_inliers < 8:
        raise RuntimeError("RANSAC failed to find a model with >= 8 inliers")

    # linear refit on all inliers, then optional nonlinear refinement
    F = eight_point(pts_a[best_mask], pts_b[best_mask])
    if refine:
        F = refine_F_lm(F, pts_a[best_mask], pts_b[best_mask])
    return F, best_mask
