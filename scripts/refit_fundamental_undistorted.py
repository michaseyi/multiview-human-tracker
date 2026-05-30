"""Refit F in undistorted pixel space from a saved board-correspondence harvest.

Loads pts_a, pts_b, K, D from F_camA_camB.npz, undistorts both point sets, then
refits F three ways and saves to F_camA_camB_undist.npz alongside the original.
Downstream callers that also undistort their points get geometrically clean
epipolar lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from multiview_tracker.epipolar import (
    eight_point_ransac,
    epipolar_constraint_residuals,
    fundamental_direct,
    fundamental_via_essential,
    normalise_F,
    symmetric_epipolar_distance,
    undistort_points,
)


def refit_pair(npz_path: Path, out_path: Path) -> None:
    z = np.load(npz_path)
    pts_a = z["pts_a"].astype(np.float32)
    pts_b = z["pts_b"].astype(np.float32)
    K_a = z["K_a"]; D_a = z["D_a"]
    K_b = z["K_b"]; D_b = z["D_b"]
    frame_idx = z["frame_idx"]
    point_ids = z["point_ids"]
    n = len(pts_a)
    print(f"\n=== refit {npz_path.name}  ({n} correspondences) ===")

    pts_a_u = undistort_points(pts_a, K_a, D_a)
    pts_b_u = undistort_points(pts_b, K_b, D_b)
    shift_a = np.linalg.norm(pts_a_u - pts_a, axis=1)
    shift_b = np.linalg.norm(pts_b_u - pts_b, axis=1)
    print(f"  cam_a points shifted by undistortion: median={np.median(shift_a):.2f}px max={shift_a.max():.2f}px")
    print(f"  cam_b points shifted by undistortion: median={np.median(shift_b):.2f}px max={shift_b.max():.2f}px")

    # direct cv2.findFundamentalMat (RANSAC) on undistorted points
    est_c1 = fundamental_direct(pts_a_u, pts_b_u)
    sed_c1 = symmetric_epipolar_distance(est_c1.F, pts_a_u, pts_b_u)

    # via essential matrix. inputs are already undistorted, so pass zero D
    # to keep the inner call from double-undistorting.
    D_zero = np.zeros_like(D_a)
    est_c2, E_c2 = fundamental_via_essential(pts_a_u, pts_b_u, K_a, K_b, D_zero, D_zero)
    sed_c2 = symmetric_epipolar_distance(est_c2.F, pts_a_u, pts_b_u)

    # manual 8-point RANSAC
    F_c3, mask_c3 = eight_point_ransac(pts_a_u, pts_b_u, threshold_px=1.0, n_iters=2000)
    sed_c3 = symmetric_epipolar_distance(F_c3, pts_a_u, pts_b_u)

    E = E_c2

    print(f"  C1 (cv2.findFundamentalMat): SED median={np.median(sed_c1):.3f}px max={sed_c1.max():.2f}")
    print(f"  C2 (via essential matrix) : SED median={np.median(sed_c2):.3f}px max={sed_c2.max():.2f}")
    print(f"  C3 (manual 8-point RANSAC): SED median={np.median(sed_c3):.3f}px max={sed_c3.max():.2f}  inliers={int(mask_c3.sum())}/{n}")

    # cross-method agreement
    F1n = normalise_F(est_c1.F); F2n = normalise_F(est_c2.F); F3n = normalise_F(F_c3)
    def diff(A, B): return float(min(np.linalg.norm(A - B), np.linalg.norm(A + B)))
    print(f"  ||F_c1 - F_c2|| = {diff(F1n, F2n):.5f}")
    print(f"  ||F_c1 - F_c3|| = {diff(F1n, F3n):.5f}")
    print(f"  ||F_c2 - F_c3|| = {diff(F2n, F3n):.5f}")

    np.savez(
        out_path,
        F_c1=est_c1.F, F_c2=est_c2.F, F_c3=F_c3, E=E,
        pts_a=pts_a, pts_b=pts_b,            # raw distorted, kept for record
        pts_a_u=pts_a_u, pts_b_u=pts_b_u,    # undistorted set
        frame_idx=frame_idx, point_ids=point_ids,
        K_a=K_a, K_b=K_b, D_a=D_a, D_b=D_b,
        sed_c1=sed_c1, sed_c2=sed_c2, sed_c3=sed_c3,
        inliers_c3=mask_c3,
        space="undistorted_pixel",
    )
    print(f"  -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="experiments/default")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for sid in args.support:
        npz = out_dir / f"F_{args.primary}_{sid}.npz"
        if not npz.exists():
            print(f"missing {npz}, skipping")
            continue
        out = out_dir / f"F_{args.primary}_{sid}_undist.npz"
        refit_pair(npz, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
