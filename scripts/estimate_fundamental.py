"""Estimate the fundamental matrix between a pair of cameras.

Reads synchronised puzzleboard correspondences, runs the direct fit, the
essential-matrix path, and the manual 8-point + Hartley + RANSAC solver, then
saves all three F matrices side by side with a comparison report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from multiview_tracker.calibration import PuzzleboardConfig, load_calibration
from multiview_tracker.epipolar import (
    eight_point_ransac,
    epipolar_constraint_residuals,
    fundamental_direct,
    fundamental_via_essential,
    harvest_correspondences,
    normalise_F,
    symmetric_epipolar_distance,
)


def parse_pair(s: str) -> tuple[str, str]:
    a, b = s.split(",")
    return a.strip(), b.strip()


def fmt_F(F: np.ndarray) -> str:
    F = normalise_F(F)
    rows = "\n".join("    [" + "  ".join(f"{v:+.5f}" for v in r) + "]" for r in F)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True, help='e.g. "cam0,cam2"')
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target-frames", type=int, default=30)
    ap.add_argument("--stride", type=int, default=50)
    args = ap.parse_args()

    cid_a, cid_b = parse_pair(args.pair)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])

    cam_a = next(c for c in cfg["cameras"] if c["id"] == cid_a)
    cam_b = next(c for c in cfg["cameras"] if c["id"] == cid_b)

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
    )

    # offsets are zero in practice but read them anyway to stay safe
    offsets_path = out_dir / "offsets.json"
    offsets = {cid_a: 0, cid_b: 0}
    if offsets_path.exists():
        all_offsets = json.loads(offsets_path.read_text())["offsets"]
        offsets = {cid_a: all_offsets[cid_a]["tau"], cid_b: all_offsets[cid_b]["tau"]}

    cs = harvest_correspondences(
        Path(cam_a["source"]), Path(cam_b["source"]),
        pb_cfg=pb_cfg,
        stride=args.stride,
        target_frames=args.target_frames,
        offset_b=offsets[cid_b] - offsets[cid_a],
    )
    print(f"\n{cs.pts_a.shape[0]} correspondences across {cs.n_frames_used} frames\n")

    # direct fit via cv2.findFundamentalMat
    est_c1 = fundamental_direct(cs.pts_a, cs.pts_b)
    res_c1 = epipolar_constraint_residuals(est_c1.F, cs.pts_a, cs.pts_b)
    sed_c1 = symmetric_epipolar_distance(est_c1.F, cs.pts_a, cs.pts_b)

    # E -> F via known intrinsics
    cal_a = load_calibration(out_dir / f"{cid_a}_calibration.npz")
    cal_b = load_calibration(out_dir / f"{cid_b}_calibration.npz")
    est_c2, E = fundamental_via_essential(
        cs.pts_a, cs.pts_b, cal_a.K, cal_b.K, cal_a.D, cal_b.D
    )
    res_c2 = epipolar_constraint_residuals(est_c2.F, cs.pts_a, cs.pts_b)
    sed_c2 = symmetric_epipolar_distance(est_c2.F, cs.pts_a, cs.pts_b)

    print(f"=== C1 (cv2.findFundamentalMat: {cid_a} -> {cid_b}) ===")
    print(f"  F (normalised, ||F||=1):\n{fmt_F(est_c1.F)}")
    print(f"  rank(F)              = {np.linalg.matrix_rank(est_c1.F, tol=1e-6)}")
    print(f"  inliers              = {est_c1.inlier_mask.sum()} / {len(est_c1.inlier_mask)}")
    print(f"  median |x'^T F x|    = {np.median(np.abs(res_c1)):.4e}")
    print(f"  median sym epi dist  = {np.median(sed_c1):.3f} px")
    print(f"  mean   sym epi dist  = {sed_c1.mean():.3f} px")
    print()

    print(f"=== C2 (E -> F via known intrinsics: {cid_a} -> {cid_b}) ===")
    print(f"  F (normalised, ||F||=1):\n{fmt_F(est_c2.F)}")
    print(f"  rank(F)              = {np.linalg.matrix_rank(est_c2.F, tol=1e-6)}")
    print(f"  inliers              = {est_c2.inlier_mask.sum()} / {len(est_c2.inlier_mask)}")
    print(f"  median |x'^T F x|    = {np.median(np.abs(res_c2)):.4e}")
    print(f"  median sym epi dist  = {np.median(sed_c2):.3f} px")
    print(f"  mean   sym epi dist  = {sed_c2.mean():.3f} px")
    print()

    # manual 8-point + Hartley normalisation + RANSAC
    F_c3, mask_c3 = eight_point_ransac(cs.pts_a, cs.pts_b, threshold_px=1.0, n_iters=2000)
    res_c3 = epipolar_constraint_residuals(F_c3, cs.pts_a, cs.pts_b)
    sed_c3 = symmetric_epipolar_distance(F_c3, cs.pts_a, cs.pts_b)

    print(f"=== C3 (manual 8-point + Hartley + RANSAC: {cid_a} -> {cid_b}) ===")
    print(f"  F (normalised, ||F||=1):\n{fmt_F(F_c3)}")
    print(f"  rank(F)              = {np.linalg.matrix_rank(F_c3, tol=1e-6)}")
    print(f"  inliers              = {mask_c3.sum()} / {len(mask_c3)}")
    print(f"  median |x'^T F x|    = {np.median(np.abs(res_c3)):.4e}")
    print(f"  median sym epi dist  = {np.median(sed_c3):.3f} px")
    print(f"  mean   sym epi dist  = {sed_c3.mean():.3f} px")
    print()

    # cross-check: how close are the three F matrices to each other?
    F1n, F2n, F3n = normalise_F(est_c1.F), normalise_F(est_c2.F), normalise_F(F_c3)
    def diff(a, b):
        return min(float(np.linalg.norm(a - b)), float(np.linalg.norm(a + b)))
    print(f"=== Cross-check (Frobenius distance, smaller of |F-F'| and |F+F'|) ===")
    print(f"  ||F_c1 - F_c2|| = {diff(F1n, F2n):.4f}")
    print(f"  ||F_c1 - F_c3|| = {diff(F1n, F3n):.4f}")
    print(f"  ||F_c2 - F_c3|| = {diff(F2n, F3n):.4f}")
    print()

    out_path = out_dir / f"F_{cid_a}_{cid_b}.npz"
    np.savez(
        out_path,
        F_c1=est_c1.F, F_c2=est_c2.F, F_c3=F_c3, E=E,
        pts_a=cs.pts_a, pts_b=cs.pts_b,
        frame_idx=cs.frame_idx, point_ids=cs.point_ids,
        K_a=cal_a.K, K_b=cal_b.K, D_a=cal_a.D, D_b=cal_b.D,
        sed_c1=sed_c1, sed_c2=sed_c2, sed_c3=sed_c3,
        inliers_c3=mask_c3,
    )
    print(f"[save] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
