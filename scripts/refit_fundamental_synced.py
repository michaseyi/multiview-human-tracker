"""Refit F from board correspondences paired via event-anchored TimeSync and
undistorted before fitting.

Inputs:
  - configs/default.yaml
  - experiments/<exp>/<cam>_calibration.npz (K, D per camera)
  - experiments/<exp>/offsets_event_anchored.json (TimeSync model)

Output per support pair: experiments/<exp>/F_cam0_camX_synced_undist.npz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.calibration import PuzzleboardConfig
from multiview_tracker.epipolar import (
    eight_point_ransac,
    fundamental_direct,
    fundamental_via_essential,
    harvest_correspondences,
    normalise_F,
    symmetric_epipolar_distance,
    undistort_points,
)
from multiview_tracker.sync import TimeSync


def fmt_F(F: np.ndarray) -> str:
    return "\n".join("  " + "  ".join(f"{v:+.6f}" for v in row) for row in F)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--target-frames", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
        discard_edge_layers=pb.get("discard_edge_layers", 0),
    )

    cams = {c["id"]: c for c in cfg["cameras"]}
    Ka = np.load(out_dir / f"{args.primary}_calibration.npz")["K"]
    Da = np.load(out_dir / f"{args.primary}_calibration.npz")["D"]

    ts = TimeSync.from_json(out_dir / "offsets_event_anchored.json",
                            primary=args.primary)
    print(f"using event-anchored sync:")
    print(ts.summary())

    for sid in args.support:
        print(f"\n=== {args.primary} <-> {sid} (synced + undistorted refit) ===")
        Kb = np.load(out_dir / f"{sid}_calibration.npz")["K"]
        Db = np.load(out_dir / f"{sid}_calibration.npz")["D"]
        cs = harvest_correspondences(
            Path(cams[args.primary]["source"]),
            Path(cams[sid]["source"]),
            pb_cfg=pb_cfg,
            stride=args.stride,
            target_frames=args.target_frames,
            support_frame_fn=lambda n, _s=sid: ts.support_frame(_s, n),
        )
        n = len(cs.pts_a)
        print(f"  {n} correspondences across {cs.n_frames_used} frames")
        pa_u = undistort_points(cs.pts_a, Ka, Da)
        pb_u = undistort_points(cs.pts_b, Kb, Db)

        est_c1 = fundamental_direct(pa_u, pb_u)
        D_zero = np.zeros_like(Da)
        est_c2, E_c2 = fundamental_via_essential(pa_u, pb_u, Ka, Kb, D_zero, D_zero)
        F_c3, mask_c3 = eight_point_ransac(pa_u, pb_u, threshold_px=1.0, n_iters=2000)

        for name, F in [("C1", est_c1.F), ("C2", est_c2.F), ("C3", F_c3)]:
            sed = symmetric_epipolar_distance(F, pa_u, pb_u)
            print(f"  {name}: SED median={np.median(sed):.3f}px  max={sed.max():.2f}px"
                  + (f"  inliers={int(mask_c3.sum())}/{n}" if name == "C3" else ""))

        F1n = normalise_F(est_c1.F); F2n = normalise_F(est_c2.F); F3n = normalise_F(F_c3)
        def diff(A, B): return float(min(np.linalg.norm(A - B), np.linalg.norm(A + B)))
        print(f"  ||F_c1 - F_c2|| = {diff(F1n, F2n):.5f}")
        print(f"  ||F_c1 - F_c3|| = {diff(F1n, F3n):.5f}")
        print(f"  ||F_c2 - F_c3|| = {diff(F2n, F3n):.5f}")

        out_path = out_dir / f"F_{args.primary}_{sid}_synced_undist.npz"
        np.savez(out_path,
                 F_c1=est_c1.F, F_c2=est_c2.F, F_c3=F_c3, E=E_c2,
                 pts_a=cs.pts_a, pts_b=cs.pts_b,
                 pts_a_u=pa_u, pts_b_u=pb_u,
                 frame_idx=cs.frame_idx, point_ids=cs.point_ids,
                 K_a=Ka, K_b=Kb, D_a=Da, D_b=Db,
                 sed_c1=symmetric_epipolar_distance(est_c1.F, pa_u, pb_u),
                 sed_c2=symmetric_epipolar_distance(est_c2.F, pa_u, pb_u),
                 sed_c3=symmetric_epipolar_distance(F_c3, pa_u, pb_u),
                 inliers_c3=mask_c3,
                 space="undistorted_pixel", sync="event_anchored_affine")
        print(f"  saved -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
