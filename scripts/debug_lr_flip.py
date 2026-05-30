"""For each support camera, test whether swapping L<->R labels on symmetric keypoints reduces SED to cam0.

Aggregates across all eligible frames; a consistent improvement under the swap
indicates YOLO is flipping side labels in that view.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.epipolar import symmetric_epipolar_distance
from multiview_tracker.sync import filter_to_actor

COCO = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shldr", "R_shldr", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--conf-min", type=float, default=0.5)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    kp_dir = Path(cfg["detection"]["output_dir"])
    cid_a = args.primary
    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    F = {sid: np.load(out_dir / f"F_{cid_a}_{sid}.npz")[f_key] for sid in args.support}

    def actor(cid):
        dets = dedupe_per_frame_detections(load_pose_detections(kp_dir / f"{cid}.npz"))
        return {d.frame_idx: d for d in filter_to_actor(dets, merge=True, static_anchor=True)}

    A = actor(cid_a)
    S = {sid: actor(sid) for sid in args.support}

    common = set(A)
    for sid in args.support:
        common &= set(S[sid])
    common = sorted(common)
    print(f"Aggregating L/R swap test across {len(common)} common frames\n")

    # per (support, L/R pair), accumulate SED_asis and SED_swap and print
    # the per-pair median SED with vs without swap
    print(f"{'support':>8s} {'pair':>14s} {'n':>5s} {'SED_asis':>9s} {'SED_swap':>9s} {'verdict':>8s}")
    print("-" * 70)

    for sid in args.support:
        for kl, kr in LR_PAIRS:
            asis, swap = [], []
            for f in common:
                da, db = A[f], S[sid][f]
                # need both L and R high-conf in both cameras for the test
                if (da.keypoints[kl, 2] < args.conf_min or
                    da.keypoints[kr, 2] < args.conf_min or
                    db.keypoints[kl, 2] < args.conf_min or
                    db.keypoints[kr, 2] < args.conf_min):
                    continue
                # as-is: (cam0 L) vs (cam_s L), and (cam0 R) vs (cam_s R)
                xa_l = da.keypoints[kl, :2].astype(np.float32)
                xa_r = da.keypoints[kr, :2].astype(np.float32)
                xb_l = db.keypoints[kl, :2].astype(np.float32)
                xb_r = db.keypoints[kr, :2].astype(np.float32)
                s_ll = float(symmetric_epipolar_distance(F[sid], xa_l[None], xb_l[None])[0])
                s_rr = float(symmetric_epipolar_distance(F[sid], xa_r[None], xb_r[None])[0])
                # swap: (cam0 L) vs (cam_s R), and (cam0 R) vs (cam_s L)
                s_lr = float(symmetric_epipolar_distance(F[sid], xa_l[None], xb_r[None])[0])
                s_rl = float(symmetric_epipolar_distance(F[sid], xa_r[None], xb_l[None])[0])
                asis.append((s_ll + s_rr) / 2.0)
                swap.append((s_lr + s_rl) / 2.0)
            if not asis:
                continue
            med_asis = float(np.median(asis))
            med_swap = float(np.median(swap))
            verdict = "FLIP" if med_swap < med_asis * 0.6 else ("close" if abs(med_swap - med_asis) / (med_asis + 1e-9) < 0.2 else "as-is")
            pair_name = f"{COCO[kl]}/{COCO[kr]}"
            print(f"{sid:>8s} {pair_name:>14s} {len(asis):>5d} {med_asis:>9.1f} {med_swap:>9.1f} {verdict:>8s}")

    print()
    print("Verdict key:")
    print("  FLIP   : swap median SED < 60% of as-is median, YOLO is consistently flipping L<->R for this pair in this support cam")
    print("  close  : swap SED ~ as-is SED (<20% diff), no consistent flip; both labels equally noisy")
    print("  as-is  : swap SED notably worse than as-is, labels are correct as YOLO reports them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
