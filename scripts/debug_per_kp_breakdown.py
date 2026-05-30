"""Per-keypoint forensic for a single frame.

For each keypoint visible in cam0 plus all supports: SED of (truth_a, support_b)
under F_a_b per support; 2-pair recovery error and 3-pair LS recovery error;
plus an L<->R swap test on symmetric keypoints (shoulders, hips, etc.) to flag
side-label flips across views.
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
from multiview_tracker.recovery import (
    epipolar_line_in_a,
    least_squares_intersection,
    line_line_intersect,
)
from multiview_tracker.sync import filter_to_actor

COCO = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shldr", "R_shldr", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]
# L/R COCO index pairs for the side-flip test
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
LR_PARTNER = {l: r for l, r in LR_PAIRS}
LR_PARTNER.update({r: l for l, r in LR_PAIRS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--frame", type=int, default=6454)
    ap.add_argument("--conf-min", type=float, default=0.5)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    kp_dir = Path(cfg["detection"]["output_dir"])
    cid_a = args.primary
    sids = args.support
    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    F = {sid: np.load(out_dir / f"F_{cid_a}_{sid}.npz")[f_key] for sid in sids}

    def actor(cid):
        dets = dedupe_per_frame_detections(load_pose_detections(kp_dir / f"{cid}.npz"))
        return {d.frame_idx: d for d in filter_to_actor(dets, merge=True, static_anchor=True)}

    A = actor(cid_a)
    S = {sid: actor(sid) for sid in sids}
    if args.frame not in A:
        raise SystemExit(f"no actor in {cid_a} at frame {args.frame}")
    for sid in sids:
        if args.frame not in S[sid]:
            raise SystemExit(f"no actor in {sid} at frame {args.frame}")

    det_a = A[args.frame]
    det_s = {sid: S[sid][args.frame] for sid in sids}

    print(f"=== Per-keypoint forensic, {cid_a} from {'+'.join(sids)}, method={args.method.upper()}, frame={args.frame} ===\n")

    hdr = f"{'kp':>4s} {'name':>8s} {'cf_a':>5s}"
    for sid in sids:
        hdr += f"  {sid+'_cf':>7s} {sid+'_SED':>7s} {sid+'_e2':>7s}"
    hdr += f"  {'eLS3':>6s}  {'eLS3_swap':>9s}"
    print(hdr)
    print("-" * len(hdr))

    for k in range(17):
        cf_a = det_a.keypoints[k, 2]
        if cf_a < args.conf_min:
            continue
        cfs = [det_s[sid].keypoints[k, 2] for sid in sids]
        if min(cfs) < args.conf_min:
            continue

        xa = det_a.keypoints[k, :2].astype(np.float32)

        row = f"{k:>4d} {COCO[k]:>8s} {cf_a:>5.2f}"

        # per-support: SED + 2-pair recovery error
        lines = []
        for sid in sids:
            xb = det_s[sid].keypoints[k, :2].astype(np.float32)
            sed = float(symmetric_epipolar_distance(F[sid], xa[None, :], xb[None, :])[0])
            line = epipolar_line_in_a(F[sid], xb)
            lines.append(line)

            # 2-pair recovery: this support + one other, median across pairings
            others = [s for s in sids if s != sid]
            errs = []
            for s2 in others:
                xb2 = det_s[s2].keypoints[k, :2].astype(np.float32)
                l1 = line
                l2 = epipolar_line_in_a(F[s2], xb2)
                pt, _ = line_line_intersect(l1, l2)
                if not np.isnan(pt).any():
                    errs.append(float(np.linalg.norm(pt - xa)))
            e2 = float(np.median(errs)) if errs else float("inf")
            cf = det_s[sid].keypoints[k, 2]
            row += f"  {cf:>7.2f} {sed:>7.1f} {e2:>7.1f}"

        # 3-pair LS recovery
        ls_pt = least_squares_intersection(np.stack(lines))
        eLS3 = float(np.linalg.norm(ls_pt - xa)) if not np.isnan(ls_pt).any() else float("inf")
        row += f"  {eLS3:>6.1f}"

        # L/R swap test: if any support has a high-conf opposite-side partner,
        # use that instead and re-recover
        if k in LR_PARTNER:
            kp = LR_PARTNER[k]
            swap_lines = []
            ok = True
            for sid in sids:
                cf_sw = det_s[sid].keypoints[kp, 2]
                if cf_sw < args.conf_min:
                    ok = False
                    break
                xb_sw = det_s[sid].keypoints[kp, :2].astype(np.float32)
                swap_lines.append(epipolar_line_in_a(F[sid], xb_sw))
            if ok:
                ls_sw = least_squares_intersection(np.stack(swap_lines))
                eSW = float(np.linalg.norm(ls_sw - xa)) if not np.isnan(ls_sw).any() else float("inf")
                row += f"  {eSW:>9.1f}"
            else:
                row += f"  {'-':>9s}"
        else:
            row += f"  {'-':>9s}"

        print(row)

    print()
    print("Column key:")
    print("  cf_a       cam0 keypoint confidence")
    print("  {sid}_cf   support cam keypoint confidence")
    print("  {sid}_SED  symmetric epipolar distance for (xa, xb) under F; expects ~0 if both detect same physical point")
    print("  {sid}_e2   2-pair recovery error using this support + one other (median over the 2 pairings)")
    print("  eLS3       3-line LS recovery error using all supports")
    print("  eLS3_swap  3-line LS recovery error if we swap L<->R label on every support; if << eLS3, YOLO flipped side labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
