"""Sweep frame offsets per support camera to test whether the cameras need a temporal shift.

For each support: at every candidate offset, pair cam0 keypoints at frame f with
the support's keypoints at f + delta, compute SED under F_cam0_cam_i, and report
the median across all pairings. The delta minimising median SED is the true offset.
Aggregates over high-confidence body keypoints (head/face skipped as ambiguous);
the static-anchor actor filter is applied.
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

# body keypoints only; nose, eyes, and ears are visually ambiguous across views
KP_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--max-lag", type=int, default=20)
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

    print(f"=== SED-vs-offset sweep, {cid_a} from {'+'.join(sids)}, method={args.method.upper()} ===")
    print(f"body keypoints only (idx 5..16, skipping head/face)\n")

    # per camera: compute SED at each offset, then find the min
    for sid in sids:
        results = []
        for delta in range(-args.max_lag, args.max_lag + 1):
            seds = []
            for f in A:
                f_s = f + delta
                if f_s not in S[sid]:
                    continue
                da, db = A[f], S[sid][f_s]
                for k in KP_BODY:
                    if da.keypoints[k, 2] < args.conf_min or db.keypoints[k, 2] < args.conf_min:
                        continue
                    xa = da.keypoints[k, :2].astype(np.float32)[None]
                    xb = db.keypoints[k, :2].astype(np.float32)[None]
                    s = float(symmetric_epipolar_distance(F[sid], xa, xb)[0])
                    seds.append(s)
            if not seds:
                continue
            results.append((delta, float(np.median(seds)), len(seds)))

        results.sort(key=lambda r: r[1])
        best_delta, best_sed, best_n = results[0]
        zero_sed = next(r[1] for r in results if r[0] == 0)
        # print neighbours of the minimum plus offset 0 for comparison
        print(f"{sid}: median body-keypoint SED vs frame offset")
        print(f"  {'Δ':>5s}  {'med SED':>9s}  {'n_pairs':>8s}  note")
        for delta, sed, n in sorted([r for r in results if abs(r[0] - best_delta) <= 5 or r[0] == 0],
                                     key=lambda r: r[0]):
            note = []
            if delta == 0:
                note.append("← current (no offset)")
            if delta == best_delta:
                note.append("← MIN")
            print(f"  {delta:+5d}  {sed:>9.1f}  {n:>8d}  {' '.join(note)}")
        print(f"  best Δ = {best_delta:+d}  (SED {best_sed:.1f}  vs offset-0 SED {zero_sed:.1f}, "
              f"{(zero_sed - best_sed) / zero_sed * 100:+.0f}% improvement)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
