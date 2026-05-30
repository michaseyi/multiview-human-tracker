"""Estimate per-camera temporal offsets relative to cam0 via NCC on actor motion signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.sync import (
    build_motion_signal,
    filter_to_actor,
    normalised_xcorr,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--reference", default="cam0", help="reference camera id")
    ap.add_argument("--max-lag", type=int, default=30, help="search range in frames")
    ap.add_argument("--static-anchor", action="store_true",
                    help="use the static-anchor actor filter")
    ap.add_argument("--merge", action="store_true",
                    help="merge fragmented tracklets before actor selection")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cameras = [c["id"] for c in cfg["cameras"]]
    src = {c["id"]: Path(c["source"]) for c in cfg["cameras"]}
    kp_dir = Path(cfg["detection"]["output_dir"])

    # use longest video as the fixed signal length
    n_frames_by_cam = {}
    for cid in cameras:
        cap = cv2.VideoCapture(str(src[cid]))
        n_frames_by_cam[cid] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    n_frames = max(n_frames_by_cam.values())
    print(f"signal length: {n_frames} frames")

    signals: dict[str, np.ndarray] = {}
    for cid in cameras:
        dets = load_pose_detections(kp_dir / f"{cid}.npz")
        if args.static_anchor or args.merge:
            dets = dedupe_per_frame_detections(dets)
        actor = filter_to_actor(
            dets,
            merge=args.merge or args.static_anchor,
            static_anchor=args.static_anchor,
        )
        sig = build_motion_signal(actor, n_frames)
        signals[cid] = sig
        print(f"  {cid}: {len(actor):5d} actor detections, signal std={sig.std():.4f}")

    if args.reference not in signals:
        raise SystemExit(f"reference {args.reference!r} not in cameras")

    ref = signals[args.reference]
    offsets: dict[str, dict] = {}
    print()
    print(f"offsets relative to {args.reference}:")
    for cid in cameras:
        if cid == args.reference:
            offsets[cid] = {"tau": 0, "peak": 1.0}
            print(f"  {cid:5s}: tau= +0 frames  peak=1.000  (reference)")
            continue
        est = normalised_xcorr(ref, signals[cid], max_lag=args.max_lag)
        offsets[cid] = {"tau": est.tau, "peak": est.peak_value}
        print(f"  {cid:5s}: tau={est.tau:+3d} frames  peak={est.peak_value:+.3f}")

    suffix = ""
    if args.static_anchor:
        suffix = "_static_anchor"
    elif args.merge:
        suffix = "_merge"
    out_path = Path(cfg["experiment"]["output_dir"]) / f"offsets{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"reference": args.reference, "offsets": offsets}, indent=2))
    print(f"\n[save] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
