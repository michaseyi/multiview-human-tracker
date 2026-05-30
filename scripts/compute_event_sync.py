"""Compute event-anchored affine temporal sync from two manually-identified events across all cameras.

Per support camera: support_frame = round(alpha + beta * primary_frame),
with beta = (M_late - M_early) / (N_late - N_early) and alpha = M_early - beta * N_early.
Output JSON is consumed by downstream scripts via TimeSync.from_json().
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

from multiview_tracker.sync import TimeSync


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument(
        "--event1", nargs="+", required=True, metavar="cid:frame",
        help="early event: one cid:frame per camera, e.g. "
             "cam0:330 cam1:341 cam2:335 cam3:336",
    )
    ap.add_argument(
        "--event2", nargs="+", required=True, metavar="cid:frame",
        help="late event, same format as --event1",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])

    def _parse(items):
        out = {}
        for item in items:
            cid, _, fr = item.partition(":")
            out[cid] = int(fr)
        return out

    event1 = _parse(args.event1)
    event2 = _parse(args.event2)
    cameras = sorted(set(event1) | set(event2))
    missing = [c for c in cameras if c not in event1 or c not in event2]
    if missing:
        raise SystemExit(f"missing events for {missing}")
    if args.primary not in event1:
        raise SystemExit(f"primary {args.primary!r} not in --event1")

    ts = TimeSync(
        events={cid: (event1[cid], event2[cid]) for cid in cameras},
        primary=args.primary,
    )

    # FPS metadata for diagnostic comparison
    fps = {}
    for cam in cfg["cameras"]:
        cap = cv2.VideoCapture(str(cam["source"]))
        fps[cam["id"]] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

    print(f"=== Event-anchored sync ({args.primary} as primary) ===")
    print(ts.summary())
    print()
    print(f"FPS metadata vs fitted beta * fps_primary (implied true fps):")
    fps_a = fps[args.primary]
    for cid in cameras:
        if cid == args.primary:
            continue
        m = ts.model_for(cid)
        print(f"  {cid}: metadata={fps[cid]:.4f}  fitted={m.beta * fps_a:.4f}  "
              f"(off by {(fps[cid] - m.beta * fps_a) * 1000 / fps[cid]:+.1f} ppm, expressed as Hz)")

    # predicted offsets at key frames
    end_n = max(ts._events[args.primary])  # primary's late event
    horizon = max(end_n, 10000)
    print()
    print(f"Predicted matched frame and offset:")
    hdr = f"  {'cam0 N':>6s} ||"
    for sid in [c for c in cameras if c != args.primary]:
        hdr += f" {sid+' (M, off)':>17s} |"
    print(hdr)
    for N in [0, ts._events[args.primary][0], 1200, 4255, end_n, horizon]:
        row = f"  {N:>6d} ||"
        for sid in [c for c in cameras if c != args.primary]:
            M = ts.support_frame(sid, N)
            row += f" {M:>10d} ({M-N:>+5d}) |"
        print(row)

    payload = {
        "method": "event-anchored, 2-event affine, beta fit empirically",
        "primary": args.primary,
        "events": [event1, event2],
        "fps_metadata": fps,
        "offsets": {
            cid: {
                "alpha": ts.model_for(cid).alpha,
                "beta": ts.model_for(cid).beta,
                "event_frames": [
                    [event1[args.primary], event1[cid]],
                    [event2[args.primary], event2[cid]],
                ],
            }
            for cid in cameras
        },
    }
    out_path = out_dir / "offsets_event_anchored.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
