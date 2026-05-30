"""YOLO-pose keypoint detection for one camera."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from multiview_tracker.detection import detect_poses, save_pose_detections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="cam0", help="camera id")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames for quick smoke tests")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det = cfg["detection"]
    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), None)
    if cam is None:
        raise SystemExit(f"camera {args.camera!r} not in config")

    detections = detect_poses(
        video_path=Path(cam["source"]),
        model_name=det["model"],
        device=det.get("device"),
        conf_threshold=det["conf_threshold"],
        imgsz=det["imgsz"],
        stride=det["stride"],
        max_frames=args.max_frames,
    )

    out_path = Path(det["output_dir"]) / f"{args.camera}.npz"
    save_pose_detections(detections, out_path)
    print(f"[save] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
