"""ViTPose-pose keypoint detection for one camera (YOLO boxes + ViTPose keypoints).

Drop-in alternative to scripts/detect_keypoints.py. Writes the same
PoseDetection .npz format to a separate directory (default
experiments/default/keypoints_vitpose) so the YOLO baseline stays intact for
A/B comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from multiview_tracker.detection.vitpose import (
    DEFAULT_BOX_MODEL,
    DEFAULT_POSE_MODEL,
    detect_poses_vitpose,
)
from multiview_tracker.detection import save_pose_detections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="cam0", help="camera id")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pose-model", default=DEFAULT_POSE_MODEL,
                    help="HuggingFace ViTPose checkpoint")
    ap.add_argument("--box-model", default=DEFAULT_BOX_MODEL,
                    help="YOLO detector used for person boxes")
    ap.add_argument("--output-dir", default="experiments/default/keypoints_vitpose",
                    help="where to write {camera}.npz")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap frames for quick smoke tests")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det = cfg["detection"]
    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), None)
    if cam is None:
        raise SystemExit(f"camera {args.camera!r} not in config")

    detections = detect_poses_vitpose(
        video_path=Path(cam["source"]),
        box_model_name=args.box_model,
        pose_model_name=args.pose_model,
        device=args.device or det.get("device"),
        box_conf_threshold=det["conf_threshold"],
        imgsz=det["imgsz"],
        stride=args.stride,
        max_frames=args.max_frames,
    )

    out_path = Path(args.output_dir) / f"{args.camera}.npz"
    save_pose_detections(detections, out_path)
    print(f"[save] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
