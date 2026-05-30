"""Monocular calibration for one camera using puzzleboard captures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from multiview_tracker.calibration import (
    PuzzleboardConfig,
    calibrate_camera,
    harvest_calibration_frames,
    report,
    save_calibration,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="cam0", help="camera id")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cal = cfg["calibration"]
    pb = cal["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
        discard_edge_layers=pb.get("discard_edge_layers", 0),
    )

    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), None)
    if cam is None:
        raise SystemExit(f"camera {args.camera!r} not in config")

    image_points, object_points, image_size = harvest_calibration_frames(
        video_path=Path(cam["source"]),
        pb_cfg=pb_cfg,
        min_corners=cal["min_corners_per_frame"],
        target_frames=cal["target_frames_per_camera"],
        stride=cal["frame_stride"],
        save_dir=Path(cal["captures_dir"]) / args.camera,
    )
    if len(image_points) < 50:
        print(f"WARNING: only {len(image_points)} frames; requires 50+")

    result = calibrate_camera(image_points, object_points, image_size)
    report(result)
    save_calibration(result, Path(cfg["experiment"]["output_dir"]) / f"{args.camera}_calibration.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
