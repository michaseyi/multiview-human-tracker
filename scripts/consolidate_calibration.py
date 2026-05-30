"""Bundle per-camera calibration .npz files into a single calibration.npz."""
from __future__ import annotations

from pathlib import Path

import yaml

from multiview_tracker.calibration import bundle_calibrations

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
out_dir = Path(cfg["experiment"]["output_dir"])

per_camera = {cam["id"]: out_dir / f"{cam['id']}_calibration.npz" for cam in cfg["cameras"]}
bundle_calibrations(per_camera, Path(cfg["calibration"]["output_path"]))
