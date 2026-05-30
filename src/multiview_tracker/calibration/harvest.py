from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from multiview_tracker.calibration.puzzleboard import (
    PuzzleboardConfig,
    detect_puzzleboard,
)


def harvest_calibration_frames(
    video_path: Path,
    pb_cfg: PuzzleboardConfig,
    min_corners: int,
    target_frames: int,
    stride: int,
    save_dir: Path | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Scan video, detect puzzleboard, return per-frame correspondences as (image_points, object_points, (W, H)). Accepted frames are written as JPEGs to save_dir if provided."""
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[harvest] {video_path.name}: {n_total} frames, stride={stride}")

    image_points: list[np.ndarray] = []
    object_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    pbar = tqdm(range(0, n_total, stride), desc="scanning", unit="frame")
    for idx in pbar:
        if len(image_points) >= target_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        det = detect_puzzleboard(frame, pb_cfg)
        if det is None or len(det.image_points) < min_corners:
            continue

        image_points.append(det.image_points)
        object_points.append(det.object_points)
        image_size = det.image_size

        if save_dir is not None:
            out_jpg = save_dir / f"frame_{idx:06d}_pts{len(det.image_points):03d}.jpg"
            cv2.imwrite(str(out_jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pbar.set_postfix(kept=len(image_points), pts=len(det.image_points))

    cap.release()
    print(f"[harvest] kept {len(image_points)} frames")

    if image_size is None or not image_points:
        raise RuntimeError("no usable calibration frames found")

    return image_points, object_points, image_size
