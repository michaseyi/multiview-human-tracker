from __future__ import annotations

from pathlib import Path

import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection


def save_pose_detections(detections: list[PoseDetection], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if detections:
        np.savez(
            out_path,
            frame_idx=np.array([d.frame_idx for d in detections], dtype=np.int64),
            person_idx=np.array([d.person_idx for d in detections], dtype=np.int64),
            person_score=np.array([d.person_score for d in detections], dtype=np.float32),
            keypoints=np.stack([d.keypoints for d in detections]),
        )
    else:
        np.savez(
            out_path,
            frame_idx=np.array([], dtype=np.int64),
            person_idx=np.array([], dtype=np.int64),
            person_score=np.array([], dtype=np.float32),
            keypoints=np.zeros((0, 17, 3), dtype=np.float32),
        )


def load_pose_detections(path: Path) -> list[PoseDetection]:
    """Load detections, reading each backing array once (per-key access on an NpzFile re-reads from disk)."""
    z = np.load(path)
    frame_idx = z["frame_idx"]
    person_idx = z["person_idx"]
    person_score = z["person_score"]
    keypoints = z["keypoints"]
    n = len(frame_idx)
    return [
        PoseDetection(
            frame_idx=int(frame_idx[i]),
            person_idx=int(person_idx[i]),
            person_score=float(person_score[i]),
            keypoints=keypoints[i],
        )
        for i in range(n)
    ]
