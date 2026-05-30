from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class PoseDetection:
    """One person's keypoints in one frame. keypoints is (17, 3) of (x, y, conf) in COCO order."""
    frame_idx: int
    person_idx: int
    person_score: float
    keypoints: np.ndarray


def best_device() -> str:
    """Pick cuda > mps > cpu based on availability."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_poses(
    video_path: Path,
    model_name: str = "yolov8s-pose.pt",
    device: str | None = None,
    conf_threshold: float = 0.35,
    imgsz: int = 1280,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[PoseDetection]:
    """Run YOLO-pose across a video; return one PoseDetection per (frame, person)."""
    from ultralytics import YOLO

    if device in (None, "", "auto"):
        device = best_device()

    model = YOLO(model_name)
    print(f"[detect] {video_path.name}  model={model_name}  device={device}  imgsz={imgsz}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_total = min(n_total, max_frames)

    detections: list[PoseDetection] = []
    pbar = tqdm(range(0, n_total, stride), desc=f"yolo {video_path.stem}", unit="frame")

    for idx in pbar:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue

        out = model.predict(
            frame, device=device, conf=conf_threshold, imgsz=imgsz, verbose=False
        )[0]

        if out.keypoints is None or len(out.keypoints) == 0:
            continue

        kp_data = out.keypoints.data.cpu().numpy()  # (P, 17, 3)
        if out.boxes is not None and out.boxes.conf is not None:
            scores = out.boxes.conf.cpu().numpy()
        else:
            scores = np.ones(len(kp_data), dtype=np.float32)

        for pi in range(len(kp_data)):
            detections.append(
                PoseDetection(
                    frame_idx=int(idx),
                    person_idx=pi,
                    person_score=float(scores[pi]),
                    keypoints=kp_data[pi].astype(np.float32),
                )
            )

        pbar.set_postfix(dets=len(detections))

    cap.release()
    print(f"[detect] {len(detections)} detections from {video_path.name}")
    return detections
