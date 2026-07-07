"""ViTPose top-down keypoint detection: YOLO person boxes -> ViTPose keypoints.

This is a drop-in producer of the same :class:`PoseDetection` (COCO-17) records
as :mod:`multiview_tracker.detection.yolo_pose`, so the downstream
calibration / sync / fundamental-matrix / recovery pipeline consumes it
unchanged. Only the keypoint *localisation* changes: person boxes still come
from the same YOLO detector used by the baseline, which keeps person
detection / recall identical and isolates the detector-quality effect on
keypoint accuracy.

ViTPose is a two-stage (top-down) estimator: a person detector proposes boxes,
then the pose network localises 17 keypoints inside each box crop via heatmaps.
We use HuggingFace ``transformers`` (``VitPoseForPoseEstimation``); the default
checkpoint is the strongest publicly runnable one, ViTPose++ Huge.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from multiview_tracker.detection.yolo_pose import PoseDetection, best_device

# Best publicly runnable ViTPose checkpoint (ViTPose++ Huge, MoE). The 1B
# ViTPose-G that holds the COCO record was never released.
DEFAULT_POSE_MODEL = "usyd-community/vitpose-plus-huge"
DEFAULT_BOX_MODEL = "yolov8s-pose.pt"  # same detector the YOLO baseline used


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(N,4) [x1,y1,x2,y2] -> [x,y,w,h], the COCO box format the processor wants."""
    b = boxes.astype(np.float32).copy()
    b[:, 2] = b[:, 2] - b[:, 0]
    b[:, 3] = b[:, 3] - b[:, 1]
    return b


def detect_poses_vitpose(
    video_path: Path,
    box_model_name: str = DEFAULT_BOX_MODEL,
    pose_model_name: str = DEFAULT_POSE_MODEL,
    device: str | None = None,
    box_conf_threshold: float = 0.35,
    imgsz: int = 1280,
    stride: int = 1,
    max_frames: int | None = None,
    dataset_index: int = 0,
) -> list[PoseDetection]:
    """Run YOLO-box + ViTPose-keypoint estimation across a video.

    Returns one :class:`PoseDetection` per (frame, person). Keypoints are in
    full-resolution pixel coordinates (the same space as the calibration and
    fundamental matrices), COCO-17 order, with the ViTPose per-keypoint
    heatmap score in the confidence channel. ``person_score`` is the YOLO box
    confidence, matching the baseline's per-person score semantics.

    ``dataset_index`` selects the MoE expert for ViTPose++ checkpoints
    (0 = COCO); it is ignored for non-MoE checkpoints.
    """
    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    from ultralytics import YOLO

    if device in (None, "", "auto"):
        device = best_device()

    box_model = YOLO(box_model_name)
    processor = AutoProcessor.from_pretrained(pose_model_name)
    model = VitPoseForPoseEstimation.from_pretrained(pose_model_name).to(device).eval()
    is_moe = "plus" in pose_model_name  # ViTPose++ mixture-of-experts

    print(f"[vitpose] {video_path.name}  box={box_model_name}  "
          f"pose={pose_model_name}  device={device}  moe={is_moe}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n_total = min(n_total, max_frames)

    detections: list[PoseDetection] = []
    # Sequential decode (no per-frame random seek) — far faster than the YOLO
    # baseline's cap.set()-per-frame pattern. Only frames on the stride grid
    # are run through the models.
    pbar = tqdm(total=n_total, desc=f"vitpose {video_path.stem}", unit="frame")
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx >= n_total:
            break
        pbar.update(1)
        if idx % stride != 0:
            continue

        # 1) person boxes from YOLO (BGR frame is fine for ultralytics)
        res = box_model.predict(
            frame, device=device, conf=box_conf_threshold, imgsz=imgsz, verbose=False
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        bscore = res.boxes.conf.cpu().numpy()
        boxes_xywh = _xyxy_to_xywh(xyxy)

        # 2) ViTPose keypoints per box (processor expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(rgb, boxes=[boxes_xywh], return_tensors="pt").to(device)
        if is_moe:
            inputs["dataset_index"] = torch.tensor(
                [dataset_index] * len(boxes_xywh), device=device
            )
        with torch.no_grad():
            outputs = model(**inputs)
        pose = processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]

        # 3) assemble one PoseDetection per detected person
        for pi, person in enumerate(pose):
            kxy = np.asarray(person["keypoints"])           # (17, 2)
            ksc = np.asarray(person["scores"]).reshape(-1)  # (17,)
            kp = np.concatenate([kxy, ksc[:, None]], axis=1).astype(np.float32)
            detections.append(
                PoseDetection(
                    frame_idx=int(idx),
                    person_idx=int(pi),
                    person_score=float(bscore[pi]),
                    keypoints=kp,
                )
            )
        pbar.set_postfix(dets=len(detections))

    cap.release()
    pbar.close()
    print(f"[vitpose] {len(detections)} detections from {video_path.name}")
    return detections
