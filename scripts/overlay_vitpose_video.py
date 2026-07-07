"""Run ViTPose on one camera's footage and write a keypoint-overlay video.

Single decode pass: for each frame, get person boxes from YOLO, localise
COCO-17 keypoints with ViTPose, draw the skeleton for every person, and write
the annotated frame to an mp4. Keypoints are computed at full resolution; the
output can be downscaled for a smaller file without affecting detection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.detection.vitpose import (
    DEFAULT_BOX_MODEL,
    DEFAULT_POSE_MODEL,
    _xyxy_to_xywh,
)
from multiview_tracker.detection.yolo_pose import PoseDetection, best_device
from multiview_tracker.visualization.keypoints import draw_pose

# Distinct per-person colours (BGR) so the actor and bystander read apart.
PERSON_COLORS = [
    ((0, 255, 0), (0, 200, 255)),      # person 0: green points / amber bones
    ((255, 128, 0), (255, 200, 0)),    # person 1: blue points / cyan bones
    ((200, 0, 255), (255, 0, 200)),    # person 2
    ((0, 255, 255), (0, 180, 255)),    # person 3
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    ap.add_argument("--box-model", default=DEFAULT_BOX_MODEL)
    ap.add_argument("--output", default=None, help="output mp4 path")
    ap.add_argument("--start", type=int, default=0, help="first frame index")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="number of frames to render from --start")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--downscale", type=float, default=0.5,
                    help="output scale factor (0.5 -> 1280x720 from 2560x1440)")
    ap.add_argument("--conf-min", type=float, default=0.3,
                    help="min ViTPose keypoint score to draw")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, VitPoseForPoseEstimation
    from ultralytics import YOLO

    cfg = yaml.safe_load(Path(args.config).read_text())
    cam = next((c for c in cfg["cameras"] if c["id"] == args.camera), None)
    if cam is None:
        raise SystemExit(f"camera {args.camera!r} not in config")
    src = Path(cam["source"])

    device = args.device or cfg["detection"].get("device")
    if device in (None, "", "auto"):
        device = best_device()

    box_model = YOLO(args.box_model)
    processor = AutoProcessor.from_pretrained(args.pose_model)
    model = VitPoseForPoseEstimation.from_pretrained(args.pose_model).to(device).eval()
    is_moe = "plus" in args.pose_model

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cam.get("fps", 24.55)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w, out_h = int(W * args.downscale), int(H * args.downscale)
    out_path = Path(args.output) if args.output else Path(
        f"experiments/default/vitpose_overlay_{args.camera}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.stride, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {out_path}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    end = n_total if args.max_frames is None else min(n_total, args.start + args.max_frames)

    print(f"[overlay] {src.name} frames [{args.start},{end}) stride={args.stride} "
          f"-> {out_path}  ({out_w}x{out_h} @ {fps/args.stride:.2f}fps)  device={device}")

    idx = args.start - 1
    written = 0
    # scale draw sizes to full-res so they survive the downscale
    pr = max(3, int(6 * (W / 1280)))
    bt = max(2, int(3 * (W / 1280)))
    while True:
        ok, frame = cap.read()
        if not ok or idx + 1 >= end:
            break
        idx += 1
        if (idx - args.start) % args.stride != 0:
            continue

        r = box_model.predict(frame, device=device,
                              conf=cfg["detection"]["conf_threshold"],
                              imgsz=cfg["detection"]["imgsz"], verbose=False)[0]
        n_persons = 0
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            boxes_xywh = _xyxy_to_xywh(xyxy)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(rgb, boxes=[boxes_xywh], return_tensors="pt").to(device)
            if is_moe:
                inputs["dataset_index"] = torch.tensor([0] * len(boxes_xywh), device=device)
            with torch.no_grad():
                outputs = model(**inputs)
            pose = processor.post_process_pose_estimation(outputs, boxes=[boxes_xywh])[0]
            n_persons = len(pose)
            for pi, person in enumerate(pose):
                kxy = np.asarray(person["keypoints"])
                ksc = np.asarray(person["scores"]).reshape(-1)
                kp = np.concatenate([kxy, ksc[:, None]], axis=1).astype(np.float32)
                det = PoseDetection(idx, pi, 1.0, kp)
                pcol, bcol = PERSON_COLORS[pi % len(PERSON_COLORS)]
                draw_pose(frame, det, conf_min=args.conf_min,
                          point_color=pcol, bone_color=bcol,
                          point_radius=pr, bone_thickness=bt)

        hud = f"{args.camera}  frame {idx}  persons={n_persons}  ViTPose++ Huge"
        scale = W / 1280
        cv2.rectangle(frame, (0, 0), (int(1000 * scale), int(44 * scale)), (0, 0, 0), -1)
        cv2.putText(frame, hud, (int(10 * scale), int(32 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                    max(1, int(2 * scale)), cv2.LINE_AA)

        out_frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(out_frame)
        written += 1
        if written % 250 == 0:
            print(f"  ... {written} frames written (at source idx {idx})", flush=True)

    cap.release()
    writer.release()
    print(f"[overlay] wrote {written} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
