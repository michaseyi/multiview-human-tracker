"""Save one annotated keypoint frame per camera."""
from __future__ import annotations

from pathlib import Path

import cv2
import yaml

from multiview_tracker.detection import load_pose_detections
from multiview_tracker.sync import filter_to_actor
from multiview_tracker.visualization import draw_pose

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
out_dir = Path("experiments/default/keypoint_samples")
out_dir.mkdir(parents=True, exist_ok=True)

for cam in cfg["cameras"]:
    cid = cam["id"]
    actor = filter_to_actor(load_pose_detections(Path(cfg["detection"]["output_dir"]) / f"{cid}.npz"))
    # highest person_score
    best = max(actor, key=lambda d: d.person_score)
    cap = cv2.VideoCapture(cam["source"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, best.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        continue
    draw_pose(frame, best, point_radius=8, bone_thickness=3)
    label = f"{cid}  frame={best.frame_idx}  score={best.person_score:.2f}"
    pad, scale, thick = 14, 2.0, 3
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(frame, label, (pad, th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thick, cv2.LINE_AA)
    out_path = out_dir / f"{cid}_kp_sample.jpg"
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  {cid}: frame {best.frame_idx}, score {best.person_score:.3f} -> {out_path}")
