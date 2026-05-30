from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection
from multiview_tracker.visualization.keypoints import draw_pose


@dataclass
class CameraScore:
    n_visible: int
    mean_conf: float


def score_detections(detections: list[PoseDetection], conf_min: float = 0.35) -> CameraScore:
    """Score = (#kp above conf_min, mean conf among those), aggregated across all persons in the frame."""
    if not detections:
        return CameraScore(0, 0.0)
    kps = np.concatenate([d.keypoints for d in detections], axis=0)
    visible_mask = kps[:, 2] >= conf_min
    n_visible = int(visible_mask.sum())
    mean_conf = float(kps[visible_mask, 2].mean()) if n_visible else 0.0
    return CameraScore(n_visible, mean_conf)


def select_best(scores: dict[str, CameraScore]) -> str:
    """Pick the camera with the most visible keypoints; break ties by mean confidence."""
    return max(scores, key=lambda c: (scores[c].n_visible, scores[c].mean_conf))


def index_by_frame(detections: list[PoseDetection]) -> dict[int, list[PoseDetection]]:
    out: dict[int, list[PoseDetection]] = defaultdict(list)
    for d in detections:
        out[d.frame_idx].append(d)
    return out


def annotate_panel(
    frame: np.ndarray,
    detections: list[PoseDetection],
    cam_id: str,
    score: CameraScore,
    *,
    is_best: bool = False,
) -> np.ndarray:
    """Return a copy of frame with keypoints and a text header drawn on it."""
    out = frame.copy()
    for d in detections:
        draw_pose(out, d)

    h, _ = out.shape[:2]
    text = f"{cam_id}  n={score.n_visible}  conf={score.mean_conf:.2f}"
    if is_best:
        text += "  [BEST]"

    # text sized relative to frame width so it stays readable after downscaling
    pad = 12
    scale = max(1.5, frame.shape[1] / 1100)  # ~3.5 at 2560 px wide
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    color = (0, 255, 255) if is_best else (255, 255, 255)
    cv2.putText(out, text, (pad, th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return out


def compose_grid(
    panels: dict[str, np.ndarray],
    best_panel: np.ndarray,
    panel_size: tuple[int, int],
    grid_order: tuple[str, str, str, str] = ("cam0", "cam1", "cam2", "cam3"),
) -> np.ndarray:
    """2x2 grid of camera panels on the left, BEST panel doubled on the right; output is 4W x 2H."""
    pw, ph = panel_size
    resized = {cid: cv2.resize(panels[cid], (pw, ph)) for cid in grid_order}
    top = np.hstack([resized[grid_order[0]], resized[grid_order[1]]])
    bot = np.hstack([resized[grid_order[2]], resized[grid_order[3]]])
    grid = np.vstack([top, bot])  # 2W x 2H
    big = cv2.resize(best_panel, (2 * pw, 2 * ph))
    return np.hstack([grid, big])  # 4W x 2H
