from __future__ import annotations

import cv2
import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection

COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),                  # head: nose-eyes-ears
    (5, 6),                                          # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),                 # arms
    (5, 11), (6, 12), (11, 12),                      # torso to hips
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
]


def draw_pose(
    frame: np.ndarray,
    detection: PoseDetection,
    *,
    conf_min: float = 0.35,
    point_color: tuple[int, int, int] = (0, 255, 0),
    bone_color: tuple[int, int, int] = (0, 200, 255),
    point_radius: int = 4,
    bone_thickness: int = 2,
) -> None:
    """Draw COCO-17 keypoints and bones in place on a BGR frame."""
    kp = detection.keypoints
    for a, b in COCO_SKELETON:
        if kp[a, 2] >= conf_min and kp[b, 2] >= conf_min:
            cv2.line(
                frame,
                (int(kp[a, 0]), int(kp[a, 1])),
                (int(kp[b, 0]), int(kp[b, 1])),
                bone_color, bone_thickness, cv2.LINE_AA,
            )
    for i in range(kp.shape[0]):
        if kp[i, 2] >= conf_min:
            cv2.circle(frame, (int(kp[i, 0]), int(kp[i, 1])),
                       point_radius, point_color, -1, cv2.LINE_AA)
