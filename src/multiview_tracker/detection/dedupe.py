"""Per-frame deduplication of YOLO-pose detections.

YOLO can emit multiple boxes for the same person (e.g. full body and torso);
this module merges them and caps detections per frame at the known number
of people.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection


def _pairwise_keypoint_distance(
    a: PoseDetection, b: PoseDetection, conf_min: float
) -> float:
    """Mean L2 distance over keypoints with confidence >= conf_min in both detections. Returns nan if fewer than 3 such keypoints exist."""
    mask = (a.keypoints[:, 2] >= conf_min) & (b.keypoints[:, 2] >= conf_min)
    if int(mask.sum()) < 3:
        return float("nan")
    diff = a.keypoints[mask, :2] - b.keypoints[mask, :2]
    return float(np.linalg.norm(diff, axis=1).mean())


def _merge_two_detections(
    kept: PoseDetection, dropped: PoseDetection
) -> PoseDetection:
    """Merge dropped into kept: per-keypoint, take the (x, y, conf) with higher conf. person_score is the max of the two."""
    new_kp = kept.keypoints.copy()
    take_dropped = dropped.keypoints[:, 2] > new_kp[:, 2]
    new_kp[take_dropped] = dropped.keypoints[take_dropped]
    return PoseDetection(
        frame_idx=kept.frame_idx,
        person_idx=kept.person_idx,
        person_score=float(max(kept.person_score, dropped.person_score)),
        keypoints=new_kp,
    )


def dedupe_per_frame_detections(
    detections: list[PoseDetection],
    *,
    similarity_threshold_px: float = 50.0,
    conf_min: float = 0.4,
    max_persons: int = 2,
) -> list[PoseDetection]:
    """Greedy-merge duplicates within each frame, then cap to max_persons.

    For each frame: repeatedly merge the closest pair whose mean keypoint
    distance is below similarity_threshold_px, then keep the top
    max_persons by person_score. Output is sorted by (frame_idx,
    -person_score).
    """
    by_frame: dict[int, list[PoseDetection]] = defaultdict(list)
    for d in detections:
        by_frame[d.frame_idx].append(d)

    out: list[PoseDetection] = []
    for frame_idx in sorted(by_frame.keys()):
        dets = list(by_frame[frame_idx])

        # greedy-merge until no pair is below the threshold
        while len(dets) > 1:
            best_pair: tuple[int, int] | None = None
            best_d = similarity_threshold_px
            n = len(dets)
            for i in range(n):
                for j in range(i + 1, n):
                    d = _pairwise_keypoint_distance(dets[i], dets[j], conf_min)
                    if not np.isnan(d) and d < best_d:
                        best_d, best_pair = d, (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            if dets[i].person_score >= dets[j].person_score:
                kept, dropped = dets[i], dets[j]
            else:
                kept, dropped = dets[j], dets[i]
            merged = _merge_two_detections(kept, dropped)
            dets = [d for k, d in enumerate(dets) if k != i and k != j] + [merged]

        # cap at max_persons by person_score descending
        dets.sort(key=lambda d: d.person_score, reverse=True)
        dets = dets[:max_persons]
        out.extend(dets)

    return out
