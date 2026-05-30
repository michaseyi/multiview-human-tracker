from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from multiview_tracker.calibration.puzzleboard import (
    PuzzleboardConfig,
    detect_puzzleboard,
)


@dataclass
class CorrespondenceSet:
    pts_a: np.ndarray             # (N, 2) pixel coords in image A
    pts_b: np.ndarray             # (N, 2) pixel coords in image B
    frame_idx: np.ndarray         # (N,) which frame each correspondence came from
    point_ids: np.ndarray         # (N, 2) puzzleboard (row, col) id
    n_frames_used: int


def _match_by_id(det_a, det_b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two puzzleboard detections by (row, col) ids, returning (pts_a, pts_b, ids) for common points."""
    ids_a = {tuple(r): i for i, r in enumerate(det_a.point_ids)}
    pa: list[np.ndarray] = []
    pb: list[np.ndarray] = []
    ids: list[tuple[int, int]] = []
    for i, r in enumerate(det_b.point_ids):
        key = tuple(r)
        if key in ids_a:
            pa.append(det_a.image_points[ids_a[key]])
            pb.append(det_b.image_points[i])
            ids.append(key)
    if not pa:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )
    return (
        np.stack(pa).astype(np.float32),
        np.stack(pb).astype(np.float32),
        np.array(ids, dtype=np.int32),
    )


def harvest_correspondences(
    video_a: Path,
    video_b: Path,
    pb_cfg: PuzzleboardConfig,
    *,
    stride: int = 50,
    min_matches_per_frame: int = 50,
    target_frames: int = 30,
    offset_b: int = 0,
    support_frame_fn=None,
) -> CorrespondenceSet:
    """Walk two videos, detect the puzzleboard in both, collect id-matched correspondences.

    cam_b frame for a given cam_a frame idx:
      - offset_b: constant integer offset (cam_b reads idx + offset_b).
      - support_frame_fn: callable idx -> int, per-frame; takes precedence over offset_b.
        Use with a TimeSync model to compensate for clock drift between cameras.
    """
    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    if not (cap_a.isOpened() and cap_b.isOpened()):
        raise RuntimeError("cannot open one of the videos")

    if support_frame_fn is not None:
        # conservative upper bound; per-frame validity is re-checked below
        n_total = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
        n_b_total = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"[xcorr] {video_a.name} <-> {video_b.name} "
            f"| {n_total} frames | stride={stride} | support_frame_fn=<callable>"
        )
    else:
        n_total = min(
            int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT)) - offset_b,
        )
        n_b_total = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
        print(
            f"[xcorr] {video_a.name} <-> {video_b.name} "
            f"| {n_total} frames | stride={stride} | offset_b={offset_b:+d}"
        )

    pa_all: list[np.ndarray] = []
    pb_all: list[np.ndarray] = []
    fr_all: list[int] = []
    ids_all: list[np.ndarray] = []
    n_frames_used = 0

    pbar = tqdm(range(0, n_total, stride), desc="scanning", unit="frame")
    for idx in pbar:
        if n_frames_used >= target_frames:
            break
        if support_frame_fn is not None:
            idx_b = int(support_frame_fn(idx))
        else:
            idx_b = idx + offset_b
        if idx_b < 0 or idx_b >= n_b_total:
            continue
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not (ok_a and ok_b):
            continue

        det_a = detect_puzzleboard(frame_a, pb_cfg)
        det_b = detect_puzzleboard(frame_b, pb_cfg)
        if det_a is None or det_b is None:
            continue

        pa, pb, ids = _match_by_id(det_a, det_b)
        if len(pa) < min_matches_per_frame:
            continue

        pa_all.append(pa)
        pb_all.append(pb)
        fr_all.extend([idx] * len(pa))
        ids_all.append(ids)
        n_frames_used += 1
        pbar.set_postfix(frames=n_frames_used, matches=len(pa))

    cap_a.release()
    cap_b.release()

    if not pa_all:
        raise RuntimeError("no usable correspondences found")

    return CorrespondenceSet(
        pts_a=np.concatenate(pa_all, axis=0),
        pts_b=np.concatenate(pb_all, axis=0),
        frame_idx=np.array(fr_all, dtype=np.int64),
        point_ids=np.concatenate(ids_all, axis=0),
        n_frames_used=n_frames_used,
    )
