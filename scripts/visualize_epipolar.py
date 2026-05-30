"""Visualise epipolar lines for puzzleboard corners and actor keypoints.

Per frame: three side-by-side comparisons (one per F method) of image A with
selected points and image B with the corresponding epipolar lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.calibration import PuzzleboardConfig, detect_puzzleboard
from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.epipolar import (
    draw_points_and_epilines,
    make_palette,
    stack_compare,
    symmetric_epipolar_distance,
)
from multiview_tracker.sync import filter_to_actor


def index_dets_by_frame_local(detections):
    out: dict[int, list] = {}
    for d in detections:
        out.setdefault(d.frame_idx, []).append(d)
    return out


def grab_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def board_points_for_pair(det_a, det_b, n_points: int):
    """Common puzzleboard ids to (pts_a, pts_b) with up to n_points samples
    spread across the board."""
    ids_a = {tuple(r): i for i, r in enumerate(det_a.point_ids)}
    common = []
    for j, r in enumerate(det_b.point_ids):
        key = tuple(r)
        if key in ids_a:
            common.append((ids_a[key], j))
    if not common:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    if len(common) > n_points:
        step = len(common) // n_points
        common = common[::step][:n_points]
    pa = np.stack([det_a.image_points[i] for i, _ in common]).astype(np.float32)
    pb = np.stack([det_b.image_points[j] for _, j in common]).astype(np.float32)
    return pa, pb


def actor_keypoints_for_pair(det_a, det_b, conf_min: float = 0.5):
    """Match COCO keypoint indices between two single-person detections.
    Pass None for either to skip."""
    if det_a is None or det_b is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    pa, pb = [], []
    for k in range(17):
        if det_a.keypoints[k, 2] >= conf_min and det_b.keypoints[k, 2] >= conf_min:
            pa.append(det_a.keypoints[k, :2])
            pb.append(det_b.keypoints[k, :2])
    if not pa:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    return np.stack(pa).astype(np.float32), np.stack(pb).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", required=True, help='e.g. "cam0,cam2"')
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frames", nargs="+", type=int, default=None,
                    help="explicit frame indices to render; if omitted, picks 4 frames spread across the saved correspondences")
    ap.add_argument("--n-board-pts", type=int, default=14)
    ap.add_argument("--panel-w", type=int, default=900)
    args = ap.parse_args()

    cid_a, cid_b = [s.strip() for s in args.pair.split(",")]
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    kp_dir = Path(cfg["detection"]["output_dir"])

    cam_a = next(c for c in cfg["cameras"] if c["id"] == cid_a)
    cam_b = next(c for c in cfg["cameras"] if c["id"] == cid_b)

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
    )

    # prefer the synced+undistorted F; fall back to older variants if absent
    for suffix in ("_synced_undist", "_undist", ""):
        f_npz = out_dir / f"F_{cid_a}_{cid_b}{suffix}.npz"
        if f_npz.exists():
            break
    else:
        raise SystemExit(f"no F file found for {cid_a}<->{cid_b}; run "
                         "refit_fundamental_synced.py first")
    print(f"using F from {f_npz.name}")
    z = np.load(f_npz)
    F_methods = {"C1 OpenCV direct": z["F_c1"],
                 "C2 OpenCV via E": z["F_c2"],
                 "C3 manual 8-point": z["F_c3"]}

    # if F was fit from synced+undistorted correspondences, the same TimeSync
    # must be applied here so epipolar lines are drawn against simultaneous samples
    ts = None
    if "synced_undist" in f_npz.name:
        from multiview_tracker.sync import TimeSync
        ts = TimeSync.from_json(out_dir / "offsets_event_anchored.json",
                                 primary=cid_a)
        print(f"applying event-anchored TimeSync (matched cam_b frame per cam_a frame)")

    cap_a = cv2.VideoCapture(str(cam_a["source"]))
    cap_b = cv2.VideoCapture(str(cam_b["source"]))

    # filter to the moving-actor track per camera so we don't match the actor
    # in one view to the seated bystander in another
    def _actor(cid):
        dets = dedupe_per_frame_detections(load_pose_detections(kp_dir / f"{cid}.npz"))
        return {d.frame_idx: d for d in filter_to_actor(dets, merge=True, static_anchor=True)}
    actor_a = _actor(cid_a)
    actor_b = _actor(cid_b)

    # pick frames in two batches: board frames (low SED expected, sanity check)
    # and actor frames (the real test)
    if args.frames:
        frames = args.frames
    else:
        board_eligible = sorted(set(int(f) for f in np.unique(z["frame_idx"])))
        actor_eligible = sorted(set(actor_a.keys()) & set(actor_b.keys()))
        n_per = 2
        b_picks = np.linspace(0, len(board_eligible) - 1, n_per, dtype=int)
        a_picks = np.linspace(0, len(actor_eligible) - 1, n_per, dtype=int)
        frames = (
            [int(board_eligible[i]) for i in b_picks] +
            [int(actor_eligible[i]) for i in a_picks]
        )
        print(f"selected frames: {frames}  "
              f"(board pool {len(board_eligible)}, actor pool {len(actor_eligible)})")

    out_dir_pngs = out_dir / "epipolar_vis"
    out_dir_pngs.mkdir(parents=True, exist_ok=True)

    for frame_idx in frames:
        print(f"\n[render] frame {frame_idx}")
        img_a = grab_frame(cap_a, frame_idx)
        # cam_b is sampled at the sync-corrected frame when TimeSync is active
        frame_idx_b = ts.support_frame(cid_b, frame_idx) if ts is not None else frame_idx
        img_b = grab_frame(cap_b, frame_idx_b)
        if img_a is None or img_b is None:
            print("  failed to read"); continue

        det_a = detect_puzzleboard(img_a, pb_cfg)
        det_b = detect_puzzleboard(img_b, pb_cfg)
        pa_b, pb_b = (board_points_for_pair(det_a, det_b, args.n_board_pts)
                      if det_a is not None and det_b is not None
                      else (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)))

        pa_k, pb_k = actor_keypoints_for_pair(actor_a.get(frame_idx), actor_b.get(frame_idx_b))

        pa = np.concatenate([pa_b, pa_k], axis=0) if len(pa_b) or len(pa_k) else None
        pb = np.concatenate([pb_b, pb_k], axis=0) if len(pb_b) or len(pb_k) else None
        if pa is None or len(pa) == 0:
            print("  no points"); continue

        # board points get cool hues, actor keypoints get warm hues
        n_b, n_k = len(pa_b), len(pa_k)
        palette = make_palette(n_b) + make_palette(n_k)[::-1]

        rows = []
        for label, F in F_methods.items():
            sed = symmetric_epipolar_distance(F, pa, pb)
            sed_b = sed[:n_b] if n_b else np.array([])
            sed_k = sed[n_b:] if n_k else np.array([])
            full_label = (
                f"{label}  "
                f"board(n={n_b}) med={np.median(sed_b):.1f}px  "
                f"actor(n={n_k}) med={np.median(sed_k):.1f}px"
                if n_k and n_b
                else f"{label}  n={len(pa)}  med SED={np.median(sed):.1f}px"
            )
            ann_a, ann_b = draw_points_and_epilines(img_a, img_b, pa, pb, F, palette=palette)
            rows.append((full_label, ann_a, ann_b))

        composite = stack_compare(rows, panel_w=args.panel_w)
        out_path = out_dir_pngs / f"f{frame_idx:06d}_{cid_a}_{cid_b}.jpg"
        cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  -> {out_path}  (board={n_b}, actor_kps={n_k})")

    cap_a.release()
    cap_b.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
