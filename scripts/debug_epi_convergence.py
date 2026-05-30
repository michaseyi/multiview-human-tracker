"""For one frame, draw on cam0: the actor's actual keypoint pixel (filled dot),
epipolar lines from cam1/cam2/cam3 projected into cam0 (color-matched), and
the LS intersection of the three lines (open ring).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.epipolar import line_image_endpoints, make_palette
from multiview_tracker.recovery import (
    epipolar_line_in_a,
    least_squares_intersection,
)
from multiview_tracker.sync import filter_to_actor

COCO_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shldr", "R_shldr", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--frame", type=int, default=None,
                    help="explicit source frame index; default is auto-pick")
    ap.add_argument("--conf-min", type=float, default=0.5)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    kp_dir = Path(cfg["detection"]["output_dir"])
    cid_a = args.primary
    sids = args.support

    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    F_supports = {sid: np.load(out_dir / f"F_{cid_a}_{sid}.npz")[f_key] for sid in sids}

    def actor_by_frame(cid):
        dets = dedupe_per_frame_detections(load_pose_detections(kp_dir / f"{cid}.npz"))
        return {d.frame_idx: d for d in filter_to_actor(dets, merge=True, static_anchor=True)}

    actor_a = actor_by_frame(cid_a)
    actor_sup = {sid: actor_by_frame(sid) for sid in sids}

    # auto-pick a frame where all 4 actor detections have plenty of high-conf keypoints
    if args.frame is None:
        common = set(actor_a)
        for sid in sids:
            common &= set(actor_sup[sid])
        if not common:
            raise SystemExit("no frame has actor in all 4 cameras")
        best_f, best_n = None, 0
        for f in sorted(common):
            m = actor_a[f].keypoints[:, 2] >= args.conf_min
            for sid in sids:
                m = m & (actor_sup[sid][f].keypoints[:, 2] >= args.conf_min)
            n = int(m.sum())
            if n > best_n:
                best_n, best_f = n, f
        frame_idx = int(best_f)
        print(f"auto-picked frame {frame_idx} with {best_n} common high-conf keypoints")
    else:
        frame_idx = args.frame

    src = next(c["source"] for c in cfg["cameras"] if c["id"] == cid_a)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame_idx} of {cid_a}")

    h, w = image.shape[:2]

    det_a = actor_a[frame_idx]
    det_sup = {sid: actor_sup[sid][frame_idx] for sid in sids}

    # keypoints to draw: those visible in primary and all supports
    mask = det_a.keypoints[:, 2] >= args.conf_min
    for sid in sids:
        mask = mask & (det_sup[sid].keypoints[:, 2] >= args.conf_min)
    kp_indices = [int(k) for k in np.where(mask)[0]]
    if not kp_indices:
        raise SystemExit("no common high-conf keypoints in this frame")

    palette = make_palette(len(kp_indices))

    # for each tested keypoint, draw the 3 epipolar lines + truth + LS intersection
    for slot, k in enumerate(kp_indices):
        col = palette[slot]
        truth = det_a.keypoints[k, :2]
        sup_pts = [(sid, det_sup[sid].keypoints[k, :2]) for sid in sids]
        lines = [epipolar_line_in_a(F_supports[sid], xp) for sid, xp in sup_pts]
        # shade each line to distinguish sources: full, dim, dimmer
        line_colors = [
            tuple(min(255, int(c * f)) for c in col)
            for f in (1.0, 0.75, 0.5)
        ]
        for line, lc in zip(lines, line_colors):
            ends = line_image_endpoints(line, w, h)
            if ends is not None:
                p, q = ends
                cv2.line(image, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                         lc, 2, cv2.LINE_AA)
        ls_pt = least_squares_intersection(np.stack(lines))
        if not np.isnan(ls_pt).any():
            cv2.circle(image, (int(ls_pt[0]), int(ls_pt[1])), 12, col, 3, cv2.LINE_AA)
        # truth: filled dot
        cv2.circle(image, (int(truth[0]), int(truth[1])), 18, col, -1, cv2.LINE_AA)
        cv2.circle(image, (int(truth[0]), int(truth[1])), 18, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"{COCO_NAMES[k]}",
                    (int(truth[0]) + 22, int(truth[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

    legend_lines = [
        f"primary={cid_a}  supports={'+'.join(sids)}  method={args.method.upper()}  frame={frame_idx}",
        "filled dot = truth (cam0 detection)",
        "open ring = LS intersection of the 3 epipolar lines",
        "3 line shades per colour = lines from cam1 / cam2 / cam3 respectively",
    ]
    pad = 14
    scale = max(1.0, w / 1500)
    thick = max(2, int(scale * 2))
    line_h = int(28 * scale)
    rect_h = pad * 2 + line_h * len(legend_lines)
    max_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
                for t in legend_lines)
    cv2.rectangle(image, (0, 0), (max_w + 2 * pad, rect_h), (0, 0, 0), -1)
    for i, t in enumerate(legend_lines):
        cv2.putText(image, t, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    out_root = out_dir / "epi_convergence"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"f{frame_idx:06d}_{cid_a}_from_{'_'.join(sids)}_{args.method}.jpg"
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved -> {out_path}")

    # per-keypoint distance from LS intersection to truth
    print(f"\n{'kp':>4s} {'name':>8s} {'sin_min':>8s} {'err_px':>8s}")
    for slot, k in enumerate(kp_indices):
        truth = det_a.keypoints[k, :2]
        sup_pts = [det_sup[sid].keypoints[k, :2] for sid in sids]
        lines = np.stack([
            epipolar_line_in_a(F_supports[sid], xp) for sid, xp in zip(sids, sup_pts)
        ])
        ls_pt = least_squares_intersection(lines)
        err = float(np.linalg.norm(ls_pt - truth)) if not np.isnan(ls_pt).any() else float("inf")
        # smallest sin angle between any pair of lines
        n = lines[:, :2]
        n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        sin_min = 1.0
        for i in range(len(n)):
            for j in range(i + 1, len(n)):
                s = abs(float(np.cross(n[i], n[j])))
                sin_min = min(sin_min, s)
        print(f"{k:>4d} {COCO_NAMES[k]:>8s} {sin_min:>8.3f} {err:>8.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
