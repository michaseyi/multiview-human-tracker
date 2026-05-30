"""Recover a held-out keypoint in cam0 from epipolar intersections of cam1+cam2 detections.

Per recovery, visualises the support epipolar lines, the recovered intersection,
and the ground-truth keypoint.
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
from multiview_tracker.epipolar import (
    distort_points,
    line_image_endpoints,
    make_palette,
    undistort_points,
)
from multiview_tracker.recovery import (
    epipolar_line_in_a,
    recover_from_views,
    recover_from_views_undist,
)
from multiview_tracker.sync import filter_to_actor

COCO_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shldr", "R_shldr", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]


def find_eligible_frames(actor_a, actor_b, actor_c, conf_min=0.5, min_kps=8):
    """Frames where all three cameras have an actor detection with at least
    min_kps keypoints at conf >= conf_min in every view."""
    common = sorted(set(actor_a) & set(actor_b) & set(actor_c))
    eligible = []
    for f in common:
        m = (
            (actor_a[f].keypoints[:, 2] >= conf_min)
            & (actor_b[f].keypoints[:, 2] >= conf_min)
            & (actor_c[f].keypoints[:, 2] >= conf_min)
        )
        if int(m.sum()) >= min_kps:
            eligible.append((f, int(m.sum())))
    return eligible


def render_recovery(
    image_a: np.ndarray,
    F_ab: np.ndarray, point_b: np.ndarray,
    F_ac: np.ndarray, point_c: np.ndarray,
    recovered: np.ndarray,
    truth: np.ndarray,
    panel_label: str,
) -> np.ndarray:
    """Annotate image_a with both epipolar lines, the recovered intersection,
    and the ground-truth point."""
    out = image_a.copy()
    h, w = out.shape[:2]
    color_b = (255, 100, 100)   # line from cam B
    color_c = (100, 255, 100)   # line from cam C
    color_rec = (0, 255, 255)   # recovered
    color_truth = (0, 0, 255)   # ground truth

    line_b = epipolar_line_in_a(F_ab, point_b)
    line_c = epipolar_line_in_a(F_ac, point_c)
    for line, col in [(line_b, color_b), (line_c, color_c)]:
        ends = line_image_endpoints(line, w, h)
        if ends is not None:
            p, q = ends
            cv2.line(out,
                     (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     col, 2, cv2.LINE_AA)

    # recovered = filled disc
    if not np.isnan(recovered).any():
        cv2.circle(out, (int(recovered[0]), int(recovered[1])), 12, color_rec, -1, cv2.LINE_AA)
        cv2.circle(out, (int(recovered[0]), int(recovered[1])), 12, (0, 0, 0), 2, cv2.LINE_AA)
    # truth = ring
    cv2.circle(out, (int(truth[0]), int(truth[1])), 14, color_truth, 3, cv2.LINE_AA)

    pad = 12
    scale = max(1.5, out.shape[1] / 1100)
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(panel_label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, panel_label, (pad, th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0", help="camera to recover keypoints in")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2"],
                    help="2 or more supporting cameras (LS used if >2)")
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"],
                    help="which F to use for epipolar lines")
    ap.add_argument("--n-frames", type=int, default=4)
    ap.add_argument("--keypoints", type=int, nargs="*", default=None,
                    help="which COCO indices to evaluate (default: all visible in all 3 views)")
    ap.add_argument("--space", choices=["distorted", "undistorted"], default="undistorted",
                    help="distorted: raw pixels with F fit on raw pixels. "
                         "undistorted: undistort, fit F in undistorted space, redistort.")
    ap.add_argument("--sync", choices=["none", "event"], default="event",
                    help="event: affine offsets from offsets_event_anchored.json. "
                         "none: support frame = primary frame.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    kp_dir = Path(cfg["detection"]["output_dir"])

    cid_a = args.primary
    support_ids = args.support

    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    # with sync=event + space=undistorted, prefer the synced+undistorted F.
    # falls back to plain undistorted F if the synced one isn't on disk.
    f_suffix = ""
    if args.space == "undistorted":
        f_suffix = "_undist"
        if args.sync == "event":
            synced_path = out_dir / f"F_{cid_a}_{support_ids[0]}_synced_undist.npz"
            if synced_path.exists():
                f_suffix = "_synced_undist"
    F_supports = {sid: np.load(out_dir / f"F_{cid_a}_{sid}{f_suffix}.npz")[f_key]
                  for sid in support_ids}
    print(f"F suffix in use: {f_suffix!r}")
    if args.space == "undistorted":
        K = {}; D = {}
        for cid in [cid_a] + support_ids:
            z = np.load(out_dir / f"{cid}_calibration.npz")
            K[cid] = z["K"]; D[cid] = z["D"]
        print(f"loaded intrinsics for {len(K)} cameras  (--space {args.space}, F suffix '{f_suffix}')")
    else:
        K = D = None
        print(f"using distorted pipeline  (--space {args.space})")

    def _actor(cid):
        dets = dedupe_per_frame_detections(load_pose_detections(kp_dir / f"{cid}.npz"))
        return {d.frame_idx: d for d in filter_to_actor(dets, merge=True, static_anchor=True)}
    actor_a = _actor(cid_a)
    actor_supports = {sid: _actor(sid)
                      for sid in support_ids}

    # support_frame_for(sid, N) maps cam0 frame N to the matched support frame
    if args.sync == "event":
        from multiview_tracker.sync import TimeSync
        offsets_path = out_dir / "offsets_event_anchored.json"
        if not offsets_path.exists():
            raise SystemExit(f"--sync event requires {offsets_path}; run "
                             "scripts/compute_event_sync.py first.")
        _sync_model = TimeSync.from_json(offsets_path, primary=cid_a)
        print(f"event-anchored affine sync ({offsets_path.name}):")
        print(_sync_model.summary())
        def support_frame_for(sid, N):
            return _sync_model.support_frame(sid, N)
    else:
        def support_frame_for(sid, N):
            return N

    # cam0 frame N is eligible if every support sid has an actor detection at
    # the offset-corrected frame and >= 4 high-conf keypoints are shared with cam0
    eligible = []
    for f in sorted(actor_a):
        det_a_f = actor_a[f]
        ok = True
        masks = [det_a_f.keypoints[:, 2] >= 0.5]
        for sid in support_ids:
            n_s = support_frame_for(sid, f)
            if n_s not in actor_supports[sid]:
                ok = False
                break
            masks.append(actor_supports[sid][n_s].keypoints[:, 2] >= 0.5)
        if not ok:
            continue
        m = masks[0]
        for x in masks[1:]:
            m = m & x
        n = int(m.sum())
        if n >= 4:
            eligible.append((f, n))
    if not eligible:
        raise SystemExit(f"no eligible frames with actor + {len(support_ids)} supports")
    print(f"{len(eligible)} eligible frames")

    # pick n_frames spread across eligible
    picks = np.linspace(0, len(eligible) - 1, args.n_frames, dtype=int)
    chosen = [eligible[i] for i in picks]
    print(f"chosen frames: {[f for f, _ in chosen]}")

    cap_a = cv2.VideoCapture(str(next(c for c in cfg["cameras"] if c["id"] == cid_a)["source"]))

    out_root = out_dir / "recovery"
    out_root.mkdir(parents=True, exist_ok=True)

    all_errs: list[float] = []
    parallel_count = 0
    cross_count = 0
    ls_count = 0

    for frame_idx, _ in chosen:
        det_a = actor_a[frame_idx]
        det_supports_f = {
            sid: actor_supports[sid][support_frame_for(sid, frame_idx)]
            for sid in support_ids
        }

        if args.keypoints is None:
            mask = det_a.keypoints[:, 2] >= 0.5
            for sid in support_ids:
                mask &= det_supports_f[sid].keypoints[:, 2] >= 0.5
            kp_indices = [int(k) for k in np.where(mask)[0]]
        else:
            kp_indices = args.keypoints

        cap_a.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image_a = cap_a.read()
        if not ok:
            continue

        # all recoveries for this frame drawn on a single cam0 image
        out_image = image_a.copy()
        palette = make_palette(len(kp_indices))
        h, w = out_image.shape[:2]
        for slot, k in enumerate(kp_indices):
            xa_truth = det_a.keypoints[k, :2]
            if args.space == "undistorted":
                support_views = [
                    (K[sid], D[sid], F_supports[sid], det_supports_f[sid].keypoints[k, :2])
                    for sid in support_ids
                ]
                res = recover_from_views_undist(support_views, K[cid_a], D[cid_a])
            else:
                view_pts = [(F_supports[sid], det_supports_f[sid].keypoints[k, :2])
                            for sid in support_ids]
                res = recover_from_views(view_pts)
            err = float(np.linalg.norm(res.point - xa_truth)) if not np.isnan(res.point).any() else float("inf")

            all_errs.append(err)
            if res.method == "cross_product":
                cross_count += 1
            else:
                ls_count += 1
            if res.parallel:
                parallel_count += 1

            for sid in support_ids:
                p_x_raw = det_supports_f[sid].keypoints[k, :2]
                if args.space == "undistorted":
                    p_x_u = undistort_points(np.array([p_x_raw], np.float32), K[sid], D[sid])[0]
                    line_u = epipolar_line_in_a(F_supports[sid], p_x_u)
                    ends_u = line_image_endpoints(line_u, w, h)
                    if ends_u is not None:
                        p_u, q_u = np.asarray(ends_u[0], np.float64), np.asarray(ends_u[1], np.float64)
                        ts = np.linspace(0, 1, 40)
                        pts_u = (p_u[None, :] + ts[:, None] * (q_u - p_u)[None, :]).astype(np.float32)
                        pts_d = distort_points(pts_u, K[cid_a], D[cid_a])
                        for i in range(len(pts_d) - 1):
                            x0, y0 = pts_d[i]; x1, y1 = pts_d[i + 1]
                            cv2.line(out_image,
                                     (int(round(x0)), int(round(y0))),
                                     (int(round(x1)), int(round(y1))),
                                     palette[slot], 1, cv2.LINE_AA)
                else:
                    line = epipolar_line_in_a(F_supports[sid], p_x_raw)
                    ends = line_image_endpoints(line, w, h)
                    if ends is not None:
                        p, q = ends
                        cv2.line(out_image,
                                 (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                                 palette[slot], 1, cv2.LINE_AA)
            # skip drawing the recovered point when it landed wildly off-image
            # (near-parallel epipolar lines make the LS solution explode)
            if not np.isnan(res.point).any() and abs(res.point[0]) < 1e6 and abs(res.point[1]) < 1e6:
                px, py = int(res.point[0]), int(res.point[1])
                if -w < px < 2 * w and -h < py < 2 * h:
                    cv2.circle(out_image, (px, py), 10, palette[slot], -1, cv2.LINE_AA)
                    cv2.circle(out_image, (px, py), 10, (0, 0, 0), 2, cv2.LINE_AA)
            # truth = ring
            cv2.circle(out_image, (int(xa_truth[0]), int(xa_truth[1])),
                       12, palette[slot], 3, cv2.LINE_AA)
            print(
                f"  frame {frame_idx} kp {k:2d} ({COCO_NAMES[k]:>8s}): "
                f"recovered ({res.point[0]:7.1f}, {res.point[1]:7.1f})  "
                f"truth ({xa_truth[0]:7.1f}, {xa_truth[1]:7.1f})  "
                f"err {err:6.1f} px  [{res.method}]"
            )

        legend = (
            "filled disc = recovered    open ring = truth    "
            f"method={args.method.upper()}    frame={frame_idx}    "
            f"supports={'+'.join(support_ids)}"
        )
        pad = 12
        scale = max(1.0, out_image.shape[1] / 1500)
        thick = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(out_image, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
        cv2.putText(out_image, legend, (pad, th + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

        out_path = out_root / f"recovery_f{frame_idx:06d}_{args.method}_{'_'.join(support_ids)}_{args.space}.jpg"
        cv2.imwrite(str(out_path), out_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  -> {out_path}")

    cap_a.release()

    # summary
    errs = np.array(all_errs)
    finite = errs[np.isfinite(errs)]
    print()
    print("=" * 60)
    print(f"  Recoveries: {len(errs)} ({cross_count} cross-product, {ls_count} LS, {parallel_count} flagged near-parallel)")
    if len(finite):
        print(f"  Recovery error (px):  median={np.median(finite):.2f}  "
              f"mean={finite.mean():.2f}  max={finite.max():.2f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
