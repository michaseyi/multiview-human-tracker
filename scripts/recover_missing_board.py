"""Sanity-check the recovery algorithm on puzzleboard corners.

Correspondences are unambiguous, so any error is from the recovery math and F
estimation rather than from mismatched inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.calibration import PuzzleboardConfig, detect_puzzleboard
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


def common_ids(det_a, det_b, det_c):
    a = {tuple(r): det_a.image_points[i] for i, r in enumerate(det_a.point_ids)}
    b = {tuple(r): det_b.image_points[i] for i, r in enumerate(det_b.point_ids)}
    c = {tuple(r): det_c.image_points[i] for i, r in enumerate(det_c.point_ids)}
    keys = sorted(set(a) & set(b) & set(c))
    return keys, a, b, c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2"],
                    help="2 or more supporting camera ids (LS used if >2)")
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--frame", type=int, default=None,
                    help="frame index; if omitted, picks the first frame where all 3 cameras detect the board")
    ap.add_argument("--n-samples", type=int, default=12,
                    help="number of corners to recover, spread across the board")
    ap.add_argument("--space", choices=["distorted", "undistorted"], default="undistorted",
                    help="distorted: raw pixels with F fit on raw pixels. "
                         "undistorted: undistort, fit F in undistorted space, redistort.")
    ap.add_argument("--sync", choices=["none", "event"], default="event",
                    help="none: support frame = primary frame. "
                         "event: TimeSync from offsets_event_anchored.json.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
    )

    cid_a = args.primary
    support_ids = args.support
    if len(support_ids) < 2:
        raise SystemExit("need at least 2 supporting cameras")

    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
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

    # with --sync event, look up cam_b frames via TimeSync
    if args.sync == "event":
        from multiview_tracker.sync import TimeSync
        _sync_model = TimeSync.from_json(out_dir / "offsets_event_anchored.json",
                                          primary=cid_a)
        def support_frame_for(sid, N):
            return _sync_model.support_frame(sid, N)
    else:
        def support_frame_for(sid, N):
            return N

    # intrinsics for the undistorted pipeline
    if args.space == "undistorted":
        K = {}; D = {}
        for cid in [cid_a] + support_ids:
            z = np.load(out_dir / f"{cid}_calibration.npz")
            K[cid] = z["K"]; D[cid] = z["D"]
        print(f"loaded intrinsics for {len(K)} cameras (undistorted pipeline)")

    src = {c["id"]: Path(c["source"]) for c in cfg["cameras"]}
    cap_a = cv2.VideoCapture(str(src[cid_a]))
    cap_supports = {sid: cv2.VideoCapture(str(src[sid])) for sid in support_ids}

    n_total = min(int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)),
                  *(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in cap_supports.values()))

    def detect_in_all(idx: int):
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_a, frame_a = cap_a.read()
        if not ok_a:
            return None
        det_a = detect_puzzleboard(frame_a, pb_cfg)
        if det_a is None:
            return None
        det_supports = {}
        frame_supports = {}
        # read each support at its sync-corrected frame so correspondences
        # refer to the same physical moment, not the same integer index
        for sid, cap in cap_supports.items():
            idx_b = support_frame_for(sid, idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx_b)
            ok, fr = cap.read()
            if not ok:
                return None
            d = detect_puzzleboard(fr, pb_cfg)
            if d is None:
                return None
            det_supports[sid] = d
            frame_supports[sid] = fr
        return frame_a, det_a, frame_supports, det_supports

    chosen_frame = args.frame
    chosen = None
    if chosen_frame is None:
        for idx in range(0, n_total, 50):
            r = detect_in_all(idx)
            if r is None:
                continue
            frame_a, det_a, frame_supports, det_supports = r
            id_sets = [{tuple(row) for row in det_a.point_ids}]
            id_sets += [{tuple(row) for row in d.point_ids} for d in det_supports.values()]
            common = sorted(set.intersection(*id_sets))
            if len(common) >= args.n_samples:
                chosen_frame = idx
                chosen = (frame_a, det_a, frame_supports, det_supports)
                break
        if chosen is None:
            raise SystemExit("no frame has board detected by primary + all supports")
    else:
        chosen = detect_in_all(chosen_frame)
        if chosen is None:
            raise SystemExit(f"frame {chosen_frame} not detected in all cameras")
    print(f"using frame {chosen_frame}")

    image_a, da, frame_supports, det_supports = chosen
    id_sets = [{tuple(row) for row in da.point_ids}]
    id_sets += [{tuple(row) for row in d.point_ids} for d in det_supports.values()]
    common_keys = sorted(set.intersection(*id_sets))

    ma = {tuple(r): da.image_points[i] for i, r in enumerate(da.point_ids)}
    m_supports = {sid: {tuple(r): d.image_points[i] for i, r in enumerate(d.point_ids)}
                  for sid, d in det_supports.items()}
    keys = common_keys
    if len(keys) > args.n_samples:
        step = len(keys) // args.n_samples
        keys = keys[::step][: args.n_samples]
    print(f"recovering {len(keys)} corners")

    out_image = image_a.copy()
    h, w = out_image.shape[:2]
    palette = make_palette(len(keys))
    errs = []
    n_cross = n_ls = n_par = 0

    for slot, k in enumerate(keys):
        truth = ma[k]  # raw distorted pixel of cam0 truth

        if args.space == "undistorted":
            support_views = [
                (K[sid], D[sid], F_supports[sid], m_supports[sid][k])
                for sid in support_ids
            ]
            res = recover_from_views_undist(support_views, K[cid_a], D[cid_a])
        else:
            view_pts = [(F_supports[sid], m_supports[sid][k]) for sid in support_ids]
            res = recover_from_views(view_pts)

        err = float(np.linalg.norm(res.point - truth)) if not np.isnan(res.point).any() else float("inf")
        errs.append(err)
        if res.method == "cross_product":
            n_cross += 1
        else:
            n_ls += 1
        if res.parallel:
            n_par += 1

        col = palette[slot]
        # undistorted mode: lines live in undistorted space and must be
        # redistorted as polylines onto the raw frame.
        # distorted mode: lines are straight in raw pixel space.
        for sid in support_ids:
            p_x_raw = m_supports[sid][k]
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
                                 col, 1, cv2.LINE_AA)
            else:
                line = epipolar_line_in_a(F_supports[sid], p_x_raw)
                ends = line_image_endpoints(line, w, h)
                if ends is not None:
                    p, q = ends
                    cv2.line(out_image, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                             col, 1, cv2.LINE_AA)

        # recovered = filled disc, truth = ring
        if not np.isnan(res.point).any():
            cv2.circle(out_image, (int(res.point[0]), int(res.point[1])), 8, col, -1, cv2.LINE_AA)
            cv2.circle(out_image, (int(res.point[0]), int(res.point[1])), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(out_image, (int(truth[0]), int(truth[1])), 10, col, 3, cv2.LINE_AA)
        print(f"  id {k}: recovered ({res.point[0]:7.1f}, {res.point[1]:7.1f})  "
              f"truth ({truth[0]:7.1f}, {truth[1]:7.1f})  err {err:5.2f} px  [{res.method}]")

    legend = (
        f"filled disc=recovered  open ring=truth  method={args.method.upper()}  "
        f"frame={chosen_frame}  supports={'+'.join(support_ids)}"
    )
    pad = 12
    scale = max(1.0, out_image.shape[1] / 1500)
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out_image, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out_image, legend, (pad, th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    out_path = out_dir / "recovery" / f"recovery_board_f{chosen_frame:06d}_{args.method}_{'_'.join(support_ids)}_{args.space}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n[save] -> {out_path}")

    finite = np.array([e for e in errs if np.isfinite(e)])
    print(f"\nRecoveries: {len(errs)} ({n_cross} cross, {n_ls} LS, {n_par} flagged parallel)")
    if len(finite):
        print(f"Recovery error (px): median={np.median(finite):.2f}  mean={finite.mean():.2f}  max={finite.max():.2f}")

    cap_a.release()
    for cap in cap_supports.values():
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
