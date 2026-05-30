"""Same visualisation as debug_epi_convergence.py but using puzzleboard corners instead of YOLO actor keypoints.

For a chosen frame: detect the board in all 4 cameras, intersect by (row, col)
point_id, sample N corners evenly across the board, then draw on cam0 the cam0
detection (filled dot), epipolar lines from each support (shaded), and the LS
intersection (open ring). Isolates "is F good?" from "are the actor keypoints
consistent across views?".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from multiview_tracker.calibration import PuzzleboardConfig, detect_puzzleboard
from multiview_tracker.epipolar import line_image_endpoints, make_palette
from multiview_tracker.recovery import (
    epipolar_line_in_a,
    least_squares_intersection,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--frame", type=int, default=None,
                    help="explicit frame; default auto-picks from the cam0-cam1 F-fit pool")
    ap.add_argument("--n-corners", type=int, default=18,
                    help="number of corners to render, sampled evenly across the board")
    ap.add_argument("--apply-offsets", choices=["none", "fps_drift", "affine"], default="none",
                    help="none = same frame index for all cameras; "
                         "fps_drift = round(n_a * fps_s/fps_a); "
                         "affine = fps_drift plus per-camera alpha from SED sweep")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    cid_a = args.primary
    sids = args.support
    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    F = {sid: np.load(out_dir / f"F_{cid_a}_{sid}.npz")[f_key] for sid in sids}

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
        discard_edge_layers=0,  # visualisation uses the full board (keep outer ring)
    )

    cams = {c["id"]: c for c in cfg["cameras"]}

    # read FPS directly from each video; yaml has the same nominal value for
    # all four cameras but the true clocks differ
    def video_fps(cid):
        c = cv2.VideoCapture(str(cams[cid]["source"]))
        v = c.get(cv2.CAP_PROP_FPS) or 24.55
        c.release()
        return float(v)

    fps_a = video_fps(cid_a)
    fps_supports = {sid: video_fps(sid) for sid in sids}

    def support_index(sid, n_a):
        if args.apply_offsets == "fps_drift":
            return int(round(n_a * fps_supports[sid] / fps_a))
        return n_a

    def grab(cid, idx):
        cap = cv2.VideoCapture(str(cams[cid]["source"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    # candidate frames: union of frames where any pair extracted board
    # correspondences (board visible in cam0 + that support); other supports
    # are probed at the same frame
    candidates = set()
    for sid in sids:
        candidates |= set(int(f) for f in np.unique(np.load(out_dir / f"F_{cid_a}_{sid}.npz")["frame_idx"]))
    candidates = sorted(candidates)
    if not candidates:
        raise SystemExit("no candidate frames in any F.npz")

    # pick the first frame with the board detected in all 4 cameras, unless --frame is given
    if args.frame is not None:
        frame_idx = args.frame
        dets = {cid_a: None}
        for sid in sids:
            dets[sid] = None
    else:
        print(f"probing {len(candidates)} candidate frames for board-in-all-4...")
        frame_idx = None
        best_n_common = 0
        for f in candidates:
            ok = True
            dets_try = {}
            for cid in [cid_a] + sids:
                img = grab(cid, f)
                if img is None:
                    ok = False
                    break
                d = detect_puzzleboard(img, pb_cfg)
                if d is None or len(d.point_ids) == 0:
                    ok = False
                    break
                dets_try[cid] = d
            if not ok:
                continue
            # common point IDs across all 4
            id_sets = [set(map(tuple, d.point_ids)) for d in dets_try.values()]
            n_common = len(set.intersection(*id_sets))
            if n_common > best_n_common:
                best_n_common = n_common
                frame_idx = f
                # accept the first frame with >= 80 common corners
                if n_common >= 80:
                    break
        if frame_idx is None:
            raise SystemExit("no frame has board in all 4 cameras")
        print(f"auto-picked frame {frame_idx} with {best_n_common} common board corners")

    # detect the board in all cameras at the chosen frame; primary uses
    # frame_idx, each support uses its offset-corrected index
    dets = {}
    frames_used = {cid_a: frame_idx}
    for cid in [cid_a] + sids:
        f_use = frame_idx if cid == cid_a else support_index(cid, frame_idx)
        frames_used[cid] = f_use
        img = grab(cid, f_use)
        if img is None:
            raise SystemExit(f"failed to read {cid} frame {f_use}")
        d = detect_puzzleboard(img, pb_cfg)
        if d is None or len(d.point_ids) == 0:
            raise SystemExit(f"no board detected in {cid} at frame {f_use}")
        dets[cid] = d
    print(f"frames used (offsets={args.apply_offsets}): " +
          "  ".join(f"{cid}={frames_used[cid]}" for cid in [cid_a] + sids))

    image_a = grab(cid_a, frame_idx)
    h, w = image_a.shape[:2]

    # build id -> image_point lookup per camera, then intersect IDs
    lookup = {cid: {tuple(r): dets[cid].image_points[i]
                    for i, r in enumerate(dets[cid].point_ids)}
              for cid in [cid_a] + sids}
    common_ids = sorted(
        set.intersection(*(set(lookup[cid].keys()) for cid in [cid_a] + sids))
    )
    if not common_ids:
        raise SystemExit(f"no corners common to {cid_a}+{'+'.join(sids)} at frame {frame_idx}")
    print(f"{len(common_ids)} corners visible in {cid_a}+{'+'.join(sids)} at frame {frame_idx}")

    # sample N corners spread across the board (every k-th in sorted order)
    if len(common_ids) > args.n_corners:
        step = len(common_ids) // args.n_corners
        sampled = common_ids[::step][:args.n_corners]
    else:
        sampled = common_ids

    palette = make_palette(len(sampled))

    # per-corner: draw truth, 3 epipolar lines, LS intersection
    errors = []
    for slot, cid_kv in enumerate(sampled):
        col = palette[slot]
        truth = lookup[cid_a][cid_kv]
        lines = []
        for sid in sids:
            xb = lookup[sid][cid_kv]
            lines.append(epipolar_line_in_a(F[sid], xb))
        # shade each line to distinguish sources: full, dim, dimmer
        line_colors = [
            tuple(min(255, int(c * f)) for c in col)
            for f in (1.0, 0.75, 0.5)
        ]
        for line, lc in zip(lines, line_colors):
            ends = line_image_endpoints(line, w, h)
            if ends is not None:
                p, q = ends
                cv2.line(image_a, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                         lc, 1, cv2.LINE_AA)
        ls_pt = least_squares_intersection(np.stack(lines))
        err = float(np.linalg.norm(ls_pt - truth)) if not np.isnan(ls_pt).any() else float("inf")
        errors.append((cid_kv, err))
        if not np.isnan(ls_pt).any():
            cv2.circle(image_a, (int(ls_pt[0]), int(ls_pt[1])), 10, col, 2, cv2.LINE_AA)
        # truth: filled dot
        cv2.circle(image_a, (int(truth[0]), int(truth[1])), 6, col, -1, cv2.LINE_AA)
        cv2.circle(image_a, (int(truth[0]), int(truth[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)

    sup_frames_label = "  ".join(f"{sid}@{frames_used[sid]}" for sid in sids)
    legend_lines = [
        f"primary={cid_a}@{frame_idx}  supports={sup_frames_label}",
        f"method={args.method.upper()}  offsets={args.apply_offsets}",
        "filled dot = truth (cam0 board corner detection)",
        f"open ring = LS intersection of {len(sids)} epipolar line(s)",
        f"{len(sids)} line shades per colour = lines from " + " / ".join(sids),
        f"corners sampled: {len(sampled)} of {len(common_ids)} common",
    ]
    pad = 14
    scale = max(1.0, w / 1500)
    thick = max(2, int(scale * 2))
    line_h = int(28 * scale)
    rect_h = pad * 2 + line_h * len(legend_lines)
    max_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
                for t in legend_lines)
    cv2.rectangle(image_a, (0, 0), (max_w + 2 * pad, rect_h), (0, 0, 0), -1)
    for i, t in enumerate(legend_lines):
        cv2.putText(image_a, t, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    out_root = out_dir / "epi_convergence_board"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"f{frame_idx:06d}_{cid_a}_from_{'_'.join(sids)}_{args.method}_offsets-{args.apply_offsets}.jpg"
    cv2.imwrite(str(out_path), image_a, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"saved -> {out_path}")

    # per-corner error summary
    errs = np.array([e for _, e in errors if np.isfinite(e)])
    print(f"\nrecovery error on {len(errs)} board corners (px):")
    print(f"  median = {np.median(errs):.2f}")
    print(f"  mean   = {errs.mean():.2f}")
    print(f"  max    = {errs.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
