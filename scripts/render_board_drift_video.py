"""Render a video showing where epipolar geometry predicts board corners in cam0
versus where cam0 actually detects them.

Per sampled cam0 frame, the cam0-detected corner is a filled dot, each support's
epipolar line is drawn, and the LS intersection is an open ring. With perfect
sync the rings overlap the dots; with sync drift the ring lags or leads when
the board moves and rejoins on static frames.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from multiview_tracker.calibration import PuzzleboardConfig, detect_puzzleboard
from multiview_tracker.epipolar import line_image_endpoints, make_palette
from multiview_tracker.recovery import (
    epipolar_line_in_a,
    least_squares_intersection,
)


# source FPS per camera, read from each video's metadata; used for FPS-drift correction
DEFAULT_FPS = {"cam0": 24.548, "cam1": 24.524, "cam2": 24.442, "cam3": 24.296}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--primary", default="cam0")
    ap.add_argument("--support", nargs="+", default=["cam1", "cam2", "cam3"])
    ap.add_argument("--method", default="c1", choices=["c1", "c2", "c3"])
    ap.add_argument("--stride", type=int, default=10,
                    help="sample every Nth cam0 frame (default 10)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--n-corners-max", type=int, default=20,
                    help="cap on corners drawn per frame (sample evenly across the board)")
    ap.add_argument("--panel-w", type=int, default=1600,
                    help="output video width (height scales to preserve aspect)")
    ap.add_argument("--apply-offsets",
                    choices=["none", "saved", "fps_drift"],
                    default="none",
                    help="none: support_idx = cam0_idx. "
                         "saved: apply integer tau from offsets.json. "
                         "fps_drift: support_idx = round(cam0_idx * fps_support/fps_primary).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    cid_a = args.primary
    sids = args.support

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
        discard_edge_layers=0,
    )

    f_key = {"c1": "F_c1", "c2": "F_c2", "c3": "F_c3"}[args.method]
    F = {sid: np.load(out_dir / f"F_{cid_a}_{sid}.npz")[f_key] for sid in sids}

    cams = {c["id"]: c for c in cfg["cameras"]}

    # offset function: cam0 frame N -> support frame.
    # read FPS directly from each video; the yaml has a single nominal 24.55
    # for all cameras but the clocks genuinely differ (cam0 24.548, cam1 24.524,
    # cam2 24.442, cam3 24.296). using the yaml made fps_drift a silent no-op.
    def video_fps(cid):
        c = cv2.VideoCapture(str(cams[cid]["source"]))
        v = c.get(cv2.CAP_PROP_FPS) or 24.55
        c.release()
        return float(v)

    fps_a = video_fps(cid_a)
    fps_supports = {sid: video_fps(sid) for sid in sids}
    print(f"FPS (from video metadata): {cid_a}={fps_a:.4f}  " +
          "  ".join(f"{sid}={fps_supports[sid]:.4f}" for sid in sids))

    def support_index(sid, n_a):
        if args.apply_offsets == "none":
            return n_a
        if args.apply_offsets == "saved":
            offsets_path = out_dir / "offsets.json"
            if not hasattr(support_index, "_tau"):
                tau = {sid: 0 for sid in sids}
                if offsets_path.exists():
                    j = json.loads(offsets_path.read_text())["offsets"]
                    tau = {sid: int(j[sid]["tau"]) - int(j[cid_a]["tau"]) for sid in sids}
                support_index._tau = tau  # type: ignore[attr-defined]
            return n_a + support_index._tau[sid]
        if args.apply_offsets == "fps_drift":
            return int(round(n_a * fps_supports[sid] / fps_a))
        return n_a

    caps = {cid: cv2.VideoCapture(str(cams[cid]["source"])) for cid in [cid_a] + sids}
    n_total = int(caps[cid_a].get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(caps[cid_a].get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(caps[cid_a].get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_w = args.panel_w
    panel_h = int(src_h * panel_w / src_w)
    out_fps = max(2.0, fps_a / args.stride)

    end_frame = args.end if args.end is not None else n_total
    indices = list(range(args.start, end_frame, args.stride))
    print(f"primary={cid_a}, supports={'+'.join(sids)}, method={args.method.upper()}")
    print(f"stride {args.stride} -> {len(indices)} output frames, video fps {out_fps:.2f}")
    print(f"apply_offsets = {args.apply_offsets}")

    out_root = out_dir / "board_drift_video"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{cid_a}_from_{'_'.join(sids)}_{args.method}_offsets-{args.apply_offsets}.mp4"
    # avc1 is OpenCV's tag for H.264, which QuickTime decodes natively.
    # mp4v (MPEG-4 Part 2) produces files QuickTime renders as blank/green frames.
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (panel_w, panel_h))
    if not writer.isOpened():
        # avc1 unavailable; fall back to mp4v
        print("WARN: avc1 codec not available; falling back to mp4v "
              "(QuickTime may render blank frames, convert with ffmpeg if so)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (panel_w, panel_h))

    n_drawn = 0
    n_static = 0
    sum_med_err = 0.0

    for n_a in tqdm(indices, desc="render", unit="frame"):
        caps[cid_a].set(cv2.CAP_PROP_POS_FRAMES, n_a)
        ok, img_a = caps[cid_a].read()
        if not ok:
            continue

        # detect board in cam0
        det_a = detect_puzzleboard(img_a, pb_cfg)
        if det_a is None:
            # write the raw frame so the timeline stays continuous
            _annotate_header(img_a, f"{cid_a} f={n_a}  cam0: NO BOARD",
                             f"offsets={args.apply_offsets}")
            writer.write(cv2.resize(img_a, (panel_w, panel_h)))
            continue

        # cam0 corner lookup
        lk_a = {tuple(r): det_a.image_points[i] for i, r in enumerate(det_a.point_ids)}

        # detect in supports at offset-corrected frames
        sup_dets = {}
        sup_frames_used = {}
        for sid in sids:
            n_s = support_index(sid, n_a)
            if n_s < 0 or n_s >= int(caps[sid].get(cv2.CAP_PROP_FRAME_COUNT)):
                continue
            caps[sid].set(cv2.CAP_PROP_POS_FRAMES, n_s)
            ok_s, img_s = caps[sid].read()
            if not ok_s:
                continue
            d_s = detect_puzzleboard(img_s, pb_cfg)
            sup_frames_used[sid] = n_s
            if d_s is None:
                continue
            sup_dets[sid] = {tuple(r): d_s.image_points[i] for i, r in enumerate(d_s.point_ids)}

        if not sup_dets:
            _annotate_header(img_a, f"{cid_a} f={n_a}  no supports w/ board",
                             f"offsets={args.apply_offsets}")
            writer.write(cv2.resize(img_a, (panel_w, panel_h)))
            continue

        # common corners: visible in cam0 and in at least 2 supports
        # (otherwise the LS intersection is rank-deficient)
        common = []
        for cid_kv in lk_a.keys():
            n_sup = sum(1 for sid in sup_dets if cid_kv in sup_dets[sid])
            if n_sup >= 2:
                common.append(cid_kv)
        if not common:
            _annotate_header(img_a, f"{cid_a} f={n_a}  no corners in cam0 + 2 supports",
                             f"offsets={args.apply_offsets}")
            writer.write(cv2.resize(img_a, (panel_w, panel_h)))
            continue

        # cap to n_corners_max sampled evenly
        common.sort()
        if len(common) > args.n_corners_max:
            step = max(1, len(common) // args.n_corners_max)
            common = common[::step][:args.n_corners_max]
        palette = make_palette(len(common))

        # draw
        h, w = img_a.shape[:2]
        errs = []
        for slot, cid_kv in enumerate(common):
            col = palette[slot]
            truth = lk_a[cid_kv]
            lines = []
            line_shades = []
            for shade_idx, sid in enumerate(sids):
                if sid not in sup_dets or cid_kv not in sup_dets[sid]:
                    continue
                xb = sup_dets[sid][cid_kv]
                lines.append(epipolar_line_in_a(F[sid], xb))
                # shade per support: 1.0 / 0.75 / 0.5
                factor = (1.0, 0.75, 0.5)[shade_idx]
                line_shades.append(tuple(min(255, int(c * factor)) for c in col))
            for line, lc in zip(lines, line_shades):
                ends = line_image_endpoints(line, w, h)
                if ends is None:
                    continue
                p, q = ends
                cv2.line(img_a, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                         lc, 1, cv2.LINE_AA)
            ls_pt = least_squares_intersection(np.stack(lines))
            err = float(np.linalg.norm(ls_pt - truth)) if not np.isnan(ls_pt).any() else float("inf")
            errs.append(err)
            if not np.isnan(ls_pt).any() and 0 <= ls_pt[0] < w and 0 <= ls_pt[1] < h:
                cv2.circle(img_a, (int(ls_pt[0]), int(ls_pt[1])), 8, col, 2, cv2.LINE_AA)
            cv2.circle(img_a, (int(truth[0]), int(truth[1])), 5, col, -1, cv2.LINE_AA)
            cv2.circle(img_a, (int(truth[0]), int(truth[1])), 5, (0, 0, 0), 1, cv2.LINE_AA)

        finite = [e for e in errs if np.isfinite(e)]
        med = float(np.median(finite)) if finite else float("nan")
        n_drawn += 1
        if finite:
            sum_med_err += med
            if med < 5.0:
                n_static += 1

        sup_info = ", ".join(f"{sid}@{sup_frames_used.get(sid, '-')}" for sid in sids)
        _annotate_header(
            img_a,
            f"{cid_a} f={n_a}    median ring->dot = {med:.1f} px    corners drawn = {len(common)}",
            f"offsets={args.apply_offsets}    supports: {sup_info}",
        )

        writer.write(cv2.resize(img_a, (panel_w, panel_h)))

    for c in caps.values():
        c.release()
    writer.release()

    avg_med = sum_med_err / max(1, n_drawn)
    print()
    print(f"wrote {out_path}")
    print(f"  frames drawn (board in cam0 + 2 supports) : {n_drawn} / {len(indices)}")
    print(f"  frames with median err < 5 px (~static)   : {n_static}")
    print(f"  mean-of-frame-medians                     : {avg_med:.1f} px")
    return 0


def _annotate_header(image: np.ndarray, line1: str, line2: str) -> None:
    pad = 14
    h, w = image.shape[:2]
    scale = max(1.0, w / 1500)
    thick = max(2, int(scale * 2))
    line_h = int(34 * scale)
    rect_h = pad * 2 + line_h * 2
    max_w = max(
        cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        for t in [line1, line2]
    )
    cv2.rectangle(image, (0, 0), (max_w + 2 * pad, rect_h), (0, 0, 0), -1)
    for i, t in enumerate([line1, line2]):
        cv2.putText(image, t, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


if __name__ == "__main__":
    sys.exit(main())
