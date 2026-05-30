"""Diagnostic video: per camera, run the puzzleboard detector at every sampled
frame and overlay what it finds.

Per frame: detected corners are drawn as dots coloured by a deterministic hash
of (row, col) so the same physical corner is the same colour across frames.
Frames with no detection are written as-is with a red NO DETECTION banner so
the timeline stays continuous.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from multiview_tracker.calibration import PuzzleboardConfig, detect_puzzleboard


def color_for_id(row: int, col: int) -> tuple[int, int, int]:
    """Deterministic BGR colour per (row, col); hashes hue across the wheel."""
    h = (row * 53 + col * 97) % 180  # OpenCV HSV hue range
    hsv = np.uint8([[[h, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_one(
    cid: str,
    src: Path,
    pb_cfg: PuzzleboardConfig,
    out_path: Path,
    stride: int,
    panel_w: int,
    start: int,
    end: int | None,
) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  {cid}: cannot open {src}"); return
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.55
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel_h = int(src_h * panel_w / src_w)
    out_fps = max(2.0, src_fps / stride)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (panel_w, panel_h))
    if not writer.isOpened():
        print(f"  {cid}: avc1 unavailable, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (panel_w, panel_h))

    n_drawn = 0
    n_detected = 0
    sum_corners = 0
    end_frame = end if end is not None else n_total
    indices = list(range(start, end_frame, stride))

    for n_a in tqdm(indices, desc=cid, unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_a)
        ok, img = cap.read()
        if not ok:
            continue
        n_drawn += 1
        det = detect_puzzleboard(img, pb_cfg)
        if det is not None and len(det.point_ids) > 0:
            n_detected += 1
            sum_corners += len(det.point_ids)
            for (row, col), (x, y) in zip(det.point_ids, det.image_points):
                col_bgr = color_for_id(int(row), int(col))
                cv2.circle(img, (int(x), int(y)), 4, col_bgr, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), 1, cv2.LINE_AA)
            header_top = f"{cid}  f={n_a}    corners detected = {len(det.point_ids)}"
        else:
            # red NO-DETECTION banner in the bottom-left
            txt = "NO DETECTION"
            scale = max(2.0, img.shape[1] / 700)
            thick = max(3, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv2.rectangle(img, (20, img.shape[0] - 40 - th - 20),
                          (40 + tw, img.shape[0] - 20), (0, 0, 100), -1)
            cv2.putText(img, txt, (30, img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 40, 255),
                        thick, cv2.LINE_AA)
            header_top = f"{cid}  f={n_a}    NO BOARD DETECTED"

        # header: black bar with white text
        header_bot = f"colour = hash(row, col); same colour means same physical corner"
        pad = 14
        h_w = img.shape[1]
        s = max(1.0, h_w / 1500)
        th_px = max(2, int(s * 2))
        line_h = int(36 * s)
        rect_h = pad * 2 + line_h * 2
        widths = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th_px)[0][0]
                  for t in (header_top, header_bot)]
        cv2.rectangle(img, (0, 0), (max(widths) + 2 * pad, rect_h), (0, 0, 0), -1)
        for i, t in enumerate((header_top, header_bot)):
            cv2.putText(img, t, (pad, pad + line_h * (i + 1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), th_px, cv2.LINE_AA)

        writer.write(cv2.resize(img, (panel_w, panel_h)))

    cap.release()
    writer.release()
    avg_corners = sum_corners / max(1, n_detected)
    print(f"  {cid}: frames sampled={n_drawn}  detected={n_detected} "
          f"({100*n_detected/max(1,n_drawn):.0f}%)  mean corners/detected={avg_corners:.1f}")
    print(f"  -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="which cameras to render (default: all)")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--panel-w", type=int, default=1600)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])

    pb = cfg["calibration"]["puzzleboard"]
    pb_cfg = PuzzleboardConfig(
        rows=pb["rows"], cols=pb["cols"],
        square_size_m=pb["square_size_m"], marker_bits=pb["marker_bits"],
        discard_edge_layers=0,  # show every corner the detector finds
    )

    cams = {c["id"]: c for c in cfg["cameras"]}
    cameras = args.cameras or [c["id"] for c in cfg["cameras"]]
    out_root = out_dir / "board_detect_video"
    out_root.mkdir(parents=True, exist_ok=True)

    for cid in cameras:
        if cid not in cams:
            print(f"skip unknown camera {cid!r}"); continue
        out_path = out_root / f"{cid}_board_detect.mp4"
        render_one(
            cid=cid,
            src=Path(cams[cid]["source"]),
            pb_cfg=pb_cfg,
            out_path=out_path,
            stride=args.stride,
            panel_w=args.panel_w,
            start=args.start,
            end=args.end,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
