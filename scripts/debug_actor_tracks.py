"""Render each camera's video with tracked persons overlaid to verify which one `filter_to_actor` chose.

Actor track = green skeleton + 'A' tag; other tracks = red skeleton + 'B' tag.
Header shows camera id, frame index, and current actor track id.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.sync.actor import (
    assign_tracks,
    cluster_tracks_to_k,
    high_conf_centroid,
    merge_tracklets,
    select_actor_track,
    static_anchor_classify,
)
from multiview_tracker.visualization import draw_pose

ACTOR_BONE = (60, 220, 60)
ACTOR_PT = (40, 255, 40)
OTHER_BONE = (60, 60, 220)
OTHER_PT = (40, 40, 255)


def render_one_camera(
    cid: str,
    video_path: Path,
    detections,
    out_path: Path,
    stride: int,
    panel_w: int,
    merge_tracklets_flag: bool = False,
    merge_max_gap: int = 5000,
    merge_max_extrap: float = 300.0,
    force_cap_to_two: bool = False,
    static_anchor: bool = False,
    static_anchor_home_radius: float = 150.0,
):
    track_ids = assign_tracks(detections)
    n_tracks_before = len({t for t in track_ids if t >= 0})
    if merge_tracklets_flag:
        track_ids = merge_tracklets(
            detections, track_ids,
            max_gap_frames=merge_max_gap,
            max_extrap_dist_px=merge_max_extrap,
        )
        n_tracks_after = len({t for t in track_ids if t >= 0})
        print(f"  {cid}: merge {n_tracks_before} tracks -> {n_tracks_after}")
        n_tracks_before = n_tracks_after
    if static_anchor:
        track_ids = static_anchor_classify(
            detections, track_ids,
            home_radius_px=static_anchor_home_radius,
        )
        n_tracks_after = len({t for t in track_ids if t >= 0})
        print(f"  {cid}: static-anchor {n_tracks_before} tracks -> {n_tracks_after}")
        actor_tid = 1  # static_anchor_classify convention
    elif force_cap_to_two:
        track_ids = cluster_tracks_to_k(detections, track_ids, k=2)
        n_tracks_after = len({t for t in track_ids if t >= 0})
        print(f"  {cid}: K=2 cluster {n_tracks_before} tracks -> {n_tracks_after}")
        actor_tid = select_actor_track(detections, track_ids)
    else:
        actor_tid = select_actor_track(detections, track_ids)
    n_actor = sum(1 for t in track_ids if t == actor_tid)
    print(f"  {cid}: actor track id = {actor_tid}, {n_actor} detections labelled actor")

    by_frame: dict[int, list[tuple]] = {}
    for det, tid in zip(detections, track_ids):
        by_frame.setdefault(det.frame_idx, []).append((det, tid))

    cap = cv2.VideoCapture(str(video_path))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.55
    out_fps = src_fps / stride
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = int(src_h * panel_w / src_w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (panel_w, out_h))

    indices = range(0, n_total, stride)
    for idx in tqdm(indices, desc=cid, unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        for det, tid in by_frame.get(idx, []):
            is_actor = tid == actor_tid
            draw_pose(
                frame, det,
                point_color=ACTOR_PT if is_actor else OTHER_PT,
                bone_color=ACTOR_BONE if is_actor else OTHER_BONE,
                point_radius=6, bone_thickness=3,
            )
            centroid = high_conf_centroid(det)
            if centroid is not None:
                tag = "A" if is_actor else f"B(t={tid})"
                cv2.putText(
                    frame, tag,
                    (int(centroid[0]) + 8, int(centroid[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    ACTOR_PT if is_actor else OTHER_PT, 3, cv2.LINE_AA,
                )

        header = f"{cid}  frame={idx}  actor_track={actor_tid}  (green=actor, red=other)"
        pad = 14
        scale = max(1.5, frame.shape[1] / 1100)
        thick = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
        cv2.putText(frame, header, (pad, th + pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

        writer.write(cv2.resize(frame, (panel_w, out_h)))

    cap.release()
    writer.release()
    print(f"  -> {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stride", type=int, default=15,
                    help="render every Nth source frame")
    ap.add_argument("--panel-w", type=int, default=1280)
    ap.add_argument("--cameras", nargs="+", default=None,
                    help="cameras to render; default is all")
    ap.add_argument("--dedupe", action="store_true",
                    help="dedupe duplicate YOLO detections per frame and cap to 2 persons")
    ap.add_argument("--dedupe-threshold-px", type=float, default=50.0,
                    help="mean per-keypoint distance below which two detections are duplicates")
    ap.add_argument("--merge-tracklets", action="store_true",
                    help="velocity-based merging of fragmented tracks before actor selection")
    ap.add_argument("--merge-max-gap-frames", type=int, default=5000,
                    help="max temporal gap when linking tracklets")
    ap.add_argument("--merge-max-extrap-dist-px", type=float, default=300.0,
                    help="max extrapolation distance when linking")
    ap.add_argument("--force-cap-to-2", action="store_true",
                    help="after merge, K-means cluster the surviving tracks into 2 buckets")
    ap.add_argument("--static-anchor", action="store_true",
                    help="static anchor classification: lowest-variance long tracklet defines a "
                         "home zone for the bystander; everything outside is the actor")
    ap.add_argument("--static-anchor-home-radius", type=float, default=150.0,
                    help="radius of the bystander home zone in px")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    kp_dir = Path(cfg["detection"]["output_dir"])
    out_dir = Path(cfg["experiment"]["output_dir"]) / "debug_actor_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = args.cameras or [c["id"] for c in cfg["cameras"]]
    cam_lookup = {c["id"]: c for c in cfg["cameras"]}

    for cid in cameras:
        if cid not in cam_lookup:
            print(f"WARNING: skipping unknown camera {cid!r}")
            continue
        detections = load_pose_detections(kp_dir / f"{cid}.npz")
        if args.dedupe:
            before = len(detections)
            detections = dedupe_per_frame_detections(
                detections, similarity_threshold_px=args.dedupe_threshold_px,
            )
            print(f"  {cid}: dedupe {before} -> {len(detections)} detections")
        parts = []
        if args.dedupe: parts.append("dedupe")
        if args.merge_tracklets: parts.append("merge")
        if args.force_cap_to_2: parts.append("cap2")
        if args.static_anchor: parts.append("anchor")
        suffix = ("_" + "_".join(parts)) if parts else ""
        render_one_camera(
            cid=cid,
            video_path=Path(cam_lookup[cid]["source"]),
            detections=detections,
            out_path=out_dir / f"{cid}_actor_track{suffix}.mp4",
            stride=args.stride,
            panel_w=args.panel_w,
            merge_tracklets_flag=args.merge_tracklets,
            merge_max_gap=args.merge_max_gap_frames,
            merge_max_extrap=args.merge_max_extrap_dist_px,
            force_cap_to_two=args.force_cap_to_2,
            static_anchor=args.static_anchor,
            static_anchor_home_radius=args.static_anchor_home_radius,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
