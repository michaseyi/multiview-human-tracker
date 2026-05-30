"""Render a video showing all 4 camera views plus the per-frame best view.

Each panel shows the actor's keypoints, camera id, and score (n_visible,
mean_conf); the selected camera gets a [BEST] tag. Frame alignment uses the
event-anchored TimeSync model, and detections are filtered to the actor track
(dedupe + static-anchor) so the score reflects the actor, not the bystander.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from multiview_tracker.detection import (
    dedupe_per_frame_detections,
    load_pose_detections,
)
from multiview_tracker.sync import filter_to_actor
from multiview_tracker.visualization import (
    annotate_panel,
    compose_grid,
    index_by_frame,
    score_detections,
    select_best,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--stride", type=int, default=5,
                    help="render every Nth frame (1 = full rate)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many output frames")
    ap.add_argument("--panel-w", type=int, default=480)
    ap.add_argument("--panel-h", type=int, default=270)
    ap.add_argument("--output", default=None,
                    help="output mp4 (default: experiments/<exp>/best_view.mp4)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["experiment"]["output_dir"])
    out_path = Path(args.output) if args.output else out_dir / "best_view.mp4"

    cameras = [c["id"] for c in cfg["cameras"]]
    sources = {c["id"]: Path(c["source"]) for c in cfg["cameras"]}
    kp_dir = Path(cfg["detection"]["output_dir"])

    # prefer event-anchored TimeSync (affine in cam0 frame index);
    # fall back to constant integer offsets.json if not yet available
    primary = cameras[0]
    sync_path_event = out_dir / "offsets_event_anchored.json"
    sync_path_ncc = out_dir / "offsets.json"
    ts = None
    if sync_path_event.exists():
        from multiview_tracker.sync import TimeSync
        ts = TimeSync.from_json(sync_path_event, primary=primary)
        print(f"sync: event-anchored TimeSync from {sync_path_event.name}")
    elif sync_path_ncc.exists():
        const_offsets = {cid: int(v["tau"]) for cid, v in
                         json.loads(sync_path_ncc.read_text())["offsets"].items()}
        print(f"sync: constant tau from {sync_path_ncc.name}: {const_offsets}")
        ts = None
    else:
        const_offsets = {cid: 0 for cid in cameras}

    def support_frame(cid: str, ref_idx: int) -> int:
        if ts is not None:
            return ts.support_frame(cid, ref_idx)
        return ref_idx + const_offsets.get(cid, 0)

    caps = {cid: cv2.VideoCapture(str(sources[cid])) for cid in cameras}
    n_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())

    # filter to the single actor track per camera. dedupe collapses YOLO's
    # duplicate boxes; static_anchor labels each detection actor vs bystander;
    # filter_to_actor keeps only the actor.
    detections_by_frame = {}
    for cid in cameras:
        raw = load_pose_detections(kp_dir / f"{cid}.npz")
        dedup = dedupe_per_frame_detections(raw)
        actor = filter_to_actor(dedup, merge=True, static_anchor=True)
        detections_by_frame[cid] = index_by_frame(actor)
    print(f"actor-only detections: "
          f"{[len(detections_by_frame[c]) for c in cameras]} frames per cam")

    pw, ph = args.panel_w, args.panel_h
    out_w, out_h = 4 * pw, 2 * ph

    src_fps = caps[cameras[0]].get(cv2.CAP_PROP_FPS) or 24.55
    out_fps = src_fps / args.stride
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    print(f"output: {out_path} ({out_w}x{out_h} @ {out_fps:.2f} fps)")

    indices = list(range(0, n_frames, args.stride))
    if args.max_frames:
        indices = indices[: args.max_frames]

    pbar = tqdm(indices, desc="rendering", unit="frame")
    for ref_idx in pbar:
        panels: dict[str, "cv2.Mat"] = {}
        scores = {}
        for cid in cameras:
            target = support_frame(cid, ref_idx)
            cap = caps[cid]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if not ok:
                # fallback: black panel
                frame = 0 * (panels[cameras[0]] if panels else None)
            dets_here = detections_by_frame[cid].get(target, [])
            score = score_detections(dets_here)
            scores[cid] = score
            panels[cid] = annotate_panel(frame, dets_here, cid, score)

        best_cid = select_best(scores)
        # re-annotate the best panel so it carries the [BEST] tag
        cap = caps[best_cid]
        best_target = support_frame(best_cid, ref_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_target)
        ok, frame = cap.read()
        best_dets = detections_by_frame[best_cid].get(best_target, [])
        best_panel = annotate_panel(frame, best_dets, best_cid, scores[best_cid], is_best=True)

        composite = compose_grid(panels, best_panel, panel_size=(pw, ph))
        writer.write(composite)

    writer.release()
    for c in caps.values():
        c.release()
    print(f"\n[done] -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
