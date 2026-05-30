"""Plot the per-camera motion signals and pairwise xcorr curves used for sync."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

from multiview_tracker.detection import load_pose_detections
from multiview_tracker.sync import build_motion_signal, filter_to_actor, normalised_xcorr

cfg = yaml.safe_load(Path("configs/default.yaml").read_text())
cameras = [c["id"] for c in cfg["cameras"]]
src = {c["id"]: Path(c["source"]) for c in cfg["cameras"]}
kp_dir = Path(cfg["detection"]["output_dir"])
out_dir = Path(cfg["experiment"]["output_dir"])

# longest video sets the signal length
n_frames = 0
for cid in cameras:
    cap = cv2.VideoCapture(str(src[cid]))
    n_frames = max(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()

signals: dict[str, np.ndarray] = {}
for cid in cameras:
    actor = filter_to_actor(load_pose_detections(kp_dir / f"{cid}.npz"))
    signals[cid] = build_motion_signal(actor, n_frames)

# signals on top, xcorr curves underneath
fig, axes = plt.subplots(2, 1, figsize=(13, 7))

ax = axes[0]
for cid in cameras:
    ax.plot(signals[cid], label=cid, alpha=0.7, linewidth=0.6)
ax.set_xlim(0, 2000)  # first ~80 s
ax.set_xlabel("frame")
ax.set_ylabel("centroid-Y velocity (px / frame)")
ax.set_title("Actor motion signals (first 2000 frames)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
ref = signals["cam0"]
max_lag = 30
for cid in cameras[1:]:
    est = normalised_xcorr(ref, signals[cid], max_lag=max_lag)
    ax.plot(est.lags, est.curve, marker=".", label=f"cam0 vs {cid}: argmax={est.tau:+d}, peak={est.peak_value:.3f}")
ax.axvline(0, color="grey", alpha=0.4, linestyle="--")
ax.set_xlabel("lag tau (frames)")
ax.set_ylabel("normalised cross-correlation")
ax.set_title("Cross-correlation curves vs cam0")
ax.legend(loc="lower center")
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = out_dir / "offset_diagnostics.png"
plt.savefig(out_path, dpi=120)
print(f"[save] -> {out_path}")
