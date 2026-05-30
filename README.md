# Multi-View Human Tracking with Puzzleboard Calibration

Four-camera pipeline that calibrates each camera with a Puzzleboard target,
runs YOLOv8-pose on every stream, fits a fundamental matrix between camera
pairs (three independent estimators), and recovers missing keypoints in one
view from epipolar intersections with the others. The full writeup is at
`report/main.pdf`.

## Setup

```bash
python -m pip install -e ".[detect,dev]"
```

`yolov8s-pose.pt` (at the project root) is the cached pose model; remove it
to force re-download on first run.

## Reproduce the canonical pipeline

YOLO detection runs once on raw video and never depends on calibration,
sync, or `F`. Everything downstream is rebuilt from those cached
detections. Start from a clean `experiments/default/` (keypoints copied
back from any archive) and run:

```bash
# 1. Per-camera intrinsic calibration (with edge-corner discard from config)
for cam in cam0 cam1 cam2 cam3; do
  python scripts/calibrate_camera.py --camera $cam
done

# 2. YOLO pose detection (only if you do not have keypoints/*.npz already)
for cam in cam0 cam1 cam2 cam3; do
  python scripts/detect_keypoints.py --camera $cam
done

# 3. Event-anchored temporal sync (pass two events identified manually)
python scripts/compute_event_sync.py \
  --event1 cam0:330 cam1:341 cam2:335 cam3:336 \
  --event2 cam0:7714 cam1:7712 cam2:7670 cam3:7607

# 4. F per support pair, harvested at sync-corrected frames and refit in
#    undistorted pixel space
python scripts/refit_fundamental_synced.py

# 5. Sample annotated keypoint frame per camera
python scripts/sample_keypoint_frames.py

# 6. Best-view composite (actor-only scoring)
python scripts/render_best_view.py

# 7. Epipolar-line validation (three F variants drawn side by side per frame)
python scripts/visualize_epipolar.py --pair cam0,cam2

# 8. Board-corner recovery (validates the recovery algorithm on
#    correspondences that are unambiguous by construction)
python scripts/recover_missing_board.py --frame 4450 \
  --support cam1 cam2 cam3 --n-samples 30

# 9. Actor-keypoint recovery (canonical configuration: 2 supports, undistorted
#    pipeline, event-anchored sync)
python scripts/recover_missing_keypoint.py \
  --primary cam0 --support cam1 cam2 --n-frames 12
```

Phase 1 is the longest, around 7 minutes wall-clock with four parallel calls.
Phase 2 takes about 2 hours per camera on MPS but can be skipped if YOLO
keypoints are already cached. Everything else finishes in about 15 minutes
total.

## Outputs

```
experiments/default/
  cam{0..3}_calibration.npz       K, D, RMS, image_size, n_frames
  offsets_event_anchored.json     two-event affine TimeSync model
  keypoints/cam{0..3}.npz         YOLO detections per camera
  keypoint_samples/cam*_kp_sample.jpg
                                  one annotated frame per camera
  F_cam0_cam{1,2,3}_synced_undist.npz
                                  F (three estimators), the points used in
                                  the fit, K and D for both cameras
  best_view.mp4                   4-up composite with the chosen panel tagged
  epipolar_vis/                   D1 figures, one per chosen frame
  recovery/                       E1 board-corner and actor-keypoint visuals
```

## Deliverables map

Every required artefact from the assignment, with its location on disk and the
report section that discusses it. Numbers in the right column refer to
sections in `report/main.pdf`.

| Task | Deliverable | File on disk | Report section |
|------|-------------|--------------|----------------|
| A1 | `K`, `D` per camera | `experiments/default/cam{0..3}_calibration.npz` | §3 (tables of intrinsics and distortion) |
| A1 | Example calibration frames | `data/calibration/cam{0..3}/frame_*.jpg` (about 80 frames per camera) | §3, Figure 1 |
| A2 | Chosen detector | `yolov8s-pose.pt` (cached weights, downloaded by Ultralytics on first use) | §5 (configuration and per-camera detection stats in Table 7) |
| A2 | Example frames with keypoints drawn | `experiments/default/keypoint_samples/cam{0..3}_kp_sample.jpg` | §5, Figure 3 |
| A3 | Estimated temporal offsets | `experiments/default/offsets_event_anchored.json` (alpha, beta per camera; events used; metadata FPS for diagnostic comparison) | §4 (Tables 4 and 5 fit + Table 6 predicted offsets at six representative cam0 frames) |
| B1 | Best-view visualisation | `experiments/default/best_view.mp4` (4-up grid + selected panel; per-panel `cam<i> n=<#kp> conf=<mean>` text overlay; `[BEST]` tag on chosen panel) | §6, Figure 4 |
| C1 | Fundamental matrix via `cv2.findFundamentalMat` | `F_c1` key inside `experiments/default/F_cam0_cam{1,2,3}_synced_undist.npz` | §7 (Direct via OpenCV) |
| C2 | Fundamental matrix via the essential matrix | `F_c2` key inside the same npz files; the essential matrix is also saved under the `E` key | §7 (Via the essential matrix) |
| C3 | Custom implementation | `F_c3` key inside the same npz files; the solver is `src/multiview_tracker/epipolar/manual_solver.py` | §7 (Manual normalised eight-point algorithm) plus the verification paragraph confirming agreement with `cv2.FM_8POINT` to 1e-10 |
| C3 | Short explanation of custom method | (writeup only) | §7 (Hartley pre-conditioning, linear system, SVD, rank-2 enforcement, denormalisation) |
| D1 | Epipolar-line visualisations | `experiments/default/epipolar_vis/f{000000,001750,010672}_cam0_cam2.jpg` (each shows all three F matrices side by side, on board and actor correspondences) | §8, Figures 5 (Puzzleboard) and 6 (actor keypoints) |
| D1 | Short discussion + which method worked best | (writeup only) | §8.2 (the via-E estimator wins on variance, the manual eight-point wins on the cleanest training frames, all three fail uniformly on cross-view actor correspondences for the reasons unpacked in §9 and §10) |
| E1 | Recovery method | `scripts/recover_missing_keypoint.py` (actor), `scripts/recover_missing_board.py` (board sanity check), `src/multiview_tracker/recovery/intersect.py` (algorithm) | §9 (algorithm) |
| E1 | Visual examples | `experiments/default/recovery/recovery_board_f004450_*.jpg` (12-corner board sanity check at a static frame, 0.66 px median) and `recovery_f{000000..010672}_c1_cam1_cam2_undistorted.jpg` (12 evenly-spaced actor recoveries) | §9.2 (board, Figure 7), §9.3 (actor) |
| E1 | Short explanation of what happens when lines do not intersect cleanly | (writeup only) | §9.4 (near-parallel rejection threshold, fallback to least-squares; rolling-shutter shear as the residual contributor) |
| Submission | Final report | `report/main.pdf` | n/a |

## Layout

```
configs/                 YAML config (cameras, board geometry, harvest params)
scripts/                 CLI entry points (one per stage)
src/multiview_tracker/
  calibration/           Puzzleboard detector wrapper + Zhang fit
  detection/             YOLO pose runner + dedupe + storage
  sync/                  actor track filter + TimeSync (event-anchored)
  epipolar/              correspondence harvest, three F solvers, undistort
  recovery/              cross-product and LS epipolar intersection
  visualization/         panel/keypoint draw helpers and best-view scoring
data/raw/                source videos (gitignored)
data/calibration/        per-camera harvested calibration frames (gitignored)
experiments/             pipeline outputs (gitignored)
report/                  LaTeX source and the compiled PDF
```
