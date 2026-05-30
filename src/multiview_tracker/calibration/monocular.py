from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    K: np.ndarray
    D: np.ndarray
    rms: float
    image_size: tuple[int, int]
    n_frames: int


def per_frame_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    """Mean reprojection distance (in pixels) for one frame."""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    diff = projected.reshape(-1, 2) - image_points
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def calibrate_camera(
    image_points: list[np.ndarray],
    object_points: list[np.ndarray],
    image_size: tuple[int, int],
    outlier_factor: float = 2.0,
) -> CalibrationResult:
    """Two-pass cv2.calibrateCamera with median-based outlier rejection. Pass 2 drops frames whose error exceeds outlier_factor times the median, then refits. Set outlier_factor <= 0 to disable."""
    print(f"[calibrate] pass 1 on {len(image_points)} frames")
    rms1, K1, D1, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    print(f"[calibrate] pass 1 RMS = {rms1:.4f} px")

    if outlier_factor <= 0:
        return CalibrationResult(
            K=K1, D=D1, rms=float(rms1),
            image_size=image_size, n_frames=len(image_points),
        )

    errs = np.array([
        per_frame_error(o, i, r, t, K1, D1)
        for o, i, r, t in zip(object_points, image_points, rvecs, tvecs)
    ])
    threshold = outlier_factor * float(np.median(errs))
    keep = errs <= threshold
    n_drop = int((~keep).sum())
    print(
        f"[calibrate] errs: median={np.median(errs):.3f}  max={errs.max():.3f}  "
        f"threshold={threshold:.3f}  dropping {n_drop}"
    )

    if n_drop == 0:
        return CalibrationResult(
            K=K1, D=D1, rms=float(rms1),
            image_size=image_size, n_frames=len(image_points),
        )

    op = [o for o, k in zip(object_points, keep) if k]
    ip = [i for i, k in zip(image_points, keep) if k]
    print(f"[calibrate] pass 2 on {len(op)} frames")
    rms2, K2, D2, _, _ = cv2.calibrateCamera(op, ip, image_size, None, None)
    print(f"[calibrate] pass 2 RMS = {rms2:.4f} px")

    return CalibrationResult(
        K=K2, D=D2, rms=float(rms2),
        image_size=image_size, n_frames=len(op),
    )


def report(result: CalibrationResult) -> None:
    """Print K, D, RMS and sanity checks."""
    K, D = result.K, result.D.ravel()
    W, H = result.image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print()
    print("=" * 64)
    print(f"  Frames used:        {result.n_frames}")
    print(f"  Image size (W x H): {W} x {H}")
    print()
    print(f"  K = [[{fx:10.3f}  {0.0:10.3f}  {cx:10.3f}],")
    print(f"       [{0.0:10.3f}  {fy:10.3f}  {cy:10.3f}],")
    print(f"       [{0.0:10.3f}  {0.0:10.3f}  {1.0:10.3f}]]")
    print()
    print(
        f"  D = [k1={D[0]:+.5f}  k2={D[1]:+.5f}  "
        f"p1={D[2]:+.5f}  p2={D[3]:+.5f}  k3={D[4]:+.5f}]"
    )
    print()
    print(f"  RMS reprojection error: {result.rms:.4f} px")
    print()

    checks = {
        "f_y / f_x":    (f"{fy/fx:.5f}",                    abs(fy/fx - 1.0) < 0.02),
        "c_x near W/2": (f"{cx:.1f} (W/2={W/2:.1f})",       abs(cx - W/2) < W * 0.10),
        "c_y near H/2": (f"{cy:.1f} (H/2={H/2:.1f})",       abs(cy - H/2) < H * 0.10),
        "f_x in range": (f"{fx:.1f}",                       1200 <= fx <= 2400),
        "RMS < 1 px":   (f"{result.rms:.4f}",               result.rms < 1.0),
    }
    print("  Sanity checks:")
    for name, (val, ok) in checks.items():
        print(f"    [{'PASS' if ok else 'WARN'}]  {name:14s} = {val}")
    print("=" * 64, "\n")
