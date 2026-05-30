from __future__ import annotations

from pathlib import Path

import numpy as np

from multiview_tracker.calibration.monocular import CalibrationResult


def save_calibration(result: CalibrationResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        K=result.K,
        D=result.D,
        rms=np.float64(result.rms),
        image_size=np.array(result.image_size, dtype=np.int64),
        n_frames=np.int64(result.n_frames),
    )


def load_calibration(path: Path) -> CalibrationResult:
    z = np.load(path)
    return CalibrationResult(
        K=z["K"],
        D=z["D"],
        rms=float(z["rms"]),
        image_size=tuple(z["image_size"].tolist()),
        n_frames=int(z["n_frames"]),
    )


def bundle_calibrations(
    per_camera: dict[str, Path],
    out_path: Path,
    *,
    verbose: bool = True,
) -> None:
    """Bundle per-camera .npz files into a single calibration.npz."""
    bundle: dict[str, np.ndarray] = {}
    for cid, src in per_camera.items():
        r = load_calibration(src)
        bundle[f"{cid}_K"] = r.K
        bundle[f"{cid}_D"] = r.D
        bundle[f"{cid}_rms"] = np.float64(r.rms)
        bundle[f"{cid}_image_size"] = np.array(r.image_size, dtype=np.int64)
        if verbose:
            print(f"  {cid}: f_x={r.K[0,0]:.1f}  f_y={r.K[1,1]:.1f}  rms={r.rms:.3f} px")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **bundle)
