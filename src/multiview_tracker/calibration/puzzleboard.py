from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PuzzleboardConfig:
    rows: int
    cols: int
    square_size_m: float
    marker_bits: int = 4
    discard_edge_layers: int = 0  # drop outer N rings of corners


@dataclass
class PuzzleboardDetection:
    image_points: np.ndarray
    object_points: np.ndarray
    point_ids: np.ndarray
    image_size: tuple[int, int]


def detect_puzzleboard(
    image: np.ndarray,
    config: PuzzleboardConfig,
) -> PuzzleboardDetection | None:
    """Wrap Stelldinger's puzzleboard detector and return correspondences ready for cv2.calibrateCamera."""
    from puzzle_board.puzzle_board_detector import detect_puzzleboard as _detect

    ids, coords = _detect(image.copy())
    if ids is None or len(ids) == 0:
        return None

    ids_arr = np.asarray(ids, dtype=np.int32)
    coords_arr = np.asarray(coords, dtype=np.float32)

    # drop the outermost N rings of corners: the board is paper and held at
    # the edges, so outer corners bow off the Z=0 plane Zhang's method assumes.
    n_drop = max(0, int(config.discard_edge_layers))
    if n_drop > 0 and len(ids_arr) > 0:
        r_min, r_max = int(ids_arr[:, 0].min()), int(ids_arr[:, 0].max())
        c_min, c_max = int(ids_arr[:, 1].min()), int(ids_arr[:, 1].max())
        keep = (
            (ids_arr[:, 0] >= r_min + n_drop)
            & (ids_arr[:, 0] <= r_max - n_drop)
            & (ids_arr[:, 1] >= c_min + n_drop)
            & (ids_arr[:, 1] <= c_max - n_drop)
        )
        ids_arr = ids_arr[keep]
        coords_arr = coords_arr[keep]
        if len(ids_arr) == 0:
            return None

    # upstream uses (y, x) while opencv expects (x, y)
    image_points = coords_arr[:, ::-1].astype(np.float32)
    grid_xy = ids_arr[:, ::-1].astype(np.float32)

    object_points = np.zeros((len(ids_arr), 3), dtype=np.float32)
    object_points[:, :2] = grid_xy * float(config.square_size_m)

    h, w = image.shape[:2]
    return PuzzleboardDetection(
        image_points=image_points,
        object_points=object_points,
        point_ids=ids_arr,
        image_size=(w, h),
    )
