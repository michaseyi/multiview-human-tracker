from __future__ import annotations

import cv2
import numpy as np


def line_image_endpoints(
    line: np.ndarray, w: int, h: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip homogeneous line (a, b, c) to [0, w] x [0, h], or None if it misses the image."""
    a, b, c = float(line[0]), float(line[1]), float(line[2])
    pts: list[tuple[float, float]] = []
    if abs(b) > 1e-10:
        for x in (0.0, float(w)):
            y = -(a * x + c) / b
            if -1 <= y <= h + 1:
                pts.append((x, y))
    if abs(a) > 1e-10:
        for y in (0.0, float(h)):
            x = -(b * y + c) / a
            if -1 <= x <= w + 1:
                pts.append((x, y))
    if len(pts) < 2:
        return None
    # two farthest unique endpoints make the cleanest segment
    pts = sorted(set((round(p[0], 1), round(p[1], 1)) for p in pts))
    return pts[0], pts[-1]


def make_palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct BGR colours via an HSV hue sweep."""
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = int(i * 180 / max(n, 1))
        bgr = cv2.cvtColor(np.uint8([[[hue, 230, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
        out.append(tuple(int(c) for c in bgr))
    return out


def draw_points_and_epilines(
    image_a: np.ndarray,
    image_b: np.ndarray,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    F: np.ndarray,
    *,
    point_radius: int = 6,
    line_thickness: int = 2,
    palette: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw each pts_a[i] on image_a, its epipolar line F x_a on image_b, and outline pts_b[i] on image_b."""
    out_a, out_b = image_a.copy(), image_b.copy()
    n = len(pts_a)
    colors = palette or make_palette(n)
    h, w = out_b.shape[:2]
    for i in range(n):
        col = colors[i]
        x_a = np.array([pts_a[i, 0], pts_a[i, 1], 1.0])
        line = F @ x_a
        cv2.circle(out_a,
                   (int(pts_a[i, 0]), int(pts_a[i, 1])),
                   point_radius, col, -1, cv2.LINE_AA)
        ends = line_image_endpoints(line, w, h)
        if ends is not None:
            p, q = ends
            cv2.line(out_b,
                     (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     col, line_thickness, cv2.LINE_AA)
        cv2.circle(out_b,
                   (int(pts_b[i, 0]), int(pts_b[i, 1])),
                   point_radius, col, 2, cv2.LINE_AA)
    return out_a, out_b


def label_panel(image: np.ndarray, text: str, *, color=(0, 255, 255)) -> np.ndarray:
    """Return a copy of image with a text header painted on top."""
    out = image.copy()
    pad = 12
    scale = max(1.5, image.shape[1] / 1100)
    thick = max(2, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return out


def stack_compare(rows: list[tuple[str, np.ndarray, np.ndarray]],
                  panel_w: int) -> np.ndarray:
    """Stack (label, image_a, image_b) rows vertically; each image resized to panel_w.

    image_a sits left of image_b; label is painted on the top-left of image_b.
    """
    out_rows: list[np.ndarray] = []
    for label, image_a, image_b in rows:
        ratio = panel_w / image_a.shape[1]
        ph = int(image_a.shape[0] * ratio)
        a = cv2.resize(image_a, (panel_w, ph))
        b = cv2.resize(image_b, (panel_w, ph))
        b = label_panel(b, label)
        out_rows.append(np.hstack([a, b]))
    return np.vstack(out_rows)
