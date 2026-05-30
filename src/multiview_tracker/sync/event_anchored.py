"""Event-anchored temporal synchronisation.

Camera clocks drift, so the offset between cam0 and each support grows
linearly through the recording. The model is affine per support:

    support_frame = round(alpha + beta * primary_frame)

Two manually-identified events per camera fix alpha and beta by linear
interpolation. This module exists so downstream scripts share one model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AffineModel:
    alpha: float   # intercept c
    beta: float    # slope m


class TimeSync:
    """Two-event affine sync model for one primary camera and N supports. Construct from per-camera (early_frame, late_frame) tuples or from the JSON written by scripts/compute_event_sync.py."""

    def __init__(self, events: dict[str, tuple[int, int]], primary: str = "cam0") -> None:
        if primary not in events:
            raise ValueError(f"primary camera {primary!r} not in events")
        self._primary = primary
        self._events = {cid: (int(e), int(l)) for cid, (e, l) in events.items()}
        N1, N2 = self._events[primary]
        if N2 == N1:
            raise ValueError("primary early and late event are the same frame")
        dx = N2 - N1
        self._models: dict[str, AffineModel] = {}
        for cid, (M1, M2) in self._events.items():
            if cid == primary:
                self._models[cid] = AffineModel(alpha=0.0, beta=1.0)
                continue
            beta = (M2 - M1) / dx
            alpha = M1 - beta * N1
            self._models[cid] = AffineModel(alpha=float(alpha), beta=float(beta))

    @classmethod
    def from_json(cls, path: Path | str, primary: str = "cam0") -> "TimeSync":
        """Construct from the file written by compute_event_sync.py."""
        data = json.loads(Path(path).read_text())
        if "events" not in data or len(data["events"]) < 2:
            raise ValueError(f"{path}: missing two events in JSON")
        e1, e2 = data["events"][0], data["events"][1]
        return cls(
            events={cid: (int(e1[cid]), int(e2[cid])) for cid in e1},
            primary=primary,
        )

    @property
    def primary(self) -> str:
        return self._primary

    def cameras(self) -> list[str]:
        return list(self._models.keys())

    def model_for(self, cid: str) -> AffineModel:
        return self._models[cid]

    def support_frame(self, cid: str, primary_frame: int) -> int:
        """Matched frame in camera cid for a given primary frame, rounded to nearest integer."""
        m = self._models[cid]
        return int(round(m.alpha + m.beta * primary_frame))

    def offset(self, cid: str, primary_frame: int) -> int:
        """Signed integer offset (matched - primary) at primary_frame."""
        return self.support_frame(cid, primary_frame) - int(primary_frame)

    def summary(self) -> str:
        """One-line-per-camera summary for logs and headers."""
        lines = [f"primary={self._primary}"]
        for cid, m in self._models.items():
            if cid == self._primary:
                continue
            lines.append(f"  {cid}: alpha={m.alpha:+.3f}  beta={m.beta:.6f}")
        return "\n".join(lines)
