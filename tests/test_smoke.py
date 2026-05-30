from __future__ import annotations

from pathlib import Path

import yaml

import multiview_tracker

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_exposed():
    assert multiview_tracker.__version__


def test_default_config_loads():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    assert "cameras" in cfg
    assert "calibration" in cfg
    assert cfg["calibration"]["puzzleboard"]["rows"] > 0
