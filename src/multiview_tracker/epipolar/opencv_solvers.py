from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FundamentalEstimate:
    F: np.ndarray             # 3x3 fundamental matrix (rank-2)
    inlier_mask: np.ndarray   # (N,) bool, True for RANSAC inliers
    method: str               # human-readable identifier


def fundamental_direct(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    *,
    method: int = cv2.FM_RANSAC,
    ransac_threshold_px: float = 1.0,
    confidence: float = 0.999,
) -> FundamentalEstimate:
    """direct fit via cv2.findFundamentalMat."""
    F, mask = cv2.findFundamentalMat(
        pts_a, pts_b,
        method=method,
        ransacReprojThreshold=ransac_threshold_px,
        confidence=confidence,
    )
    if F is None:
        raise RuntimeError("findFundamentalMat returned None")
    return FundamentalEstimate(
        F=F.astype(np.float64),
        inlier_mask=mask.ravel().astype(bool),
        method="opencv_direct",
    )


def fundamental_via_essential(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    K_a: np.ndarray,
    K_b: np.ndarray,
    D_a: np.ndarray,
    D_b: np.ndarray,
    *,
    ransac_threshold: float = 1e-3,
    confidence: float = 0.999,
) -> tuple[FundamentalEstimate, np.ndarray]:
    """Estimate via essential matrix: undistort, findEssentialMat, rebuild F = K_b^-T E K_a^-1.

    Returns (FundamentalEstimate, E).
    """
    pts_a_n = cv2.undistortPoints(
        pts_a.reshape(-1, 1, 2), K_a, D_a
    ).reshape(-1, 2)
    pts_b_n = cv2.undistortPoints(
        pts_b.reshape(-1, 1, 2), K_b, D_b
    ).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts_a_n, pts_b_n,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=confidence,
        threshold=ransac_threshold,
    )
    if E is None:
        raise RuntimeError("findEssentialMat returned None")

    F = np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)
    return (
        FundamentalEstimate(
            F=F.astype(np.float64),
            inlier_mask=mask.ravel().astype(bool),
            method="opencv_essential_to_fundamental",
        ),
        E.astype(np.float64),
    )


def normalise_F(F: np.ndarray) -> np.ndarray:
    """Scale F to unit Frobenius norm with positive last entry."""
    F = F / np.linalg.norm(F)
    if F[2, 2] < 0:
        F = -F
    return F


def epipolar_constraint_residuals(
    F: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """For each correspondence return x_b^T F x_a (close to 0 for good F)."""
    pa = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pb = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    Fxa = pa @ F.T            # each row is F x_a
    res = np.einsum("ij,ij->i", pb, Fxa)
    return res


def symmetric_epipolar_distance(
    F: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray
) -> np.ndarray:
    """Mean point-to-epipolar-line distance, averaged across both directions.

    For point a in image A, line in image B is l_b = F a; distance is |b^T l_b| / sqrt(l_b[0]^2 + l_b[1]^2).
    """
    pa = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pb = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    lines_b = pa @ F.T            # epipolar lines in image B for each a
    lines_a = pb @ F              # epipolar lines in image A for each b
    num_b = np.abs(np.einsum("ij,ij->i", pb, lines_b))
    num_a = np.abs(np.einsum("ij,ij->i", pa, lines_a))
    den_b = np.linalg.norm(lines_b[:, :2], axis=1) + 1e-12
    den_a = np.linalg.norm(lines_a[:, :2], axis=1) + 1e-12
    return 0.5 * (num_b / den_b + num_a / den_a)
