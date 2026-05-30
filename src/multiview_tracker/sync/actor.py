from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from multiview_tracker.detection.yolo_pose import PoseDetection


def high_conf_centroid(det: PoseDetection, conf_min: float = 0.4) -> np.ndarray | None:
    """Mean (x, y) over keypoints with confidence >= conf_min."""
    kp = det.keypoints
    mask = kp[:, 2] >= conf_min
    if mask.sum() < 3:
        return None
    return kp[mask, :2].mean(axis=0)


@dataclass
class _Track:
    last_centroid: np.ndarray
    last_seen_frame: int
    total_motion: float = 0.0
    n_observations: int = 1


def assign_tracks(
    detections: list[PoseDetection],
    max_match_dist_px: float = 250.0,
    max_gap_frames: int = 30,
) -> list[int]:
    """Greedy nearest-centroid tracking. Returns a track_id per detection in input order; -1 means no usable centroid."""
    by_frame: dict[int, list[tuple[int, PoseDetection]]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_frame[d.frame_idx].append((i, d))

    tracks: list[_Track] = []
    track_ids = [-1] * len(detections)

    for frame_idx in sorted(by_frame.keys()):
        for det_idx, det in by_frame[frame_idx]:
            centroid = high_conf_centroid(det)
            if centroid is None:
                continue

            best_tid, best_dist = -1, max_match_dist_px
            for tid, t in enumerate(tracks):
                if frame_idx - t.last_seen_frame > max_gap_frames:
                    continue
                dist = float(np.linalg.norm(centroid - t.last_centroid))
                if dist < best_dist:
                    best_dist, best_tid = dist, tid

            if best_tid < 0:
                tracks.append(_Track(
                    last_centroid=centroid,
                    last_seen_frame=frame_idx,
                ))
                track_ids[det_idx] = len(tracks) - 1
            else:
                t = tracks[best_tid]
                t.total_motion += float(np.linalg.norm(centroid - t.last_centroid))
                t.last_centroid = centroid
                t.last_seen_frame = frame_idx
                t.n_observations += 1
                track_ids[det_idx] = best_tid

    return track_ids


def select_actor_track(
    detections: list[PoseDetection],
    track_ids: list[int],
    min_observations: int = 100,
) -> int:
    """Pick the track id with the largest total centroid motion. Tracks with fewer than min_observations detections are ignored to skip false-detection flicker."""
    totals: dict[int, float] = defaultdict(float)
    last: dict[int, np.ndarray] = {}
    counts: dict[int, int] = defaultdict(int)

    by_frame: dict[int, list[tuple[int, PoseDetection]]] = defaultdict(list)
    for i, d in enumerate(detections):
        by_frame[d.frame_idx].append((i, d))

    for frame_idx in sorted(by_frame.keys()):
        for det_idx, det in by_frame[frame_idx]:
            tid = track_ids[det_idx]
            if tid < 0:
                continue
            c = high_conf_centroid(det)
            if c is None:
                continue
            counts[tid] += 1
            if tid in last:
                totals[tid] += float(np.linalg.norm(c - last[tid]))
            last[tid] = c

    eligible = {tid: m for tid, m in totals.items() if counts[tid] >= min_observations}
    if not eligible:
        # fall back: pick the longest-lived track
        return max(counts, key=counts.get)
    return max(eligible, key=eligible.get)


@dataclass
class _Tracklet:
    """Per-track summary used during the merging pass."""
    tid: int
    first_frame: int
    last_frame: int
    first_centroid: np.ndarray
    last_centroid: np.ndarray
    velocity: np.ndarray              # px/frame, from the most recent observations
    det_indices: list[int]


def _estimate_velocity(
    observations: list[tuple[int, int, np.ndarray]], window: int
) -> np.ndarray:
    """Average velocity (px/frame) over the last ``window`` observations, a frame-sorted list of (frame_idx, det_idx, centroid)."""
    if len(observations) < 2:
        return np.zeros(2)
    recent = observations[-window:] if len(observations) >= window else observations
    if len(recent) < 2:
        return np.zeros(2)
    df = recent[-1][0] - recent[0][0]
    if df <= 0:
        return np.zeros(2)
    dc = recent[-1][2] - recent[0][2]
    return dc / df


def merge_tracklets(
    detections: list[PoseDetection],
    track_ids: list[int],
    *,
    max_gap_frames: int = 200,
    max_extrap_dist_px: float = 150.0,
    velocity_window: int = 10,
    gap_weight: float = 0.5,
) -> list[int]:
    """Post-hoc tracklet merging via velocity-extrapolated trajectory continuity.

    Greedily merges pairs where one tracklet ends shortly before another
    begins and the velocity-extrapolated end position lands near the next
    tracklet's start. Merged tracklets adopt the earlier tracklet's id.

    Parameters
    ----------
    max_gap_frames: max temporal gap to consider linking two tracklets.
    max_extrap_dist_px: max L2 distance from extrapolated end to next start.
    velocity_window: recent observations used to estimate end velocity.
    gap_weight: penalty per gap-frame added to spatial cost when ranking merges.
    """
    by_tid: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for di, tid in enumerate(track_ids):
        if tid < 0:
            continue
        c = high_conf_centroid(detections[di])
        if c is None:
            continue
        by_tid[tid].append((detections[di].frame_idx, di, c))

    tracklets: dict[int, _Tracklet] = {}
    for tid, obs in by_tid.items():
        obs.sort(key=lambda x: x[0])
        tracklets[tid] = _Tracklet(
            tid=tid,
            first_frame=obs[0][0],
            last_frame=obs[-1][0],
            first_centroid=obs[0][2],
            last_centroid=obs[-1][2],
            velocity=_estimate_velocity(obs, velocity_window),
            det_indices=[o[1] for o in obs],
        )

    # greedy: repeatedly merge the best-scoring linkable pair until none remain
    while True:
        best_cost = float("inf")
        best_pair: tuple[int, int] | None = None
        tids = list(tracklets.keys())
        for i, ta in enumerate(tids):
            for tb in tids[i + 1:]:
                ea, eb = tracklets[ta], tracklets[tb]
                # order by finish time; reject overlap (cannot be the same person)
                if ea.last_frame < eb.first_frame:
                    early, late = ea, eb
                elif eb.last_frame < ea.first_frame:
                    early, late = eb, ea
                else:
                    continue

                gap = late.first_frame - early.last_frame
                if gap > max_gap_frames:
                    continue
                predicted = early.last_centroid + early.velocity * gap
                dist = float(np.linalg.norm(predicted - late.first_centroid))
                if dist > max_extrap_dist_px:
                    continue
                cost = dist + gap * gap_weight
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (early.tid, late.tid)

        if best_pair is None:
            break
        early_tid, late_tid = best_pair
        e = tracklets[early_tid]
        l = tracklets[late_tid]
        e.last_frame = l.last_frame
        e.last_centroid = l.last_centroid
        e.velocity = l.velocity  # carry late's velocity (more recent)
        e.det_indices.extend(l.det_indices)
        del tracklets[late_tid]

    # re-label
    new_ids = list(track_ids)
    for tid, t in tracklets.items():
        for di in t.det_indices:
            new_ids[di] = tid
    return new_ids


@dataclass
class _MiniTracklet:
    """Summary of a tracklet used by the anchor-assignment pass."""
    tid: int
    first_frame: int
    last_frame: int
    first_centroid: np.ndarray
    last_centroid: np.ndarray
    det_indices: list[int]
    length: int


def _build_mini_tracklets(
    detections: list[PoseDetection], track_ids: list[int]
) -> list[_MiniTracklet]:
    by_tid: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    for di, tid in enumerate(track_ids):
        if tid < 0:
            continue
        c = high_conf_centroid(detections[di])
        if c is None:
            continue
        by_tid[tid].append((detections[di].frame_idx, di, c))

    out: list[_MiniTracklet] = []
    for tid, obs in by_tid.items():
        obs.sort(key=lambda x: x[0])
        out.append(_MiniTracklet(
            tid=tid,
            first_frame=obs[0][0],
            last_frame=obs[-1][0],
            first_centroid=obs[0][2],
            last_centroid=obs[-1][2],
            det_indices=[o[1] for o in obs],
            length=len(obs),
        ))
    return out


def cluster_tracks_to_k(
    detections: list[PoseDetection],
    track_ids: list[int],
    *,
    k: int = 2,
) -> list[int]:
    """Force tracklets into exactly k non-overlapping anchors by greedy kinematic-cost assignment.

    Seeds k anchors with the k longest tracklets, then assigns each remaining
    tracklet to the cheapest non-overlapping anchor by required px/frame
    speed to bridge to the temporal neighbours within that anchor. Tracklets
    that overlap every anchor are dropped (-1). Output ids are contiguous
    0..k-1.
    """
    tracklets = _build_mini_tracklets(detections, track_ids)
    if len(tracklets) <= k:
        # already at or under k tracks; relabel sequentially for tidiness
        out = list(track_ids)
        rename = {t.tid: i for i, t in enumerate(tracklets)}
        return [rename.get(t, t) if t >= 0 else -1 for t in out]

    # most-reliable first
    tracklets.sort(key=lambda t: t.length, reverse=True)
    anchors: list[list[_MiniTracklet]] = [[tracklets[i]] for i in range(k)]

    for cand in tracklets[k:]:
        best_anchor = -1
        best_cost = float("inf")
        for a_idx, anchor in enumerate(anchors):
            # hard constraint: no temporal overlap with anything in the anchor
            overlap = any(
                not (cand.last_frame < t.first_frame or cand.first_frame > t.last_frame)
                for t in anchor
            )
            if overlap:
                continue

            # immediate temporal neighbours within this anchor
            prev_t: _MiniTracklet | None = None
            next_t: _MiniTracklet | None = None
            for t in anchor:
                if t.last_frame < cand.first_frame:
                    if prev_t is None or t.last_frame > prev_t.last_frame:
                        prev_t = t
                elif t.first_frame > cand.last_frame:
                    if next_t is None or t.first_frame < next_t.first_frame:
                        next_t = t

            # kinematic cost: required px/frame speed to bridge the gap
            cost = 0.0
            comparisons = 0
            if prev_t is not None:
                gap = max(1, cand.first_frame - prev_t.last_frame)
                dist = float(np.linalg.norm(cand.first_centroid - prev_t.last_centroid))
                cost += dist / gap
                comparisons += 1
            if next_t is not None:
                gap = max(1, next_t.first_frame - cand.last_frame)
                dist = float(np.linalg.norm(next_t.first_centroid - cand.last_centroid))
                cost += dist / gap
                comparisons += 1
            if comparisons:
                cost /= comparisons

            if cost < best_cost:
                best_cost = cost
                best_anchor = a_idx

        if best_anchor != -1:
            anchors[best_anchor].append(cand)
        # else: dropped as noise (overlaps every anchor)

    # relabel with contiguous anchor ids 0..k-1
    final = [-1] * len(track_ids)
    for a_idx, anchor in enumerate(anchors):
        for t in anchor:
            for di in t.det_indices:
                final[di] = a_idx
    return final


def static_anchor_classify(
    detections: list[PoseDetection],
    track_ids: list[int],
    *,
    min_obs_for_anchor: int = 100,
    home_radius_px: float = 150.0,
) -> list[int]:
    """Static Anchor heuristic for a scene with one stationary bystander and one moving actor.

    Bystander = long tracklet with smallest spatial std. Its median centroid
    defines a circular home zone of radius home_radius_px; detections inside
    are labelled 0 (bystander), outside are 1 (actor), unclassifiable are -1.

    Classification is per-detection rather than per-tracklet so an actor
    fragment that briefly enters the home zone only flips those specific
    frames, not the whole tracklet.
    """
    by_tid: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for di, tid in enumerate(track_ids):
        if tid < 0:
            continue
        c = high_conf_centroid(detections[di])
        if c is None:
            continue
        by_tid[tid].append((di, c))

    if not by_tid:
        return list(track_ids)

    stats: dict[int, dict] = {}
    for tid, obs in by_tid.items():
        cents = np.asarray([c for _, c in obs])
        median = np.median(cents, axis=0)
        spatial_std = float(np.std(np.linalg.norm(cents - median, axis=1)))
        stats[tid] = {
            "median": median,
            "std": spatial_std,
            "n_obs": len(cents),
            "obs": obs,
        }

    # bystander = lowest spatial std among long tracklets
    long_tids = [tid for tid in stats if stats[tid]["n_obs"] >= min_obs_for_anchor]
    if not long_tids:
        long_tids = list(stats.keys())
    bystander_tid = min(long_tids, key=lambda t: stats[t]["std"])
    home_center = stats[bystander_tid]["median"]

    # per-detection labelling (see docstring for why)
    new_track_ids = list(track_ids)
    for di, tid in enumerate(track_ids):
        if tid < 0:
            continue
        c = high_conf_centroid(detections[di])
        if c is None:
            new_track_ids[di] = -1
            continue
        dist = float(np.linalg.norm(c - home_center))
        new_track_ids[di] = 0 if dist < home_radius_px else 1

    return new_track_ids


def filter_to_actor(
    detections: list[PoseDetection],
    *,
    merge: bool = False,
    force_cap_to: int | None = None,
    static_anchor: bool = False,
) -> list[PoseDetection]:
    """Tracking + motion selection: return only the actor's detections. If merge=True, runs merge_tracklets between assign_tracks and select_actor_track."""
    track_ids = assign_tracks(detections)
    if merge:
        track_ids = merge_tracklets(detections, track_ids)
    if static_anchor:
        track_ids = static_anchor_classify(detections, track_ids)
        actor_tid = 1
    elif force_cap_to is not None and force_cap_to > 0:
        track_ids = cluster_tracks_to_k(detections, track_ids, k=force_cap_to)
        actor_tid = select_actor_track(detections, track_ids)
    else:
        actor_tid = select_actor_track(detections, track_ids)
    return [d for d, tid in zip(detections, track_ids) if tid == actor_tid]
