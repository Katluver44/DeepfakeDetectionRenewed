"""Pure-numpy geometric feature helpers for WavLM frame-embedding trajectories.

Deliberately free of any torch/transformers imports so it can be unit-tested
and imported even before the venv with torch is ready. Used by
extract_wavlm_features.py (which does the actual model forward passes).

Definitions (see laughsmi_plan.md §5.2, H2):
- Radius of gyration (rg): rg = sqrt(mean_t ||h_t - h_mean||^2), a compactness
  measure of a set of frame embeddings around their centroid.
- Compactness feature C: C = -rg (negative radius of gyration; higher C means
  more compact / less spread out).
- Frame-to-frame delta norms: d_t = ||h_{t+1} - h_t||, t = 1..T-1.
- Velocity entropy T: Shannon entropy (base 2, normalized to [0, 1] by dividing
  by log2(n_bins)) of the histogram of delta norms, using a fixed number of
  bins over the observed range of delta norms for that segment.
- Mean frame-to-frame cosine distance: mean_t (1 - cos(h_t, h_{t+1})), a sanity
  metric of how much the trajectory turns from frame to frame.
"""
from __future__ import annotations

import numpy as np


def radius_of_gyration(embeddings: np.ndarray) -> float:
    """Radius of gyration of a set of frame embeddings.

    Args:
        embeddings: (T, D) array of per-frame embeddings for one segment.

    Returns:
        rg = sqrt(mean_t ||h_t - h_mean||^2) as a float. Returns 0.0 if T < 1.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return 0.0
    mean_vec = embeddings.mean(axis=0, keepdims=True)
    sq_dev = np.sum((embeddings - mean_vec) ** 2, axis=1)
    rg = float(np.sqrt(np.mean(sq_dev)))
    return rg


def compactness_C(embeddings: np.ndarray) -> float:
    """C = -radius_of_gyration(embeddings). Higher C = more compact."""
    return -radius_of_gyration(embeddings)


def frame_delta_norms(embeddings: np.ndarray) -> np.ndarray:
    """Frame-to-frame delta norms ||h_{t+1} - h_t|| for t = 1..T-1.

    Args:
        embeddings: (T, D) array.

    Returns:
        (T-1,) array of delta norms. Empty array if T < 2.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    deltas = embeddings[1:] - embeddings[:-1]
    return np.linalg.norm(deltas, axis=1)


def velocity_entropy(embeddings: np.ndarray, n_bins: int = 32) -> float:
    """Shannon entropy of the histogram of frame-to-frame delta norms.

    Entropy is computed in bits (log base 2) over `n_bins` equal-width bins
    spanning [0, max(delta_norms)] for this segment, then normalized by
    log2(n_bins) so the result lies in [0, 1] (0 = all mass in one bin /
    degenerate, 1 = uniform spread across all bins).

    Args:
        embeddings: (T, D) array of per-frame embeddings for one segment.
        n_bins: number of histogram bins (default 32, per plan §5.2).

    Returns:
        Normalized entropy in [0, 1]. Returns 0.0 for segments with fewer
        than 2 frames or with all-zero deltas (degenerate/no motion).
    """
    deltas = frame_delta_norms(embeddings)
    if deltas.size < 2:
        return 0.0
    max_val = float(deltas.max())
    if max_val <= 0.0:
        return 0.0
    hist, _ = np.histogram(deltas, bins=n_bins, range=(0.0, max_val))
    counts = hist.astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(n_bins)
    if max_entropy <= 0:
        return 0.0
    return entropy_bits / max_entropy


def mean_frame_cosine_distance(embeddings: np.ndarray) -> float:
    """Mean frame-to-frame cosine distance: mean_t (1 - cos(h_t, h_{t+1})).

    Args:
        embeddings: (T, D) array.

    Returns:
        Mean cosine distance across consecutive frame pairs. 0.0 if T < 2.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    a = embeddings[:-1]
    b = embeddings[1:]
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norm
    valid = denom > 0
    if not np.any(valid):
        return 0.0
    cos_sim = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    cos_dist = 1.0 - cos_sim
    return float(np.mean(cos_dist))
