"""Unit tests for geom_features.py using random numpy arrays (no torch needed).

Run directly:  python3 scripts/tests/test_geom_features.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geom_features import (
    compactness_C,
    frame_delta_norms,
    mean_frame_cosine_distance,
    radius_of_gyration,
    velocity_entropy,
)


def check(name: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        raise AssertionError(name)


def test_radius_of_gyration_constant_embedding_is_zero():
    emb = np.tile(np.array([1.0, 2.0, 3.0]), (10, 1))
    rg = radius_of_gyration(emb)
    check("rg of constant embedding == 0", math.isclose(rg, 0.0, abs_tol=1e-9))


def test_radius_of_gyration_known_value():
    # Two points at +-d/2 from mean along one axis -> rg = d/2.
    emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
    rg = radius_of_gyration(emb)
    check("rg of +-1 two-point set == 1.0", math.isclose(rg, 1.0, rel_tol=1e-9))


def test_compactness_is_negative_rg():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(20, 8))
    check("C == -rg", math.isclose(compactness_C(emb), -radius_of_gyration(emb)))


def test_rg_empty_and_singleton():
    check("rg([]) == 0", radius_of_gyration(np.zeros((0, 4))) == 0.0)
    check("rg(single point) == 0", math.isclose(radius_of_gyration(np.array([[1.0, 2.0]])), 0.0))


def test_frame_delta_norms_shape():
    emb = np.arange(30).reshape(10, 3).astype(float)
    deltas = frame_delta_norms(emb)
    check("delta norms length == T-1", len(deltas) == 9)
    # Each step here is a constant vector [3,3,3] -> norm = 3*sqrt(3)
    check("delta norms constant-step value", np.allclose(deltas, 3 * math.sqrt(3)))


def test_velocity_entropy_degenerate_is_zero():
    # All frames identical -> zero deltas -> entropy 0
    emb = np.ones((15, 5))
    ent = velocity_entropy(emb)
    check("velocity entropy of static embedding == 0", math.isclose(ent, 0.0, abs_tol=1e-9))


def test_velocity_entropy_bounds():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(200, 16))
    ent = velocity_entropy(emb, n_bins=32)
    check("velocity entropy in [0,1]", 0.0 <= ent <= 1.0 + 1e-9)


def test_velocity_entropy_uniform_higher_than_spiky():
    rng = np.random.default_rng(2)
    # Spiky: one huge jump, rest identical -> low entropy (mass concentrated in bins).
    emb_spiky = np.zeros((50, 4))
    emb_spiky[25] += 100.0
    ent_spiky = velocity_entropy(emb_spiky)

    # More uniform deltas across the range -> should have higher (or at least
    # not trivially lower) entropy than a single-spike trajectory of the same length.
    emb_uniform = np.cumsum(rng.uniform(0.5, 1.5, size=(50, 4)), axis=0)
    ent_uniform = velocity_entropy(emb_uniform)

    check("uniform-step trajectory entropy >= spiky trajectory entropy",
          ent_uniform >= ent_spiky)


def test_mean_frame_cosine_distance_identical_is_zero():
    emb = np.tile(np.array([1.0, 0.0, 0.0]), (10, 1))
    d = mean_frame_cosine_distance(emb)
    check("cosine distance of identical frames == 0", math.isclose(d, 0.0, abs_tol=1e-9))


def test_mean_frame_cosine_distance_orthogonal_is_one():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    d = mean_frame_cosine_distance(emb)
    check("cosine distance of alternating orthogonal frames == 1.0", math.isclose(d, 1.0, abs_tol=1e-9))


def test_short_sequences_do_not_crash():
    for T in (0, 1):
        emb = np.zeros((T, 4))
        check(f"rg with T={T} handled", radius_of_gyration(emb) == 0.0)
        check(f"entropy with T={T} handled", velocity_entropy(emb) == 0.0)
        check(f"cosine dist with T={T} handled", mean_frame_cosine_distance(emb) == 0.0)


def main():
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for t in tests:
        t()
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
