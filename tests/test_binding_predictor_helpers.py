"""Lightweight checks for binding-predictor label transforms."""
from __future__ import annotations

import numpy as np


def test_penalty_clamp_nonnegative():
    raw = np.asarray([-5.2, -0.1, 0.0, 1.5, 16.7])
    clamped = np.maximum(0.0, raw)
    assert clamped.min() == 0.0
    assert clamped[-1] == 16.7
    assert (clamped >= 0).all()


def test_begm_compose():
    be = np.asarray([-5.0, -3.0])
    penalty = np.asarray([0.5, 2.0])
    begm = be + penalty
    assert np.allclose(begm, [-4.5, -1.0])


def test_log1p_roundtrip():
    y = np.asarray([0.0, 0.5, 3.0, 10.0])
    z = np.log1p(y)
    back = np.expm1(z)
    assert np.allclose(back, y)
