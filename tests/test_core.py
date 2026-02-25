import numpy as np
from jtk_cycle_redo.core import jtk_score_series, jtk_scan


def test_jtk_score_series_detects_rhythm():
    t = np.arange(24)
    y = np.cos((2 * np.pi / 24.0) * (t - 6))
    out = jtk_score_series(y, t, period=24.0)
    assert abs(out["tau"]) > 0.6


def test_jtk_scan_runs_matrix():
    t = np.arange(24)
    X = np.vstack([
        np.cos((2 * np.pi / 24.0) * (t - 2)),
        np.random.default_rng(1).normal(size=24),
    ])
    out = jtk_scan(X, t, period_grid=(24.0,))
    assert len(out) == 2
    assert out[0]["period"] == 24.0
