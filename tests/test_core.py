import numpy as np
from jtk_cycle_redo.core import jtk_score_series, jtk_scan


def test_jtk_score_series_detects_rhythm():
    t = np.arange(24)
    y = np.cos((2 * np.pi / 24.0) * (t - 6))
    out = jtk_score_series(y, t, period=24.0)
    assert abs(out["tau"]) > 0.6
    assert out["magnitude_p90_p10"] > 0


def test_jtk_scan_runs_matrix():
    t = np.arange(24)
    X = np.vstack([
        np.cos((2 * np.pi / 24.0) * (t - 2)),
        np.random.default_rng(1).normal(size=24),
    ])
    out = jtk_scan(X, t, period_grid=(24.0,))
    assert len(out) == 2
    assert out[0]["period"] == 24.0


def test_waveform_flex_and_missing_data():
    t = np.arange(48)
    # spiky waveform (rectified high-power cosine), with missing values
    y = np.clip(np.cos((2 * np.pi / 24.0) * (t - 8)), 0, None) ** 4
    y[[3, 7, 19, 31, 44]] = np.nan

    out = jtk_score_series(y, t, period=24.0, harmonics_grid=(1, 2, 3, 4), min_obs=20)
    assert out["tau"] is not None
    assert out["n_obs"] == 43
    assert 0 < out["missing_frac"] < 1
    assert out["harmonics"] in (1, 2, 3, 4)
