import numpy as np
from scipy.stats import kendalltau


def _fourier_design(t, period, harmonics):
    cols = [np.ones_like(t, dtype=float)]
    for k in range(1, int(harmonics) + 1):
        w = 2.0 * np.pi * k / float(period)
        cols.append(np.cos(w * t))
        cols.append(np.sin(w * t))
    return np.column_stack(cols)


def _fit_periodic_fourier(y, t, period, harmonics):
    x = _fourier_design(t, period, harmonics)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    yhat = x @ beta
    return yhat, beta


def _peak_phase_from_fit(period, harmonics, beta, n_grid=240):
    g = np.linspace(0.0, float(period), int(n_grid), endpoint=False)
    xg = _fourier_design(g, period, harmonics)
    yh = xg @ beta
    return float(g[np.argmax(yh)])


def _amplitude_metrics(y):
    y = np.asarray(y, dtype=float)
    baseline = float(np.nanmedian(y))
    p10 = float(np.nanpercentile(y, 10))
    p90 = float(np.nanpercentile(y, 90))
    magnitude_p90_p10 = p90 - p10
    magnitude_max_min = float(np.nanmax(y) - np.nanmin(y))
    relative_amplitude = magnitude_p90_p10 / abs(baseline) if abs(baseline) > 1e-8 else np.nan
    return {
        "baseline_median": baseline,
        "magnitude_p90_p10": magnitude_p90_p10,
        "magnitude_max_min": magnitude_max_min,
        "relative_amplitude": float(relative_amplitude) if np.isfinite(relative_amplitude) else np.nan,
    }


def jtk_score_series(y, t, period=24.0, harmonics_grid=(1, 2, 3), min_obs=8):
    """Waveform-flexible rhythm score for one series.

    - Uses multi-harmonic periodic regression (shape-flexible; not fixed cosine-only).
    - Handles missing values by fitting on observed points only.
    - Reports amplitude metrics relative to baseline.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    ok = np.isfinite(y) & np.isfinite(t)
    n_total = int(y.size)
    n_obs = int(ok.sum())
    missing_frac = 1.0 - (n_obs / n_total if n_total else 0.0)

    base = {
        "period": float(period),
        "phase": None,
        "tau": None,
        "pvalue": 1.0,
        "harmonics": None,
        "n_obs": n_obs,
        "missing_frac": float(missing_frac),
    }

    if n_obs < int(min_obs):
        return {**base, **_amplitude_metrics(y)}

    yo = y[ok]
    to = t[ok]

    best = None
    for h in harmonics_grid:
        yhat, beta = _fit_periodic_fourier(yo, to, period=period, harmonics=h)
        tau, p = kendalltau(yo, yhat, nan_policy="omit")
        if np.isnan(tau):
            continue
        cand = {
            "period": float(period),
            "phase": _peak_phase_from_fit(period, h, beta),
            "tau": float(tau),
            "pvalue": float(p),
            "harmonics": int(h),
            "n_obs": n_obs,
            "missing_frac": float(missing_frac),
        }
        if best is None or abs(cand["tau"]) > abs(best["tau"]):
            best = cand

    if best is None:
        best = base

    return {**best, **_amplitude_metrics(y)}


def jtk_scan(matrix, t, period_grid=(24.0,), harmonics_grid=(1, 2, 3), min_obs=8):
    """Scan many rows (genes) and return per-row best score.

    Includes waveform-flexible periodic fit, missingness stats, and amplitude metrics.
    """
    x = np.asarray(matrix, dtype=float)
    t = np.asarray(t, dtype=float)

    out = []
    for i in range(x.shape[0]):
        row_best = {
            "series": i,
            "period": None,
            "phase": None,
            "tau": None,
            "pvalue": 1.0,
            "harmonics": None,
            "n_obs": 0,
            "missing_frac": 1.0,
            "baseline_median": np.nan,
            "magnitude_p90_p10": np.nan,
            "magnitude_max_min": np.nan,
            "relative_amplitude": np.nan,
        }
        for p in period_grid:
            s = jtk_score_series(x[i], t, period=p, harmonics_grid=harmonics_grid, min_obs=min_obs)
            if s["tau"] is None:
                if row_best["tau"] is None:
                    row_best = {**row_best, **s, "series": i}
                continue
            if row_best["tau"] is None or abs(s["tau"]) > abs(row_best["tau"]):
                row_best = {"series": i, **s}
        out.append(row_best)
    return out
