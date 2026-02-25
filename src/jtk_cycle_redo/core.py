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


def _aic(y, yhat, k):
    n = len(y)
    rss = np.sum((y - yhat) ** 2)
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + 2 * k


def _amplitude_metrics(y, yhat=None):
    y = np.asarray(y, dtype=float)
    baseline = float(np.nanmedian(y))

    p10 = float(np.nanpercentile(y, 10))
    p90 = float(np.nanpercentile(y, 90))
    obs_mag_p90_p10 = p90 - p10
    obs_mag_max_min = float(np.nanmax(y) - np.nanmin(y))

    fitted_mag = np.nan
    if yhat is not None and len(yhat) > 0:
        fitted_mag = float(np.nanmax(yhat) - np.nanmin(yhat))

    rel_amp_obs = obs_mag_p90_p10 / abs(baseline) if abs(baseline) > 1e-8 else np.nan
    rel_amp_fit = fitted_mag / abs(baseline) if np.isfinite(fitted_mag) and abs(baseline) > 1e-8 else np.nan

    return {
        "baseline_median": baseline,
        "magnitude_p90_p10": obs_mag_p90_p10,
        "magnitude_max_min": obs_mag_max_min,
        "fitted_magnitude": fitted_mag,
        "relative_amplitude": float(rel_amp_obs) if np.isfinite(rel_amp_obs) else np.nan,
        "relative_fitted_amplitude": float(rel_amp_fit) if np.isfinite(rel_amp_fit) else np.nan,
    }


def _choose_harmonics_grid(n_obs, design_robust=True, harmonics_grid=None):
    if harmonics_grid is not None:
        return tuple(harmonics_grid)
    if not design_robust:
        return (1, 2, 3)

    # adaptive by design density
    if n_obs <= 14:       # ~12-point designs
        return (1, 2)
    if n_obs <= 30:       # ~24-point designs
        return (1, 2, 3)
    return (1, 2, 3, 4)   # dense (~48)


def jtk_score_series(
    y,
    t,
    period=24.0,
    harmonics_grid=None,
    min_obs=8,
    design_robust=True,
):
    """Waveform-flexible rhythm score for one series.

    Features:
    - Multi-harmonic periodic fit (shape-flexible)
    - Missing-data tolerant (nan-aware; fit on observed points)
    - Adaptive harmonic complexity for sparse/medium/dense designs
    - Amplitude metrics including fitted magnitude
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
        "design_mode": "robust" if design_robust else "fixed",
    }

    if n_obs < int(min_obs):
        return {**base, **_amplitude_metrics(y)}

    yo = y[ok]
    to = t[ok]
    h_grid = _choose_harmonics_grid(n_obs=n_obs, design_robust=design_robust, harmonics_grid=harmonics_grid)

    best = None
    for h in h_grid:
        yhat, beta = _fit_periodic_fourier(yo, to, period=period, harmonics=h)
        tau, p = kendalltau(yo, yhat, nan_policy="omit")
        if np.isnan(tau):
            continue

        # prefer higher |tau|; on ties prefer lower AIC (simpler/better fit)
        k = 1 + 2 * h
        aic = _aic(yo, yhat, k)

        cand = {
            "period": float(period),
            "phase": _peak_phase_from_fit(period, h, beta),
            "tau": float(tau),
            "pvalue": float(p),
            "harmonics": int(h),
            "aic": float(aic),
            "n_obs": n_obs,
            "missing_frac": float(missing_frac),
            "design_mode": "robust" if design_robust else "fixed",
            "_yhat_obs": yhat,
        }

        if best is None:
            best = cand
        else:
            if abs(cand["tau"]) > abs(best["tau"]):
                best = cand
            elif np.isclose(abs(cand["tau"]), abs(best["tau"])) and cand["aic"] < best["aic"]:
                best = cand

    if best is None:
        return {**base, **_amplitude_metrics(y)}

    amp = _amplitude_metrics(y, yhat=best.pop("_yhat_obs"))
    return {**best, **amp}


def jtk_scan(
    matrix,
    t,
    period_grid=(24.0,),
    harmonics_grid=None,
    min_obs=8,
    design_robust=True,
):
    """Scan many rows (genes) and return per-row best score."""
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
            "aic": np.nan,
            "n_obs": 0,
            "missing_frac": 1.0,
            "design_mode": "robust" if design_robust else "fixed",
            "baseline_median": np.nan,
            "magnitude_p90_p10": np.nan,
            "magnitude_max_min": np.nan,
            "fitted_magnitude": np.nan,
            "relative_amplitude": np.nan,
            "relative_fitted_amplitude": np.nan,
        }

        for p in period_grid:
            s = jtk_score_series(
                x[i],
                t,
                period=p,
                harmonics_grid=harmonics_grid,
                min_obs=min_obs,
                design_robust=design_robust,
            )
            if s["tau"] is None:
                if row_best["tau"] is None:
                    row_best = {**row_best, **s, "series": i}
                continue

            if row_best["tau"] is None or abs(s["tau"]) > abs(row_best["tau"]):
                row_best = {"series": i, **s}

        out.append(row_best)
    return out
