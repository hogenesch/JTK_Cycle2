import numpy as np
from scipy.stats import kendalltau


def _cos_template(t, period, phase):
    return np.cos((2 * np.pi / period) * (t - phase))


def jtk_score_series(y, t, period=24.0, phase_grid=None):
    """Score one series against cosine templates over phase shifts.

    Returns best phase, tau, and p-value (approx from kendalltau).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if phase_grid is None:
        phase_grid = np.arange(0, period, 1.0)

    best = {"phase": None, "tau": None, "pvalue": 1.0}

    for ph in phase_grid:
        tmpl = _cos_template(t, period=period, phase=ph)
        tau, p = kendalltau(y, tmpl, nan_policy="omit")
        if np.isnan(tau):
            continue
        if best["tau"] is None or abs(tau) > abs(best["tau"]):
            best = {"phase": float(ph), "tau": float(tau), "pvalue": float(p)}

    return best


def jtk_scan(matrix, t, period_grid=(24.0,), phase_step=1.0):
    """Scan many rows (genes) and return per-row best score.

    matrix: 2D array shape [n_series, n_timepoints]
    """
    x = np.asarray(matrix, dtype=float)
    t = np.asarray(t, dtype=float)

    out = []
    for i in range(x.shape[0]):
        row_best = {"series": i, "period": None, "phase": None, "tau": None, "pvalue": 1.0}
        for p in period_grid:
            phase_grid = np.arange(0, p, phase_step)
            s = jtk_score_series(x[i], t, period=p, phase_grid=phase_grid)
            if s["tau"] is None:
                continue
            if row_best["tau"] is None or abs(s["tau"]) > abs(row_best["tau"]):
                row_best = {
                    "series": i,
                    "period": float(p),
                    "phase": s["phase"],
                    "tau": s["tau"],
                    "pvalue": s["pvalue"],
                }
        out.append(row_best)
    return out
