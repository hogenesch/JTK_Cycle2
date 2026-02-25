import numpy as np


def bh_qvalues(pvalues):
    """Benjamini-Hochberg FDR q-values.

    pvalues: array-like with np.nan allowed.
    returns np.ndarray q-values aligned to input.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q

    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]

    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)

    inv = np.empty_like(order)
    inv[order] = np.arange(m)
    q_ok = q_ranked[inv]
    q[ok] = q_ok
    return q


def permutation_pvalues_scan(
    matrix,
    t,
    score_fn,
    n_perm=200,
    random_seed=42,
):
    """Permutation p-values using max-|tau| null per series.

    score_fn must be callable: score_fn(y, t) -> dict with key 'tau'.
    """
    x = np.asarray(matrix, dtype=float)
    t = np.asarray(t, dtype=float)
    rng = np.random.default_rng(random_seed)

    pvals = np.full(x.shape[0], np.nan, dtype=float)

    for i in range(x.shape[0]):
        y = x[i]
        obs = score_fn(y, t)
        tau_obs = obs.get("tau", np.nan)
        if tau_obs is None or not np.isfinite(tau_obs):
            continue

        null = np.zeros(n_perm, dtype=float)
        for b in range(n_perm):
            yp = y.copy()
            rng.shuffle(yp)
            s = score_fn(yp, t)
            tb = s.get("tau", np.nan)
            null[b] = abs(tb) if np.isfinite(tb) else 0.0

        pvals[i] = (1.0 + np.sum(null >= abs(tau_obs))) / (n_perm + 1.0)

    return pvals
