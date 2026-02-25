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
