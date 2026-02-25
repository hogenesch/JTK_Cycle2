from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    """
    Minimal synthetic dataset container.

    t: (n_timepoints,) time vector
    y: (n_genes, n_timepoints) data matrix
    truth: metadata including which genes are rhythmic
    """

    t: np.ndarray
    y: np.ndarray
    truth: Dict[str, Any]


def make_synthetic_dataset(
    n_genes: int = 200,
    n_timepoints: int = 12,
    period: float = 24.0,
    dt: float = 2.0,
    rhythmic_fraction: float = 0.25,
    amplitude_range: Tuple[float, float] = (0.5, 2.0),
    baseline_range: Tuple[float, float] = (0.0, 2.0),
    noise_sd: float = 0.75,
    seed: Optional[int] = 123,
    missing_rate: float = 0.0,
) -> SyntheticDataset:
    """
    Generates sinusoidal rhythms + Gaussian noise, with optional missingness.
    Designed for demos/tests, not for method claims.
    """

    rng = np.random.default_rng(seed)

    t = np.arange(n_timepoints, dtype=float) * dt  # hours
    phases = rng.uniform(0.0, period, size=n_genes)
    amps = rng.uniform(amplitude_range[0], amplitude_range[1], size=n_genes)
    bases = rng.uniform(baseline_range[0], baseline_range[1], size=n_genes)
    rhythmic = rng.random(n_genes) < rhythmic_fraction

    y = np.zeros((n_genes, n_timepoints), dtype=float)
    w = 2.0 * np.pi / period

    for i in range(n_genes):
        if rhythmic[i]:
            y[i] = bases[i] + amps[i] * np.cos(w * (t - phases[i]))
        else:
            y[i] = bases[i]
        y[i] += rng.normal(0.0, noise_sd, size=n_timepoints)

    if missing_rate > 0:
        mask = rng.random(y.shape) < missing_rate
        y = y.copy()
        y[mask] = np.nan

    truth = {
        "rhythmic": rhythmic,
        "period": period,
        "phase": phases,
        "amplitude": amps,
        "baseline": bases,
        "dt": dt,
    }

    return SyntheticDataset(t=t, y=y, truth=truth)
