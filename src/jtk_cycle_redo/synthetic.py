import numpy as np


def make_synthetic_dataset(
    n_series=200,
    n_timepoints=24,
    period=24.0,
    rhythmic_fraction=0.3,
    noise_sd=0.5,
    seed=13,
):
    rng = np.random.default_rng(seed)
    t = np.arange(n_timepoints, dtype=float)

    x = np.zeros((n_series, n_timepoints), dtype=float)
    labels = np.zeros(n_series, dtype=int)

    n_rhy = int(round(n_series * rhythmic_fraction))
    rhythmic_idx = rng.choice(n_series, size=n_rhy, replace=False)

    for i in range(n_series):
        baseline = rng.normal(0, 0.3)
        if i in rhythmic_idx:
            amp = rng.uniform(0.6, 1.8)
            phase = rng.uniform(0, period)
            signal = amp * np.cos((2 * np.pi / period) * (t - phase))
            labels[i] = 1
        else:
            signal = 0.0
        noise = rng.normal(0, noise_sd, size=n_timepoints)
        x[i] = baseline + signal + noise

    return {"X": x, "t": t, "labels": labels}
