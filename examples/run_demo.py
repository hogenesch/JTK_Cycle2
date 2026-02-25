from jtk_cycle_redo.synthetic import make_synthetic_dataset
from jtk_cycle_redo.core import jtk_scan


def main():
    ds = make_synthetic_dataset(n_series=100, n_timepoints=24, rhythmic_fraction=0.4, seed=42)
    out = jtk_scan(ds["X"], ds["t"], period_grid=(24.0,), phase_step=1.0)

    top = sorted(out, key=lambda r: abs(r["tau"]), reverse=True)[:10]
    print("Top 10 series by |tau|:")
    for r in top:
        print(r)


if __name__ == "__main__":
    main()
