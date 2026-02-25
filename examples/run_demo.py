from jtk_cycle_redo.synthetic import make_synthetic_dataset
from jtk_cycle_redo.core import jtk_scan


def main():
    ds = make_synthetic_dataset(n_genes=100, n_timepoints=24, rhythmic_fraction=0.4, seed=42)
    out = jtk_scan(ds.y, ds.t, period_grid=(24.0,))

    top = sorted(out, key=lambda r: abs(r["tau"]) if r["tau"] is not None else -1, reverse=True)[:10]
    print("Top 10 series by |tau|:")
    for r in top:
        print(r)


if __name__ == "__main__":
    main()
