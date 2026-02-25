#!/usr/bin/env python3
"""Benchmark permutation p-value runtime for Python vs Rust backends.

This script evaluates runtime across a grid of (topn, n_perm) settings and
writes a CSV summary with speedups and output-difference checks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geo_gse11923_preview import build_expression_ct
from jtk_cycle_redo.core import jtk_score_series
from jtk_cycle_redo.stats import (
    has_rust_permutation_backend,
    permutation_pvalues_scan,
    permutation_pvalues_scan_rust,
)


def _parse_int_list(raw: str) -> list[int]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    if not vals:
        raise ValueError("list cannot be empty")
    return vals


def _time_python(sub: np.ndarray, t_mod: np.ndarray, n_perm: int, seed: int) -> tuple[np.ndarray, float]:
    score_fn = lambda y, tt: jtk_score_series(y, tt, period=24.0, harmonics_grid=None, min_obs=24, design_robust=True)
    t0 = time.perf_counter()
    pvals = permutation_pvalues_scan(sub, t_mod, score_fn=score_fn, n_perm=n_perm, random_seed=seed)
    dt = time.perf_counter() - t0
    return pvals, dt


def _time_rust(
    sub: np.ndarray,
    t_mod: np.ndarray,
    n_perm: int,
    seed: int,
    rust_threads: int,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pvals = permutation_pvalues_scan_rust(
        sub,
        t_mod,
        period=24.0,
        harmonics_grid=None,
        min_obs=24,
        design_robust=True,
        n_perm=n_perm,
        random_seed=seed,
        n_threads=rust_threads,
    )
    dt = time.perf_counter() - t0
    return pvals, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="data/GSE11923")
    ap.add_argument("--topn-list", default="100,500,1000")
    ap.add_argument("--n-perm-list", default="50,200")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--rust-threads", type=int, default=0)
    ap.add_argument(
        "--output-csv",
        default=None,
        help="Default: <workdir>/perm_backend_benchmark.csv",
    )
    args = ap.parse_args()

    topn_list = _parse_int_list(args.topn_list)
    n_perm_list = _parse_int_list(args.n_perm_list)
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    w = Path(args.workdir)
    expr_out = w / "GSE11923_expression_CT.tsv"
    if expr_out.exists():
        expr = pd.read_csv(expr_out, sep="\t")
    else:
        series = w / "GSE11923_series_matrix.txt.gz"
        if not series.exists():
            raise FileNotFoundError(
                f"Missing {series}. Provide an existing expression file or GEO series matrix."
            )
        expr = build_expression_ct(series, expr_out)

    val_cols = [c for c in expr.columns if c.startswith("CT")]
    if not val_cols:
        raise RuntimeError("No CT columns found in expression table")
    x = expr[val_cols].to_numpy(dtype=float)
    var = np.nanvar(x, axis=1)
    order = np.argsort(var)[::-1]
    t = np.array([float(c.replace("CT", "")) for c in val_cols])
    t_mod = np.mod(t, 24.0)

    rust_available = has_rust_permutation_backend()
    print(f"Rust backend available: {rust_available}")
    print(f"topn grid: {topn_list}")
    print(f"n_perm grid: {n_perm_list}")
    print(f"repeats: {args.repeats}")

    rows: list[dict] = []
    for topn in topn_list:
        if topn <= 0:
            raise ValueError("topn must be > 0")
        if topn > x.shape[0]:
            raise ValueError(f"topn={topn} exceeds available series ({x.shape[0]})")
        idx = order[:topn]
        sub = x[idx]

        for n_perm in n_perm_list:
            if n_perm < 1:
                raise ValueError("n_perm must be >= 1")

            py_times = []
            rust_times = []
            py_ref = None
            rust_ref = None

            for rep in range(args.repeats):
                seed = args.random_seed + rep
                py_ref, dt_py = _time_python(sub, t_mod, n_perm=n_perm, seed=seed)
                py_times.append(dt_py)

                if rust_available:
                    rust_ref, dt_rust = _time_rust(
                        sub,
                        t_mod,
                        n_perm=n_perm,
                        seed=seed,
                        rust_threads=args.rust_threads,
                    )
                    rust_times.append(dt_rust)

            py_mean = float(np.mean(py_times))
            py_median = float(np.median(py_times))

            row = {
                "topn": int(topn),
                "n_perm": int(n_perm),
                "repeats": int(args.repeats),
                "python_time_mean_s": py_mean,
                "python_time_median_s": py_median,
                "rust_available": bool(rust_available),
                "rust_threads": int(args.rust_threads),
            }

            if rust_available:
                rust_mean = float(np.mean(rust_times))
                rust_median = float(np.median(rust_times))
                speedup = py_mean / rust_mean if rust_mean > 0 else np.inf
                max_abs_diff = float(np.nanmax(np.abs(py_ref - rust_ref)))
                row.update(
                    {
                        "rust_time_mean_s": rust_mean,
                        "rust_time_median_s": rust_median,
                        "speedup_python_over_rust": float(speedup),
                        "max_abs_diff_last_repeat": max_abs_diff,
                    }
                )
                print(
                    f"topn={topn:4d} n_perm={n_perm:4d} "
                    f"python={py_mean:.3f}s rust={rust_mean:.3f}s speedup={speedup:.2f}x "
                    f"max_abs_diff={max_abs_diff:.4g}"
                )
            else:
                print(f"topn={topn:4d} n_perm={n_perm:4d} python={py_mean:.3f}s rust=NA")

            rows.append(row)

    out = pd.DataFrame(rows)
    out_csv = Path(args.output_csv) if args.output_csv else (w / "perm_backend_benchmark.csv")
    out.to_csv(out_csv, index=False)
    print(f"Wrote benchmark CSV: {out_csv}")


if __name__ == "__main__":
    main()
