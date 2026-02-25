#!/usr/bin/env python3
"""Build a quick JTK_Cycle2 preview table for GEO GSE11923.

Inputs expected in a working directory (default: ~/Downloads/GSE11923):
- GSE11923_series_matrix.txt.gz
- GPL1261.annot.gz

Outputs:
- GSE11923_expression_CT.tsv
- GSE11923_top1000var_jtkcycle2_preview_with_symbols_amplitude.tsv
"""

from __future__ import annotations
import argparse
import gzip
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from jtk_cycle_redo.core import jtk_scan, jtk_score_series
from jtk_cycle_redo.stats import (
    bh_qvalues,
    has_rust_permutation_backend,
    permutation_pvalues_scan,
    permutation_pvalues_scan_rust,
)


def build_expression_ct(series_matrix_gz: Path, out_tsv: Path) -> pd.DataFrame:
    sample_titles = None
    rows = []
    in_table = False
    header = None

    with gzip.open(series_matrix_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_title"):
                sample_titles = [p.strip('"') for p in line.split("\t")[1:]]
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if in_table:
                parts = [p.strip('"') for p in line.split("\t")]
                if parts[0] == "ID_REF":
                    header = parts
                    continue
                rows.append(parts)

    if not rows or header is None:
        raise RuntimeError("No matrix table found in series matrix file")

    df = pd.DataFrame(rows, columns=header)
    if sample_titles and len(sample_titles) == (len(header) - 1):
        ct = []
        for s in sample_titles:
            m = re.search(r"CT(\d+)", s)
            ct.append(int(m.group(1)) if m else None)
        new_cols = ["ID_REF"] + [f"CT{c}" if c is not None else gsm for c, gsm in zip(ct, header[1:])]
        df.columns = new_cols

    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(out_tsv, sep="\t", index=False)
    return df


def read_platform_annotation(annot_gz: Path) -> pd.DataFrame:
    lines = []
    in_table = False
    with gzip.open(annot_gz, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                lines.append(line)

    ann = pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str, low_memory=False)
    ann = ann[["ID", "Gene symbol", "Gene title"]].rename(
        columns={"ID": "ID_REF", "Gene symbol": "GeneSymbol", "Gene title": "GeneTitle"}
    )
    ann = ann.drop_duplicates(subset=["ID_REF"], keep="first")
    for c in ["GeneSymbol", "GeneTitle"]:
        ann[c] = ann[c].fillna("").astype(str).str.replace("///", "; ", regex=False).str.strip()
    return ann


def _replicate_consistency(sub, t_mod):
    """Estimate day-to-day replicate consistency for repeated phases."""
    uniq = sorted(np.unique(t_mod))
    phase_to_cols = {ph: np.where(t_mod == ph)[0] for ph in uniq}
    # require exactly 2 replicates per phase for this metric
    valid = [ph for ph in uniq if len(phase_to_cols[ph]) == 2]
    if len(valid) < 6:
        return np.full(sub.shape[0], np.nan)

    c1 = np.array([phase_to_cols[ph][0] for ph in valid])
    c2 = np.array([phase_to_cols[ph][1] for ph in valid])

    out = np.full(sub.shape[0], np.nan)
    for i in range(sub.shape[0]):
        a = sub[i, c1]
        b = sub[i, c2]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 6:
            continue
        if np.nanstd(a[ok]) < 1e-12 or np.nanstd(b[ok]) < 1e-12:
            continue
        out[i] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/Users/hogenesch/Downloads/GSE11923")
    ap.add_argument("--topn", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--perm-backend", choices=("python", "rust"), default="python")
    ap.add_argument(
        "--compare-perm-backends",
        action="store_true",
        help="Time permutation p-values with Python and Rust backends on the same subset.",
    )
    ap.add_argument(
        "--rust-threads",
        type=int,
        default=0,
        help="Thread count for Rust backend (0 lets Rust decide).",
    )
    args = ap.parse_args()

    w = Path(args.workdir)
    series = w / "GSE11923_series_matrix.txt.gz"
    annot = w / "GPL1261.annot.gz"

    expr_out = w / "GSE11923_expression_CT.tsv"
    expr = build_expression_ct(series, expr_out)

    val_cols = [c for c in expr.columns if c.startswith("CT")]
    X = expr[val_cols].to_numpy(dtype=float)
    var = np.nanvar(X, axis=1)
    idx = np.argsort(var)[-args.topn :][::-1]

    t = np.array([float(c.replace("CT", "")) for c in val_cols])
    t_mod = np.mod(t, 24.0)

    sub = X[idx]

    # design-robust scoring (adaptive harmonics + fitted amplitude)
    res = jtk_scan(sub, t_mod, period_grid=(24.0,), harmonics_grid=None, min_obs=24, design_robust=True)
    out = pd.DataFrame(res)
    out.insert(0, "ID_REF", expr.iloc[idx, 0].astype(str).to_numpy())
    out["abs_tau"] = out["tau"].abs()

    # permutation p-values + BH q-values
    score_fn = lambda y, tt: jtk_score_series(y, tt, period=24.0, harmonics_grid=None, min_obs=24, design_robust=True)

    if args.compare_perm_backends:
        t0 = time.perf_counter()
        p_python = permutation_pvalues_scan(sub, t_mod, score_fn=score_fn, n_perm=args.n_perm, random_seed=42)
        dt_python = time.perf_counter() - t0
        print(f"Permutation timing (python): {dt_python:.3f}s")

        if has_rust_permutation_backend():
            t0 = time.perf_counter()
            p_rust = permutation_pvalues_scan_rust(
                sub,
                t_mod,
                period=24.0,
                harmonics_grid=None,
                min_obs=24,
                design_robust=True,
                n_perm=args.n_perm,
                random_seed=42,
                n_threads=args.rust_threads,
            )
            dt_rust = time.perf_counter() - t0
            speedup = dt_python / dt_rust if dt_rust > 0 else np.inf
            print(f"Permutation timing (rust): {dt_rust:.3f}s")
            print(f"Permutation speedup (python/rust): {speedup:.2f}x")
            out["perm_pvalue_python"] = p_python
            out["perm_pvalue_rust"] = p_rust
        else:
            print("Permutation timing (rust): unavailable (extension not installed)")
            out["perm_pvalue_python"] = p_python

    if args.perm_backend == "python":
        out["perm_pvalue"] = permutation_pvalues_scan(sub, t_mod, score_fn=score_fn, n_perm=args.n_perm, random_seed=42)
    else:
        out["perm_pvalue"] = permutation_pvalues_scan_rust(
            sub,
            t_mod,
            period=24.0,
            harmonics_grid=None,
            min_obs=24,
            design_robust=True,
            n_perm=args.n_perm,
            random_seed=42,
            n_threads=args.rust_threads,
        )

    out["qvalue"] = bh_qvalues(out["perm_pvalue"].to_numpy())

    # replicate consistency (two-day repeated phases)
    out["replicate_corr"] = _replicate_consistency(sub, t_mod)

    # compatibility aliases for previous output names
    out["BaselineMedian"] = out["baseline_median"]
    out["Magnitude_P90_P10"] = out["magnitude_p90_p10"]
    out["Magnitude_Max_Min"] = out["magnitude_max_min"]
    out["RelativeAmplitude"] = out["relative_amplitude"]
    out["FittedMagnitude"] = out["fitted_magnitude"]
    out["RelativeFittedAmplitude"] = out["relative_fitted_amplitude"]

    # Option A ranking: gate by q-value, then magnitude-first ranking
    q_gate = 0.20
    out["passes_q_gate"] = out["qvalue"] <= q_gate

    # design-aware magnitude column: sparse designs prefer fitted magnitude
    n_unique = len(np.unique(t_mod))
    rank_mag_col = "FittedMagnitude" if n_unique <= 12 else "Magnitude_P90_P10"
    out["rank_magnitude_col"] = rank_mag_col

    out = out.sort_values(
        ["passes_q_gate", rank_mag_col, "RelativeAmplitude", "qvalue"],
        ascending=[False, False, False, True],
    )

    ann = read_platform_annotation(annot)
    out = out.merge(ann, on="ID_REF", how="left")

    final = w / "GSE11923_top1000var_jtkcycle2_preview_with_symbols_amplitude.tsv"
    out.to_csv(final, sep="\t", index=False)
    print(f"Wrote: {expr_out}")
    print(f"Wrote: {final}")


if __name__ == "__main__":
    main()
