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
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from jtk_cycle_redo.core import jtk_scan
from jtk_cycle_redo.stats import bh_qvalues


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/Users/hogenesch/Downloads/GSE11923")
    ap.add_argument("--topn", type=int, default=1000)
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

    res = jtk_scan(X[idx], t_mod, period_grid=(24.0,), harmonics_grid=None, min_obs=24, design_robust=True)
    out = pd.DataFrame(res)
    out.insert(0, "ID_REF", expr.iloc[idx, 0].astype(str).to_numpy())
    out["abs_tau"] = out["tau"].abs()
    out["qvalue"] = bh_qvalues(out["pvalue"].to_numpy())

    # amplitude terms
    sub = X[idx]
    p10 = np.nanpercentile(sub, 10, axis=1)
    p90 = np.nanpercentile(sub, 90, axis=1)
    median = np.nanmedian(sub, axis=1)
    ptp = np.nanmax(sub, axis=1) - np.nanmin(sub, axis=1)
    robust_mag = p90 - p10
    rel_amp = robust_mag / np.where(np.abs(median) < 1e-8, np.nan, np.abs(median))

    out["BaselineMedian"] = median
    out["Magnitude_P90_P10"] = robust_mag
    out["Magnitude_Max_Min"] = ptp
    out["RelativeAmplitude"] = rel_amp
    out["RelAmpRank"] = out["RelativeAmplitude"].rank(method="min", ascending=False)

    ann = read_platform_annotation(annot)
    out = out.merge(ann, on="ID_REF", how="left")

    # Option A ranking: gate by q-value, then magnitude-first ranking
    q_gate = 0.20
    out["passes_q_gate"] = out["qvalue"] <= q_gate
    out = out.sort_values(
        ["passes_q_gate", "Magnitude_P90_P10", "RelativeAmplitude", "qvalue"],
        ascending=[False, False, False, True],
    )

    final = w / "GSE11923_top1000var_jtkcycle2_preview_with_symbols_amplitude.tsv"
    out.to_csv(final, sep="\t", index=False)
    print(f"Wrote: {expr_out}")
    print(f"Wrote: {final}")


if __name__ == "__main__":
    main()
