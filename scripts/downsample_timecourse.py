#!/usr/bin/env python3
"""Downsample dense CT time-course matrix to coarser designs.

Input: TSV with columns ID_REF, CT18..CT65 (or similar CT columns)
Output: TSV with selected CT columns by stride (e.g., 2h, 4h)
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Input expression TSV')
    ap.add_argument('--output', required=True, help='Output expression TSV')
    ap.add_argument('--step', type=int, required=True, help='Sampling step in hours (e.g., 1,2,4)')
    ap.add_argument('--start', type=int, default=None, help='Optional starting CT hour; default uses first available')
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep='\t')
    ct_cols = [c for c in df.columns if re.fullmatch(r'CT\d+', str(c))]
    if not ct_cols:
        raise SystemExit('No CT columns found')

    ct_vals = sorted(int(c.replace('CT', '')) for c in ct_cols)
    start = ct_vals[0] if args.start is None else args.start
    keep_vals = set([v for v in ct_vals if (v - start) % args.step == 0])

    keep_cols = ['ID_REF'] + [f'CT{v}' for v in ct_vals if v in keep_vals and f'CT{v}' in df.columns]
    out = df[keep_cols].copy()
    out.to_csv(args.output, sep='\t', index=False)
    print(f'Wrote {args.output} with {len(keep_cols)-1} timepoints')


if __name__ == '__main__':
    main()
