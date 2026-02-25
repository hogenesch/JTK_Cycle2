#!/usr/bin/env python3
"""Apply q-gate + magnitude-first ranking to result tables."""

from __future__ import annotations
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--qgate', type=float, default=0.20)
    ap.add_argument('--magcol', default='Magnitude_P90_P10')
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep='\t')
    if 'qvalue' not in df.columns:
        raise SystemExit('Input must contain qvalue column')
    if args.magcol not in df.columns:
        raise SystemExit(f'Missing magnitude column: {args.magcol}')

    df['passes_q_gate'] = df['qvalue'] <= args.qgate
    df = df.sort_values(
        ['passes_q_gate', args.magcol, 'RelativeAmplitude', 'qvalue'],
        ascending=[False, False, False, True],
    )
    df.to_csv(args.output, sep='\t', index=False)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
