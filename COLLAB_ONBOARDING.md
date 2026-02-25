# JTK_Cycle2 — Quick Onboarding (Marc + Andrew)

Welcome. This repo is building a **biologically practical** rhythm detector, not a straight JTK port.

## Start here (10 minutes)
1. Read `README.md`
2. Read `data/README.md`
3. Run GEO preview:
   ```bash
   python scripts/geo_gse11923_preview.py --workdir ~/Downloads/GSE11923 --topn 1000 --n-perm 200
   ```

## Current principles
- Minimize false negatives
- Magnitude-first prioritization (not p-value-only)
- Waveform flexibility (not fixed cosine only)
- Missing-data tolerance
- Design robustness for 12/24/48-point studies

## Current ranking policy (Option A)
- Gate: `qvalue <= 0.20`
- Rank by magnitude first, then relative amplitude, then q

## Key outputs
- `perm_pvalue`, `qvalue`
- `Magnitude_P90_P10`, `FittedMagnitude`
- `RelativeAmplitude`, `RelativeFittedAmplitude`
- `replicate_corr`, `missing_frac`, `n_obs`

## Current scope boundary
- CT-focused framework (not ZT)
- Human chronotype/phase-jitter modeling deferred for later phase

## What feedback is most useful now
- Does ranking prioritize biologically actionable genes?
- Does behavior stay stable under 48→24→12 downsampling?
- Are replicate consistency and magnitude metrics intuitive/useful?
