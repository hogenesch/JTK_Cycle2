# Benchmark Protocol (Anti-Noise / Anti-Hype)

Purpose: make low-quality contributions obvious and keep scientifically meaningful changes.

## Required for any method change
1. **Reproducible command block** (exact commands + seed)
2. **Dataset declaration** (source + preprocessing)
3. **Metric deltas** vs baseline commit
4. **Biological interpretation** (not just numeric gains)

## Baseline datasets
- GSE11923 dense (48 points)
- Derived downsample sets (24-point, 12-point)

## Core benchmark metrics
- Recovery of canonical rhythmic genes (e.g., Dbp/Arntl class)
- Rank stability under downsampling
- Magnitude-priority behavior (top ranks should have meaningful wave size)
- Missing-data robustness (controlled masking tests)
- Replicate consistency effects (`replicate_corr` behavior)

## Reporting template (required in PR description)
- Change summary (1 paragraph)
- Why this should improve biology-facing decisions
- Benchmark table (before vs after)
- Failure modes / tradeoffs
- Reproducibility commands

## Red flags (reject unless justified)
- Better p-values but lower biological actionability
- Gains only on dense designs, collapse on 12-point designs
- No fixed random seed / irreproducible result
- No explanation for changed top-ranked genes
