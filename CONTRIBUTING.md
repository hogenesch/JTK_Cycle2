# Contributing to JTK_Cycle2

This project values biological utility over leaderboard optimization.

## Ground rules
- Keep changes reproducible.
- Explain biological implications, not only statistics.
- Prefer robust defaults for realistic designs (12/24/48 points).

## Pull request requirements
- Link to issue/problem statement
- Reproducible run commands
- Benchmark deltas using `BENCHMARK_PROTOCOL.md`
- Short rationale for ranking changes

## Design principles to preserve
- q-gated magnitude-first ranking
- waveform-flexible scoring
- missing-data tolerance
- transparent output columns

## Code quality
- Add/update tests for every behavior change
- Keep scripts deterministic where possible (fixed seeds)
- Avoid hidden magic constants without comments

## Scope notes
- Multi-period (12h/8h) support is optional extension, not default path
- Human phase-jitter handling is deferred to human-data phase
