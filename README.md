# jtk-cycle-redo (Python)

A modern Python reimplementation scaffold for JTK_CYCLE-style rhythmicity detection.

## Goals
- Recreate core JTK_CYCLE behavior for periodic signal detection
- Keep implementation transparent and testable
- Provide reproducible synthetic benchmarks

## Quick start

```bash
cd jtk-cycle-redo
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
python examples/run_demo.py
```

## Current status
- ✅ Project scaffold
- ✅ Waveform-flexible periodic scoring (multi-harmonic, not fixed cosine-only)
- ✅ Missing-data tolerant scoring (`nan`-aware + minimum-observation control)
- ✅ Amplitude-first metrics (baseline, robust magnitude, relative amplitude)
- ✅ Synthetic rhythmic data generator
- ✅ Basic tests
- ✅ GEO preview script with probe→symbol mapping and amplitude columns (`scripts/geo_gse11923_preview.py`)
- ✅ Permutation-based p-values + BH q-values in GEO preview (`--n-perm`)
- ✅ Magnitude-first, q-gated ranking (Option A)
