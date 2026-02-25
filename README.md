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
- ✅ Core Kendall tau + phase-shift scoring
- ✅ Synthetic rhythmic data generator
- ✅ Basic tests
- ⏭ Next: empirical p-values / FDR calibration against null permutations
