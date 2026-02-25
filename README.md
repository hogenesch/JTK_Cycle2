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

## Compare Python vs Rust permutation runtime

Use the GEO preview script with backend flags:

```bash
# Python backend (baseline)
PYTHONPATH=src python3 scripts/geo_gse11923_preview.py \
  --workdir /path/to/GSE11923 \
  --perm-backend python

# Rust backend (requires jtk_cycle_redo._rust_stats extension)
PYTHONPATH=src python3 scripts/geo_gse11923_preview.py \
  --workdir /path/to/GSE11923 \
  --perm-backend rust

# Side-by-side timing on the same input subset
PYTHONPATH=src python3 scripts/geo_gse11923_preview.py \
  --workdir /path/to/GSE11923 \
  --compare-perm-backends
```

### Build the Rust extension

The Rust backend module is `jtk_cycle_redo._rust_stats` under `rust/`.

```bash
cd /home/asu/Science/JTK_Cycle2
source .venv/bin/activate
python -m pip install maturin
maturin develop --manifest-path rust/Cargo.toml
```

Then verify:

```bash
python -c "from jtk_cycle_redo import has_rust_permutation_backend; print(has_rust_permutation_backend())"
```

### Grid benchmark (CSV output)

Run a reproducible timing grid over `topn` and `n_perm`:

```bash
cd /home/asu/Science/JTK_Cycle2
source .venv/bin/activate
PYTHONPATH=src python3 scripts/benchmark_perm_backends.py \
  --workdir data/GSE11923 \
  --topn-list 100,500,1000 \
  --n-perm-list 50,200 \
  --repeats 3
```

Output CSV (default):
- `data/GSE11923/perm_backend_benchmark.csv`
