# Reproduction package

This package implements the exact depth-one multi-angle QAOA observable map,
Jacobian-vector and transpose-Jacobian actions, authenticated Stanford Gset
loading, source-span diagnostics, task-only and complement-robust action
factors, dense audit baselines, and generated manuscript evidence.

## Installation

```bash
python3 -m venv /tmp/observable-subspaces-venv
/tmp/observable-subspaces-venv/bin/python -m pip install -e "./code[test]"
```

## Data authentication and replay

```bash
RSQAOA_GSET_CACHE=/tmp/observable-subspaces-gset \
  /tmp/observable-subspaces-venv/bin/python code/scripts/download_data.py
RSQAOA_GSET_CACHE=/tmp/observable-subspaces-gset \
  /tmp/observable-subspaces-venv/bin/python code/scripts/reproduce.py
```

Every registered file is downloaded only from Stanford and rejected unless
its SHA-256 digest matches. Replay reconstructs all neighborhoods and task
banks, checks span separation, independently compares action and dense
objectives, evaluates every method and rank, and verifies the locked evidence
manifest. Host-dependent timing values are stored with environment metadata
but excluded from deterministic numeric comparison.

To regenerate all records, timings, tables, and plots:

```bash
RSQAOA_GSET_CACHE=/tmp/observable-subspaces-gset \
  /tmp/observable-subspaces-venv/bin/python code/scripts/make_figures.py
```

## Tests and release validation

```bash
RSQAOA_GSET_CACHE=/tmp/observable-subspaces-gset \
  /tmp/observable-subspaces-venv/bin/python -m pytest code/tests -q
/tmp/observable-subspaces-venv/bin/python code/scripts/validate_release.py
```

Tests cover provenance metadata, revised out-of-span task banks, the rejected
in-span counterexample, the leave-one-out radius, task and robust action-factor
identities, exact VJP counts, the target-span triangle bound, sensitivity
adjointness, locked replay, and manuscript asset closure.
