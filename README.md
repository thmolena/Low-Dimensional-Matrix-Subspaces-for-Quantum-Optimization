# Complement-Robust Sensitivity Subspaces for Repeated Quantum Objectives

This repository contains the manuscript and deterministic reproduction for a complement-robust sensitivity subspace. The method augments a development task moment with an outcome-free innovation radius on the orthogonal complement of the observed task span.

Eight checksum-authenticated Stanford Gset graphs supply ten-vertex depth-one multi-angle QAOA objectives. At rank four, mean held-out discarded-gradient energy decreases from 0.0590 to 0.0381 for full objectives and from 0.8989 to 0.4396 for localized contrasts. The exact robust construction uses more transpose-Jacobian products and is slower than matched dense differentiation on every tested instance.

## Reproduction

    python -m venv .venv
    .venv/bin/pip install -e 'code[test]'
    .venv/bin/observable-subspaces-reproduce --offline --verify
    .venv/bin/pytest code/tests
    tectonic main.tex

The locked evidence records source hashes, selected neighborhoods, task factors, span residuals, innovation radii, rank frontiers, adjointness checks, derivative counts, action--dense agreement, and repeated timings. The experiment concerns ten-qubit noiseless state-vector objectives.
