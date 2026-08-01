# Data provenance

The numerical study uses eight MaxCut benchmark files from the Stanford Gset
collection: G1, G6, G11, G14, G18, G22, G32, and G43.

Original source: https://web.stanford.edu/~yyye/yyye/Gset/

The software downloads each file into an external cache. A file is accepted
only when its SHA-256 digest matches `checksums.sha256`. The repository does
not redistribute Gset bytes.

```bash
python -m pip install -e code
RSQAOA_GSET_CACHE=/tmp/observable-subspaces-gset \
  python code/scripts/download_data.py
```

One deterministic ten-vertex neighborhood is extracted from each graph.
Even-vertex development and odd-vertex held-out task vectors are generated
from the original signed edge weights under full-objective and localized-
contrast protocols. The code also reconstructs the rejected legacy split only
to verify its zero source-span residual. The complete construction is in the
manuscript and `src/hqml_observable_subspaces/authentic_study.py`.
