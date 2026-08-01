"""Exact CPU observable maps and matrix-free QAOA sensitivity products."""

from .circuits import (
    MaxCutProblem,
    cut_table,
    cut_value,
    edge_expectations,
    n_params,
    statevector,
    unpack,
)
from .operator import (MatrixFreeSensitivity, QAOASensitivity, OpCounts,
                       adjointness_gap)
__version__ = "0.5.0"

__all__ = [
    "MaxCutProblem", "cut_table", "cut_value", "edge_expectations", "n_params",
    "statevector", "unpack",
    "MatrixFreeSensitivity", "QAOASensitivity", "OpCounts", "adjointness_gap",
    "__version__",
]
