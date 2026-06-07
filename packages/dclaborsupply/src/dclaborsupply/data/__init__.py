"""Package-native engine-ready data loading.

Country-general construction of the core PrecomputedData containers from harmonised
engine-ready parquet files or DataFrames. Imports only stdlib + numpy + pandas (no jax,
no MNL/app/EUROMOD). See ``loader`` for the read-vs-recompute reproduction contract.
"""
from __future__ import annotations

from .loader import (
    load_singles,
    load_couples,
    load_engine_ready_stem,
    DataSource,
)

__all__ = ["load_singles", "load_couples", "load_engine_ready_stem", "DataSource"]
