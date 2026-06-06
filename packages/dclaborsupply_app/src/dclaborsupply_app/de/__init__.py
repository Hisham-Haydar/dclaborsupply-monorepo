"""Germany (DE) country adapter for dclaborsupply.

DE 2017 data-prep: transforms the EUROMOD microdata ``DE_2017_a2`` into processed
singles / couples for a minimal DE labour-supply smoke test, mirroring the FR
validated-sample definition (see ``data_prep`` for the frozen eligibility rule).

Self-contained: depends only on pandas + numpy. Does NOT import the France adapter
or any MNL source. Country-specific conventions (no yem00/yemxp split; identity
yemse = yem + yse; region fields constant; ISCO-1d occupation) are documented in
``data_prep``.
"""
from __future__ import annotations

from .data_prep import (
    DE_CONFIG,
    prepare_de_2017,
    classify_households,
    collapse_loc_to_loc4,
    compute_is_worker,
    reshape_couples_to_wide,
)

__all__ = [
    "DE_CONFIG",
    "prepare_de_2017",
    "classify_households",
    "collapse_loc_to_loc4",
    "compute_is_worker",
    "reshape_couples_to_wide",
]
