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
from .draws_prep import (
    COUPLES_GRID_NOTE,
    POST_MANAGED_FIELDS,
    canonicalize_choice_state,
    canonicalize_post_draws,
    assert_choice_state_consistent,
)
from .pricing import de_earnings_policy
from .engine_ready import (
    EngineReadyResult,
    assemble,
    assemble_singles,
    assemble_couples,
    aggregate_consumption,
    apply_consumption_floor,
    restore_cluster_id,
    loc4_one_hots,
    DCM_MIN_POSITIVE,
    TOTAL_LEISURE_HOURS,
)

__all__ = [
    "DE_CONFIG",
    "prepare_de_2017",
    "classify_households",
    "collapse_loc_to_loc4",
    "compute_is_worker",
    "reshape_couples_to_wide",
    "COUPLES_GRID_NOTE",
    "POST_MANAGED_FIELDS",
    "canonicalize_choice_state",
    "canonicalize_post_draws",
    "assert_choice_state_consistent",
    "de_earnings_policy",
    "EngineReadyResult",
    "assemble",
    "assemble_singles",
    "assemble_couples",
    "aggregate_consumption",
    "apply_consumption_floor",
    "restore_cluster_id",
    "loc4_one_hots",
    "DCM_MIN_POSITIVE",
    "TOTAL_LEISURE_HOURS",
]
