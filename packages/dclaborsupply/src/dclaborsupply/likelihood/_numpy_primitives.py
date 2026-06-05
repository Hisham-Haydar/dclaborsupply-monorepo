"""NumPy engine primitives (lifted, migration matrix Wave 1.3).

PrecomputedData containers + grouped log-sum-exp, lifted byte-faithfully from
MNL/scripts/enhanced/estimation_utils.py. Copy + import adaptation only; no math
change. Zero MNL/old-repo imports; numba is optional (pure-NumPy fallback).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Optional Numba acceleration (NOT a base dependency). Pure-NumPy fallback when
# numba is absent, so `import dclaborsupply` stays light. Lifted verbatim from
# estimation_utils.py.
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    numba = None
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Epsilon for numerical stability (from estimation_utils).
EPS = 1e-12


@dataclass
class PrecomputedDataSingles:
    """
    All arrays for singles estimation, pre-extracted for vectorized computation.

    All arrays are (n_obs,) unless otherwise noted.
    Arrays are NumPy arrays for performance.
    """
    # Core utility components
    consumption: np.ndarray  # Raw consumption (before normalization/transform)
    leisure: np.ndarray      # Raw leisure
    log_c: np.ndarray        # log(consumption) - for Box-Cox derivatives
    log_l: np.ndarray        # log(leisure)

    # Demographics (preference shifters)
    age_norm: np.ndarray     # Demeaned age
    age_norm2: np.ndarray    # Age squared
    n_children: np.ndarray   # Number of children (0 for males, actual for females)
    educL: np.ndarray        # Low education dummy
    educM: np.ndarray        # Medium education dummy
    educH: np.ndarray        # High education dummy    # Hours opportunity shifters
    working: np.ndarray      # 1 if hours > 0, else 0
    working_pt1: np.ndarray  # Part-time 1 indicator (~20h focal)
    working_pt2: np.ndarray  # Part-time 2 indicator (~30h focal)
    working_ft: np.ndarray   # Full-time indicator (~40h focal)
    working_lh: np.ndarray   # Long-hours indicator (hours in [44.5,70]); F35 ref carries no flag
    gsur: np.ndarray         # Group-specific unemployment rate (0 if missing)
      # NEW: Interaction variables for hours opportunity
    female: np.ndarray       # 1 if female, 0 if male
    in_couple: np.ndarray    # 1 if in couple, 0 if single (always 0 for singles)
    drgn1: np.ndarray        # Region code (1-8) - kept for backward compatibility
    
    # Region dummies (Île-de-France = region 1 is reference)
    reg2: np.ndarray         # Region 2 dummy
    reg3: np.ndarray         # Region 3 dummy
    reg4: np.ndarray         # Region 4 dummy
    reg5: np.ndarray         # Region 5 dummy
    reg6: np.ndarray         # Region 6 dummy
    reg7: np.ndarray         # Region 7 dummy
    reg8: np.ndarray         # Region 8 dummy

    # Household access shifters: urbanisation (D5; rural drgru = reference) + year indicators
    drgur: np.ndarray        # Urban (db100==1)
    drgmd: np.ndarray        # Middle density (db100==2)
    drgru: np.ndarray        # Rural (db100==3) = reference
    year_2015_indicator: np.ndarray
    year_2017_indicator: np.ndarray

    # Wage opportunity (if vw or loc_empirical)
    log_wage: Optional[np.ndarray]  # log(wage) for workers, 0 for non-workers
    pexp_years: Optional[np.ndarray]  # Potential experience in years
    pexp_years2: Optional[np.ndarray]  # Experience squared

    # Occupation (if loc_empirical)
    loc4: Optional[np.ndarray]  # 4-category occupation code
    loc4_1: Optional[np.ndarray]  # Occupation group 1 dummy
    loc4_2: Optional[np.ndarray]  # Occupation group 2 dummy
    loc4_3: Optional[np.ndarray]  # Occupation group 3 dummy
    loc4_4: Optional[np.ndarray]  # Occupation group 4 dummy

    # Normalization & prior
    prior: np.ndarray        # Importance sampling prior
    c_scale: float           # Consumption normalization constant
    l_scale: float           # Leisure normalization constant    # Group structure (for softmax)
    group_ids: np.ndarray    # Person IDs (unique identifiers)
    group_starts: np.ndarray # Index where each group starts
    group_ends: np.ndarray   # Index where each group ends (exclusive)
    n_groups: int            # Number of individuals
    n_obs: int               # Total observations (n_groups × n_draws)
    
    # Chosen alternative indicator (for GAMSPy estimation)
    actual_choice: np.ndarray  # 1.0 if this is the observed choice, 0.0 otherwise

    # Cluster ids for clustered sandwich SE (one per choice-set group, aligned to group_starts)
    # Contains idorighh for each group; shape (n_groups,).
    cluster_ids: np.ndarray

    # Metadata
    is_male: bool            # True if male dataset, False if female


@dataclass
class PrecomputedDataCouples:
    """
    All arrays for couples estimation (wide format).

    All arrays are (n_obs,) unless otherwise noted.
    """
    # Core utility components
    # NOTE: Consumption is HOUSEHOLD-LEVEL (sum of male + female disposable income)
    # normalized_consumption_couples = (ils_dispy_male + ils_dispy_female) / mean(ils_dispy_male + ils_dispy_female)
    consumption: np.ndarray    # Household consumption (normalized sum)
    log_c: np.ndarray          # log(consumption)

    # Male leisure
    leisure_male: np.ndarray
    log_l_male: np.ndarray

    # Female leisure
    leisure_female: np.ndarray
    log_l_female: np.ndarray

    # Male demographics
    age_norm_male: np.ndarray
    age_norm2_male: np.ndarray
    educL_male: np.ndarray
    educM_male: np.ndarray
    educH_male: np.ndarray

    # Female demographics
    age_norm_female: np.ndarray
    age_norm2_female: np.ndarray
    n_children: np.ndarray      # Only for female
    educL_female: np.ndarray
    educM_female: np.ndarray
    educH_female: np.ndarray

    # Male hours opportunity
    working_male: np.ndarray
    working_pt1_male: np.ndarray
    working_pt2_male: np.ndarray
    working_ft_male: np.ndarray
    working_lh_male: np.ndarray   # Long-hours indicator (hours in [44.5,70]); F35 ref carries no flag
    gsur_male: np.ndarray    # Female hours opportunity
    working_female: np.ndarray
    working_pt1_female: np.ndarray
    working_pt2_female: np.ndarray
    working_ft_female: np.ndarray
    working_lh_female: np.ndarray  # Long-hours indicator (hours in [44.5,70]); F35 ref carries no flag
    gsur_female: np.ndarray    # NEW: Interaction variables for hours opportunity (couples)
    female_male: np.ndarray   # Always 0 for male partner
    female_female: np.ndarray # Always 1 for female partner
    in_couple_male: np.ndarray  # Always 1 for couples
    in_couple_female: np.ndarray  # Always 1 for couples
    drgn1_male: np.ndarray    # Region code (household-level)
    drgn1_female: np.ndarray  # Same as drgn1_male (household-level)
    
    # Region dummies (Île-de-France = region 1 is reference)
    reg2: np.ndarray         # Region 2 dummy (household-level)
    reg3: np.ndarray         # Region 3 dummy
    reg4: np.ndarray         # Region 4 dummy
    reg5: np.ndarray         # Region 5 dummy
    reg6: np.ndarray         # Region 6 dummy
    reg7: np.ndarray         # Region 7 dummy
    reg8: np.ndarray         # Region 8 dummy

    # Household access shifters: urbanisation (D5; rural drgru = reference) + year indicators
    drgur: np.ndarray        # Urban (db100==1), household-level
    drgmd: np.ndarray        # Middle density (db100==2), household-level
    drgru: np.ndarray        # Rural (db100==3) = reference, household-level
    year_2015_indicator: np.ndarray
    year_2017_indicator: np.ndarray

    # Male wage opportunity (if vw or loc_empirical)
    log_wage_male: Optional[np.ndarray]
    pexp_years_male: Optional[np.ndarray]
    pexp_years2_male: Optional[np.ndarray]
    loc4_male: Optional[np.ndarray]
    loc4_1_male: Optional[np.ndarray]
    loc4_2_male: Optional[np.ndarray]
    loc4_3_male: Optional[np.ndarray]
    loc4_4_male: Optional[np.ndarray]

    # Female wage opportunity
    log_wage_female: Optional[np.ndarray]
    pexp_years_female: Optional[np.ndarray]
    pexp_years2_female: Optional[np.ndarray]
    loc4_female: Optional[np.ndarray]
    loc4_1_female: Optional[np.ndarray]
    loc4_2_female: Optional[np.ndarray]
    loc4_3_female: Optional[np.ndarray]
    loc4_4_female: Optional[np.ndarray]

    # Normalization & prior
    prior: np.ndarray
    c_scale: float
    l_scale: float    # Group structure
    group_ids: np.ndarray
    group_starts: np.ndarray
    group_ends: np.ndarray
    n_groups: int
    n_obs: int
    
    # Chosen alternative indicator (for GAMSPy estimation)
    actual_choice: np.ndarray  # 1.0 if this is the observed choice, 0.0 otherwise

    # Cluster ids for clustered sandwich SE (one per choice-set group, aligned to group_starts)
    # Contains idorighh for each group; shape (n_groups,).
    cluster_ids: np.ndarray


# ==============================================================================
# Grouped log-sum-exp
# ==============================================================================

# Numba-accelerated log-sum-exp (if available)
@jit(nopython=True, parallel=True, cache=True)
def _compute_lse_numba(V, group_starts, group_ends):
    """
    Numba JIT-compiled log-sum-exp per group (10-50x faster than pure Python).
    """
    n_groups = len(group_starts)
    lse = np.zeros(n_groups)
    
    for i in prange(n_groups):
        start = group_starts[i]
        end = group_ends[i]
        
        # Find max for numerical stability
        max_V = V[start]
        for j in range(start + 1, end):
            if V[j] > max_V:
                max_V = V[j]
        
        # Compute sum of exp(V - max_V)
        sum_exp = 0.0
        for j in range(start, end):
            sum_exp += np.exp(V[j] - max_V)
        
        lse[i] = max_V + np.log(sum_exp)
    
    return lse


def compute_log_sum_exp_by_group(
    V: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray
) -> np.ndarray:
    """
    Numerically stable log-sum-exp per group.

    For each group i:
        max_V_i = max(V[start_i:end_i])
        lse_i = max_V_i + log(Σ_j exp(V_j - max_V_i))

    Parameters
    ----------
    V : np.ndarray, shape (n_obs,)
        Value function for all alternatives
    group_starts : np.ndarray, shape (n_groups,)
        Starting index for each group
    group_ends : np.ndarray, shape (n_groups,)
        Ending index (exclusive) for each group

    Returns
    -------
    lse : np.ndarray, shape (n_groups,)
        Log-sum-exp for each group
    """
    # Use Numba-accelerated version if available
    if HAS_NUMBA:
        return _compute_lse_numba(V, group_starts, group_ends)
    
    # Fallback to pure Python/NumPy
    n_groups = len(group_starts)
    lse = np.zeros(n_groups)

    for i in range(n_groups):
        start, end = group_starts[i], group_ends[i]
        
        # Check for empty groups (should not happen in valid data)
        if start >= end:
            raise ValueError(
                f"Group {i} is empty (start={start}, end={end}). "
                "This indicates a data preprocessing error. "
                "Each household should have at least one alternative."
            )
        
        V_group = V[start:end]
        max_V = V_group.max()
        lse[i] = max_V + np.log(np.sum(np.exp(V_group - max_V)))

    return lse
