"""
==============================================================================
Vectorized GAMSPy-based MNL Estimation for RURO Pipeline
==============================================================================
OPTIMIZED implementation using GAMSPy indexed operations for production use.

Key improvements over gamspy_estimation.py:
- Uses GAMSPy Sets and Parameters (indexed operations)
- 3-5x faster expression building (A→B stage)
- 2-4x faster GAMS compilation (B→C stage)
- Much smaller GAMS files (10-50 MB vs 200-500 MB)
- Scalable to occupation choice (400 alternatives)

Architecture:
- Data organized as 2D arrays (individuals × alternatives)
- Utility functions built as indexed expressions
- Vectorized log-sum-exp using Sum operator
- Compatible with existing YAML specifications

Author: Enhanced RURO Pipeline
Created: 2026-01-28
==============================================================================
"""

from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    # Type hints only. These are strings at runtime (`from __future__ import
    # annotations`), so importing them here at runtime is avoided — it would pull
    # numba (via _numpy_primitives) and yaml (via spec.parser) and bloat
    # `import dclaborsupply.solvers`. No runtime isinstance/construction uses them.
    from dclaborsupply.likelihood._numpy_primitives import (
        PrecomputedDataSingles,
        PrecomputedDataCouples,
    )
    from dclaborsupply.spec.parser import EstimationSpec

# GAMSPy is an OPTIONAL extra, imported LAZILY (never at module import). The
# module-level symbols stay None until _load_gamspy() binds them inside an
# estimation entry; with `from __future__ import annotations` the gamspy type
# hints below are strings, so this module imports cleanly with gamspy absent.
# The GAMSPy expression-constraint builder is inlined at the bottom of this file
# (lifted from expression_constraints.py — solver layer per the Wave-1.3 boundary).
HAS_GAMSPY = importlib.util.find_spec("gamspy") is not None
Container = Model = Variable = Set = Parameter = Equation = Alias = GamsSum = Options = None
gp_exp = gp_log = None


def _load_gamspy() -> None:
    """Import GAMSPy lazily and bind the module-level symbols (idempotent).

    Raises a documented ImportError pointing at the optional extra when gamspy is
    not installed. ``gams`` is only pulled in transitively by gamspy here, so it
    too stays unimported until this runs.
    """
    global Container, Model, Variable, Set, Parameter, Equation, Alias, GamsSum, Options
    global gp_exp, gp_log
    if Container is not None:
        return
    try:
        from gamspy import (
            Container as _Container, Model as _Model, Variable as _Variable,
            Set as _Set, Parameter as _Parameter, Equation as _Equation,
            Alias as _Alias, Sum as _Sum, Options as _Options,
        )
        from gamspy.math import exp as _gp_exp, log as _gp_log
    except ImportError as exc:  # pragma: no cover - exercised only without gamspy
        raise ImportError(
            "GAMSPy solver requires gamspy (and a GAMS install). "
            "Install with `pip install dclaborsupply[gamspy]`."
        ) from exc
    Container, Model, Variable, Set = _Container, _Model, _Variable, _Set
    Parameter, Equation, Alias = _Parameter, _Equation, _Alias
    GamsSum, Options = _Sum, _Options
    gp_exp, gp_log = _gp_exp, _gp_log


# ==============================================================================
# Constants
# ==============================================================================

# Small epsilon for numerical stability
LOG_EPS = 1e-12

# Solver mapping
SOLVER_MAP = {
    "conopt": "conopt",
    "ipopt": "ipopt",
    "minos": "minos",
    "snopt": "snopt",
}

# Group to parameter suffix mapping (4-group architecture)
SUFFIX_MAP = {
    "singles_male": "_sm",
    "singles_female": "_sf",
    "singles_pooled": "_sm",
    "couples_male": "_m",
    "couples_female": "_f",
    "couples_household": "",  # No suffix for household-level parameters
}


def get_param_name(base_name: str, group: str, param_vars: dict) -> str:
    """
    Resolve parameter name for a group-specific context.

    Strategy:
    1) Try group-specific suffix (if any)
    2) Fall back to base name
    """
    suffix = SUFFIX_MAP.get(group, "")
    if suffix:
        param_with_suffix = f"{base_name}{suffix}"
        if param_with_suffix in param_vars:
            return param_with_suffix
    if base_name in param_vars:
        return base_name
    tried_names = [f"{base_name}{suffix}", base_name] if suffix else [base_name]
    raise ValueError(
        f"Parameter '{base_name}' for group '{group}' not found. "
        f"Tried: {', '.join(tried_names)}"
    )


def _normalize_interaction_terms(interaction_cfg: Any) -> List[str]:
    """
    Normalize a YAML interaction config to a flat list of variable names.

    Supported formats:
    - None
    - "working"
    - ["working", "educL"]
    """
    if interaction_cfg is None:
        return []
    if isinstance(interaction_cfg, (list, tuple, set)):
        terms = [str(term).strip() for term in interaction_cfg if str(term).strip()]
        return terms
    term = str(interaction_cfg).strip()
    return [term] if term else []


def _apply_market_centering(
    container: Container,
    log_market_expr: Any,
    prior_param: Parameter,
    i_set: Set,
    j_set: Set,
    n_alts: int,
    spec: EstimationSpec,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Optionally center market-opportunity index within each choice set.

    loglambda_tilde_ij = loglambda_ij - E_k[loglambda_ik]
    where expectation uses uniform or proposal (prior) weights.
    """
    if not getattr(spec, "market_opportunity_center_within_choice_set", False):
        return log_market_expr

    weights = str(getattr(spec, "market_opportunity_center_weights", "uniform")).strip().lower()
    j_alias = Alias(container, name=f"{j_set.name}_c", alias_with=j_set)
    log_market_alias = log_market_expr[i_set, j_alias]
    if weights == "proposal":
        denom = GamsSum(j_alias, prior_param[i_set, j_alias]) + LOG_EPS
        center_expr = GamsSum(j_alias, prior_param[i_set, j_alias] * log_market_alias) / denom
    else:
        center_expr = GamsSum(j_alias, log_market_alias) / float(max(n_alts, 1))

    if logger:
        logger.info(
            "    Centered market opportunity within choice set (weights=%s).",
            "proposal" if weights == "proposal" else "uniform",
        )
    return log_market_expr - center_expr


def _apply_market_scale(var_param: Any, var_name: str, scale_map: Dict[str, float]) -> Any:
    if not scale_map:
        return var_param
    scale_value = scale_map.get(str(var_name).strip())
    if scale_value is None:
        return var_param
    try:
        scale_value = float(scale_value)
    except (TypeError, ValueError):
        return var_param
    if scale_value == 1.0:
        return var_param
    return var_param * scale_value


# ==============================================================================
# Utility Functions
# ==============================================================================

def _set_gamspy_workdir(gamspy_workdir: Optional[str] = None) -> str:
    r"""Resolve the GAMSPy working directory WITHOUT mutating the process CWD (R4).

    GAMSPy's Container() refuses to start when the process CWD is a UNC path
    (\\server\share). The old-repo helper (ensure_local_workdir) silently
    os.chdir()'d to a local dir; CORE DOES NOT chdir. Instead:
      - if ``gamspy_workdir`` is given, use it (created if needed);
      - else if the CWD is local, use ``<cwd>/_gams_work``;
      - else (UNC CWD, no workdir) raise a clear error asking the caller to pass
        ``gamspy_workdir=`` (the app layer resolves this via path_helpers, which is
        deliberately NOT lifted into core — inventory R4 boundary decision).

    Sets GAMSPY_WORKING_DIR and returns the resolved directory.
    """
    import os
    if gamspy_workdir is not None:
        local_dir = Path(gamspy_workdir)
        local_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GAMSPY_WORKING_DIR"] = str(local_dir)
        return str(local_dir)
    cwd = Path.cwd()
    if str(cwd).startswith("\\\\"):
        raise RuntimeError(
            f"GAMSPy cannot start from a UNC working directory ({cwd}); core does "
            "not chdir. Pass an explicit local `gamspy_workdir=...` to the "
            "estimation function (R4: path_helpers is app-layer, not in core)."
        )
    local_dir = cwd / "_gams_work"
    local_dir.mkdir(exist_ok=True)
    os.environ["GAMSPY_WORKING_DIR"] = str(local_dir)
    return str(local_dir)


def box_cox_transform(x, theta, eps=LOG_EPS):
    """
    Box-Cox transformation: BC(x, θ) = (x^θ - 1) / θ

    Handles θ ≈ 0 case: BC(x, 0) = log(x)
    """
    # Use Taylor expansion around theta=0 for correct limit and smoothness.
    # Matches boxcox_gamspy (non-vectorized) and SciPy behavior.
    log_x = gp_log(x + eps)
    log_x2 = log_x * log_x
    log_x3 = log_x2 * log_x
    log_x4 = log_x3 * log_x
    return log_x * (
        1.0
        + theta * log_x / 2.0
        + theta * theta * log_x2 / 6.0
        + theta * theta * theta * log_x3 / 24.0
        + theta * theta * theta * theta * log_x4 / 120.0
    )


def _sanitize_eq_name(name: str, prefix: str = "") -> str:
    raw = f"{prefix}{name}"
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)
    if not safe:
        safe = "expr_constraint"
    # GAMS identifiers are limited; keep a conservative margin.
    return safe[:55]


def _apply_expression_constraints(
    container: Container,
    spec: EstimationSpec,
    param_vars: Dict[str, Variable],
    ll_expr: Any,
    active_groups: Optional[Tuple[str, ...]],
    name_prefix: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, List[Equation]]:
    if not getattr(spec, "expression_constraints_enabled", False):
        return ll_expr, []

    built = build_expression_constraints_gamspy(
        spec=spec,
        param_vars=param_vars,
        box_cox_transform_fn=box_cox_transform,
        gp_exp=gp_exp,
        gp_log=gp_log,
        log_eps=LOG_EPS,
        active_groups=list(active_groups) if active_groups is not None else None,
    )

    adjusted_ll = ll_expr - built["soft_penalty_expr"]
    hard_equations: List[Equation] = []
    for idx, bound in enumerate(built["hard_bounds"]):
        base = _sanitize_eq_name(
            f"{bound.get('name', f'expr_{idx}')}_{idx}",
            prefix=f"{name_prefix}_",
        )
        value_expr = bound["value_expr"]
        lower = bound.get("lower")
        upper = bound.get("upper")
        if lower is not None:
            hard_equations.append(
                Equation(
                    container,
                    name=f"{base}_lb",
                    definition=(value_expr >= float(lower)),
                )
            )
        if upper is not None:
            hard_equations.append(
                Equation(
                    container,
                    name=f"{base}_ub",
                    definition=(value_expr <= float(upper)),
                )
            )

    if logger and built["n_active"] > 0:
        logger.info(
            "  Applied expression constraints: "
            f"{built['n_active']} active, {len(hard_equations)} hard inequalities"
        )

    return adjusted_ll, hard_equations


def _extract_var_level(var) -> float:
    """
    Extract scalar level from a GAMSPy Variable across versions.
    """
    if hasattr(var, "records") and var.records is not None:
        if hasattr(var.records, "level"):
            level_series = var.records.level
            if hasattr(level_series, "iloc") and len(level_series) > 0:
                return float(level_series.iloc[0])
    if hasattr(var, "level"):
        return float(var.level)
    if hasattr(var, "l"):
        try:
            return float(var.l)
        except Exception:
            pass
    if hasattr(var, "records") and hasattr(var.records, "iloc"):
        if len(var.records) > 0:
            last_col = var.records.columns[-1]
            return float(var.records.iloc[0][last_col])
    logging.warning(f"Could not extract level for variable {getattr(var, 'name', '<unknown>')}, defaulting to 0.0")
    return 0.0


def _extract_num_iterations(model: Optional[Model] = None, solve_result: Any = None) -> Optional[int]:
    """
    Extract iteration count from GAMSPy model/solve result across versions.
    """
    for obj in (model, solve_result):
        if obj is None:
            continue
        for attr in ("num_iterations", "iteration_count", "iter_used", "_num_iterations"):
            val = getattr(obj, attr, None)
            if val is None:
                continue
            try:
                return int(val)
            except Exception:
                try:
                    return int(float(val))
                except Exception:
                    continue
    return None


def _build_singles_ll_vectorized(
    container: Container,
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
    param_vars: Dict[str, Variable],
    group: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, int]:
    """
    Build vectorized log-likelihood expression for singles group.
    Returns (ll_expr, n_alts).
    """
    n_groups = data.n_groups
    n_alts = data.n_obs // n_groups

    if logger:
        logger.info("  Building indexed data structure...")

    i_set = Set(container, name=f"{prefix}i", records=[str(i) for i in range(n_groups)])
    j_set = Set(container, name=f"{prefix}j", records=[str(j) for j in range(n_alts)])

    def _reshape(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(n_groups, n_alts)

    def _param2d(name: str, arr2d: np.ndarray) -> Parameter:
        return Parameter(container, name=f"{prefix}{name}", domain=[i_set, j_set], records=arr2d)

    # Core data parameters
    consumption_param = _param2d("consumption", _reshape(data.consumption))
    leisure_param = _param2d("leisure", _reshape(data.leisure))
    chosen_param = _param2d("chosen", _reshape(data.actual_choice))
    prior_param = _param2d("prior", _reshape(data.prior))

    var_cache: Dict[str, Parameter] = {
        "consumption": consumption_param,
        "leisure": leisure_param,
        "chosen": chosen_param,
        "prior": prior_param,
    }

    def get_var_param(var_name: str) -> Optional[Parameter]:
        if var_name in var_cache:
            return var_cache[var_name]
        if not hasattr(data, var_name):
            return None
        arr = getattr(data, var_name)
        if arr is None:
            return None
        param = _param2d(var_name, _reshape(arr))
        var_cache[var_name] = param
        return param

    if logger:
        logger.info(f"    Created indexed data: {n_groups} individuals × {n_alts} alternatives")

    if spec.utility_form != "box_cox":
        raise NotImplementedError(f"Utility form {spec.utility_form} not implemented in vectorized GAMSPy")

    # === Consumption utility ===
    # If beta_c is fixed (scale-normalisation numéraire), it is a compile-time
    # constant and is NOT in param_vars. Mirror the numpy engine pattern at
    # estimation_engine.py:506-514 and the expression-constraints handling at
    # expression_constraints.py:292+. Phase 1 (commit 31eaecc) wired this through
    # the numpy LL+gradient path and both expression-constraint paths; the GAMSPy
    # LL builder was missed and is completed here.
    _fixed_beta_c = getattr(spec, "utility_consumption_coef_fixed", None)
    if _fixed_beta_c is not None:
        beta_c = float(_fixed_beta_c)
    else:
        beta_c_name = get_param_name(spec.utility_consumption_coef, group, param_vars)
        beta_c = param_vars[beta_c_name]

    if spec.utility_consumption_theta:
        # spec.theta_c_param_name routes singles_{male,female} to the shared
        # `theta_c_singles` when M0a-clean is active; couples_household keeps
        # the legacy shared `theta_c`.
        theta_c_base = spec.theta_c_param_name(group) or spec.utility_consumption_theta
        theta_c_name = get_param_name(theta_c_base, group, param_vars)
        theta_c = param_vars[theta_c_name]
        bc_c = box_cox_transform(consumption_param, theta_c)
    else:
        bc_c = gp_log(consumption_param + LOG_EPS)

    u_consumption = beta_c * bc_c

    # === Leisure utility (with shifters) ===
    beta_l0_name = get_param_name(spec.utility_leisure_intercept, group, param_vars)
    beta_l_coeff = param_vars[beta_l0_name]

    for shifter in spec.utility_leisure_shifters:
        var_name = shifter["variable"]
        if shifter.get("gender_specific") and var_name == "n_children":
            if group in ("singles_male", "singles_pooled"):
                continue
        coef_base = shifter["coefficient"]
        try:
            coef_name = get_param_name(coef_base, group, param_vars)
        except ValueError:
            continue
        var_param = get_var_param(var_name)
        if var_param is None:
            continue
        beta_l_coeff = beta_l_coeff + param_vars[coef_name] * var_param

    if spec.utility_leisure_theta:
        theta_l_name = get_param_name(spec.utility_leisure_theta, group, param_vars)
        theta_l = param_vars[theta_l_name]
        bc_l = box_cox_transform(leisure_param, theta_l)
    else:
        bc_l = gp_log(leisure_param + LOG_EPS)

    u_leisure = beta_l_coeff * bc_l

    # Optional consumption-leisure interaction: beta_cl * BC(C) * BC(L)
    u_consumption_leisure = 0.0
    if spec.utility_consumption_leisure_interaction_coef:
        try:
            beta_cl_name = get_param_name(
                spec.utility_consumption_leisure_interaction_coef, group, param_vars
            )
            u_consumption_leisure = param_vars[beta_cl_name] * bc_c * bc_l
        except ValueError:
            pass

    # === Hours opportunity density ===
    log_h = 0.0
    working_param = None

    if spec.hours_shifters:
        if group in ("singles_male", "singles_pooled"):
            hours_suffix = "_male"
        elif group == "singles_female":
            hours_suffix = "_female"
        else:
            hours_suffix = ""

        for shifter in spec.hours_shifters:
            var_name = shifter["variable"]
            coef_name = shifter["coefficient"]
            interaction = shifter.get("interaction", None)

            coef_name_gender = f"{coef_name}{hours_suffix}" if hours_suffix else coef_name
            if coef_name_gender in param_vars:
                param = param_vars[coef_name_gender]
            elif coef_name in param_vars:
                param = param_vars[coef_name]
            else:
                continue

            var_param = get_var_param(var_name)
            if var_param is None:
                continue

            if interaction == "working":
                if working_param is None:
                    working_param = get_var_param("working")
                if working_param is not None:
                    var_param = var_param * working_param

            log_h = log_h + param * var_param

    # === Wage opportunity density ===
    log_w = 0.0
    if spec.wage_spec == "vw" and data.log_wage is not None and spec.wage_variance_param in param_vars:
        log_wage_param = get_var_param("log_wage")
        if log_wage_param is not None:
            mu_w = 0.0
            for shifter in spec.wage_mean_shifters:
                var_name = shifter["variable"]
                coef_name = shifter["coefficient"]
                if coef_name not in param_vars:
                    continue
                if var_name == "intercept":
                    mu_w = mu_w + param_vars[coef_name]
                else:
                    var_param = get_var_param(var_name)
                    if var_param is None:
                        continue
                    mu_w = mu_w + param_vars[coef_name] * var_param

            sigma_param = param_vars[spec.wage_variance_param]
            residual = log_wage_param - mu_w

            log_w_density = (
                -0.5 * (residual * residual) / (sigma_param * sigma_param + LOG_EPS)
                - gp_log(sigma_param + LOG_EPS)
                - 0.5 * gp_log(2.0 * 3.141592653589793)
                - log_wage_param
            )

            if working_param is None:
                working_param = get_var_param("working")
            if working_param is not None:
                log_w = working_param * log_w_density

    elif spec.wage_spec == "loc_empirical" and data.log_wage is not None and spec.wage_loc_groups:
        log_wage_param = get_var_param("log_wage")
        if log_wage_param is not None:
            common_shift = 0.0
            for shifter in spec.wage_mean_shifters:
                var_name = shifter["variable"]
                coef_name = shifter["coefficient"]
                if coef_name not in param_vars:
                    continue
                if var_name == "intercept":
                    common_shift = common_shift + param_vars[coef_name]
                else:
                    var_param = get_var_param(var_name)
                    if var_param is None:
                        continue
                    common_shift = common_shift + param_vars[coef_name] * var_param

            log_w_total = 0.0
            for group_cfg in spec.wage_loc_groups:
                loc_var = group_cfg["variable"]
                intercept_name = group_cfg["intercept"]
                sigma_name = group_cfg["sigma"]
                if intercept_name not in param_vars or sigma_name not in param_vars:
                    continue
                loc_param = get_var_param(loc_var)
                if loc_param is None:
                    continue
                mu_g = param_vars[intercept_name] + common_shift
                sigma_g = param_vars[sigma_name]
                residual = log_wage_param - mu_g
                log_w_g = (
                    -0.5 * (residual * residual) / (sigma_g * sigma_g + LOG_EPS)
                    - gp_log(sigma_g + LOG_EPS)
                    - 0.5 * gp_log(2.0 * 3.141592653589793)
                    - log_wage_param
                )
                log_w_total = log_w_total + loc_param * log_w_g

            if working_param is None:
                working_param = get_var_param("working")
            if working_param is not None:
                log_w = working_param * log_w_total

    # === Job/market opportunity density ===
    log_market = 0.0
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}
    if getattr(spec, "market_opportunity_shifters", None):
        for shifter in spec.market_opportunity_shifters:
            var_name = shifter.get("variable")
            coef_name = shifter.get("coefficient")
            interaction_terms = _normalize_interaction_terms(shifter.get("interaction", None))
            applies_to = str(shifter.get("applies_to", "both")).strip().lower()
            if not var_name or not coef_name:
                continue
            if coef_name not in param_vars:
                continue
            if applies_to in {"male", "sm"} and not getattr(data, "is_male", False):
                continue
            if applies_to in {"female", "sf"} and getattr(data, "is_male", False):
                continue
            if applies_to in {"cm", "cf"}:
                continue
            # "household" falls through to the same single-variable path as "both":
            # singles data has reg2/drgur/year_* directly (no gender split needed).
            var_param = get_var_param(var_name)
            if var_param is None:
                continue
            var_param = _apply_market_scale(var_param, var_name, scale_map)

            interaction_missing = False
            for interaction_name in interaction_terms:
                if interaction_name == "working":
                    if working_param is None:
                        working_param = get_var_param("working")
                    interaction_param = working_param
                else:
                    interaction_param = get_var_param(interaction_name)
                if interaction_param is None:
                    interaction_missing = True
                    break
                interaction_param = _apply_market_scale(
                    interaction_param, interaction_name, scale_map
                )
                var_param = var_param * interaction_param
            if interaction_missing:
                continue

            log_market = log_market + param_vars[coef_name] * var_param

    if getattr(spec, "market_opportunity_center_within_choice_set", False):
        log_market = _apply_market_centering(
            container=container,
            log_market_expr=log_market,
            prior_param=prior_param,
            i_set=i_set,
            j_set=j_set,
            n_alts=n_alts,
            spec=spec,
            logger=logger,
        )

    # Composite utility
    utility = (
        u_consumption
        + u_leisure
        + u_consumption_leisure
        + log_h
        + log_w
        + log_market
        - gp_log(prior_param + LOG_EPS)
    )

    if logger:
        logger.info("    Utility expression built (vectorized)")

    # Log-likelihood
    if logger:
        logger.info("  Building vectorized log-likelihood...")

    chosen_utility = GamsSum(j_set, chosen_param * utility)
    denom = GamsSum(j_set, gp_exp(utility))
    ll_expr = GamsSum(i_set, chosen_utility - gp_log(denom + LOG_EPS))

    if logger:
        logger.info("    Log-likelihood expression built (vectorized)")

    return ll_expr, n_alts


def _build_couples_ll_vectorized(
    container: Container,
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    param_vars: Dict[str, Variable],
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, int]:
    """
    Build vectorized log-likelihood expression for couples group.
    Returns (ll_expr, n_alts).
    """
    n_groups = data.n_groups
    n_alts = data.n_obs // n_groups

    if logger:
        logger.info("  Building indexed data structure for couples...")

    i_set = Set(container, name=f"{prefix}i", records=[str(i) for i in range(n_groups)])
    j_set = Set(container, name=f"{prefix}j", records=[str(j) for j in range(n_alts)])

    def _reshape(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(n_groups, n_alts)

    def _param2d(name: str, arr2d: np.ndarray) -> Parameter:
        return Parameter(container, name=f"{prefix}{name}", domain=[i_set, j_set], records=arr2d)

    consumption_param = _param2d("consumption", _reshape(data.consumption))
    leisure_m_param = _param2d("leisure_male", _reshape(data.leisure_male))
    leisure_f_param = _param2d("leisure_female", _reshape(data.leisure_female))
    chosen_param = _param2d("chosen", _reshape(data.actual_choice))
    prior_param = _param2d("prior", _reshape(data.prior))

    var_cache: Dict[str, Parameter] = {
        "consumption": consumption_param,
        "leisure_male": leisure_m_param,
        "leisure_female": leisure_f_param,
        "chosen": chosen_param,
        "prior": prior_param,
    }

    def get_var_param(
        base_name: str,
        gender: Optional[str] = None,
        fallback_to_base: bool = True,
    ) -> Optional[Parameter]:
        if gender:
            if base_name == "female":
                attr = f"female_{gender}"
            elif base_name in ("couple", "in_couple"):
                attr = f"in_couple_{gender}"
            else:
                attr_candidate = f"{base_name}_{gender}"
                if hasattr(data, attr_candidate):
                    attr = attr_candidate
                elif fallback_to_base:
                    attr = base_name
                else:
                    return None
        else:
            attr = base_name

        if attr in var_cache:
            return var_cache[attr]
        if not hasattr(data, attr):
            return None
        arr = getattr(data, attr)
        if arr is None:
            return None
        param = _param2d(attr, _reshape(arr))
        var_cache[attr] = param
        return param

    if logger:
        logger.info(f"    Created indexed data: {n_groups} households × {n_alts} alternatives")

    if spec.utility_form != "box_cox":
        raise NotImplementedError(f"Utility form {spec.utility_form} not implemented in vectorized GAMSPy")

    # Consumption utility (household-level)
    # If beta_c is fixed (scale-normalisation numéraire), it is a compile-time
    # constant and is NOT in param_vars. Mirror the singles path above and the
    # numpy engine's couples LL at estimation_engine.py:1462-1473. Phase 1
    # (commit 31eaecc) wired this through the numpy LL+gradient and expression
    # constraints; the GAMSPy couples LL builder is completed here.
    _fixed_beta_c = getattr(spec, "utility_consumption_coef_fixed", None)
    if _fixed_beta_c is not None:
        beta_c = float(_fixed_beta_c)
    else:
        beta_c_name = get_param_name(spec.utility_consumption_coef, "couples_household", param_vars)
        beta_c = param_vars[beta_c_name]

    _couples_fixed_theta = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    if _couples_fixed_theta is not None:
        # M0c-b: theta_c is structurally fixed — use compile-time constant, no param lookup.
        bc_c = box_cox_transform(consumption_param, float(_couples_fixed_theta))
    elif spec.utility_consumption_theta:
        # Couples group -> spec.theta_c_param_name returns the legacy shared
        # `theta_c`; routed via the helper for symmetry with the singles path.
        theta_c_base = (
            spec.theta_c_param_name("couples_household")
            or spec.utility_consumption_theta
        )
        theta_c_name = get_param_name(theta_c_base, "couples_household", param_vars)
        theta_c = param_vars[theta_c_name]
        bc_c = box_cox_transform(consumption_param, theta_c)
    else:
        bc_c = gp_log(consumption_param + LOG_EPS)

    u_consumption = beta_c * bc_c

    # Leisure utility - male
    beta_l0_m_name = get_param_name(spec.utility_leisure_intercept, "couples_male", param_vars)
    beta_l_coeff_m = param_vars[beta_l0_m_name]

    for shifter in spec.utility_leisure_shifters:
        var_name = shifter["variable"]
        if shifter.get("gender_specific") and var_name == "n_children":
            continue
        coef_base = shifter["coefficient"]
        try:
            coef_name = get_param_name(coef_base, "couples_male", param_vars)
        except ValueError:
            continue
        var_param = get_var_param(var_name, gender="male")
        if var_param is None:
            continue
        beta_l_coeff_m = beta_l_coeff_m + param_vars[coef_name] * var_param

    if spec.utility_leisure_theta:
        theta_l_m_name = get_param_name(spec.utility_leisure_theta, "couples_male", param_vars)
        theta_l_m = param_vars[theta_l_m_name]
        bc_l_m = box_cox_transform(leisure_m_param, theta_l_m)
    else:
        bc_l_m = gp_log(leisure_m_param + LOG_EPS)

    u_leisure_m = beta_l_coeff_m * bc_l_m

    # Leisure utility - female
    beta_l0_f_name = get_param_name(spec.utility_leisure_intercept, "couples_female", param_vars)
    beta_l_coeff_f = param_vars[beta_l0_f_name]

    for shifter in spec.utility_leisure_shifters:
        var_name = shifter["variable"]
        coef_base = shifter["coefficient"]
        try:
            coef_name = get_param_name(coef_base, "couples_female", param_vars)
        except ValueError:
            continue
        if var_name == "n_children":
            var_param = get_var_param("n_children")
        else:
            var_param = get_var_param(var_name, gender="female")
        if var_param is None:
            continue
        beta_l_coeff_f = beta_l_coeff_f + param_vars[coef_name] * var_param

    if spec.utility_leisure_theta:
        theta_l_f_name = get_param_name(spec.utility_leisure_theta, "couples_female", param_vars)
        theta_l_f = param_vars[theta_l_f_name]
        bc_l_f = box_cox_transform(leisure_f_param, theta_l_f)
    else:
        bc_l_f = gp_log(leisure_f_param + LOG_EPS)

    u_leisure_f = beta_l_coeff_f * bc_l_f

    # Optional consumption-leisure interaction terms:
    # beta_cl_m * BC(C) * BC(L_m) + beta_cl_f * BC(C) * BC(L_f)
    u_consumption_leisure = 0.0
    if spec.utility_consumption_leisure_interaction_coef:
        try:
            beta_cl_m_name = get_param_name(
                spec.utility_consumption_leisure_interaction_coef, "couples_male", param_vars
            )
            u_consumption_leisure = u_consumption_leisure + param_vars[beta_cl_m_name] * bc_c * bc_l_m
        except ValueError:
            pass
        try:
            beta_cl_f_name = get_param_name(
                spec.utility_consumption_leisure_interaction_coef, "couples_female", param_vars
            )
            u_consumption_leisure = u_consumption_leisure + param_vars[beta_cl_f_name] * bc_c * bc_l_f
        except ValueError:
            pass

    # Interaction term (if specified)
    u_interact = 0.0
    if spec.couples_interaction_coef and spec.couples_interaction_coef in param_vars:
        u_interact = param_vars[spec.couples_interaction_coef] * bc_l_m * bc_l_f

    utility = u_consumption + u_leisure_m + u_leisure_f + u_consumption_leisure + u_interact

    # Hours opportunity - male and female
    log_h_m = 0.0
    log_h_f = 0.0
    working_m = get_var_param("working", gender="male")
    working_f = get_var_param("working", gender="female")

    if spec.hours_shifters:
        for shifter in spec.hours_shifters:
            var_name = shifter["variable"]
            coef_name = shifter["coefficient"]
            interaction = shifter.get("interaction", None)

            coef_m = f"{coef_name}_male"
            if coef_m in param_vars:
                param_m = param_vars[coef_m]
            elif coef_name in param_vars:
                param_m = param_vars[coef_name]
            else:
                param_m = None

            if param_m is not None:
                var_param_m = get_var_param(var_name, gender="male")
                if var_param_m is not None:
                    if interaction == "working" and working_m is not None:
                        var_param_m = var_param_m * working_m
                    log_h_m = log_h_m + param_m * var_param_m

            coef_f = f"{coef_name}_female"
            if coef_f in param_vars:
                param_f = param_vars[coef_f]
            elif coef_name in param_vars:
                param_f = param_vars[coef_name]
            else:
                param_f = None

            if param_f is not None:
                var_param_f = get_var_param(var_name, gender="female")
                if var_param_f is not None:
                    if interaction == "working" and working_f is not None:
                        var_param_f = var_param_f * working_f
                    log_h_f = log_h_f + param_f * var_param_f

    # Wage opportunity - male and female
    log_w = 0.0
    if spec.wage_spec == "vw" and spec.wage_variance_param in param_vars:
        sigma_param = param_vars[spec.wage_variance_param]

        def build_mu_w(gender: str) -> Any:
            mu_w = 0.0
            for shifter in spec.wage_mean_shifters:
                var_name = shifter["variable"]
                coef_name = shifter["coefficient"]
                if coef_name not in param_vars:
                    continue
                if var_name == "intercept":
                    mu_w = mu_w + param_vars[coef_name]
                else:
                    var_param = get_var_param(var_name, gender=gender)
                    if var_param is None:
                        continue
                    mu_w = mu_w + param_vars[coef_name] * var_param
            return mu_w

        log_w_m = 0.0
        log_w_f = 0.0

        log_wage_m = get_var_param("log_wage", gender="male")
        if log_wage_m is not None and working_m is not None:
            mu_w_m = build_mu_w("male")
            residual_m = log_wage_m - mu_w_m
            log_w_density_m = (
                -0.5 * (residual_m * residual_m) / (sigma_param * sigma_param + LOG_EPS)
                - gp_log(sigma_param + LOG_EPS)
                - 0.5 * gp_log(2.0 * 3.141592653589793)
                - log_wage_m
            )
            log_w_m = working_m * log_w_density_m

        log_wage_f = get_var_param("log_wage", gender="female")
        if log_wage_f is not None and working_f is not None:
            mu_w_f = build_mu_w("female")
            residual_f = log_wage_f - mu_w_f
            log_w_density_f = (
                -0.5 * (residual_f * residual_f) / (sigma_param * sigma_param + LOG_EPS)
                - gp_log(sigma_param + LOG_EPS)
                - 0.5 * gp_log(2.0 * 3.141592653589793)
                - log_wage_f
            )
            log_w_f = working_f * log_w_density_f

        log_w = log_w_m + log_w_f

    elif spec.wage_spec == "loc_empirical" and data.log_wage_male is not None and spec.wage_loc_groups:
        # Occupation-specific wages for couples (male + female)
        def build_common_shift(gender: str) -> Any:
            common_shift = 0.0
            for shifter in spec.wage_mean_shifters:
                var_name = shifter["variable"]
                coef_name = shifter["coefficient"]
                if coef_name not in param_vars:
                    continue
                if var_name == "intercept":
                    common_shift = common_shift + param_vars[coef_name]
                else:
                    var_param = get_var_param(var_name, gender=gender)
                    if var_param is None:
                        continue
                    common_shift = common_shift + param_vars[coef_name] * var_param
            return common_shift

        def build_loc_logw(gender: str, working_param: Optional[Parameter]) -> Any:
            log_wage_param = get_var_param("log_wage", gender=gender)
            if log_wage_param is None or working_param is None:
                return 0.0
            common_shift = build_common_shift(gender)
            log_w_total = 0.0
            for group_cfg in spec.wage_loc_groups:
                loc_var = group_cfg["variable"]
                intercept_name = group_cfg["intercept"]
                sigma_name = group_cfg["sigma"]
                if intercept_name not in param_vars or sigma_name not in param_vars:
                    continue
                loc_param = get_var_param(loc_var, gender=gender)
                if loc_param is None:
                    continue
                mu_g = param_vars[intercept_name] + common_shift
                sigma_g = param_vars[sigma_name]
                residual = log_wage_param - mu_g
                log_w_g = (
                    -0.5 * (residual * residual) / (sigma_g * sigma_g + LOG_EPS)
                    - gp_log(sigma_g + LOG_EPS)
                    - 0.5 * gp_log(2.0 * 3.141592653589793)
                    - log_wage_param
                )
                log_w_total = log_w_total + loc_param * log_w_g
            return working_param * log_w_total

        log_w = build_loc_logw("male", working_m) + build_loc_logw("female", working_f)

    # Job/market opportunity density
    log_market = 0.0
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}
    if getattr(spec, "market_opportunity_shifters", None):
        for shifter in spec.market_opportunity_shifters:
            var_name = shifter.get("variable")
            coef_name = shifter.get("coefficient")
            interaction_terms = _normalize_interaction_terms(shifter.get("interaction", None))
            applies_to = str(shifter.get("applies_to", "both")).strip().lower()
            if not var_name or not coef_name:
                continue
            if coef_name not in param_vars:
                continue

            if applies_to == "household":
                var_param = get_var_param(var_name)
                if var_param is None:
                    continue
                var_param = _apply_market_scale(var_param, var_name, scale_map)
                interaction_missing = False
                for interaction_name in interaction_terms:
                    if interaction_name == "working":
                        if working_m is not None and working_f is not None:
                            interaction_param = working_m + working_f
                        elif working_m is not None:
                            interaction_param = working_m
                        elif working_f is not None:
                            interaction_param = working_f
                        else:
                            interaction_param = None
                    else:
                        interaction_param = get_var_param(interaction_name)
                    if interaction_param is None:
                        interaction_missing = True
                        break
                    interaction_param = _apply_market_scale(
                        interaction_param, interaction_name, scale_map
                    )
                    var_param = var_param * interaction_param
                if interaction_missing:
                    continue
                log_market = log_market + param_vars[coef_name] * var_param
                continue

            if applies_to in ("male", "cm", "both"):
                var_param_m = get_var_param(var_name, gender="male", fallback_to_base=False)
                if var_param_m is not None:
                    var_param_m = _apply_market_scale(var_param_m, var_name, scale_map)
                    interaction_missing = False
                    for interaction_name in interaction_terms:
                        if interaction_name == "working":
                            interaction_param = working_m
                        else:
                            interaction_param = get_var_param(interaction_name, gender="male")
                        if interaction_param is None:
                            interaction_missing = True
                            break
                        interaction_param = _apply_market_scale(
                            interaction_param, interaction_name, scale_map
                        )
                        var_param_m = var_param_m * interaction_param
                    if not interaction_missing:
                        log_market = log_market + param_vars[coef_name] * var_param_m

            if applies_to in ("female", "cf", "both"):
                var_param_f = get_var_param(var_name, gender="female", fallback_to_base=False)
                if var_param_f is not None:
                    var_param_f = _apply_market_scale(var_param_f, var_name, scale_map)
                    interaction_missing = False
                    for interaction_name in interaction_terms:
                        if interaction_name == "working":
                            interaction_param = working_f
                        else:
                            interaction_param = get_var_param(interaction_name, gender="female")
                        if interaction_param is None:
                            interaction_missing = True
                            break
                        interaction_param = _apply_market_scale(
                            interaction_param, interaction_name, scale_map
                        )
                        var_param_f = var_param_f * interaction_param
                    if not interaction_missing:
                        log_market = log_market + param_vars[coef_name] * var_param_f

    if getattr(spec, "market_opportunity_center_within_choice_set", False):
        log_market = _apply_market_centering(
            container=container,
            log_market_expr=log_market,
            prior_param=prior_param,
            i_set=i_set,
            j_set=j_set,
            n_alts=n_alts,
            spec=spec,
            logger=logger,
        )

    # Composite utility
    utility = utility + log_h_m + log_h_f + log_w + log_market - gp_log(prior_param + LOG_EPS)

    if logger:
        logger.info("    Couples utility expression built (vectorized)")

    if logger:
        logger.info("  Building vectorized log-likelihood for couples...")

    chosen_utility = GamsSum(j_set, chosen_param * utility)
    denom = GamsSum(j_set, gp_exp(utility))
    ll_expr = GamsSum(i_set, chosen_utility - gp_log(denom + LOG_EPS))

    if logger:
        logger.info("    Couples log-likelihood expression built (vectorized)")

    return ll_expr, n_alts


# ==============================================================================
# Vectorized Singles Estimation
# ==============================================================================

def estimate_singles_vectorized_gamspy(
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
    theta_init: np.ndarray,
    group: str = "singles_male",
    solver: str = "conopt",
    verbose: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
    solver_artifacts: Optional[Dict[str, str]] = None,
    gamspy_workdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate singles MNL model using vectorized GAMSPy operations.

    This is the OPTIMIZED version using indexed Sets and Parameters.
    Expected speedup: 3-5x faster than line-by-line approach.

    Parameters
    ----------
    data : PrecomputedDataSingles
        Precomputed data for singles
    spec : EstimationSpec
        Specification from YAML config
    theta_init : np.ndarray
        Initial parameter values
    group : str, default="singles_male"
        Estimation group: "singles_male" or "singles_female"
    solver : str, default="conopt"
        GAMSPy solver
    verbose : bool, default=True
        Print solver output
    solver_options : dict, optional
        Solver-specific options

    Returns
    -------
    dict with estimation results
    """
    _load_gamspy()

    logger = logging.getLogger(__name__)

    # Validate solver
    if solver not in SOLVER_MAP:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: {list(SOLVER_MAP.keys())}")

    solver_name = SOLVER_MAP[solver]

    # Guard against the silent group/data mismatch that estimated female data
    # against the _sm leisure block (see workitem-bpool-singles-female-leisure-suffix).
    # `group` defaults to "singles_male"; if a caller forgets to pass it while handing
    # in female data, fail loudly instead of silently using the wrong coefficients.
    _is_male = getattr(data, "is_male", None)
    if _is_male is not None:
        if group == "singles_female" and _is_male:
            raise ValueError(
                "group='singles_female' but data.is_male=True — group/data mismatch "
                "would estimate the wrong leisure block."
            )
        if group in ("singles_male", "singles_pooled") and not _is_male:
            raise ValueError(
                f"group={group!r} but data.is_male=False — group/data mismatch. "
                "Pass group='singles_female' for female singles data "
                "(see workitem-bpool-singles-female-leisure-suffix)."
            )

    # Compute number of alternatives
    n_alts = data.n_obs // data.n_groups

    logger.info(f"Starting VECTORIZED GAMSPy singles estimation (solver={solver_name.upper()})")
    logger.info(f"  Observations: {data.n_obs:,}")
    logger.info(f"  Groups: {data.n_groups:,}")
    logger.info(f"  Alternatives: {n_alts}")
    logger.info(f"  Parameters: {len(spec.all_param_names)}")

    start_time = time.time()

    # Ensure local working directory
    _set_gamspy_workdir(gamspy_workdir)

    # Create GAMSPy container
    container = Container()

    # ========================================================================
    # 1. Define indexed structure
    # ========================================================================

    logger.info("  Building indexed data structure...")

    # Define sets
    i_set = Set(container, name="individuals", records=[str(i) for i in range(data.n_groups)])
    j_set = Set(container, name="alternatives", records=[str(j) for j in range(n_alts)])

    # Reshape data to 2D (individuals × alternatives)
    n_groups = data.n_groups

    # Extract data arrays and reshape
    consumption_2d = data.consumption.reshape(n_groups, n_alts)
    leisure_2d = data.leisure.reshape(n_groups, n_alts)

    # Chosen alternative indicator (1 where choice was made, 0 elsewhere)
    chosen_2d = data.actual_choice.reshape(n_groups, n_alts)

    # Prior probabilities (for importance sampling correction)
    prior_2d = data.prior.reshape(n_groups, n_alts)

    # Define Parameters (2D indexed data)
    consumption_param = Parameter(
        container,
        name="consumption",
        domain=[i_set, j_set],
        records=consumption_2d
    )

    leisure_param = Parameter(
        container,
        name="leisure",
        domain=[i_set, j_set],
        records=leisure_2d
    )

    chosen_param = Parameter(
        container,
        name="chosen",
        domain=[i_set, j_set],
        records=chosen_2d
    )

    prior_param = Parameter(
        container,
        name="prior",
        domain=[i_set, j_set],
        records=prior_2d
    )

    # Scaling constants
    c_scale = float(data.c_scale)
    l_scale = float(data.l_scale)

    logger.info(f"    Created indexed data: {n_groups} individuals × {n_alts} alternatives")

    # ========================================================================
    # 2. Create parameter variables
    # ========================================================================

    param_vars = {}

    for i, param_name in enumerate(spec.all_param_names):
        var = Variable(container, param_name, type="free")
        var.l = float(theta_init[i])

        if param_name in spec.bounds:
            lb, ub = spec.bounds[param_name]
            if lb is not None:
                var.lo = float(lb)
            if ub is not None:
                var.up = float(ub)

        param_vars[param_name] = var

    logger.info(f"  Created {len(param_vars)} parameter variables")

    # ========================================================================
    # 3. Build log-likelihood expression (shared builder)
    # ========================================================================
    ll_expr, _ = _build_singles_ll_vectorized(
        container=container,
        data=data,
        spec=spec,
        param_vars=param_vars,
        group=group,
        prefix="sg_",
        logger=logger,
    )

    ll_expr, hard_eqs = _apply_expression_constraints(
        container=container,
        spec=spec,
        param_vars=param_vars,
        ll_expr=ll_expr,
        active_groups=(group,),
        name_prefix="sg",
        logger=logger,
    )

    # ========================================================================
    # 5. Create model and solve
    # ========================================================================

    if hard_eqs:
        model = Model(
            container,
            name="ruro_singles_mnl_vectorized",
            problem="nlp",
            sense="max",
            equations=hard_eqs,
            objective=ll_expr,
        )
    else:
        model = Model(
            container,
            name="ruro_singles_mnl_vectorized",
            problem="nlp",
            sense="max",
            objective=ll_expr,
        )

    logger.info(f"    Model created (problem type: NLP, sense: MAX)")
    logger.info(f"  Solving with {solver_name.upper()}...")
    logger.info("  (Vectorized approach should be 3-5x faster than line-by-line)")
    logger.info("  Proposal correction active: utility includes -log(prior) exactly once.")

    # Build GAMS options for artifact capture when requested
    gams_options = None
    if solver_artifacts:
        logger.info(f"  Artifact capture: solver_log={solver_artifacts.get('solver_log')}")
        logger.info(f"  Artifact capture: listing_file={solver_artifacts.get('listing_file')}")
        gams_options = Options(
            log_file=solver_artifacts.get('solver_log'),
            listing_file=solver_artifacts.get('listing_file'),
            write_listing_file=True,
            report_solution=1,
        )

    # Solve
    if solver_options and gams_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options, options=gams_options)
    elif solver_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options)
    elif gams_options:
        solve_result = model.solve(solver=solver_name, options=gams_options)
    else:
        solve_result = model.solve(solver=solver_name)

    walltime = time.time() - start_time

    # ========================================================================
    # 6. Extract results
    # ========================================================================

    theta_final = np.array([
        _extract_var_level(param_vars[pname])
        for pname in spec.all_param_names
    ])

    ll_final = getattr(model, "objective_value", None)
    if ll_final is None:
        ll_final = getattr(solve_result, "objective_value", None)

    solve_status_enum = getattr(model, "solve_status", None)
    model_status_enum = getattr(model, "status", None)

    solver_status = str(solve_status_enum) if solve_status_enum else "Unknown"
    model_status = str(model_status_enum) if model_status_enum else "Unknown"

    n_iterations = _extract_num_iterations(model, solve_result)

    logger.info("=" * 80)
    logger.info("VECTORIZED ESTIMATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Solver status: {solver_status}")
    logger.info(f"  Model status: {model_status}")
    if ll_final is not None:
        logger.info(f"  Objective value (LL): {ll_final:.4f}")
    logger.info(f"  Wall time: {walltime:.2f} seconds")

    return {
        "theta": theta_final,
        "log_likelihood": ll_final,
        "solver_status": solver_status,
        "model_status": model_status,
        "walltime": walltime,
        "n_iterations": n_iterations,
        "gamspy_result": solve_result,
        "solver": solver_name,
        "n_obs": data.n_obs,
        "n_groups": data.n_groups,
        "n_alts": n_alts,
        "spec_name": spec.name,
        "ll": ll_final,
        "prior_correction_applied": True,
        "prior_correction_form": "-log(prior)",
        "market_centering_applied": bool(getattr(spec, "market_opportunity_center_within_choice_set", False)),
    }


# ==============================================================================
# Vectorized Couples Estimation
# ==============================================================================

def estimate_couples_vectorized_gamspy(
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    theta_init: np.ndarray,
    solver: str = "conopt",
    verbose: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
    solver_artifacts: Optional[Dict[str, str]] = None,
    gamspy_workdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate couples MNL model using vectorized GAMSPy operations.

    This uses indexed Sets and Parameters for fast expression building.
    """
    _load_gamspy()

    logger = logging.getLogger(__name__)

    if solver not in SOLVER_MAP:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: {list(SOLVER_MAP.keys())}")

    solver_name = SOLVER_MAP[solver]
    n_alts = data.n_obs // data.n_groups

    logger.info(f"Starting VECTORIZED GAMSPy couples estimation (solver={solver_name.upper()})")
    logger.info(f"  Observations: {data.n_obs:,}")
    logger.info(f"  Groups: {data.n_groups:,}")
    logger.info(f"  Alternatives: {n_alts}")
    logger.info(f"  Parameters: {len(spec.all_param_names)}")

    start_time = time.time()

    # Ensure local working directory
    _set_gamspy_workdir(gamspy_workdir)

    # Create GAMSPy container
    container = Container()

    # ========================================================================
    # 1. Create parameter variables
    # ========================================================================
    param_vars: Dict[str, Variable] = {}

    for i, param_name in enumerate(spec.all_param_names):
        var = Variable(container, param_name, type="free")
        var.l = float(theta_init[i])

        if param_name in spec.bounds:
            lb, ub = spec.bounds[param_name]
            if lb is not None:
                var.lo = float(lb)
            if ub is not None:
                var.up = float(ub)

        param_vars[param_name] = var

    logger.info(f"  Created {len(param_vars)} parameter variables")

    # ========================================================================
    # 2. Build log-likelihood expression (vectorized)
    # ========================================================================
    ll_expr, _ = _build_couples_ll_vectorized(
        container=container,
        data=data,
        spec=spec,
        param_vars=param_vars,
        prefix="cou_",
        logger=logger,
    )

    ll_expr, hard_eqs = _apply_expression_constraints(
        container=container,
        spec=spec,
        param_vars=param_vars,
        ll_expr=ll_expr,
        active_groups=("couples_male", "couples_female", "couples_household"),
        name_prefix="cou",
        logger=logger,
    )

    # ========================================================================
    # 3. Create model and solve
    # ========================================================================
    if hard_eqs:
        model = Model(
            container,
            name="ruro_couples_mnl_vectorized",
            problem="nlp",
            sense="max",
            equations=hard_eqs,
            objective=ll_expr,
        )
    else:
        model = Model(
            container,
            name="ruro_couples_mnl_vectorized",
            problem="nlp",
            sense="max",
            objective=ll_expr,
        )

    logger.info(f"  Solving with {solver_name.upper()}...")
    logger.info("  (Vectorized approach should be 3-5x faster than line-by-line)")
    logger.info("  Proposal correction active: utility includes -log(prior) exactly once.")

    # Build GAMS options for artifact capture when requested
    gams_options = None
    if solver_artifacts:
        logger.info(f"  Artifact capture: solver_log={solver_artifacts.get('solver_log')}")
        logger.info(f"  Artifact capture: listing_file={solver_artifacts.get('listing_file')}")
        gams_options = Options(
            log_file=solver_artifacts.get('solver_log'),
            listing_file=solver_artifacts.get('listing_file'),
            write_listing_file=True,
            report_solution=1,
        )

    if solver_options and gams_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options, options=gams_options)
    elif solver_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options)
    elif gams_options:
        solve_result = model.solve(solver=solver_name, options=gams_options)
    else:
        solve_result = model.solve(solver=solver_name)

    walltime = time.time() - start_time

    # ========================================================================
    # 4. Extract results
    # ========================================================================
    theta_final = np.array([
        _extract_var_level(param_vars[pname])
        for pname in spec.all_param_names
    ])

    ll_final = getattr(model, "objective_value", None)
    if ll_final is None:
        ll_final = getattr(solve_result, "objective_value", None)

    solve_status_enum = getattr(model, "solve_status", None)
    model_status_enum = getattr(model, "status", None)
    solver_status = str(solve_status_enum) if solve_status_enum else "Unknown"
    model_status = str(model_status_enum) if model_status_enum else "Unknown"

    n_iterations = _extract_num_iterations(model, solve_result)

    logger.info("=" * 80)
    logger.info("VECTORIZED COUPLES ESTIMATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Solver status: {solver_status}")
    logger.info(f"  Model status: {model_status}")
    if ll_final is not None:
        logger.info(f"  Objective value (LL): {ll_final:.4f}")
    logger.info(f"  Wall time: {walltime:.2f} seconds")

    return {
        "theta": theta_final,
        "log_likelihood": ll_final,
        "solver_status": solver_status,
        "model_status": model_status,
        "walltime": walltime,
        "n_iterations": n_iterations,
        "gamspy_result": solve_result,
        "solver": solver_name,
        "n_obs": data.n_obs,
        "n_groups": data.n_groups,
        "n_alts": n_alts,
        "spec_name": spec.name,
        "ll": ll_final,
        "prior_correction_applied": True,
        "prior_correction_form": "-log(prior)",
        "market_centering_applied": bool(getattr(spec, "market_opportunity_center_within_choice_set", False)),
    }


# ==============================================================================
# Joint Estimation (Singles + Couples) - VECTORIZED
# ==============================================================================

def estimate_joint_vectorized_gamspy(
    data_singles_male: PrecomputedDataSingles,
    data_singles_female: PrecomputedDataSingles,
    data_couples: PrecomputedDataCouples,
    spec: EstimationSpec,
    theta_init: np.ndarray,
    solver: str = "conopt",
    verbose: bool = True,
    solver_options: Optional[Dict[str, Any]] = None,
    solver_artifacts: Optional[Dict[str, str]] = None,
    gamspy_workdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Joint estimation (singles male + singles female + couples) using vectorized GAMSPy.

    This is the OPTIMIZED version. Expected total speedup:
    - A→B (expression combination): 30-60s → 5-10s (5-6x faster)
    - B→C (GAMS compilation): 5-7 min → 1-2 min (3-5x faster)
    - Total: 5-8 min → 1-3 min

    For occupation choice (400 alts):
    - Total: 15-30 min → 3-7 min

    Parameters
    ----------
    data_singles_male, data_singles_female : PrecomputedDataSingles
        Singles male and female data
    data_couples : PrecomputedDataCouples
        Couples data
    spec : EstimationSpec
        Specification from YAML
    theta_init : np.ndarray
        Initial parameters
    solver : str
        GAMSPy solver
    verbose : bool
        Print output
    solver_options : dict, optional
        Solver options

    Returns
    -------
    dict with estimation results
    """
    _load_gamspy()

    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("VECTORIZED JOINT ESTIMATION")
    logger.info("=" * 80)
    logger.info(f"  Singles Male: {data_singles_male.n_groups:,} individuals")
    logger.info(f"  Singles Female: {data_singles_female.n_groups:,} individuals")
    logger.info(f"  Couples: {data_couples.n_groups:,} households")
    logger.info(f"  Alternatives: {data_singles_male.n_obs // data_singles_male.n_groups}")
    logger.info(f"  Parameters: {len(spec.all_param_names)}")
    if solver not in SOLVER_MAP:
        raise ValueError(f"Unknown solver '{solver}'. Choose from: {list(SOLVER_MAP.keys())}")
    solver_name = SOLVER_MAP[solver]
    logger.info(f"  Solver: {solver_name.upper()}")

    start_time = time.time()

    # Ensure local working directory
    _set_gamspy_workdir(gamspy_workdir)

    # Create GAMSPy container
    container = Container()

    # ========================================================================
    # 1. Create shared parameter variables
    # ========================================================================
    param_vars: Dict[str, Variable] = {}

    for i, param_name in enumerate(spec.all_param_names):
        var = Variable(container, param_name, type="free")
        var.l = float(theta_init[i])

        if param_name in spec.bounds:
            lb, ub = spec.bounds[param_name]
            if lb is not None:
                var.lo = float(lb)
            if ub is not None:
                var.up = float(ub)

        param_vars[param_name] = var

    logger.info(f"  Created {len(param_vars)} shared parameter variables")

    # ========================================================================
    # 2. Build log-likelihood expressions (vectorized)
    # ========================================================================
    ll_sm, n_alts_sm = _build_singles_ll_vectorized(
        container=container,
        data=data_singles_male,
        spec=spec,
        param_vars=param_vars,
        group="singles_male",
        prefix="sm_",
        logger=logger,
    )
    ll_sf, n_alts_sf = _build_singles_ll_vectorized(
        container=container,
        data=data_singles_female,
        spec=spec,
        param_vars=param_vars,
        group="singles_female",
        prefix="sf_",
        logger=logger,
    )
    ll_cou, n_alts_cou = _build_couples_ll_vectorized(
        container=container,
        data=data_couples,
        spec=spec,
        param_vars=param_vars,
        prefix="cou_",
        logger=logger,
    )

    if len({n_alts_sm, n_alts_sf, n_alts_cou}) != 1:
        logger.warning(
            f"Different number of alternatives across groups: "
            f"sm={n_alts_sm}, sf={n_alts_sf}, cou={n_alts_cou}"
        )

    ll_joint = ll_sm + ll_sf + ll_cou

    ll_joint, hard_eqs = _apply_expression_constraints(
        container=container,
        spec=spec,
        param_vars=param_vars,
        ll_expr=ll_joint,
        active_groups=(
            "singles_male",
            "singles_female",
            "couples_male",
            "couples_female",
            "couples_household",
        ),
        name_prefix="joint",
        logger=logger,
    )

    # ========================================================================
    # 3. Create model and solve
    # ========================================================================
    if hard_eqs:
        model = Model(
            container,
            name="ruro_joint_mnl_vectorized",
            problem="nlp",
            sense="max",
            equations=hard_eqs,
            objective=ll_joint,
        )
    else:
        model = Model(
            container,
            name="ruro_joint_mnl_vectorized",
            problem="nlp",
            sense="max",
            objective=ll_joint,
        )

    logger.info("  Solving joint model...")
    logger.info("  Proposal correction active: utility includes -log(prior) exactly once per group.")

    # Build GAMS options for artifact capture when requested
    gams_options = None
    if solver_artifacts:
        logger.info(f"  Artifact capture: solver_log={solver_artifacts.get('solver_log')}")
        logger.info(f"  Artifact capture: listing_file={solver_artifacts.get('listing_file')}")
        gams_options = Options(
            log_file=solver_artifacts.get('solver_log'),
            listing_file=solver_artifacts.get('listing_file'),
            write_listing_file=True,
            report_solution=1,
        )

    if solver_options and gams_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options, options=gams_options)
    elif solver_options:
        logger.info(f"  Solver options: {solver_options}")
        solve_result = model.solve(solver=solver_name, solver_options=solver_options)
    elif gams_options:
        solve_result = model.solve(solver=solver_name, options=gams_options)
    else:
        solve_result = model.solve(solver=solver_name)

    walltime = time.time() - start_time

    # ========================================================================
    # 4. Extract results
    # ========================================================================
    theta_final = np.array([
        _extract_var_level(param_vars[pname])
        for pname in spec.all_param_names
    ])

    ll_final = getattr(model, "objective_value", None)
    if ll_final is None:
        ll_final = getattr(solve_result, "objective_value", None)

    solve_status_enum = getattr(model, "solve_status", None)
    model_status_enum = getattr(model, "status", None)
    solver_status = str(solve_status_enum) if solve_status_enum else "Unknown"
    model_status = str(model_status_enum) if model_status_enum else "Unknown"

    n_iterations = _extract_num_iterations(model, solve_result)

    logger.info("=" * 80)
    logger.info("VECTORIZED JOINT ESTIMATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Solver status: {solver_status}")
    logger.info(f"  Model status: {model_status}")
    if ll_final is not None:
        logger.info(f"  Objective value (LL): {ll_final:.4f}")
    logger.info(f"  Wall time: {walltime:.2f} seconds")

    return {
        "theta": theta_final,
        "log_likelihood": ll_final,
        "solver_status": solver_status,
        "model_status": model_status,
        "walltime": walltime,
        "n_iterations": n_iterations,
        "gamspy_result": solve_result,
        "solver": solver_name,
        "n_obs": data_singles_male.n_obs + data_singles_female.n_obs + data_couples.n_obs,
        "n_groups": data_singles_male.n_groups + data_singles_female.n_groups + data_couples.n_groups,
        "n_alts": n_alts_sm,
        "spec_name": spec.name,
        "ll": ll_final,
        "ll_singles_male": None,
        "ll_singles_female": None,
        "ll_couples": None,
        "prior_correction_applied": True,
        "prior_correction_form": "-log(prior)",
        "market_centering_applied": bool(getattr(spec, "market_opportunity_center_within_choice_set", False)),
    }


# compare_performance() (vectorized-vs-line-by-line timing, importing the old-runner
# gamspy_estimation.py) was intentionally NOT lifted — old provenance/comparison code.


# ==============================================================================
# Expression constraints - GAMSPy symbolic path (lifted Wave 2.4)
# ==============================================================================
# Lifted byte-faithfully from MNL/scripts/enhanced/expression_constraints.py:
# ONLY the GAMSPy expression-constraint builder + its minimal shared helpers.
# The NumPy penalty path (evaluate_constraint_value_numpy /
# compute_expression_constraints_penalty_numpy and the *_scalar helpers) was
# NOT lifted - it is the optimization-time likelihood penalty deferred from the
# core engines in Wave 1.3 and must NOT be wired back into the likelihood engines.
# gp_exp / gp_log / box_cox_transform_fn are injected by the caller
# (_apply_expression_constraints above), so this path needs no module-level gamspy.
# ==============================================================================


SUPPORTED_EXPRESSIONS = {"muc", "mul", "dmuc_dc", "dmul_dl", "param_diff"}
SUPPORTED_GROUPS = {
    "singles_male",
    "singles_female",
    "couples_male",
    "couples_female",
    "couples_household",
    "couples",
    "global",
}

_SUFFIX_BY_GROUP = {
    "singles_male": "_sm",
    "singles_female": "_sf",
    "couples_male": "_m",
    "couples_female": "_f",
    "couples_household": "",
    "couples": "",
    "global": "",
}


def _normalize_group_name(group: str) -> str:
    return str(group).strip().lower()


def _normalize_active_groups(active_groups: Optional[Iterable[str]]) -> Optional[set[str]]:
    if active_groups is None:
        return None
    return {_normalize_group_name(g) for g in active_groups}


def get_active_constraints(
    spec: EstimationSpec,
    active_groups: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Return constraints that apply to the active estimation groups."""
    if not getattr(spec, "expression_constraints_enabled", False):
        return []
    constraints = list(getattr(spec, "expression_constraints", []) or [])
    if not constraints:
        return []

    active = _normalize_active_groups(active_groups)
    if active is None:
        return constraints

    filtered: List[Dict[str, Any]] = []
    for constraint in constraints:
        group = _normalize_group_name(constraint.get("group", ""))
        if group in active:
            filtered.append(constraint)
            continue
        if group == "global":
            filtered.append(constraint)
            continue
        if group == "couples":
            # Household-level alias: apply whenever either couples group is active.
            if "couples_male" in active or "couples_female" in active or "couples_household" in active:
                filtered.append(constraint)
    return filtered


def _candidate_param_names(base_name: str, group: str) -> List[str]:
    suffix = _SUFFIX_BY_GROUP.get(group, "")
    if suffix:
        return [f"{base_name}{suffix}", base_name]
    return [base_name]


def _resolve_param_symbol(
    param_vars: Dict[str, Any],
    base_name: Optional[str],
    group: str,
    required: bool = False,
) -> Optional[Any]:
    if not base_name:
        return None
    for name in _candidate_param_names(base_name, group):
        if name in param_vars:
            return param_vars[name]
    if required:
        raise ValueError(
            f"Expression constraint references missing parameter '{base_name}' for group '{group}'."
        )
    return None


def _at_value(at: Dict[str, float], key: str, default: float) -> float:
    return float(at.get(key, default))


def _coalesce_none(value: Optional[Any], default: Any = 0.0) -> Any:
    """Return value unless it is None (safe for symbolic objects)."""
    return default if value is None else value


def _first_non_none(*values: Optional[Any], default: Any = 0.0) -> Any:
    """Return first value that is not None (safe for symbolic objects)."""
    for value in values:
        if value is not None:
            return value
    return default


def _resolve_single_leisure_beta_symbol(
    param_vars: Dict[str, Any],
    spec: EstimationSpec,
    group: str,
    at: Dict[str, float],
) -> Any:
    beta_l = _resolve_param_symbol(
        param_vars, spec.utility_leisure_intercept, group, required=True
    )
    for shifter in spec.utility_leisure_shifters:
        var_name = shifter["variable"]
        if shifter.get("gender_specific") and var_name == "n_children" and group in {"singles_male", "couples_male"}:
            continue
        coef_sym = _resolve_param_symbol(param_vars, shifter["coefficient"], group, required=False)
        if coef_sym is None:
            continue
        beta_l = beta_l + coef_sym * _at_value(at, var_name, 0.0)
    return beta_l


def _dbc_dx_symbol(x_value: float, theta_symbol: Any, gp_exp: Any, gp_log: Any, log_eps: float) -> Any:
    # x^(theta-1) implemented via exp((theta-1) * log(x)) to avoid POWER with variable exponent.
    return gp_exp((theta_symbol - 1.0) * gp_log(float(x_value) + log_eps))


def _d2bc_dx2_symbol(x_value: float, theta_symbol: Any, gp_exp: Any, gp_log: Any, log_eps: float) -> Any:
    # (theta-1) * x^(theta-2)
    return (theta_symbol - 1.0) * gp_exp((theta_symbol - 2.0) * gp_log(float(x_value) + log_eps))


def _utility_component_symbol(
    x_value: float,
    theta_symbol: Optional[Any],
    utility_form: str,
    box_cox_transform_fn: Any,
    gp_log: Any,
    log_eps: float,
) -> Any:
    x_safe = float(x_value) + log_eps
    if utility_form == "linear":
        return float(x_value)
    if utility_form == "log":
        return gp_log(x_safe)
    if theta_symbol is None:
        return gp_log(x_safe)
    return box_cox_transform_fn(float(x_value), theta_symbol)


def _utility_derivative_symbol(
    x_value: float,
    theta_symbol: Optional[Any],
    utility_form: str,
    gp_exp: Any,
    gp_log: Any,
    log_eps: float,
) -> Any:
    x_safe = float(x_value) + log_eps
    if utility_form == "linear":
        return 1.0
    if utility_form == "log":
        return 1.0 / x_safe
    if theta_symbol is None:
        return 1.0 / x_safe
    return _dbc_dx_symbol(float(x_value), theta_symbol, gp_exp, gp_log, log_eps)


def _utility_second_derivative_symbol(
    x_value: float,
    theta_symbol: Optional[Any],
    utility_form: str,
    gp_exp: Any,
    gp_log: Any,
    log_eps: float,
) -> Any:
    x_safe = float(x_value) + log_eps
    if utility_form == "linear":
        return 0.0
    if utility_form == "log":
        return -1.0 / (x_safe * x_safe)
    if theta_symbol is None:
        return -1.0 / (x_safe * x_safe)
    return _d2bc_dx2_symbol(float(x_value), theta_symbol, gp_exp, gp_log, log_eps)


def evaluate_constraint_value_gamspy(
    spec: EstimationSpec,
    param_vars: Dict[str, Any],
    constraint: Dict[str, Any],
    box_cox_transform_fn: Any,
    gp_exp: Any,
    gp_log: Any,
    log_eps: float,
) -> Any:
    """
    Build GAMSPy expression for one constraint value.
    """
    expression = _normalize_group_name(constraint["expression"])
    group = _normalize_group_name(constraint["group"])
    at = constraint.get("at", {}) or {}

    if expression not in SUPPORTED_EXPRESSIONS:
        raise ValueError(f"Unsupported expression constraint '{expression}'.")
    if group not in SUPPORTED_GROUPS:
        raise ValueError(f"Unsupported constraint group '{group}'.")

    if expression == "param_diff":
        lhs_param = constraint.get("lhs_param")
        rhs_param = constraint.get("rhs_param")
        lhs_sym = _resolve_param_symbol(param_vars, lhs_param, group, required=True)
        rhs_sym = _resolve_param_symbol(param_vars, rhs_param, group, required=False)
        if rhs_sym is None:
            rhs_sym = 0.0
        return lhs_sym - rhs_sym

    c = _at_value(at, "consumption", 1.0)
    theta_c_group = "couples_household" if group.startswith("couples") else group
    fixed_couples_theta = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    if group.startswith("couples") and fixed_couples_theta is not None:
        theta_c = float(fixed_couples_theta)
    else:
        theta_c_base = (
            spec.theta_c_param_name(theta_c_group)
            if hasattr(spec, "theta_c_param_name")
            else spec.utility_consumption_theta
        )
        theta_c = _resolve_param_symbol(
            param_vars,
            theta_c_base,
            theta_c_group,
            required=(spec.utility_form == "box_cox" and theta_c_base is not None),
        )
    # beta_c may be fixed (scale-normalisation numeraire) — then it is a compile-time
    # constant, not a GAMSPy variable. Mirror the fixed-theta_c handling above.
    _fixed_beta_c = getattr(spec, "utility_consumption_coef_fixed", None)
    if _fixed_beta_c is not None:
        beta_c = float(_fixed_beta_c)
    else:
        beta_c = _resolve_param_symbol(param_vars, spec.utility_consumption_coef, theta_c_group, required=True)
    bc_c = _utility_component_symbol(
        c, theta_c, spec.utility_form, box_cox_transform_fn, gp_log, log_eps
    )
    dbc_c = _utility_derivative_symbol(c, theta_c, spec.utility_form, gp_exp, gp_log, log_eps)

    if group in {"singles_male", "singles_female"}:
        l = _at_value(at, "leisure", 1.0)
        theta_l = _resolve_param_symbol(
            param_vars,
            spec.utility_leisure_theta,
            group,
            required=(spec.utility_form == "box_cox" and spec.utility_leisure_theta is not None),
        )
        bc_l = _utility_component_symbol(
            l, theta_l, spec.utility_form, box_cox_transform_fn, gp_log, log_eps
        )
        dbc_l = _utility_derivative_symbol(l, theta_l, spec.utility_form, gp_exp, gp_log, log_eps)
        beta_cl = 0.0
        if spec.utility_consumption_leisure_interaction_coef:
            beta_cl = _coalesce_none(_resolve_param_symbol(
                param_vars,
                spec.utility_consumption_leisure_interaction_coef,
                group,
                required=False,
            ))
        beta_l = _resolve_single_leisure_beta_symbol(param_vars, spec, group, at)
        if expression == "muc":
            return (beta_c + beta_cl * bc_l) * dbc_c
        if expression == "dmuc_dc":
            d2bc_c = _utility_second_derivative_symbol(c, theta_c, spec.utility_form, gp_exp, gp_log, log_eps)
            return (beta_c + beta_cl * bc_l) * d2bc_c
        if expression == "mul":
            return (beta_l + beta_cl * bc_c) * dbc_l
        d2bc_l = _utility_second_derivative_symbol(l, theta_l, spec.utility_form, gp_exp, gp_log, log_eps)
        return (beta_l + beta_cl * bc_c) * d2bc_l

    # Couples
    l_m = _at_value(at, "leisure_male", _at_value(at, "leisure", 1.0))
    l_f = _at_value(
        at,
        "leisure_female",
        _at_value(at, "leisure_partner", _at_value(at, "leisure", 1.0)),
    )
    theta_l_m = _resolve_param_symbol(
        param_vars,
        spec.utility_leisure_theta,
        "couples_male",
        required=(spec.utility_form == "box_cox" and spec.utility_leisure_theta is not None),
    )
    theta_l_f = _resolve_param_symbol(
        param_vars,
        spec.utility_leisure_theta,
        "couples_female",
        required=(spec.utility_form == "box_cox" and spec.utility_leisure_theta is not None),
    )
    bc_l_m = _utility_component_symbol(
        l_m, theta_l_m, spec.utility_form, box_cox_transform_fn, gp_log, log_eps
    )
    bc_l_f = _utility_component_symbol(
        l_f, theta_l_f, spec.utility_form, box_cox_transform_fn, gp_log, log_eps
    )

    beta_cl_m = 0.0
    beta_cl_f = 0.0
    if spec.utility_consumption_leisure_interaction_coef:
        beta_cl_m = _coalesce_none(_resolve_param_symbol(
            param_vars,
            spec.utility_consumption_leisure_interaction_coef,
            "couples_male",
            required=False,
        ))
        beta_cl_f = _coalesce_none(_resolve_param_symbol(
            param_vars,
            spec.utility_consumption_leisure_interaction_coef,
            "couples_female",
            required=False,
        ))

    if expression == "muc":
        return (beta_c + beta_cl_m * bc_l_m + beta_cl_f * bc_l_f) * dbc_c
    if expression == "dmuc_dc":
        d2bc_c = _utility_second_derivative_symbol(c, theta_c, spec.utility_form, gp_exp, gp_log, log_eps)
        return (beta_c + beta_cl_m * bc_l_m + beta_cl_f * bc_l_f) * d2bc_c

    if group not in {"couples_male", "couples_female"}:
        raise ValueError(
            "MUL/DMUL constraints for couples must use group 'couples_male' or 'couples_female'."
        )

    own_group = group
    partner_group = "couples_female" if own_group == "couples_male" else "couples_male"
    l_own = l_m if own_group == "couples_male" else l_f
    l_partner = l_f if own_group == "couples_male" else l_m
    theta_l_own = theta_l_m if own_group == "couples_male" else theta_l_f
    theta_l_partner = theta_l_f if own_group == "couples_male" else theta_l_m
    dbc_l_own = _utility_derivative_symbol(
        l_own, theta_l_own, spec.utility_form, gp_exp, gp_log, log_eps
    )
    bc_l_partner = _utility_component_symbol(
        l_partner, theta_l_partner, spec.utility_form, box_cox_transform_fn, gp_log, log_eps
    )
    beta_l_own = _resolve_single_leisure_beta_symbol(param_vars, spec, own_group, at)
    beta_cl_own = beta_cl_m if own_group == "couples_male" else beta_cl_f
    beta_interact = 0.0
    if spec.couples_interaction_coef:
        beta_interact = _first_non_none(
            _resolve_param_symbol(
                param_vars, spec.couples_interaction_coef, partner_group, required=False
            ),
            _resolve_param_symbol(
                param_vars, spec.couples_interaction_coef, "couples_household", required=False
            ),
        )
    pref_term = beta_l_own + beta_cl_own * bc_c + beta_interact * bc_l_partner
    if expression == "mul":
        return pref_term * dbc_l_own
    d2bc_l_own = _utility_second_derivative_symbol(
        l_own, theta_l_own, spec.utility_form, gp_exp, gp_log, log_eps
    )
    return pref_term * d2bc_l_own


def build_expression_constraints_gamspy(
    spec: EstimationSpec,
    param_vars: Dict[str, Any],
    box_cox_transform_fn: Any,
    gp_exp: Any,
    gp_log: Any,
    log_eps: float,
    active_groups: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build GAMSPy-ready soft penalties and hard-bound definitions.

    Returns dict with:
      - "soft_penalty_expr": objective penalty expression (subtract from LL)
      - "hard_bounds": list of dict(name, value_expr, lower, upper)
      - "n_active": number of active constraints
    """
    constraints = get_active_constraints(spec, active_groups=active_groups)
    if not constraints:
        return {"soft_penalty_expr": 0.0, "hard_bounds": [], "n_active": 0}

    # Smooth approximation of max(0, x): log(1 + exp(kx)) / k.
    kappa = 50.0
    soft_penalty_expr = 0.0
    hard_bounds: List[Dict[str, Any]] = []

    for idx, constraint in enumerate(constraints):
        value_expr = evaluate_constraint_value_gamspy(
            spec=spec,
            param_vars=param_vars,
            constraint=constraint,
            box_cox_transform_fn=box_cox_transform_fn,
            gp_exp=gp_exp,
            gp_log=gp_log,
            log_eps=log_eps,
        )
        mode = _normalize_group_name(constraint.get("mode", spec.expression_constraints_default_mode))
        lower = constraint.get("lower")
        upper = constraint.get("upper")
        weight = float(constraint.get("weight", spec.expression_constraints_default_weight))
        name = str(constraint.get("name", f"expr_constraint_{idx}")).strip() or f"expr_constraint_{idx}"

        if mode == "hard":
            hard_bounds.append({
                "name": name,
                "value_expr": value_expr,
                "lower": lower,
                "upper": upper,
            })
            continue

        if lower is not None:
            x_low = float(lower) - value_expr
            low_pos = gp_log(1.0 + gp_exp(kappa * x_low)) / kappa
            soft_penalty_expr = soft_penalty_expr + weight * low_pos * low_pos
        if upper is not None:
            x_up = value_expr - float(upper)
            up_pos = gp_log(1.0 + gp_exp(kappa * x_up)) / kappa
            soft_penalty_expr = soft_penalty_expr + weight * up_pos * up_pos

    return {
        "soft_penalty_expr": soft_penalty_expr,
        "hard_bounds": hard_bounds,
        "n_active": len(constraints),
    }
