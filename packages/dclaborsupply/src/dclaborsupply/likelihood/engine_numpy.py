"""
==============================================================================
RURO MNL Estimation Engine
==============================================================================
Likelihood and gradient computation for MNL estimation.

Supports:
- Singles estimation (male/female/pooled)
- Couples estimation with leisure interaction
- Multiple wage specifications (fw, vw, loc_empirical)
- Analytical gradients for L-BFGS-B optimization
- Vectorized computation for performance

Author: Enhanced RURO Pipeline
Created: 2026-01-03
==============================================================================
"""

import logging
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import numpy as np

# Wave 1.3 import adaptation (copy + adapt imports only; no math change):
# BC math -> dclaborsupply.utility.boxcox; containers/LSE/EPS/HAS_NUMBA ->
# dclaborsupply.likelihood._numpy_primitives; spec parser -> dclaborsupply.spec.parser.
# expression_constraints is a solver/optimization-layer concern (matrix Wave 2.4),
# NOT part of the core reference likelihood -> dropped (penalty hooks removed below).
from dclaborsupply.utility.boxcox import (
    box_cox_transform,
    box_cox_derivative_x,
    box_cox_derivative_theta,
)
from dclaborsupply.likelihood._numpy_primitives import (
    PrecomputedDataSingles,
    PrecomputedDataCouples,
    compute_log_sum_exp_by_group,
    EPS,
    HAS_NUMBA,
)
from dclaborsupply.spec.parser import EstimationSpec


def _normalize_interaction_terms(interaction_cfg) -> list:
    if interaction_cfg is None:
        return []
    if isinstance(interaction_cfg, (list, tuple, set)):
        return [str(term).strip() for term in interaction_cfg if str(term).strip()]
    term = str(interaction_cfg).strip()
    return [term] if term else []


def _apply_market_scale_numpy(values: np.ndarray, var_name: str, scale_map: Dict[str, float]) -> np.ndarray:
    if not scale_map:
        return values
    scale_value = scale_map.get(str(var_name).strip())
    if scale_value is None:
        return values
    try:
        scale_value = float(scale_value)
    except (TypeError, ValueError):
        return values
    if scale_value == 1.0:
        return values
    return values * scale_value


def _center_within_choice_set(
    values: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    centered = values.copy()
    for g in range(len(group_starts)):
        start, end = int(group_starts[g]), int(group_ends[g])
        if weights is None:
            mean_val = np.mean(values[start:end])
        else:
            w = weights[start:end]
            denom = np.sum(w) + EPS
            mean_val = np.sum(values[start:end] * w) / denom
        centered[start:end] = values[start:end] - mean_val
    return centered


def _compute_market_opportunity_singles(
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    log_market = np.zeros(data.n_obs)
    components: Dict[str, np.ndarray] = {}
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}

    if getattr(spec, "market_opportunity_shifters", None):
        for shifter in spec.market_opportunity_shifters:
            var_name = shifter.get("variable")
            coef_name = shifter.get("coefficient")
            interaction_terms = _normalize_interaction_terms(shifter.get("interaction", None))
            applies_to = str(shifter.get("applies_to", "both")).strip().lower()
            if not var_name or not coef_name:
                continue
            if coef_name not in params:
                continue
            if applies_to in {"male", "sm"} and not data.is_male:
                continue
            if applies_to in {"female", "sf"} and data.is_male:
                continue
            if applies_to in {"cm", "cf"}:
                continue
            # "household" falls through to the same single-variable path as "both":
            # singles data has reg2/drgur/year_* directly (no gender split needed).
            if not hasattr(data, var_name):
                logger.warning(
                    "Market opportunity (singles): skipping shifter '%s' "
                    "-- variable '%s' not found on data object", coef_name, var_name
                )
                continue
            var_param = getattr(data, var_name)
            if var_param is None:
                continue
            var_param = _apply_market_scale_numpy(var_param, var_name, scale_map)

            interaction_missing = False
            for interaction_name in interaction_terms:
                if interaction_name == "working":
                    interaction_param = data.working
                else:
                    interaction_param = getattr(data, interaction_name, None)
                if interaction_param is None:
                    logger.warning(
                        "Market opportunity (singles): skipping shifter '%s' "
                        "-- interaction variable '%s' not found on data",
                        coef_name, interaction_name
                    )
                    interaction_missing = True
                    break
                interaction_param = _apply_market_scale_numpy(
                    interaction_param, interaction_name, scale_map
                )
                var_param = var_param * interaction_param

            if interaction_missing:
                continue

            log_market = log_market + params[coef_name] * var_param
            if coef_name in components:
                components[coef_name] = components[coef_name] + var_param
            else:
                components[coef_name] = var_param

    if getattr(spec, "market_opportunity_center_within_choice_set", False):
        weights = data.prior if spec.market_opportunity_center_weights == "proposal" else None
        log_market = _center_within_choice_set(
            log_market, data.group_starts, data.group_ends, weights
        )
        for coef_name, var_param in list(components.items()):
            components[coef_name] = _center_within_choice_set(
                var_param, data.group_starts, data.group_ends, weights
            )

    return log_market, components


def _compute_market_opportunity_couples(
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    log_market = np.zeros(data.n_obs)
    components: Dict[str, np.ndarray] = {}
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}

    def _get_gender_var(base: str, gender: str) -> Optional[np.ndarray]:
        attr_name = f"{base}_{gender}"
        return getattr(data, attr_name, None) if hasattr(data, attr_name) else None

    if getattr(spec, "market_opportunity_shifters", None):
        for shifter in spec.market_opportunity_shifters:
            var_name = shifter.get("variable")
            coef_name = shifter.get("coefficient")
            interaction_terms = _normalize_interaction_terms(shifter.get("interaction", None))
            applies_to = str(shifter.get("applies_to", "both")).strip().lower()
            if not var_name or not coef_name:
                continue
            if coef_name not in params:
                continue

            if applies_to == "household":
                if not hasattr(data, var_name):
                    logger.warning(
                        "Market opportunity (couples/household): skipping '%s' "
                        "-- variable '%s' not found on data", coef_name, var_name
                    )
                    continue
                var_param = getattr(data, var_name)
                if var_param is None:
                    continue
                var_param = _apply_market_scale_numpy(var_param, var_name, scale_map)
                interaction_missing = False
                for interaction_name in interaction_terms:
                    if interaction_name == "working":
                        interaction_param = data.working_male + data.working_female
                    else:
                        interaction_param = getattr(data, interaction_name, None)
                    if interaction_param is None:
                        interaction_missing = True
                        break
                    interaction_param = _apply_market_scale_numpy(
                        interaction_param, interaction_name, scale_map
                    )
                    var_param = var_param * interaction_param
                if interaction_missing:
                    continue
                log_market = log_market + params[coef_name] * var_param
                components[coef_name] = components.get(coef_name, 0.0) + var_param
                continue

            if applies_to in ("male", "cm", "both"):
                var_param_m = _get_gender_var(var_name, "male")
                if var_param_m is None:
                    logger.warning(
                        "Market opportunity (couples/male): skipping '%s' "
                        "-- variable '%s_male' not found on data", coef_name, var_name
                    )
                if var_param_m is not None:
                    var_param_m = _apply_market_scale_numpy(var_param_m, var_name, scale_map)
                    interaction_missing = False
                    for interaction_name in interaction_terms:
                        if interaction_name == "working":
                            interaction_param = data.working_male
                        else:
                            interaction_param = _get_gender_var(interaction_name, "male")
                        if interaction_param is None:
                            logger.warning(
                                "Market opportunity (couples/male): skipping '%s' "
                                "-- interaction '%s_male' not found", coef_name, interaction_name
                            )
                            interaction_missing = True
                            break
                        interaction_param = _apply_market_scale_numpy(
                            interaction_param, interaction_name, scale_map
                        )
                        var_param_m = var_param_m * interaction_param
                    if not interaction_missing:
                        log_market = log_market + params[coef_name] * var_param_m
                        components[coef_name] = components.get(coef_name, 0.0) + var_param_m

            if applies_to in ("female", "cf", "both"):
                var_param_f = _get_gender_var(var_name, "female")
                if var_param_f is None:
                    logger.warning(
                        "Market opportunity (couples/female): skipping '%s' "
                        "-- variable '%s_female' not found on data", coef_name, var_name
                    )
                if var_param_f is not None:
                    var_param_f = _apply_market_scale_numpy(var_param_f, var_name, scale_map)
                    interaction_missing = False
                    for interaction_name in interaction_terms:
                        if interaction_name == "working":
                            interaction_param = data.working_female
                        else:
                            interaction_param = _get_gender_var(interaction_name, "female")
                        if interaction_param is None:
                            logger.warning(
                                "Market opportunity (couples/female): skipping '%s' "
                                "-- interaction '%s_female' not found", coef_name, interaction_name
                            )
                            interaction_missing = True
                            break
                        interaction_param = _apply_market_scale_numpy(
                            interaction_param, interaction_name, scale_map
                        )
                        var_param_f = var_param_f * interaction_param
                    if not interaction_missing:
                        log_market = log_market + params[coef_name] * var_param_f
                        components[coef_name] = components.get(coef_name, 0.0) + var_param_f

    if getattr(spec, "market_opportunity_center_within_choice_set", False):
        weights = data.prior if spec.market_opportunity_center_weights == "proposal" else None
        log_market = _center_within_choice_set(
            log_market, data.group_starts, data.group_ends, weights
        )
        for coef_name, var_param in list(components.items()):
            components[coef_name] = _center_within_choice_set(
                var_param, data.group_starts, data.group_ends, weights
            )

    return log_market, components


# ==============================================================================
# Singles Estimation - Likelihood
# ==============================================================================

def compute_likelihood_singles(
    theta: np.ndarray,
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
    return_components: bool = False
) -> Union[float, Dict[str, np.ndarray]]:
    """
    Compute negative log-likelihood for singles estimation.

    The likelihood is based on the MNL model:
        P_i(j) = exp(V_ij) / Σ_k exp(V_ik)

    Where the composite value function is:
        V_ij = u(c_ij, l_ij; X_i, θ_pref) + log h(h_ij|X_i; θ_h)
               + log w(w_ij|X_i; θ_w) - log π(h_ij, w_ij)

    Components:
    - u: Utility function (Box-Cox with demographic shifters)
    - log h: Hours opportunity density
    - log w: Wage opportunity density (if vw or loc_empirical)
    - log π: Prior (importance sampling correction)

    Parameters
    ----------
    theta : np.ndarray, shape (n_params,)
        Parameter vector
    data : PrecomputedDataSingles
        Precomputed data arrays
    spec : EstimationSpec
        Specification configuration
    return_components : bool, default=False
        If True, return dict with V, u, log_h, log_w components

    Returns
    -------
    float or dict
        If return_components=False: negative log-likelihood (for minimization)
        If return_components=True: dict with components and likelihood
    """
    # Unpack parameters
    params = spec.unpack_parameters(theta)

    # ===== 1. COMPUTE UTILITY =====
    u = _compute_utility_singles(params, data, spec)

    # ===== 2. COMPUTE HOURS OPPORTUNITY =====
    # Use gender-specific parameters: _male for males, _female for females
    log_h = _compute_hours_opportunity_singles(params, data, spec, is_male=data.is_male)

    # ===== 3. COMPUTE WAGE OPPORTUNITY =====
    if spec.wage_spec == "fw":
        log_w = np.zeros(data.n_obs)  # Fixed wages: no wage component
    elif spec.wage_spec == "vw":
        log_w = _compute_wage_opportunity_vw_singles(params, data, spec)
    elif spec.wage_spec == "loc_empirical":
        log_w = _compute_wage_opportunity_loc_singles(params, data, spec)
    else:
        raise ValueError(f"Unknown wage_spec: {spec.wage_spec}")

    # ===== 4. COMPOSITE VALUE FUNCTION =====
    # V = u + log h + log w - log π
    log_market, _ = _compute_market_opportunity_singles(params, data, spec)
    V = u + log_h + log_w + log_market - np.log(data.prior)

    # Validate likelihood computation
    if not np.all(np.isfinite(V)):
        n_nan = np.sum(np.isnan(V))
        n_inf = np.sum(np.isinf(V))
        logger.error(f"Non-finite V: {n_nan} NaN, {n_inf} Inf")
        logger.error(f"u finite: {np.all(np.isfinite(u))}")
        logger.error(f"log_h finite: {np.all(np.isfinite(log_h))}")
        logger.error(f"log_w finite: {np.all(np.isfinite(log_w))}")
        logger.error(f"prior finite: {np.all(np.isfinite(data.prior))}")
        raise ValueError("Likelihood contains NaN/Inf - check inputs")

    # ===== 6. COMPUTE LOG-LIKELIHOOD =====
    # Log-sum-exp for each choice set
    lse = compute_log_sum_exp_by_group(V, data.group_starts, data.group_ends)

    # Extract V for observed choices (draw==0, which is first in each group)
    V_obs = np.array([V[start] for start in data.group_starts])

    # Log-likelihood: Σ_i [V_obs_i - log_sum_exp_i]
    ll = np.sum(V_obs - lse)

    # Expression-constraint penalty is a solver/optimization-layer concern
    # (migration matrix Wave 2.4), NOT part of the core reference likelihood.
    # The lifted engine returns the PURE negLL (matches the JAX engine and the
    # certified figure 238504.636097). Kept as 0.0 for return-shape parity.
    penalty = 0.0

    neg_ll = -ll + penalty

    if return_components:
        return {
            'V': V,
            'u': u,
            'log_h': log_h,
            'log_w': log_w,
            'log_market': log_market,
            'lse': lse,
            'V_obs': V_obs,
            'll': ll,
            'neg_ll': neg_ll,
            'expr_penalty': penalty,
        }
    else:
        return neg_ll  # Negative LL plus optional expression-constraint penalty


def _compute_utility_singles(
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> np.ndarray:
    """
    Compute utility function for singles.

    For Box-Cox specification:
        u = [β_l0 + Σ β_l_X * X] * BC(l; θ_l) + β_c * BC(c; θ_c)

    With 4-group architecture, singles male and singles female have separate parameters:
    - Singles Male: use parameters with _sm suffix
    - Singles Female: use parameters with _sf suffix
    
    For AC2013 specification:
        - Age enters as log(age) and log(age)² instead of linear
        - Children can enter as C1, C2, C3 age groups

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Utility values
    """
    if spec.utility_form != "box_cox":
        raise NotImplementedError(f"Utility form {spec.utility_form} not implemented")

    # Determine gender suffix for parameter names
    gender_suffix = "_sm" if data.is_male else "_sf"
    singles_group = "singles_male" if data.is_male else "singles_female"

    # Box-Cox transformations with gender-specific exponents
    # Handle optional theta parameters (can be None for log utility)
    if spec.utility_leisure_theta:
        theta_l_name = f"{spec.utility_leisure_theta}{gender_suffix}"
        theta_l = params[theta_l_name]
    else:
        theta_l = 0.0  # Log utility if theta not specified

    # Consumption theta name: M0a-clean shares theta_c_singles across sm/sf;
    # otherwise this returns the gender-suffixed legacy name.
    theta_c_name = spec.theta_c_param_name(singles_group)
    if theta_c_name:
        theta_c = params[theta_c_name]
    else:
        theta_c = 0.0  # Log utility if theta not specified

    bc_l = box_cox_transform(data.leisure, theta_l)
    bc_c = box_cox_transform(data.consumption, theta_c)

    # Leisure coefficient (intercept + demographic shifters) - gender-specific
    beta_l0_name = f"{spec.utility_leisure_intercept}{gender_suffix}"
    beta_l_coeff = params[beta_l0_name]  # beta_l0_sm or beta_l0_sf

    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_base = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)

        # Get data array
        if hasattr(data, var_name):
            var_data = getattr(data, var_name)
        else:
            raise ValueError(f"Variable {var_name} not found in precomputed data")

        # Add to coefficient (with gender restriction if needed)
        if is_gender_specific:
            # n_children: only for females
            if var_name == "n_children" and data.is_male:
                continue  # Skip for males

        # Use gender-specific parameter name
        coef_name = f"{coef_base}{gender_suffix}"
        
        # AC2013: Transform age variables to log form.
        # AC2013 utilities (estimation_utils_AC2013) were NOT lifted into core
        # (out of scope per memo §L: non-baseline model variants). The certified
        # baseline is legacy (is_ac2013() is False), so this branch is never taken;
        # guard it explicitly instead of importing an old-repo module.
        if spec.is_ac2013() and var_name == "age_norm":
            raise NotImplementedError(
                "AC2013 log-age utilities are not lifted into dclaborsupply core "
                "(legacy/baseline path only). See migration matrix §L."
            )
        
        beta_l_coeff = beta_l_coeff + params[coef_name] * var_data

    # Consumption coefficient - gender-specific.
    # If beta_c is fixed (scale normalisation), use the compile-time constant;
    # it was removed from all_param_names so it is NOT in `params`.
    beta_c_name = f"{spec.utility_consumption_coef}{gender_suffix}"
    if getattr(spec, "utility_consumption_coef_fixed", None) is not None:
        beta_c = spec.utility_consumption_coef_fixed
    else:
        beta_c = params[beta_c_name]

    # Optional consumption-leisure interaction (group-specific if available)
    beta_cl = 0.0
    if spec.utility_consumption_leisure_interaction_coef:
        beta_cl_name = f"{spec.utility_consumption_leisure_interaction_coef}{gender_suffix}"
        if beta_cl_name in params:
            beta_cl = params[beta_cl_name]
        elif spec.utility_consumption_leisure_interaction_coef in params:
            beta_cl = params[spec.utility_consumption_leisure_interaction_coef]

    # Total utility
    u = beta_l_coeff * bc_l + beta_c * bc_c + beta_cl * bc_c * bc_l

    return u


def _compute_hours_opportunity_singles(
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
    is_male: bool = True
) -> np.ndarray:
    """
    Compute log hours opportunity density for singles.

    log h(h|X) = Σ β_h * X_h

    Where X_h includes:
    - working indicator
    - focal points (PT1, PT2, FT)
    - GSUR × working
    - education × working

    Uses gender-specific parameters: _male suffix for males, _female for females.
    This allows different hours opportunity effects by gender while sharing
    parameters between singles and couples of the same gender.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    is_male : bool
        True for male, False for female (determines parameter suffix)

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Log hours opportunity density
    """
    log_h = np.zeros(data.n_obs)
    gender_suffix = "_male" if is_male else "_female"

    for shifter_config in spec.hours_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        interaction = shifter_config.get("interaction", None)

        # Use gender-specific coefficient name
        coef_name_gender = f"{coef_name}{gender_suffix}"

        # Fall back to base name if gender-specific not found (for backward compatibility)
        if coef_name_gender not in params and coef_name in params:
            coef_name_gender = coef_name

        # Get data array
        if hasattr(data, var_name):
            var_data = getattr(data, var_name)
        else:
            raise ValueError(f"Variable {var_name} not found in precomputed data")

        # Apply interaction if specified
        if interaction:
            if interaction == "working":
                var_data = var_data * data.working
            else:
                raise ValueError(f"Unknown interaction: {interaction}")

        if coef_name_gender in params:
            log_h = log_h + params[coef_name_gender] * var_data

    return log_h


def _compute_wage_opportunity_vw_singles(
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> np.ndarray:
    """
    Compute log wage opportunity density for singles (variable wages).

    Mincer equation:
        log w ~ N(μ(X), σ²)
        μ(X) = β_w0 + β_educL * educL + β_educH * educH
               + β_pexp * pexp + β_pexp2 * pexp²

    Log-likelihood contribution:
        log w(w|X) = -0.5 * [(log w - μ)² / σ²] - log(σ) - 0.5 * log(2π)

    Only computed for workers (hours > 0), zero otherwise.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Log wage opportunity density
    """
    if data.log_wage is None:
        raise ValueError("log_wage not available in data (required for vw specification)")

    # Compute mean wage (μ)
    mu_w = np.zeros(data.n_obs)

    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        if var_name == "intercept":
            mu_w = mu_w + params[coef_name]
        elif hasattr(data, var_name):
            var_data = getattr(data, var_name)
            if var_data is not None:
                mu_w = mu_w + params[coef_name] * var_data
        else:
            raise ValueError(f"Variable {var_name} not found in precomputed data")

    # Standard deviation
    sigma = params[spec.wage_variance_param]

    # Log-normal density
    residual = (data.log_wage - mu_w) / sigma
    log_w = -0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi) - data.log_wage

    # Zero for non-workers
    log_w = np.where(data.working > 0, log_w, 0.0)

    return log_w


def _compute_wage_opportunity_loc_singles(
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> np.ndarray:
    """
    Compute log wage opportunity density for singles (occupation-based).

    For each LOC group g:
        log w ~ N(μ_g(X), σ_g²)
        μ_g(X) = β_w0_g + Σ β_common * X

    Total density:
        log w(w|X) = Σ_g 1{loc=g} * log N(log w; μ_g, σ_g²)

    Only computed for workers, zero otherwise.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Log wage opportunity density
    """
    if data.log_wage is None or data.loc4 is None:
        raise ValueError("log_wage and loc4 required for loc_empirical specification")

    log_w = np.zeros(data.n_obs)

    # Common shifters (computed once, used for all groups)
    common_shift = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        if hasattr(data, var_name):
            var_data = getattr(data, var_name)
            if var_data is not None:
                common_shift = common_shift + params[coef_name] * var_data
        else:
            raise ValueError(f"Variable {var_name} not found in precomputed data")

    # LOC-specific densities
    for group_config in spec.wage_loc_groups:
        group_id = group_config["group_id"]
        var_name = group_config["variable"]  # loc4_1, loc4_2, etc.
        intercept_name = group_config["intercept"]
        sigma_name = group_config["sigma"]

        # Get indicator for this occupation
        if hasattr(data, var_name):
            loc_indicator = getattr(data, var_name)
        else:
            raise ValueError(f"Variable {var_name} not found in precomputed data")

        # Mean for this group
        mu_g = params[intercept_name] + common_shift

        # Standard deviation for this group
        sigma_g = params[sigma_name]

        # Log-normal density for this group
        residual = (data.log_wage - mu_g) / sigma_g
        log_w_g = -0.5 * residual**2 - np.log(sigma_g) - 0.5 * np.log(2 * np.pi) - data.log_wage

        # Add contribution (weighted by indicator)
        log_w = log_w + loc_indicator * log_w_g

    # Zero for non-workers
    log_w = np.where(data.working > 0, log_w, 0.0)

    return log_w


# ==============================================================================
# Singles Estimation - Analytical Gradient
# ==============================================================================

def compute_gradient_singles(
    theta: np.ndarray,
    data: PrecomputedDataSingles,
    spec: EstimationSpec,
    return_scores: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute analytical gradient ∇_θ(-LL) for singles estimation.

    Uses the chain rule:
        ∂(-LL)/∂θ_k = -Σ_i [∂V_obs_i/∂θ_k - E_{j~i}[∂V_j/∂θ_k]]

    Where E_{j~i}[·] is the softmax-weighted expectation over choice set i:
        E_{j~i}[∂V_j/∂θ_k] = Σ_j P_ij * ∂V_ij/∂θ_k

    The gradient is computed by:
    1. Building dV/dθ matrix for all observations
    2. Computing softmax probabilities for each choice set
    3. Computing weighted average (expectation) per group
    4. Computing difference between observed and expected derivatives

    Parameters
    ----------
    theta : np.ndarray, shape (n_params,)
        Parameter vector
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    return_scores : bool, default=False
        If True, also return per-choice-set score matrix for the POSITIVE
        log-likelihood. Shape: (n_groups, n_params). The score for group g is
        s_g = (dV_obs_g - dV_exp_g), i.e. the negation of the per-group
        contribution to the negative-LL gradient. Satisfies:
        scores.sum(axis=0) == -grad_func(theta)  (i.e. == gradient of +LL).

    Returns
    -------
    np.ndarray, shape (n_params,)  [return_scores=False]
        Gradient vector of negative log-likelihood (for minimization).
    Tuple[np.ndarray, np.ndarray]  [return_scores=True]
        (grad, scores) where grad has shape (n_params,) and scores has
        shape (n_groups, n_params).
    """
    n_params = len(spec.all_param_names)
    params = spec.unpack_parameters(theta)

    # ===== 1. COMPUTE VALUE FUNCTION AND PROBABILITIES =====
    # Get V and components
    components = compute_likelihood_singles(theta, data, spec, return_components=True)
    V = components['V']
    lse = components['lse']

    # ===== 2. BUILD dV/dθ MATRIX =====
    dV_dtheta = np.zeros((data.n_obs, n_params))

    # 2a. Utility derivatives
    _compute_utility_derivatives_singles(dV_dtheta, params, data, spec)

    # 2b. Hours opportunity derivatives
    _compute_hours_derivatives_singles(dV_dtheta, params, data, spec)

    # 2c. Wage opportunity derivatives
    if spec.wage_spec == "vw":
        _compute_wage_derivatives_vw_singles(dV_dtheta, params, data, spec)
    elif spec.wage_spec == "loc_empirical":
        _compute_wage_derivatives_loc_singles(dV_dtheta, params, data, spec)
    # fw: no wage derivatives (all zeros)

    # 2d. Market opportunity derivatives
    _compute_market_derivatives_singles(dV_dtheta, params, data, spec)

    # ===== 3. COMPUTE GRADIENT VIA SOFTMAX WEIGHTING =====
    # Loop-based approach is faster than vectorized due to efficient NumPy @ operator
    grad = np.zeros(n_params)

    if return_scores:
        scores = np.zeros((data.n_groups, n_params))

    for g in range(data.n_groups):
        start, end = data.group_starts[g], data.group_ends[g]

        # Softmax probabilities for this group
        V_group = V[start:end]
        P_group = np.exp(V_group - lse[g])

        # Observed derivative (first alternative in group = draw 0)
        dV_obs = dV_dtheta[start, :]

        # Expected derivative (softmax-weighted average)
        dV_exp = P_group @ dV_dtheta[start:end, :]

        # Per-group score for the POSITIVE log-likelihood: s_g = dV_obs - dV_exp
        score_g = dV_obs - dV_exp

        # Add to gradient (of negative LL)
        grad += score_g

        if return_scores:
            scores[g, :] = score_g

    # Validate gradient computation
    if not np.all(np.isfinite(grad)):
        n_bad = np.sum(~np.isfinite(grad))
        logger = logging.getLogger(__name__)
        logger.error(f"Gradient contains {n_bad} non-finite entries")
        for i in range(n_params):
            if not np.isfinite(grad[i]):
                logger.error(f"  {spec.all_param_names[i]}: {grad[i]}")
        raise ValueError("Gradient contains NaN/Inf")

    neg_grad = -grad  # Gradient of negative LL for minimization

    if return_scores:
        # scores[g] = s_g = dV_obs_g - dV_exp_g  (gradient of POSITIVE LL per group)
        # Verify: scores.sum(axis=0) should equal -neg_grad = grad (positive LL gradient)
        return neg_grad, scores
    return neg_grad


def _compute_utility_derivatives_singles(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> None:
    """
    Compute utility derivatives and add to dV/dθ matrix (in-place).

    For Box-Cox utility:
        u = [β_l0 + Σ β_l_X * X] * BC(l; θ_l) + β_c * BC(c; θ_c)

    With 4-group architecture, singles male and singles female have separate parameters:
    - Singles Male: use parameters with _sm suffix
    - Singles Female: use parameters with _sf suffix

    Derivatives:
        ∂u/∂β_l0 = BC(l; θ_l)
        ∂u/∂β_l_X = X * BC(l; θ_l)
        ∂u/∂β_c = BC(c; θ_c)
        ∂u/∂θ_l = [β_l0 + Σ β_l_X * X] * ∂BC(l; θ_l)/∂θ_l
        ∂u/∂θ_c = β_c * ∂BC(c; θ_c)/∂θ_c

    Parameters
    ----------
    dV_dtheta : np.ndarray, shape (n_obs, n_params)
        Derivative matrix to update (in-place)
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    """
    # Determine gender suffix for parameter names
    gender_suffix = "_sm" if data.is_male else "_sf"
    singles_group = "singles_male" if data.is_male else "singles_female"

    # Box-Cox transformations with gender-specific exponents
    # Handle optional theta parameters (can be None for log utility)
    if spec.utility_leisure_theta:
        theta_l_name = f"{spec.utility_leisure_theta}{gender_suffix}"
        theta_l = params[theta_l_name]
    else:
        theta_l = 0.0  # Log utility if theta not specified

    # Consumption theta name (M0a-clean honours singles-shared exponent).
    theta_c_name = spec.theta_c_param_name(singles_group)
    if theta_c_name:
        theta_c = params[theta_c_name]
    else:
        theta_c = 0.0  # Log utility if theta not specified

    bc_l = box_cox_transform(data.leisure, theta_l)
    bc_c = box_cox_transform(data.consumption, theta_c)

    # Compute leisure coefficient - gender-specific
    beta_l0_name = f"{spec.utility_leisure_intercept}{gender_suffix}"
    beta_l_coeff = params[beta_l0_name]

    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_base = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)

        if is_gender_specific and var_name == "n_children" and data.is_male:
            continue

        if hasattr(data, var_name):
            var_data = getattr(data, var_name)
            coef_name = f"{coef_base}{gender_suffix}"
            beta_l_coeff = beta_l_coeff + params[coef_name] * var_data

    # Consumption coefficient - gender-specific.
    # If beta_c is fixed (scale normalisation), use the compile-time constant;
    # it is not in `params` and has no gradient column.
    beta_c_name = f"{spec.utility_consumption_coef}{gender_suffix}"
    beta_c_is_fixed = getattr(spec, "utility_consumption_coef_fixed", None) is not None
    if beta_c_is_fixed:
        beta_c = spec.utility_consumption_coef_fixed
    else:
        beta_c = params[beta_c_name]

    # Optional consumption-leisure interaction coefficient
    beta_cl = 0.0
    beta_cl_name: Optional[str] = None
    if spec.utility_consumption_leisure_interaction_coef:
        candidate_name = f"{spec.utility_consumption_leisure_interaction_coef}{gender_suffix}"
        if candidate_name in params:
            beta_cl_name = candidate_name
        elif spec.utility_consumption_leisure_interaction_coef in params:
            beta_cl_name = spec.utility_consumption_leisure_interaction_coef
        if beta_cl_name is not None:
            beta_cl = params[beta_cl_name]

    # Derivative w.r.t. leisure intercept - gender-specific
    idx_beta_l0 = spec.get_param_index(beta_l0_name)
    dV_dtheta[:, idx_beta_l0] = bc_l

    # Derivatives w.r.t. leisure shifters - gender-specific
    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_base = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)

        if is_gender_specific and var_name == "n_children" and data.is_male:
            continue

        coef_name = f"{coef_base}{gender_suffix}"
        idx = spec.get_param_index(coef_name)
        var_data = getattr(data, var_name)
        dV_dtheta[:, idx] = var_data * bc_l

    # Derivative w.r.t. consumption coefficient - gender-specific.
    # Skipped when beta_c is fixed (no parameter, no gradient column).
    if not beta_c_is_fixed:
        idx_beta_c = spec.get_param_index(beta_c_name)
        dV_dtheta[:, idx_beta_c] = bc_c

    # Derivative w.r.t. consumption-leisure interaction coefficient (if present)
    if beta_cl_name is not None:
        idx_beta_cl = spec.get_param_index(beta_cl_name)
        dV_dtheta[:, idx_beta_cl] = bc_c * bc_l

    # Derivative w.r.t. theta_l - gender-specific (only if theta_l exists)
    if spec.utility_leisure_theta:
        idx_theta_l = spec.get_param_index(theta_l_name)
        dbc_l_dtheta = box_cox_derivative_theta(data.leisure, theta_l)
        dV_dtheta[:, idx_theta_l] = (beta_l_coeff + beta_cl * bc_c) * dbc_l_dtheta

    # Derivative w.r.t. theta_c - gender-specific (only if theta_c exists)
    if spec.utility_consumption_theta:
        idx_theta_c = spec.get_param_index(theta_c_name)
        dbc_c_dtheta = box_cox_derivative_theta(data.consumption, theta_c)
        dV_dtheta[:, idx_theta_c] = (beta_c + beta_cl * bc_l) * dbc_c_dtheta


def _compute_hours_derivatives_singles(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> None:
    """
    Compute hours opportunity derivatives (in-place).

    For log h(h|X) = Σ β_h * X_h:
        ∂log h/∂β_h = X_h (with interactions if specified)

    Parameters
    ----------
    dV_dtheta : np.ndarray
        Derivative matrix to update
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    """
    for shifter_config in spec.hours_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        interaction = shifter_config.get("interaction", None)

        var_data = getattr(data, var_name)

        # Apply interaction
        if interaction == "working":
            var_data = var_data * data.working

        idx = spec.get_param_index(coef_name)
        dV_dtheta[:, idx] = var_data


def _compute_wage_derivatives_vw_singles(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> None:
    """
    Compute wage opportunity derivatives for variable wages (in-place).

    For log w ~ N(μ(X), σ²):
        log w(w|X) = -0.5 * [(log w - μ)² / σ²] - log(σ) - 0.5 * log(2π)

    Derivatives:
        ∂log w/∂β_w = (log w - μ) / σ² * ∂μ/∂β_w = residual / σ * X
        ∂log w/∂σ = -1/σ + (log w - μ)² / σ³ = -1/σ + residual² / σ

    Only non-zero for workers.

    Parameters
    ----------
    dV_dtheta : np.ndarray
        Derivative matrix to update
    params : dict
        Parameter dictionary    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    """
    # Compute μ(X)
    mu_w = np.zeros(data.n_obs)
    
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        if var_name == "intercept":
            mu_w = mu_w + params[coef_name]
        else:
            var_data = getattr(data, var_name)
            if var_data is not None:
                mu_w = mu_w + params[coef_name] * var_data
    
    sigma = params[spec.wage_variance_param]
    residual = (data.log_wage - mu_w) / sigma

    # Derivatives w.r.t. mean parameters
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        idx = spec.get_param_index(coef_name)

        if var_name == "intercept":
            deriv = residual / sigma
            dV_dtheta[:, idx] = np.where(data.working > 0, deriv, 0.0)
        else:
            var_data = getattr(data, var_name, None)
            if var_data is not None:
                deriv = residual / sigma * var_data
                dV_dtheta[:, idx] = np.where(data.working > 0, deriv, 0.0)

    # Derivative w.r.t. sigma
    idx_sigma = spec.get_param_index(spec.wage_variance_param)
    deriv_sigma = -1.0 / sigma + residual**2 / sigma
    dV_dtheta[:, idx_sigma] = np.where(data.working > 0, deriv_sigma, 0.0)


def _compute_wage_derivatives_loc_singles(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> None:
    """
    Compute wage opportunity derivatives for occupation-based wages (in-place).

    For each occupation group g:
        log w_g ~ N(μ_g(X), σ_g²)
        μ_g = β_w0_g + Σ β_common * X

    Derivatives are weighted by occupation indicators.

    Parameters
    ----------
    dV_dtheta : np.ndarray
        Derivative matrix to update
    params : dict
        Parameter dictionary
    data : PrecomputedDataSingles
        Precomputed data
    spec : EstimationSpec
        Specification
    """
    # Common shifters (computed for all groups)
    common_shift = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        var_data = getattr(data, var_name)
        if var_data is not None:
            common_shift = common_shift + params[coef_name] * var_data

    # Derivatives for each LOC group
    for group_config in spec.wage_loc_groups:
        var_name = group_config["variable"]
        intercept_name = group_config["intercept"]
        sigma_name = group_config["sigma"]

        loc_indicator = getattr(data, var_name)
        mu_g = params[intercept_name] + common_shift
        sigma_g = params[sigma_name]
        residual_g = (data.log_wage - mu_g) / sigma_g

        # Derivative w.r.t. group intercept
        idx_intercept = spec.get_param_index(intercept_name)
        deriv_intercept = residual_g / sigma_g * loc_indicator
        dV_dtheta[:, idx_intercept] = np.where(data.working > 0, deriv_intercept, 0.0)

        # Derivative w.r.t. group sigma
        idx_sigma = spec.get_param_index(sigma_name)
        deriv_sigma = (-1.0 / sigma_g + residual_g**2 / sigma_g) * loc_indicator
        dV_dtheta[:, idx_sigma] = np.where(data.working > 0, deriv_sigma, 0.0)

    # Derivatives w.r.t. common shifters (sum over all groups)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        idx = spec.get_param_index(coef_name)
        var_data = getattr(data, var_name)

        deriv_common = np.zeros(data.n_obs)

        for group_config in spec.wage_loc_groups:
            loc_var = group_config["variable"]
            intercept_name = group_config["intercept"]
            sigma_name = group_config["sigma"]

            loc_indicator = getattr(data, loc_var)
            mu_g = params[intercept_name] + common_shift
            sigma_g = params[sigma_name]
            residual_g = (data.log_wage - mu_g) / sigma_g

            if var_data is not None:


                deriv_common += residual_g / sigma_g * var_data * loc_indicator

        dV_dtheta[:, idx] = np.where(data.working > 0, deriv_common, 0.0)


def _compute_market_derivatives_singles(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataSingles,
    spec: EstimationSpec
) -> None:
    """
    Compute market-opportunity derivatives for singles (in-place).
    """
    _, components = _compute_market_opportunity_singles(params, data, spec)
    for coef_name, var_param in components.items():
        if coef_name not in spec.all_param_names:
            continue
        idx = spec.get_param_index(coef_name)
        dV_dtheta[:, idx] = dV_dtheta[:, idx] + var_param


def _compute_market_derivatives_couples(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec
) -> None:
    """
    Compute market-opportunity derivatives for couples (in-place).
    """
    _, components = _compute_market_opportunity_couples(params, data, spec)
    for coef_name, var_param in components.items():
        if coef_name not in spec.all_param_names:
            continue
        idx = spec.get_param_index(coef_name)
        dV_dtheta[:, idx] = dV_dtheta[:, idx] + var_param


# ==============================================================================
# Couples Estimation - Likelihood
# ==============================================================================

def compute_likelihood_couples(
    theta: np.ndarray,
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    return_components: bool = False
) -> Union[float, Dict[str, np.ndarray]]:
    """
    Compute negative log-likelihood for couples estimation.

    The value function for couples includes:
    - Separate utility for male and female leisure
    - Interaction term between male and female leisure
    - Shared household consumption
    - Separate hours and wage opportunities for male and female

    V = u_male(l_m, c; X_m) + u_female(l_f, c; X_f) + β_interact * BC(l_m) * BC(l_f)
        + log h_m(h_m|X_m) + log h_f(h_f|X_f)
        + log w_m(w_m|X_m) + log w_f(w_f|X_f)
        - log π

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    data : PrecomputedDataCouples
        Precomputed couples data
    spec : EstimationSpec
        Specification
    return_components : bool
        If True, return components dict

    Returns
    -------
    float or dict
        Negative log-likelihood or components dict
    """
    params = spec.unpack_parameters(theta)

    # ===== 1. COMPUTE UTILITY =====
    u = _compute_utility_couples(params, data, spec)

    # ===== 2. COMPUTE HOURS OPPORTUNITY =====
    log_h_male = _compute_hours_opportunity_couples_gender(params, data, spec, is_male=True)
    log_h_female = _compute_hours_opportunity_couples_gender(params, data, spec, is_male=False)
    log_h = log_h_male + log_h_female

    # ===== 3. COMPUTE WAGE OPPORTUNITY =====
    if spec.wage_spec == "fw":
        log_w = np.zeros(data.n_obs)
    elif spec.wage_spec == "vw":
        log_w_male = _compute_wage_opportunity_vw_couples_gender(params, data, spec, is_male=True)
        log_w_female = _compute_wage_opportunity_vw_couples_gender(params, data, spec, is_male=False)
        log_w = log_w_male + log_w_female
    elif spec.wage_spec == "loc_empirical":
        log_w_male = _compute_wage_opportunity_loc_couples_gender(params, data, spec, is_male=True)
        log_w_female = _compute_wage_opportunity_loc_couples_gender(params, data, spec, is_male=False)
        log_w = log_w_male + log_w_female
    else:
        raise ValueError(f"Unknown wage_spec: {spec.wage_spec}")

    # ===== 4. COMPUTE MARKET OPPORTUNITY =====
    log_market, _ = _compute_market_opportunity_couples(params, data, spec)

    # ===== 5. COMPOSITE VALUE FUNCTION =====
    V = u + log_h + log_w + log_market - np.log(data.prior)

    # ===== 6. COMPUTE LOG-LIKELIHOOD =====
    lse = compute_log_sum_exp_by_group(V, data.group_starts, data.group_ends)
    V_obs = np.array([V[start] for start in data.group_starts])
    ll = np.sum(V_obs - lse)

    # Expression-constraint penalty -> solver layer (matrix Wave 2.4); the core
    # reference likelihood is the PURE negLL. Kept as 0.0 for return-shape parity.
    penalty = 0.0

    neg_ll = -ll + penalty

    if return_components:
        return {
            'V': V,
            'u': u,
            'log_h_male': log_h_male,
            'log_h_female': log_h_female,
            'log_w_male': log_w_male if spec.wage_spec != "fw" else None,
            'log_w_female': log_w_female if spec.wage_spec != "fw" else None,
            'log_market': log_market,
            'lse': lse,
            'V_obs': V_obs,
            'll': ll,
            'neg_ll': neg_ll,
            'expr_penalty': penalty,
        }
    else:
        return neg_ll


def _compute_utility_couples(
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec
) -> np.ndarray:
    """
    Compute utility function for couples.

    u = u_male + u_female + β_interact * BC(l_m) * BC(l_f)

    Where:
        u_male = [β_l0 + Σ β_l_X * X_male] * BC(l_male; θ_l) + β_c * BC(c; θ_c)
        u_female = [β_l0 + Σ β_l_X * X_female + β_l_nchild * n_children] * BC(l_female; θ_l) + β_c * BC(c; θ_c)

    Note: Consumption is shared, leisure is separate.
    Note: n_children only affects female utility (asymmetric).

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification

    Returns
    -------
    np.ndarray
        Total utility
    """
    if spec.utility_form != "box_cox":
        raise NotImplementedError(f"Utility form {spec.utility_form} not implemented")

    # Check if we have gender-specific couples parameters
    has_gender_specific = spec.has_couples_gender_specific_params()

    # Get Box-Cox exponents
    # Handle optional theta parameters (can be None for log utility)
    if spec.utility_leisure_theta:
        if has_gender_specific:
            # Use separate theta_l for male and female
            theta_l_m_name = f"{spec.utility_leisure_theta}_m"
            theta_l_f_name = f"{spec.utility_leisure_theta}_f"
            theta_l_male = params[theta_l_m_name]
            theta_l_female = params[theta_l_f_name]
        else:
            # Use shared theta_l (old behavior)
            theta_l_male = params[spec.utility_leisure_theta]
            theta_l_female = params[spec.utility_leisure_theta]
    else:
        # Log utility if theta not specified
        theta_l_male = 0.0
        theta_l_female = 0.0

    _couples_fixed_theta = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    if _couples_fixed_theta is not None:
        theta_c = float(_couples_fixed_theta)
    elif spec.utility_consumption_theta:
        theta_c = params[spec.utility_consumption_theta]
    else:
        theta_c = 0.0  # Log utility if theta not specified

    # Box-Cox transformations with gender-specific exponents
    bc_l_male = box_cox_transform(data.leisure_male, theta_l_male)
    bc_l_female = box_cox_transform(data.leisure_female, theta_l_female)
    # CRITICAL: For couples, consumption is HOUSEHOLD-LEVEL (normalized sum of male+female)
    # normalized_consumption_couples = (ils_dispy_male + ils_dispy_female) / mean(ils_dispy_male + ils_dispy_female)
    # The data.consumption field already contains the normalized household sum
    bc_c = box_cox_transform(data.consumption, theta_c)

    # Male leisure coefficient - use gender-specific parameters if available
    if has_gender_specific:
        beta_l_intercept_m = params[f"{spec.utility_leisure_intercept}_m"]
        beta_l_coeff_male = beta_l_intercept_m
    else:
        beta_l_coeff_male = params[spec.utility_leisure_intercept]

    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)

        # Skip n_children for males
        if is_gender_specific and var_name == "n_children":
            continue

        var_name_male = f"{var_name}_male"
        if hasattr(data, var_name_male):
            var_data = getattr(data, var_name_male)
            # Use gender-specific coefficient if available
            if has_gender_specific:
                coef_name_m = f"{coef_name}_m"
                beta_l_coeff_male = beta_l_coeff_male + params[coef_name_m] * var_data
            else:
                beta_l_coeff_male = beta_l_coeff_male + params[coef_name] * var_data

    # Female leisure coefficient (includes n_children) - use gender-specific parameters if available
    if has_gender_specific:
        beta_l_intercept_f = params[f"{spec.utility_leisure_intercept}_f"]
        beta_l_coeff_female = beta_l_intercept_f
    else:
        beta_l_coeff_female = params[spec.utility_leisure_intercept]

    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        if var_name == "n_children":
            # n_children has no _female suffix (it's household-level)
            var_data = data.n_children
        else:
            var_name_female = f"{var_name}_female"
            if hasattr(data, var_name_female):
                var_data = getattr(data, var_name_female)
            else:
                continue

        # Use gender-specific coefficient if available
        if has_gender_specific:
            coef_name_f = f"{coef_name}_f"
            beta_l_coeff_female = beta_l_coeff_female + params[coef_name_f] * var_data
        else:
            beta_l_coeff_female = beta_l_coeff_female + params[coef_name] * var_data

    # Consumption coefficient (shared).
    # If beta_c is fixed (scale normalisation), use the compile-time constant;
    # it was removed from all_param_names so it is NOT in `params`.
    if getattr(spec, "utility_consumption_coef_fixed", None) is not None:
        beta_c = spec.utility_consumption_coef_fixed
    else:
        beta_c = params[spec.utility_consumption_coef]

    # Male and female leisure utility (CONSUMPTION ADDED ONLY ONCE, NOT TWICE!)
    # This matches the R reference code where consumption appears once in total utility
    u_male_leisure = beta_l_coeff_male * bc_l_male
    u_female_leisure = beta_l_coeff_female * bc_l_female
    u_consumption = beta_c * bc_c  # Consumption is household public good, added once

    # Optional consumption-leisure interactions
    beta_cl_male = 0.0
    beta_cl_female = 0.0
    beta_cl_name_m: Optional[str] = None
    beta_cl_name_f: Optional[str] = None
    if spec.utility_consumption_leisure_interaction_coef:
        base_name = spec.utility_consumption_leisure_interaction_coef
        if has_gender_specific:
            candidate_m = f"{base_name}_m"
            candidate_f = f"{base_name}_f"
            if candidate_m in params:
                beta_cl_name_m = candidate_m
            elif base_name in params:
                beta_cl_name_m = base_name
            if candidate_f in params:
                beta_cl_name_f = candidate_f
            elif base_name in params:
                beta_cl_name_f = base_name
        else:
            if base_name in params:
                beta_cl_name_m = base_name
                beta_cl_name_f = base_name
            else:
                if f"{base_name}_m" in params:
                    beta_cl_name_m = f"{base_name}_m"
                if f"{base_name}_f" in params:
                    beta_cl_name_f = f"{base_name}_f"

        if beta_cl_name_m is not None:
            beta_cl_male = params[beta_cl_name_m]
        if beta_cl_name_f is not None:
            beta_cl_female = params[beta_cl_name_f]

    u_consumption_leisure = beta_cl_male * bc_c * bc_l_male + beta_cl_female * bc_c * bc_l_female

    # Interaction term (if specified)
    if spec.couples_interaction_coef:
        beta_interact = params[spec.couples_interaction_coef]
        u_interact = beta_interact * bc_l_male * bc_l_female
    else:
        u_interact = 0.0

    # Total utility: male_leisure + female_leisure + consumption + interaction
    # (NOT male_total + female_total which would double-count consumption)
    return u_male_leisure + u_female_leisure + u_consumption + u_consumption_leisure + u_interact


def _compute_hours_opportunity_couples_gender(
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> np.ndarray:
    """
    Compute log hours opportunity for one gender in couples.

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    is_male : bool
        True for male, False for female

    Returns
    -------
    np.ndarray
        Log hours opportunity
    """
    suffix = "_male" if is_male else "_female"
    gender_suffix = "_male" if is_male else "_female"  # For parameter names
    log_h = np.zeros(data.n_obs)

    for shifter_config in spec.hours_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        interaction = shifter_config.get("interaction", None)

        # Use gender-specific coefficient name
        coef_name_gender = f"{coef_name}{gender_suffix}"

        # Fall back to base name if gender-specific not found (for backward compatibility)
        if coef_name_gender not in params and coef_name in params:
            coef_name_gender = coef_name

        var_name_gender = f"{var_name}{suffix}"
        if hasattr(data, var_name_gender):
            var_data = getattr(data, var_name_gender)
        else:
            continue

        # Apply interaction
        if interaction == "working":
            working_var = f"working{suffix}"
            var_data = var_data * getattr(data, working_var)

        if coef_name_gender in params:
            log_h = log_h + params[coef_name_gender] * var_data

    return log_h


def _compute_wage_opportunity_vw_couples_gender(
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> np.ndarray:
    """
    Compute log wage opportunity for one gender in couples (vw).

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    is_male : bool
        True for male, False for female

    Returns
    -------
    np.ndarray
        Log wage opportunity
    """
    suffix = "_male" if is_male else "_female"

    log_wage_var = f"log_wage{suffix}"
    if not hasattr(data, log_wage_var) or getattr(data, log_wage_var) is None:
        raise ValueError(f"{log_wage_var} not available")

    log_wage = getattr(data, log_wage_var)
    working = getattr(data, f"working{suffix}")

    # Compute mean
    mu_w = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        if var_name == "intercept":
            mu_w = mu_w + params[coef_name]
        else:
            var_name_gender = f"{var_name}{suffix}"
            if hasattr(data, var_name_gender):
                var_data = getattr(data, var_name_gender)
                if var_data is not None:
                    mu_w = mu_w + params[coef_name] * var_data

    sigma = params[spec.wage_variance_param]
    residual = (log_wage - mu_w) / sigma
    log_w = -0.5 * residual**2 - np.log(sigma) - 0.5 * np.log(2 * np.pi) - log_wage

    return np.where(working > 0, log_w, 0.0)


def _compute_wage_opportunity_loc_couples_gender(
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> np.ndarray:
    """
    Compute log wage opportunity for one gender in couples (loc_empirical).

    Parameters
    ----------
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    is_male : bool
        True for male, False for female

    Returns
    -------
    np.ndarray
        Log wage opportunity
    """
    suffix = "_male" if is_male else "_female"

    log_wage_var = f"log_wage{suffix}"
    if not hasattr(data, log_wage_var) or getattr(data, log_wage_var) is None:
        raise ValueError(f"{log_wage_var} not available")

    log_wage = getattr(data, log_wage_var)
    working = getattr(data, f"working{suffix}")
    log_w = np.zeros(data.n_obs)

    # Common shifters
    common_shift = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]

        var_name_gender = f"{var_name}{suffix}"
        if hasattr(data, var_name_gender):
            var_data = getattr(data, var_name_gender)
            if var_data is not None:
                common_shift = common_shift + params[coef_name] * var_data

    # LOC-specific densities
    for group_config in spec.wage_loc_groups:
        var_name = group_config["variable"]
        intercept_name = group_config["intercept"]
        sigma_name = group_config["sigma"]

        var_name_gender = f"{var_name}{suffix}"
        if hasattr(data, var_name_gender):
            loc_indicator = getattr(data, var_name_gender)
        else:
            continue

        mu_g = params[intercept_name] + common_shift
        sigma_g = params[sigma_name]
        residual = (log_wage - mu_g) / sigma_g
        log_w_g = -0.5 * residual**2 - np.log(sigma_g) - 0.5 * np.log(2 * np.pi) - log_wage

        log_w = log_w + loc_indicator * log_w_g

    return np.where(working > 0, log_w, 0.0)


# ==============================================================================
# Couples Estimation - Analytical Gradient
# ==============================================================================

def compute_gradient_couples(
    theta: np.ndarray,
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    return_scores: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute analytical gradient for couples estimation.

    Similar to singles, but with:
    - Separate male/female components
    - Interaction term derivatives
    - Shared consumption derivatives

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    return_scores : bool, default=False
        If True, also return per-choice-set score matrix for the POSITIVE
        log-likelihood. Shape: (n_groups, n_params). The score for group g is
        s_g = (dV_obs_g - dV_exp_g). Satisfies:
        scores.sum(axis=0) == -grad_func(theta)  (gradient of +LL).

    Returns
    -------
    np.ndarray  [return_scores=False]
        Gradient vector of negative log-likelihood (for minimization).
    Tuple[np.ndarray, np.ndarray]  [return_scores=True]
        (grad, scores) where grad has shape (n_params,) and scores has
        shape (n_groups, n_params).
    """
    n_params = len(spec.all_param_names)
    params = spec.unpack_parameters(theta)

    # Compute V and probabilities
    components = compute_likelihood_couples(theta, data, spec, return_components=True)
    V = components['V']
    lse = components['lse']

    # Build dV/dθ matrix
    dV_dtheta = np.zeros((data.n_obs, n_params))

    # Utility derivatives
    _compute_utility_derivatives_couples(dV_dtheta, params, data, spec)

    # Hours derivatives (male + female)
    _compute_hours_derivatives_couples_gender(dV_dtheta, params, data, spec, is_male=True)
    _compute_hours_derivatives_couples_gender(dV_dtheta, params, data, spec, is_male=False)

    # Wage derivatives
    if spec.wage_spec == "vw":
        _compute_wage_derivatives_vw_couples_gender(dV_dtheta, params, data, spec, is_male=True)
        _compute_wage_derivatives_vw_couples_gender(dV_dtheta, params, data, spec, is_male=False)
    elif spec.wage_spec == "loc_empirical":
        _compute_wage_derivatives_loc_couples_gender(dV_dtheta, params, data, spec, is_male=True)
        _compute_wage_derivatives_loc_couples_gender(dV_dtheta, params, data, spec, is_male=False)

    # Market opportunity derivatives
    _compute_market_derivatives_couples(dV_dtheta, params, data, spec)

    # ===== 3. COMPUTE GRADIENT VIA SOFTMAX WEIGHTING =====
    # Loop-based approach is faster than vectorized due to efficient NumPy @ operator
    grad = np.zeros(n_params)

    if return_scores:
        scores = np.zeros((data.n_groups, n_params))

    for g in range(data.n_groups):
        start, end = data.group_starts[g], data.group_ends[g]

        # Softmax probabilities for this group
        V_group = V[start:end]
        P_group = np.exp(V_group - lse[g])

        # Observed derivative (first alternative in group = draw 0)
        dV_obs = dV_dtheta[start, :]

        # Expected derivative (softmax-weighted average)
        dV_exp = P_group @ dV_dtheta[start:end, :]

        # Per-group score for the POSITIVE log-likelihood
        score_g = dV_obs - dV_exp

        grad += score_g

        if return_scores:
            scores[g, :] = score_g

    neg_grad = -grad

    if return_scores:
        return neg_grad, scores
    return neg_grad


def _compute_utility_derivatives_couples(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec
) -> None:
    """
    Compute utility derivatives for couples (in-place).

    Handles:
    - Male and female leisure separately
    - Shared consumption (derivative counted twice)
    - Interaction term

    Parameters
    ----------
    dV_dtheta : np.ndarray
        Derivative matrix to update
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    """
    # Check if we have gender-specific couples parameters
    has_gender_specific = spec.has_couples_gender_specific_params()

    # Get Box-Cox exponents
    # Handle optional theta parameters (can be None for log utility)
    if spec.utility_leisure_theta:
        if has_gender_specific:
            theta_l_male = params[f"{spec.utility_leisure_theta}_m"]
            theta_l_female = params[f"{spec.utility_leisure_theta}_f"]
        else:
            theta_l_male = params[spec.utility_leisure_theta]
            theta_l_female = params[spec.utility_leisure_theta]
    else:
        # Log utility if theta not specified
        theta_l_male = 0.0
        theta_l_female = 0.0

    _couples_fixed_theta = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    if _couples_fixed_theta is not None:
        theta_c = float(_couples_fixed_theta)
    elif spec.utility_consumption_theta:
        theta_c = params[spec.utility_consumption_theta]
    else:
        theta_c = 0.0  # Log utility if theta not specified

    # Box-Cox transformations
    bc_l_male = box_cox_transform(data.leisure_male, theta_l_male)
    bc_l_female = box_cox_transform(data.leisure_female, theta_l_female)
    # CRITICAL: Consumption is HOUSEHOLD-LEVEL (normalized sum already computed)
    bc_c = box_cox_transform(data.consumption, theta_c)

    # Compute male and female leisure coefficients
    if has_gender_specific:
        beta_l_coeff_male = params[f"{spec.utility_leisure_intercept}_m"]
        beta_l_coeff_female = params[f"{spec.utility_leisure_intercept}_f"]
    else:
        beta_l_coeff_male = params[spec.utility_leisure_intercept]
        beta_l_coeff_female = params[spec.utility_leisure_intercept]

    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)

        # Male (skip n_children)
        if not (is_gender_specific and var_name == "n_children"):
            var_name_male = f"{var_name}_male"
            if hasattr(data, var_name_male):
                var_data = getattr(data, var_name_male)
                if has_gender_specific:
                    beta_l_coeff_male = beta_l_coeff_male + params[f"{coef_name}_m"] * var_data
                else:
                    beta_l_coeff_male = beta_l_coeff_male + params[coef_name] * var_data

        # Female (include n_children)
        if var_name == "n_children":
            var_data = data.n_children
        else:
            var_name_female = f"{var_name}_female"
            if hasattr(data, var_name_female):
                var_data = getattr(data, var_name_female)
            else:
                continue

        if has_gender_specific:
            beta_l_coeff_female = beta_l_coeff_female + params[f"{coef_name}_f"] * var_data
        else:
            beta_l_coeff_female = beta_l_coeff_female + params[coef_name] * var_data

    # If beta_c is fixed (scale normalisation), use the compile-time constant;
    # it is not in `params` and has no gradient column.
    beta_c_is_fixed = getattr(spec, "utility_consumption_coef_fixed", None) is not None
    if beta_c_is_fixed:
        beta_c = spec.utility_consumption_coef_fixed
    else:
        beta_c = params[spec.utility_consumption_coef]

    # Optional consumption-leisure interaction coefficients
    beta_cl_male = 0.0
    beta_cl_female = 0.0
    beta_cl_name_m: Optional[str] = None
    beta_cl_name_f: Optional[str] = None
    if spec.utility_consumption_leisure_interaction_coef:
        base_name = spec.utility_consumption_leisure_interaction_coef
        if has_gender_specific:
            candidate_m = f"{base_name}_m"
            candidate_f = f"{base_name}_f"
            if candidate_m in params:
                beta_cl_name_m = candidate_m
            elif base_name in params:
                beta_cl_name_m = base_name
            if candidate_f in params:
                beta_cl_name_f = candidate_f
            elif base_name in params:
                beta_cl_name_f = base_name
        else:
            if base_name in params:
                beta_cl_name_m = base_name
                beta_cl_name_f = base_name
            else:
                if f"{base_name}_m" in params:
                    beta_cl_name_m = f"{base_name}_m"
                if f"{base_name}_f" in params:
                    beta_cl_name_f = f"{base_name}_f"

        if beta_cl_name_m is not None:
            beta_cl_male = params[beta_cl_name_m]
        if beta_cl_name_f is not None:
            beta_cl_female = params[beta_cl_name_f]

    # DERIVATIVES - Gender-specific handling
    if has_gender_specific:
        # Separate derivatives for male and female intercepts
        idx_beta_l0_m = spec.get_param_index(f"{spec.utility_leisure_intercept}_m")
        idx_beta_l0_f = spec.get_param_index(f"{spec.utility_leisure_intercept}_f")
        dV_dtheta[:, idx_beta_l0_m] = bc_l_male
        dV_dtheta[:, idx_beta_l0_f] = bc_l_female
    else:
        # Shared intercept (old behavior)
        idx_beta_l0 = spec.get_param_index(spec.utility_leisure_intercept)
        dV_dtheta[:, idx_beta_l0] = bc_l_male + bc_l_female    # Derivatives w.r.t. leisure shifters
    for shifter_config in spec.utility_leisure_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        is_gender_specific = shifter_config.get("gender_specific", False)
        
        if has_gender_specific:
            # Separate derivatives for male and female parameters
            # Male derivative (skip n_children)
            if not (is_gender_specific and var_name == "n_children"):
                idx_m = spec.get_param_index(f"{coef_name}_m")
                var_name_male = f"{var_name}_male"
                if hasattr(data, var_name_male):
                    var_data = getattr(data, var_name_male)
                    if var_data is not None:
                        dV_dtheta[:, idx_m] = var_data * bc_l_male

            # Female derivative
            idx_f = spec.get_param_index(f"{coef_name}_f")
            if var_name == "n_children":
                var_data = data.n_children
            else:
                var_name_female = f"{var_name}_female"
                if hasattr(data, var_name_female):
                    var_data = getattr(data, var_name_female)
                else:
                    continue
            
            if var_data is not None:
                dV_dtheta[:, idx_f] = var_data * bc_l_female
        else:
            # Shared parameter (old behavior)
            idx = spec.get_param_index(coef_name)
            deriv = np.zeros(data.n_obs)            # Male contribution (skip n_children)
            if not (is_gender_specific and var_name == "n_children"):
                var_name_male = f"{var_name}_male"
                if hasattr(data, var_name_male):
                    var_data = getattr(data, var_name_male)
                    if var_data is not None:
                        deriv += var_data * bc_l_male
            
            # Female contribution
            if var_name == "n_children":
                var_data = data.n_children
            else:
                var_name_female = f"{var_name}_female"
                if hasattr(data, var_name_female):
                    var_data = getattr(data, var_name_female)
                else:
                    continue
            
            if var_data is not None:
                deriv += var_data * bc_l_female

            dV_dtheta[:, idx] = deriv

    # Derivative w.r.t. beta_c (consumption coefficient - shared household public good)
    # FIXED: Consumption appears ONCE in utility (not twice), matching R reference code.
    # Skipped when beta_c is fixed (no parameter, no gradient column).
    if not beta_c_is_fixed:
        idx_beta_c = spec.get_param_index(spec.utility_consumption_coef)
        dV_dtheta[:, idx_beta_c] = bc_c  # Once, not twice!

    # Derivatives w.r.t. consumption-leisure interaction coefficients
    if beta_cl_name_m is not None and beta_cl_name_f is not None and beta_cl_name_m == beta_cl_name_f:
        idx_beta_cl = spec.get_param_index(beta_cl_name_m)
        dV_dtheta[:, idx_beta_cl] = bc_c * (bc_l_male + bc_l_female)
    else:
        if beta_cl_name_m is not None:
            idx_beta_cl_m = spec.get_param_index(beta_cl_name_m)
            dV_dtheta[:, idx_beta_cl_m] = bc_c * bc_l_male
        if beta_cl_name_f is not None:
            idx_beta_cl_f = spec.get_param_index(beta_cl_name_f)
            dV_dtheta[:, idx_beta_cl_f] = bc_c * bc_l_female

    # Derivative w.r.t. theta_l - gender-specific handling (only if theta_l exists)
    if spec.utility_leisure_theta:
        if has_gender_specific:
            # Separate theta_l derivatives for male and female
            idx_theta_l_m = spec.get_param_index(f"{spec.utility_leisure_theta}_m")
            idx_theta_l_f = spec.get_param_index(f"{spec.utility_leisure_theta}_f")
            dbc_l_male_dtheta = box_cox_derivative_theta(data.leisure_male, theta_l_male)
            dbc_l_female_dtheta = box_cox_derivative_theta(data.leisure_female, theta_l_female)
            dV_dtheta[:, idx_theta_l_m] = beta_l_coeff_male * dbc_l_male_dtheta
            dV_dtheta[:, idx_theta_l_f] = beta_l_coeff_female * dbc_l_female_dtheta
            if beta_cl_name_m is not None:
                dV_dtheta[:, idx_theta_l_m] += beta_cl_male * bc_c * dbc_l_male_dtheta
            if beta_cl_name_f is not None:
                dV_dtheta[:, idx_theta_l_f] += beta_cl_female * bc_c * dbc_l_female_dtheta
        else:
            # Shared theta_l (old behavior - affects both male and female)
            idx_theta_l = spec.get_param_index(spec.utility_leisure_theta)
            dbc_l_male_dtheta = box_cox_derivative_theta(data.leisure_male, theta_l_male)
            dbc_l_female_dtheta = box_cox_derivative_theta(data.leisure_female, theta_l_female)
            dV_dtheta[:, idx_theta_l] = (beta_l_coeff_male * dbc_l_male_dtheta +
                                          beta_l_coeff_female * dbc_l_female_dtheta)
            dV_dtheta[:, idx_theta_l] += (
                beta_cl_male * bc_c * dbc_l_male_dtheta +
                beta_cl_female * bc_c * dbc_l_female_dtheta
            )

            # Add interaction term contribution to shared theta_l derivative
            if spec.couples_interaction_coef:
                beta_interact = params[spec.couples_interaction_coef]
                dV_dtheta[:, idx_theta_l] += beta_interact * (
                    dbc_l_male_dtheta * bc_l_female + bc_l_male * dbc_l_female_dtheta
                )

        # Add interaction term contribution to gender-specific theta_l derivatives
        if has_gender_specific and spec.couples_interaction_coef:
            beta_interact = params[spec.couples_interaction_coef]
            dV_dtheta[:, idx_theta_l_m] += beta_interact * dbc_l_male_dtheta * bc_l_female
            dV_dtheta[:, idx_theta_l_f] += beta_interact * bc_l_male * dbc_l_female_dtheta

    # Derivative w.r.t. theta_c (consumption Box-Cox parameter) - only if theta_c is estimated.
    # When utility_consumption_theta_couples_fixed is set, theta_c is a constant — no gradient.
    # CRITICAL: Consumption is HOUSEHOLD-LEVEL (normalized sum already computed)
    _couples_fixed_theta_d = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    if spec.utility_consumption_theta and _couples_fixed_theta_d is None:
        idx_theta_c = spec.get_param_index(spec.utility_consumption_theta)
        dbc_c_dtheta = box_cox_derivative_theta(data.consumption, theta_c)
        dV_dtheta[:, idx_theta_c] = (
            beta_c + beta_cl_male * bc_l_male + beta_cl_female * bc_l_female
        ) * dbc_c_dtheta

    # Derivative w.r.t. interaction coefficient
    if spec.couples_interaction_coef:
        idx_interact = spec.get_param_index(spec.couples_interaction_coef)
        dV_dtheta[:, idx_interact] = bc_l_male * bc_l_female


def _compute_hours_derivatives_couples_gender(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> None:
    """
    Compute hours opportunity derivatives for one gender in couples (in-place).

    Parameters
    ----------
    dV_dtheta : np.ndarray
        Derivative matrix to update
    params : dict
        Parameter dictionary
    data : PrecomputedDataCouples
        Precomputed data
    spec : EstimationSpec
        Specification
    is_male : bool
        True for male, False for female
    """
    suffix = "_male" if is_male else "_female"

    for shifter_config in spec.hours_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        interaction = shifter_config.get("interaction", None)

        var_name_gender = f"{var_name}{suffix}"
        if not hasattr(data, var_name_gender):
            continue

        var_data = getattr(data, var_name_gender)

        if interaction == "working":
            working_var = f"working{suffix}"
            var_data = var_data * getattr(data, working_var)

        idx = spec.get_param_index(coef_name)
        dV_dtheta[:, idx] += var_data  # += because both genders contribute


def _compute_wage_derivatives_vw_couples_gender(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> None:
    """
    Compute wage derivatives for one gender in couples (vw, in-place).
    """
    suffix = "_male" if is_male else "_female"

    log_wage = getattr(data, f"log_wage{suffix}")
    working = getattr(data, f"working{suffix}")    # Compute mean
    mu_w = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        
        if var_name == "intercept":
            mu_w = mu_w + params[coef_name]
        else:
            var_name_gender = f"{var_name}{suffix}"
            if hasattr(data, var_name_gender):
                var_data = getattr(data, var_name_gender)
                if var_data is not None:
                    mu_w = mu_w + params[coef_name] * var_data

    sigma = params[spec.wage_variance_param]
    residual = (log_wage - mu_w) / sigma
    
    # Derivatives w.r.t. mean parameters
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        idx = spec.get_param_index(coef_name)

        if var_name == "intercept":
            deriv = residual / sigma
            dV_dtheta[:, idx] += np.where(working > 0, deriv, 0.0)
        else:
            var_name_gender = f"{var_name}{suffix}"
            if hasattr(data, var_name_gender):
                var_data = getattr(data, var_name_gender)
                if var_data is not None:
                    deriv = residual / sigma * var_data
                    dV_dtheta[:, idx] += np.where(working > 0, deriv, 0.0)  # += for both genders

    # Derivative w.r.t. sigma
    idx_sigma = spec.get_param_index(spec.wage_variance_param)
    deriv_sigma = -1.0 / sigma + residual**2 / sigma
    dV_dtheta[:, idx_sigma] += np.where(working > 0, deriv_sigma, 0.0)


def _compute_wage_derivatives_loc_couples_gender(
    dV_dtheta: np.ndarray,
    params: Dict[str, float],
    data: PrecomputedDataCouples,
    spec: EstimationSpec,
    is_male: bool
) -> None:
    """
    Compute wage derivatives for one gender in couples (loc_empirical, in-place).
    """
    suffix = "_male" if is_male else "_female"

    log_wage = getattr(data, f"log_wage{suffix}")
    working = getattr(data, f"working{suffix}")

    # Common shifters
    common_shift = np.zeros(data.n_obs)
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        var_name_gender = f"{var_name}{suffix}"
        if hasattr(data, var_name_gender):
            var_data = getattr(data, var_name_gender)
            if var_data is not None:
                common_shift = common_shift + params[coef_name] * var_data

    # Derivatives for each LOC group
    for group_config in spec.wage_loc_groups:
        var_name = group_config["variable"]
        intercept_name = group_config["intercept"]
        sigma_name = group_config["sigma"]

        var_name_gender = f"{var_name}{suffix}"
        if not hasattr(data, var_name_gender):
            continue

        loc_indicator = getattr(data, var_name_gender)
        mu_g = params[intercept_name] + common_shift
        sigma_g = params[sigma_name]
        residual_g = (log_wage - mu_g) / sigma_g

        # Derivative w.r.t. group intercept
        idx_intercept = spec.get_param_index(intercept_name)
        deriv_intercept = residual_g / sigma_g * loc_indicator
        dV_dtheta[:, idx_intercept] += np.where(working > 0, deriv_intercept, 0.0)

        # Derivative w.r.t. group sigma
        idx_sigma = spec.get_param_index(sigma_name)
        deriv_sigma = (-1.0 / sigma_g + residual_g**2 / sigma_g) * loc_indicator
        dV_dtheta[:, idx_sigma] += np.where(working > 0, deriv_sigma, 0.0)

    # Derivatives w.r.t. common shifters
    for shifter_config in spec.wage_mean_shifters:
        var_name = shifter_config["variable"]
        coef_name = shifter_config["coefficient"]
        idx = spec.get_param_index(coef_name)

        var_name_gender = f"{var_name}{suffix}"
        if not hasattr(data, var_name_gender):
            continue
        var_data = getattr(data, var_name_gender)

        deriv_common = np.zeros(data.n_obs)
        for group_config in spec.wage_loc_groups:
            loc_var = f"{group_config['variable']}{suffix}"
            if not hasattr(data, loc_var):
                continue

            loc_indicator = getattr(data, loc_var)
            intercept_name = group_config["intercept"]
            sigma_name = group_config["sigma"]

            mu_g = params[intercept_name] + common_shift
            sigma_g = params[sigma_name]
            residual_g = (log_wage - mu_g) / sigma_g

            if var_data is not None:


                deriv_common += residual_g / sigma_g * var_data * loc_indicator

        dV_dtheta[:, idx] += np.where(working > 0, deriv_common, 0.0)


# ==============================================================================
# Joint Estimation (4-Group Architecture)
# ==============================================================================

def compute_likelihood_joint(
    theta: np.ndarray,
    data_singles_male: Optional[PrecomputedDataSingles],
    data_singles_female: Optional[PrecomputedDataSingles],
    data_couples: Optional[PrecomputedDataCouples],
    spec: EstimationSpec
) -> float:
    """
    Compute negative log-likelihood for joint estimation of all groups.

    This function sums the likelihoods from:
    - Singles male (using _sm parameters)
    - Singles female (using _sf parameters)
    - Couples (using _m, _f, and shared parameters)

    Parameters
    ----------
    theta : np.ndarray
        Full parameter vector (46 parameters for 4-group architecture)
    data_singles_male : PrecomputedDataSingles or None
        Male singles data
    data_singles_female : PrecomputedDataSingles or None
        Female singles data
    data_couples : PrecomputedDataCouples or None
        Couples data
    spec : EstimationSpec
        Specification

    Returns
    -------
    float
        Negative log-likelihood (for minimization)
    """
    ll_total = 0.0

    # Singles male contribution
    if data_singles_male is not None:
        ll_sm = compute_likelihood_singles(theta, data_singles_male, spec)
        ll_total += ll_sm

    # Singles female contribution
    if data_singles_female is not None:
        ll_sf = compute_likelihood_singles(theta, data_singles_female, spec)
        ll_total += ll_sf

    # Couples contribution
    if data_couples is not None:
        ll_c = compute_likelihood_couples(theta, data_couples, spec)
        ll_total += ll_c

    return ll_total  # Already negative for minimization


def compute_gradient_joint(
    theta: np.ndarray,
    data_singles_male: Optional[PrecomputedDataSingles],
    data_singles_female: Optional[PrecomputedDataSingles],
    data_couples: Optional[PrecomputedDataCouples],
    spec: EstimationSpec
) -> np.ndarray:
    """
    Compute gradient of negative log-likelihood for joint estimation.

    This function sums the gradients from all groups. Each group's gradient
    will have zeros for parameters not used by that group, which is correct
    because those parameters don't affect that group's likelihood.

    Parameters
    ----------
    theta : np.ndarray
        Full parameter vector (46 parameters)
    data_singles_male : PrecomputedDataSingles or None
        Male singles data
    data_singles_female : PrecomputedDataSingles or None
        Female singles data
    data_couples : PrecomputedDataCouples or None
        Couples data
    spec : EstimationSpec
        Specification

    Returns
    -------
    np.ndarray, shape (46,)
        Gradient of negative log-likelihood
    """
    grad_total = np.zeros(len(theta))

    # Singles male contribution
    if data_singles_male is not None:
        grad_sm = compute_gradient_singles(theta, data_singles_male, spec)
        grad_total += grad_sm

    # Singles female contribution
    if data_singles_female is not None:
        grad_sf = compute_gradient_singles(theta, data_singles_female, spec)
        grad_total += grad_sf

    # Couples contribution
    if data_couples is not None:
        grad_c = compute_gradient_couples(theta, data_couples, spec)
        grad_total += grad_c

    return grad_total


def compute_scores_joint(
    theta: np.ndarray,
    data_singles_male: Optional[PrecomputedDataSingles],
    data_singles_female: Optional[PrecomputedDataSingles],
    data_couples: Optional[PrecomputedDataCouples],
    spec: EstimationSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect per-choice-set score matrix and cluster ids for all groups.

    Returns the score matrix for the POSITIVE log-likelihood and the matching
    cluster id (idorighh) for each choice-set row. Used to assemble the meat
    matrix B for the clustered sandwich covariance estimator.

    Sandwich formula:
        V_cluster = H^{-1} B H^{-1},  B = sum_j s_j s_j',
        s_j = sum_{g: cluster_ids[g]==j} scores_all[g]

    Sign check (T1):
        scores_all.sum(axis=0) == -compute_gradient_joint(theta, ...)
        (i.e. equals the gradient of the positive LL)

    GA15 note:
        Singles data derives consumption from ils_dispy_real (singles only).
        Couples data derives consumption from ils_dispy_male + ils_dispy_female.
        The two paths are independent and must not be confused.

    Parameters
    ----------
    theta : np.ndarray, shape (n_params,)
        Parameter vector (may be initial_values for smoke test; need not be converged).
    data_singles_male : PrecomputedDataSingles or None
    data_singles_female : PrecomputedDataSingles or None
    data_couples : PrecomputedDataCouples or None
    spec : EstimationSpec

    Returns
    -------
    scores_all : np.ndarray, shape (n_groups_total, n_params)
        Per-choice-set score vectors for the positive log-likelihood.
        Row order: singles-male groups, singles-female groups, couples groups.
    cluster_ids_all : np.ndarray, shape (n_groups_total,)
        idorighh value for each choice-set row (aligned to scores_all).
    """
    scores_list = []
    cluster_ids_list = []

    if data_singles_male is not None:
        _, scores_sm = compute_gradient_singles(theta, data_singles_male, spec, return_scores=True)
        scores_list.append(scores_sm)
        cluster_ids_list.append(data_singles_male.cluster_ids)

    if data_singles_female is not None:
        _, scores_sf = compute_gradient_singles(theta, data_singles_female, spec, return_scores=True)
        scores_list.append(scores_sf)
        cluster_ids_list.append(data_singles_female.cluster_ids)

    if data_couples is not None:
        _, scores_c = compute_gradient_couples(theta, data_couples, spec, return_scores=True)
        scores_list.append(scores_c)
        cluster_ids_list.append(data_couples.cluster_ids)

    if not scores_list:
        raise ValueError("compute_scores_joint: all data objects are None")

    scores_all = np.vstack(scores_list)
    cluster_ids_all = np.concatenate(cluster_ids_list)

    return scores_all, cluster_ids_all


# ==============================================================================
# End of estimation_engine.py
# ==============================================================================
