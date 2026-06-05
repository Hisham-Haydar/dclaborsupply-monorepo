"""
==============================================================================
RURO MNL Specification Parser
==============================================================================
Parses and validates YAML specification files for MNL estimation.

Provides:
- YAML loading and validation
- EstimationSpec dataclass with all configuration
- Parameter name extraction and ordering
- Initial value and bounds extraction
- Specification validation

Author: Enhanced RURO Pipeline
Created: 2026-01-03

Lifted into the dclaborsupply core package (migration matrix Wave 1.1) from
MNL/scripts/enhanced/estimation_spec_parser.py. Adaptation (imports/paths only,
no parsing-logic change): dropped the repo-relative SPECIFICATIONS_DIR /
resolve_specification_path helper (old-repo path aliasing, unnecessary for
parsing an explicit path) and added the EstimationSpec.from_yaml public entry.
==============================================================================
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class EstimationSpec:
    """
    Parsed estimation specification.

    Contains all information needed for estimation:
    - Specification metadata (name, wage_spec)
    - Parameter names and structure
    - Initial values and bounds
    - Shifter configurations

    Supports model versions:
    - "legacy": original continuous RURO specification
    - "AC2013": Aaberge-Colombino (2013) aligned specification
    - "occupation_choice": Occupation as choice dimension (Aaberge-Colombino 2011)
    """
    # Metadata (required fields - no defaults)
    name: str
    description: str
    wage_spec: str  # fw | vw | vw_occupation | loc_empirical

    # Utility configuration (required fields)
    utility_form: str  # box_cox | log | linear
    utility_consumption_coef: str
    utility_consumption_theta: Optional[str]
    utility_leisure_intercept: str
    utility_leisure_theta: Optional[str]
    utility_leisure_shifters: List[Dict[str, Any]]
    utility_consumption_leisure_interaction_coef: Optional[str]

    # Hours opportunity configuration
    hours_shifters: List[Dict[str, Any]]

    # Job/market opportunity configuration (optional)
    market_opportunity_shifters: List[Dict[str, Any]]

    # Wage opportunity configuration
    wage_form: str  # log_normal | occupation_specific_log_normal | occupation_groups
    wage_mean_shifters: List[Dict[str, Any]]
    wage_variance_param: Optional[str]
    wage_loc_groups: Optional[List[Dict[str, Any]]]  # For loc_empirical

    # Couples configuration
    couples_interaction_coef: Optional[str]

    # === Fields with defaults below this line ===

    # Model version (NEW: AC2013 support)
    model_version: str = "legacy"  # "legacy" or "AC2013"
    model_family: str = "regular"  # regular | job_choice

    # Job-choice opportunity identification settings (optional)
    market_opportunity_tier: Optional[str] = None
    market_opportunity_offer_only_vars: List[str] = field(default_factory=list)
    market_opportunity_center_within_choice_set: bool = False
    market_opportunity_center_weights: str = "uniform"  # uniform | proposal
    market_opportunity_extra_dimension: Optional[str] = None
    market_opportunity_enforce_job_varying: bool = False
    market_opportunity_variable_scales: Dict[str, float] = field(default_factory=dict)

    # Consumption pooling (NEW: identification fix)
    pool_consumption_across_groups: bool = False  # If True, all groups share (beta_c, theta_c)

    # Singles-only shared consumption Box-Cox exponent (M0a-clean identification repair).
    # When set, both singles_male and singles_female read this single parameter instead
    # of separate theta_c_sm / theta_c_sf. Couples consumption theta is untouched.
    # YAML key: utility.consumption.singles_box_cox_exponent
    utility_consumption_theta_singles_shared: Optional[str] = None

    # Fixed (non-estimated) couples consumption Box-Cox exponent (M0c-b).
    # When set to a float, this value is used as a compile-time constant for
    # the couples BC-C transform; theta_c is NOT added to all_param_names.
    # Singles are unaffected (they still use utility_consumption_theta_singles_shared).
    # YAML key: utility.consumption.couples_fixed_box_cox_exponent
    utility_consumption_theta_couples_fixed: Optional[float] = None

    # Fixed (non-estimated) consumption COEFFICIENT beta_c (scale normalisation).
    # When set to a float, beta_c is the utility numeraire: it is used as a
    # compile-time constant in ALL blocks (beta_c_sm, beta_c_sf, couples beta_c)
    # and is NOT added to all_param_names. Breaks the consumption/leisure scale
    # ridge (beta_c co-scaling with beta_l0). YAML key:
    # utility.consumption.fixed_value. Mirrors the couples_fixed_box_cox_exponent
    # mechanism (compile-time constant, removed from the estimated vector).
    utility_consumption_coef_fixed: Optional[float] = None

    # Occupation choice configuration (NEW)
    occupation_choice: bool = False
    occupation_preferences: List[Dict[str, Any]] = field(default_factory=list)
    occupation_specific_hours: bool = False
    occupation_hour_configs: List[Dict[str, Any]] = field(default_factory=list)
    occupation_specific_wages: bool = False
    occupation_wage_configs: List[Dict[str, Any]] = field(default_factory=list)
    occupation_availability: List[Dict[str, Any]] = field(default_factory=list)

    # McFadden sampling configuration
    sampling_method: str = "standard"
    sampling_n_alternatives_per_occ: int = 100
    sampling_total_alternatives: int = 400
    sampling_stratified_by_occ: bool = False

    # Parameter management
    all_param_names: List[str] = field(default_factory=list)
    initial_values: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Generic fixed-parameter mechanism (package-agnostic): any parameter listed
    # in a top-level `fixed_params:` YAML block is PINNED to the given value,
    # removed from all_param_names/bounds, and resolved by the JAX backend's P()
    # helper. Use to pin weakly-identified params (e.g. a Box-Cox curvature whose
    # coefficient is ~0) without touching the model structure or the numpy engine.
    fixed_params: Dict[str, float] = field(default_factory=dict)
    gender_split: List[str] = field(default_factory=list)
    expression_constraints_enabled: bool = False
    expression_constraints_default_mode: str = "soft"
    expression_constraints_default_weight: float = 1e4
    expression_constraints: List[Dict[str, Any]] = field(default_factory=list)
    reporting: Dict[str, Any] = field(default_factory=dict)

    # Optimization settings
    opt_method: str = "L-BFGS-B"
    opt_analytical_gradient: bool = True
    opt_max_iterations: int = 10000
    opt_tolerance: float = 1e-6
    opt_gradient_tolerance: float = 1e-6  # gtol - NEW FIELD
    opt_display_convergence: bool = False  # NEW FIELD for disp
    opt_iprint: int = -1  # NEW FIELD for L-BFGS-B iteration printing    # Gradient verification settings
    grad_verify_enabled: bool = False
    grad_verify_method: str = "central"
    grad_verify_epsilon: float = 1e-7
    grad_verify_tolerance: float = 1e-4
    grad_verify_at_init: bool = True
    grad_verify_random_points: int = 0
    grad_verify_seed: int = 42
    grad_verify_verbose: bool = False
    
    # A-C 2013 specific settings (NEW)
    ac2013_use_log_age: bool = False           # Use log(age) instead of linear age
    ac2013_children_age_groups: bool = False   # Use C1, C2, C3 instead of n_children
    ac2013_experience_in_wage: bool = False    # Use exp, exp² in wage equation
    ac2013_couples_cross_leisure: bool = False # Use α_ll cross-leisure term
    ac2013_couples_mu_0: bool = False          # Use μ₀ joint market availability

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "EstimationSpec":
        """
        Parse a YAML specification file into a fully-populated EstimationSpec.

        Public entry point for the package; surfaces the lifted parser
        (:func:`parse_specification`). Replaces the v0.1 skeleton stub, which
        only stored the raw YAML dict without parsing it.
        """
        return parse_specification(Path(path))

    def get_initial_vector(self) -> np.ndarray:
        """
        Get initial values as numpy array in parameter order.

        Returns
        -------
        np.ndarray
            Initial values vector
        """
        return np.array([self.initial_values[name] for name in self.all_param_names])

    def is_ac2013(self) -> bool:
        """
        Check if this specification uses Aaberge-Colombino (2013) style.

        Returns
        -------
        bool
            True if model_version is "AC2013"
        """
        return self.model_version == "AC2013"

    def theta_c_param_name(self, group: Optional[str]) -> Optional[str]:
        """
        Return the parameter-vector name of the consumption Box-Cox exponent
        for a given group, handling the M0a-clean singles-shared case.

        Parameters
        ----------
        group : str or None
            One of "singles_male", "singles_female", "singles_pooled",
            "sm", "sf", "couples_male", "couples_female", "couples_household",
            "couples", "m", "f". For singles_* / sm / sf the singles-shared
            name (when set on this spec) takes precedence over the per-gender
            suffix variant. For couples / couples_* / m / f the legacy
            shared `utility_consumption_theta` (e.g. ``theta_c``) is returned
            unchanged. ``None`` is also accepted and returns the legacy
            shared name.

        Returns
        -------
        Optional[str]
            The parameter name, or None if the spec has no consumption theta.
        """
        if not self.utility_consumption_theta:
            return None
        singles_groups = {
            "singles_male", "singles_female", "singles_pooled", "sm", "sf",
        }
        if (
            self.utility_consumption_theta_singles_shared
            and group in singles_groups
        ):
            return self.utility_consumption_theta_singles_shared
        if group in {"couples_household", "couples", "couples_male",
                     "couples_female", "m", "f", None}:
            return self.utility_consumption_theta
        # Singles legacy: append suffix to the base name.
        suffix_map = {"singles_male": "_sm", "singles_female": "_sf",
                      "sm": "_sm", "sf": "_sf"}
        suffix = suffix_map.get(group, "")
        return f"{self.utility_consumption_theta}{suffix}"

    def get_bounds_tuple(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Get bounds in scipy.optimize format.

        Returns list of (lower, upper) tuples, with None for unbounded parameters.

        Returns
        -------
        list of tuple
            Bounds for each parameter in order
        """
        bounds_list = []
        for name in self.all_param_names:
            if name in self.bounds:
                bounds_list.append(self.bounds[name])
            else:
                bounds_list.append((None, None))  # Unbounded
        return bounds_list

    def get_param_index(self, param_name: str) -> int:
        """
        Get index of parameter in parameter vector.

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        int
            Index in all_param_names list

        Raises
        ------
        ValueError
            If parameter name not found
        """
        try:
            return self.all_param_names.index(param_name)
        except ValueError:
            raise ValueError(f"Parameter '{param_name}' not found in specification")

    def unpack_parameters(self, theta: np.ndarray) -> Dict[str, float]:
        """
        Unpack parameter vector into dictionary.

        Parameters
        ----------
        theta : np.ndarray
            Parameter vector

        Returns
        -------
        dict
            Dictionary mapping parameter names to values
        """
        if len(theta) != len(self.all_param_names):
            raise ValueError(
                f"Parameter vector length ({len(theta)}) does not match "
                f"number of parameters ({len(self.all_param_names)})"
            )

        return {name: theta[i] for i, name in enumerate(self.all_param_names)}

    def has_couples_gender_specific_params(self) -> bool:
        """
        Check if specification includes gender-specific couples parameters.

        Returns
        -------
        bool
            True if couples gender-specific parameters are present
        """
        # Check for existence of _m or _f suffixed leisure parameters
        return any(name.endswith('_m') or name.endswith('_f')
                   for name in self.all_param_names)

    def get_couples_param_map(self) -> Dict[str, str]:
        """
        Get mapping from singles parameter names to couples-specific parameter names.

        For couples estimation with gender-specific parameters, this maps:
        - Male: base param name -> param_name_m
        - Female: base param name -> param_name_f

        Returns
        -------
        dict
            Mapping like {'beta_l0_male': 'beta_l0_m', 'beta_l0_female': 'beta_l0_f', ...}
        """
        if not self.has_couples_gender_specific_params():
            return {}

        param_map = {}

        # Map leisure intercept
        if self.utility_leisure_intercept in self.all_param_names:
            param_map[f"{self.utility_leisure_intercept}_male"] = f"{self.utility_leisure_intercept}_m"
            param_map[f"{self.utility_leisure_intercept}_female"] = f"{self.utility_leisure_intercept}_f"

        # Map leisure shifters
        for shifter in self.utility_leisure_shifters:
            coef = shifter["coefficient"]
            # Skip n_children for males
            if not (shifter.get("gender_specific") and shifter["variable"] == "n_children"):
                param_map[f"{coef}_male"] = f"{coef}_m"
            param_map[f"{coef}_female"] = f"{coef}_f"        # Map leisure theta
        if self.utility_leisure_theta:
            param_map[f"{self.utility_leisure_theta}_male"] = f"{self.utility_leisure_theta}_m"
            param_map[f"{self.utility_leisure_theta}_female"] = f"{self.utility_leisure_theta}_f"

        return param_map

    def get_ac2013_features(self) -> Dict[str, bool]:
        """
        Get dictionary of which A-C 2013 features are enabled.
        
        Returns
        -------
        dict
            Feature name -> enabled status
        """
        return {
            'use_log_age': self.ac2013_use_log_age,
            'children_age_groups': self.ac2013_children_age_groups,
            'experience_in_wage': self.ac2013_experience_in_wage,
            'couples_cross_leisure': self.ac2013_couples_cross_leisure,
            'couples_mu_0': self.ac2013_couples_mu_0
        }


def parse_specification(yaml_path: Path) -> EstimationSpec:
    """
    Load and validate YAML specification file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML specification file

    Returns
    -------
    EstimationSpec
        Parsed specification object

    Raises
    ------
    FileNotFoundError
        If YAML file doesn't exist
    ValueError
        If specification is invalid    """
    logger = logging.getLogger(__name__)
    yaml_path = Path(yaml_path)
    logger.info("="*80)
    logger.info(f"Parsing specification: {yaml_path}")
    logger.info("="*80)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Specification file not found: {yaml_path}")
    
    # Load YAML (with explicit UTF-8 encoding for Unicode chars like θ, μ)
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Check for model version (NEW: AC2013 support)
    model_version = config.get("model_version", "legacy")
    if model_version not in ["legacy", "AC2013"]:
        logger.warning(f"Unknown model_version '{model_version}', treating as 'legacy'")
        model_version = "legacy"
    
    logger.info(f"Model version: {model_version}")

    # Extract metadata - handle both old and new YAML formats
    spec_meta = config.get("specification", {})
    name = spec_meta.get("name", config.get("name", "unknown"))
    description = spec_meta.get("description", config.get("description", ""))
    wage_spec = spec_meta.get("wage_spec", config.get("wage_spec", "fw"))
    model_family = str(
        spec_meta.get("model_family", config.get("model_family", "regular"))
    ).strip().lower()

    logger.info(f"Specification: {name}")
    logger.info(f"Description: {description}")
    logger.info(f"Wage specification: {wage_spec}")
    logger.info(f"Model family: {model_family}")

    # Optional reporting metadata. Estimation ignores this block; reporting
    # modules use it to avoid hardcoded country/year/spec labels and bins.
    reporting_config = config.get("reporting", {}) or {}
    if not isinstance(reporting_config, dict):
        raise ValueError("Top-level reporting block must be a mapping if provided.")
    hours_bins_config = reporting_config.get("hours_bins", None)
    if hours_bins_config is not None:
        if not isinstance(hours_bins_config, list):
            raise ValueError("reporting.hours_bins must be a list of mappings.")
        for idx, bin_cfg in enumerate(hours_bins_config):
            if not isinstance(bin_cfg, dict):
                raise ValueError(
                    f"reporting.hours_bins[{idx}] must be a mapping, "
                    f"got {type(bin_cfg).__name__}."
                )

    # Check if occupation choice is enabled
    occupation_choice = spec_meta.get("occupation_choice", False)
    logger.info(f"Occupation choice enabled: {occupation_choice}")

    # Validate wage_spec
    if wage_spec not in ["fw", "vw", "vw_occupation", "loc_empirical"]:
        raise ValueError(f"Invalid wage_spec: {wage_spec}. Must be fw, vw, vw_occupation, or loc_empirical")
    if model_family not in {"regular", "job_choice"}:
        raise ValueError("model_family must be 'regular' or 'job_choice'.")

    # Parse utility function
    utility_config = config.get("utility", {})
    utility_form = utility_config.get("functional_form", "box_cox")

    consumption_config = utility_config.get("consumption", {})
    utility_consumption_coef = consumption_config.get("coefficient", "beta_c")
    utility_consumption_theta = consumption_config.get("box_cox_exponent", None)
    pool_consumption_across_groups = bool(
        consumption_config.get("pool_across_groups", False)
    )
    if pool_consumption_across_groups:
        logger.info("Consumption pooling ENABLED: all groups share (beta_c, theta_c)")

    # Singles-only shared consumption Box-Cox exponent (M0a-clean).
    # If set, both singles_male and singles_female read this one parameter
    # instead of separate theta_c_sm / theta_c_sf. Couples remain on the
    # ordinary `box_cox_exponent` (e.g. theta_c). Mutually exclusive with
    # pool_consumption_across_groups (which pools across all four groups).
    utility_consumption_theta_singles_shared = consumption_config.get(
        "singles_box_cox_exponent", None
    )
    if utility_consumption_theta_singles_shared and pool_consumption_across_groups:
        raise ValueError(
            "utility.consumption.singles_box_cox_exponent and "
            "utility.consumption.pool_across_groups are mutually exclusive. "
            "Use one or the other, not both."
        )
    if utility_consumption_theta_singles_shared:
        logger.info(
            f"Singles theta_c POOLED: singles_male and singles_female share "
            f"'{utility_consumption_theta_singles_shared}'."
        )

    # Fixed (non-estimated) couples consumption Box-Cox exponent (M0c-b).
    # When set, couples BC-C uses this float constant; theta_c is omitted from
    # all_param_names. Mutually exclusive with pool_consumption_across_groups.
    _raw_couples_fixed = consumption_config.get("couples_fixed_box_cox_exponent", None)
    utility_consumption_theta_couples_fixed: Optional[float] = None
    if _raw_couples_fixed is not None:
        try:
            utility_consumption_theta_couples_fixed = float(_raw_couples_fixed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "utility.consumption.couples_fixed_box_cox_exponent must be numeric."
            ) from exc
        if pool_consumption_across_groups:
            raise ValueError(
                "utility.consumption.couples_fixed_box_cox_exponent and "
                "utility.consumption.pool_across_groups are mutually exclusive."
            )
        logger.info(
            "Couples theta_c FIXED: couples consumption BC exponent = %g "
            "(not estimated; theta_c removed from parameter vector).",
            utility_consumption_theta_couples_fixed,
        )

    # Fixed (non-estimated) consumption COEFFICIENT beta_c — scale normalisation.
    # When set, beta_c becomes the utility numeraire: a compile-time constant in
    # every block; beta_c / beta_c_sm / beta_c_sf are removed from all_param_names.
    _raw_coef_fixed = consumption_config.get("fixed_value", None)
    utility_consumption_coef_fixed: Optional[float] = None
    if _raw_coef_fixed is not None:
        try:
            utility_consumption_coef_fixed = float(_raw_coef_fixed)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "utility.consumption.fixed_value must be numeric."
            ) from exc
        if pool_consumption_across_groups:
            raise ValueError(
                "utility.consumption.fixed_value and "
                "utility.consumption.pool_across_groups are mutually exclusive."
            )
        logger.info(
            "beta_c FIXED (scale normalisation): consumption coefficient = %g "
            "(not estimated; beta_c/beta_c_sm/beta_c_sf removed from parameter vector).",
            utility_consumption_coef_fixed,
        )

    consumption_leisure_interaction_config = utility_config.get("consumption_leisure_interaction", {})
    if isinstance(consumption_leisure_interaction_config, dict):
        utility_consumption_leisure_interaction_coef = consumption_leisure_interaction_config.get("coefficient", None)
    else:
        utility_consumption_leisure_interaction_coef = None

    leisure_config = utility_config.get("leisure", {})
    utility_leisure_intercept = leisure_config.get("intercept", "beta_l0")
    utility_leisure_theta = leisure_config.get("box_cox_exponent", None)
    utility_leisure_shifters = leisure_config.get("shifters", [])

    # Parse occupation preferences (NEW: occupation choice)
    occupation_preferences = config.get("occupation_preferences", [])

    # Parse hours opportunity
    hours_config = config.get("hours_opportunity", {})
    hours_shifters = hours_config.get("shifters", [])

    # Parse occupation-specific hours (NEW: occupation choice)
    occupation_specific_hours = hours_config.get("occupation_specific", False)
    occupation_hour_configs = hours_config.get("occupations", [])

    # Parse wage opportunity
    wage_config = config.get("wage_opportunity", {})
    wage_form = wage_config.get("specification", "log_normal")
    wage_mean_shifters = wage_config.get("mean_shifters", [])
    wage_variance_param = None
    wage_loc_groups = None

    # Parse occupation-specific wages (NEW: occupation choice)
    occupation_specific_wages = (wage_form == "occupation_specific_log_normal")
    occupation_wage_configs = wage_config.get("occupations", [])

    if wage_spec in ["vw", "vw_occupation"]:
        variance_config = wage_config.get("variance", {})
        wage_variance_param = variance_config.get("parameter", "sigma")

    if wage_spec == "loc_empirical":
        wage_loc_groups = wage_config.get("groups", [])
        if not wage_loc_groups:
            raise ValueError("loc_empirical specification requires 'groups' in wage_opportunity")

    # Parse couples configuration
    couples_config = config.get("couples", {})
    couples_interaction_coef = None
    if couples_config:
        interaction_config = couples_config.get("leisure_interaction", {})
        couples_interaction_coef = interaction_config.get("coefficient", "beta_interact")

    # Parse market opportunity
    market_config = config.get("market_opportunity", {})
    market_opportunity_tier_raw = market_config.get("tier", None)
    market_opportunity_tier = (
        str(market_opportunity_tier_raw).strip().upper()
        if market_opportunity_tier_raw is not None
        else None
    )
    if market_opportunity_tier is not None and market_opportunity_tier not in {"M0", "M1", "M2", "M3", "M4"}:
        raise ValueError("market_opportunity.tier must be one of M0, M1, M2, M3, M4.")

    market_opportunity_shifters = market_config.get("shifters", [])
    if market_opportunity_tier and not market_opportunity_shifters:
        market_opportunity_shifters = _build_market_shifters_from_tier(market_config)
        logger.info(
            "Market opportunity tier %s expanded to %d shifters.",
            market_opportunity_tier,
            len(market_opportunity_shifters),
        )

    market_opportunity_offer_only_vars = _normalize_string_list(
        market_config.get("offer_only_vars", []),
        "market_opportunity.offer_only_vars",
    )
    market_opportunity_center_within_choice_set = bool(
        market_config.get("center_within_choice_set", False)
    )
    market_opportunity_center_weights = str(
        market_config.get("center_weights", "uniform")
    ).strip().lower()
    if market_opportunity_center_weights not in {"uniform", "proposal"}:
        raise ValueError(
            "market_opportunity.center_weights must be 'uniform' or 'proposal'."
        )
    raw_variable_scales = market_config.get("variable_scales", {}) or {}
    if not isinstance(raw_variable_scales, dict):
        raise ValueError("market_opportunity.variable_scales must be a mapping.")
    market_opportunity_variable_scales: Dict[str, float] = {}
    for raw_name, raw_value in raw_variable_scales.items():
        # NOTE: do not shadow the outer `name` (specification.name).
        scale_name = str(raw_name).strip()
        if not scale_name:
            continue
        try:
            scale_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"market_opportunity.variable_scales['{scale_name}'] must be numeric."
            ) from exc
        if scale_value <= 0.0:
            raise ValueError(
                f"market_opportunity.variable_scales['{scale_name}'] must be > 0."
            )
        market_opportunity_variable_scales[scale_name] = scale_value
    market_opportunity_extra_dimension = market_config.get("extra_dimension")
    if market_opportunity_extra_dimension is not None:
        market_opportunity_extra_dimension = str(market_opportunity_extra_dimension).strip().lower()
        if market_opportunity_extra_dimension not in {"hours_bin", "wage_bin"}:
            raise ValueError(
                "market_opportunity.extra_dimension must be 'hours_bin', 'wage_bin', or omitted."
            )
    market_opportunity_enforce_job_varying = bool(
        market_config.get("enforce_job_varying", False)
    )

    # -------------------------------------------------------------------------
    # Parse occupation_opportunity block for RURO occupation-opportunity M0 specs.
    # Dedicated, distinct from market_opportunity. Per contract v3/v4:
    #   - variable is restricted to `loc4` (M0) or `loc` (M2).
    #   - loc4/loc must NOT appear in utility, hours_opportunity, wage_opportunity,
    #     or market_opportunity (exclusion restriction).
    #   - shifters are internally appended to market_opportunity_shifters so the
    #     existing engines and parameter-list builder pick them up unchanged.
    # -------------------------------------------------------------------------
    occupation_opp_config = config.get("occupation_opportunity", {}) or {}
    occupation_opp_shifters = occupation_opp_config.get("shifters", []) or []
    if occupation_opp_shifters:
        occ_var_field = str(occupation_opp_config.get("variable", "loc4")).strip()
        if occ_var_field not in {"loc4", "loc"}:
            raise ValueError(
                f"occupation_opportunity.variable must be 'loc4' or 'loc', got '{occ_var_field}'."
            )

        def _shifter_vars(shifters: List[Dict[str, Any]]) -> set:
            seen = set()
            for sh in shifters or []:
                v = sh.get("variable") if isinstance(sh, dict) else None
                if isinstance(v, str):
                    seen.add(v)
            return seen

        forbidden_prefixes = ("loc4", "loc")
        def _has_loc(vs: set) -> List[str]:
            return [v for v in vs if any(v == p or v.startswith(p + "_") for p in forbidden_prefixes)]

        util_vars = _shifter_vars(utility_leisure_shifters)
        hours_vars = _shifter_vars(hours_shifters)
        wage_mean_vars = _shifter_vars(wage_mean_shifters)
        market_vars = _shifter_vars(market_opportunity_shifters)
        for block_name, vs in (
            ("utility.leisure.shifters", util_vars),
            ("hours_opportunity.shifters", hours_vars),
            ("wage_opportunity.mean_shifters", wage_mean_vars),
            ("market_opportunity.shifters", market_vars),
        ):
            violators = _has_loc(vs)
            if violators:
                raise ValueError(
                    f"Occupation variables {violators} appear in {block_name}; "
                    "loc4/loc may only appear in occupation_opportunity at M0."
                )

        # Append occupation shifters to market_opportunity_shifters so the
        # existing engine path computes them with the correct `working` gate.
        # The math is identical to a dedicated O^Occ block.
        for sh in occupation_opp_shifters:
            if not isinstance(sh, dict):
                continue
            sh_copy = dict(sh)
            sh_copy.setdefault("interaction", ["working"])
            applies_to = str(sh_copy.get("applies_to", "both")).strip().lower() or "both"
            if applies_to not in {"both", "sm", "sf", "cm", "cf", "male", "female", "household"}:
                raise ValueError(
                    "occupation_opportunity.shifters[*].applies_to must be one of "
                    "both, sm, sf, cm, cf, male, female, or household."
                )
            sh_copy["applies_to"] = applies_to
            market_opportunity_shifters.append(sh_copy)
        logger.info(
            "occupation_opportunity: parsed %d shifter(s), variable='%s', "
            "ref=%s; appended to market_opportunity for engine evaluation.",
            len(occupation_opp_shifters),
            occ_var_field,
            occupation_opp_config.get("reference", "1"),
        )

    _validate_market_opportunity_configuration(
        utility_leisure_shifters=utility_leisure_shifters,
        market_opportunity_shifters=market_opportunity_shifters,
        offer_only_vars=market_opportunity_offer_only_vars,
        enforce_job_varying=market_opportunity_enforce_job_varying,
    )

    # Parse occupation availability (legacy occupation-choice path)
    occupation_availability = market_config.get("occupation_availability", [])

    # Parse sampling configuration (NEW: McFadden sampling)
    sampling_config = config.get("sampling", {})
    sampling_method = sampling_config.get("method", "standard")
    sampling_n_alternatives_per_occ = sampling_config.get("n_alternatives_per_occupation", 100)
    sampling_total_alternatives = sampling_config.get("total_alternatives", 400)
    sampling_stratified_by_occ = sampling_config.get("stratified_by_occupation", False)

    # Parse initial values - support both flat and nested AC2013 formats
    initial_values = config.get("initial_values", {})
    bounds = {}
    
    if not initial_values and model_version == "AC2013":
        # AC2013 format: extract init/bounds from nested 'parameters' section
        logger.info("AC2013 format detected - extracting init/bounds from nested structure")
        initial_values, bounds = _extract_ac2013_parameters(config)
    
    if not initial_values:
        logger.warning("No initial values specified in YAML")

    # Parse bounds from optimization section (may override nested bounds)
    opt_config = config.get("optimization", {})
    bounds = opt_config.get("bounds", {})    # Parse optimization settings
    opt_method = opt_config.get("method", "L-BFGS-B")
    opt_analytical_gradient = opt_config.get("analytical_gradient", True)
    opt_max_iterations = opt_config.get("max_iterations", 10000)
    opt_tolerance = float(opt_config.get("tolerance", 1e-9))
    opt_gradient_tolerance = float(opt_config.get("gradient_tolerance", 1e-6))  # NEW
    opt_display = opt_config.get("disp", False)  # NEW
    opt_iprint = int(opt_config.get("iprint", -1))  # NEW
    (
        expr_constraints_enabled,
        expr_constraints_default_mode,
        expr_constraints_default_weight,
        expr_constraints
    ) = _parse_expression_constraints(opt_config, logger)

    # Parse gradient verification settings
    grad_verify = config.get('gradient_verification', {})

    # Build parameter list (order matters!)
    # For AC2013, use extracted parameter names; for legacy, build from spec
    if model_version == "AC2013" and initial_values:
        # AC2013: parameter names come from the extracted initial_values
        all_param_names = list(initial_values.keys())
        logger.info(f"AC2013: Using {len(all_param_names)} parameters from YAML")
    else:
        # Legacy or occupation choice: build parameter list from spec structure
        all_param_names = _build_parameter_list(
            utility_form=utility_form,
            utility_consumption_coef=utility_consumption_coef,
            utility_consumption_theta=utility_consumption_theta,
            utility_leisure_intercept=utility_leisure_intercept,
            utility_leisure_theta=utility_leisure_theta,
            utility_leisure_shifters=utility_leisure_shifters,
            utility_consumption_leisure_interaction_coef=utility_consumption_leisure_interaction_coef,
            hours_shifters=hours_shifters,
            market_opportunity_shifters=market_opportunity_shifters,
            wage_spec=wage_spec,
            wage_form=wage_form,
            wage_mean_shifters=wage_mean_shifters,
            wage_variance_param=wage_variance_param,
            wage_loc_groups=wage_loc_groups,
            couples_interaction_coef=couples_interaction_coef,
            # NEW: Occupation choice parameters
            occupation_choice=occupation_choice,
            occupation_preferences=occupation_preferences,
            occupation_specific_hours=occupation_specific_hours,
            occupation_hour_configs=occupation_hour_configs,
            occupation_specific_wages=occupation_specific_wages,
            occupation_wage_configs=occupation_wage_configs,
            occupation_availability=occupation_availability,
            pool_consumption=pool_consumption_across_groups,
            singles_shared_consumption_theta=utility_consumption_theta_singles_shared,
            couples_fixed_theta=utility_consumption_theta_couples_fixed,
            consumption_coef_fixed=utility_consumption_coef_fixed,
        )

    # ------------------------------------------------------------------
    # Generic fixed_params: pin any parameter to a value and remove it from
    # the estimated vector. Package-agnostic (any param / country / spec).
    # Resolved by the JAX backend's P() helper; the numpy engine is untouched.
    # ------------------------------------------------------------------
    fixed_params_raw = config.get("fixed_params", {}) or {}
    fixed_params: Dict[str, float] = {}
    if fixed_params_raw:
        if not isinstance(fixed_params_raw, dict):
            raise ValueError("fixed_params must be a mapping {param: value}")
        for k, v in fixed_params_raw.items():
            try:
                fixed_params[str(k)] = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"fixed_params[{k}] must be numeric, got {v!r}")
        before = len(all_param_names)
        all_param_names = [p for p in all_param_names if p not in fixed_params]
        for p in fixed_params:
            bounds.pop(p, None)
        logger.info(f"fixed_params: pinned {sorted(fixed_params)} "
                    f"({before} -> {len(all_param_names)} free params)")

    # ------------------------------------------------------------------
    # Generic gender_split: relax a SHARED coefficient to male/female
    # (coef -> coef_m, coef_f), applied at LL-build time by the JAX builders'
    # gender_split= hook. Package-agnostic (any coef / country / spec). The
    # base coef is REMOVED from the estimated vector and the two gendered names
    # are added (with bounds/initials taken from the gendered entries in the
    # YAML, falling back to the base coef's). The numpy engine is untouched
    # (JAX-backend-only, like fixed_params).
    # ------------------------------------------------------------------
    gender_split_raw = config.get("gender_split", []) or []
    gender_split: List[str] = []
    if gender_split_raw:
        if not isinstance(gender_split_raw, (list, tuple)):
            raise ValueError("gender_split must be a list of base coef names")
        gender_split = [str(c) for c in gender_split_raw]
        for base in gender_split:
            m, f = base + "_m", base + "_f"
            # remove the shared base, insert the two gendered names in place
            if base in all_param_names:
                idx = all_param_names.index(base)
                all_param_names[idx:idx + 1] = [m, f]
            else:
                # base not in the list (already split in the spec) — ensure present
                for nm in (m, f):
                    if nm not in all_param_names:
                        all_param_names.append(nm)
            # bounds/initials: gendered entries win; else inherit the base's
            base_bnd = bounds.get(base)
            base_iv = initial_values.get(base)
            for nm in (m, f):
                if nm not in bounds and base_bnd is not None:
                    bounds[nm] = base_bnd
                if nm not in initial_values and base_iv is not None:
                    initial_values[nm] = base_iv
            bounds.pop(base, None)
            initial_values.pop(base, None)
        logger.info(f"gender_split: relaxed {gender_split} -> _m/_f "
                    f"({len(all_param_names)} free params)")

    logger.info(f"Total parameters: {len(all_param_names)}")

    # Validate initial values (skip fixed params — they are not estimated)
    missing_initial = [p for p in all_param_names if p not in initial_values]
    if missing_initial:
        raise ValueError(f"Missing initial values for parameters: {missing_initial}")

    # Validate bounds
    for param_name, bound in bounds.items():
        if param_name not in all_param_names:
            logger.warning(f"Bound specified for unknown parameter: {param_name}")
        if not isinstance(bound, list) or len(bound) != 2:
            raise ValueError(f"Bound for {param_name} must be [lower, upper], got {bound}")
        if bound[0] >= bound[1]:
            raise ValueError(f"Invalid bound for {param_name}: lower ({bound[0]}) >= upper ({bound[1]})")

    # Convert bounds to tuples
    bounds_dict = {name: tuple(bound) for name, bound in bounds.items()}

    if expr_constraints_enabled:
        logger.info(
            "Expression constraints enabled: "
            f"{len(expr_constraints)} constraints "
            f"(default mode={expr_constraints_default_mode}, "
            f"default weight={expr_constraints_default_weight:.2e})"
        )

    logger.info("="*80)
    logger.info("Specification parsing complete")
    logger.info("="*80)

    return EstimationSpec(
        name=name,
        description=description,
        wage_spec=wage_spec,
        utility_form=utility_form,
        utility_consumption_coef=utility_consumption_coef,
        utility_consumption_theta=utility_consumption_theta,
        utility_leisure_intercept=utility_leisure_intercept,
        utility_leisure_theta=utility_leisure_theta,
        utility_leisure_shifters=utility_leisure_shifters,
        utility_consumption_leisure_interaction_coef=utility_consumption_leisure_interaction_coef,
        hours_shifters=hours_shifters,
        market_opportunity_shifters=market_opportunity_shifters,
        wage_form=wage_form,
        wage_mean_shifters=wage_mean_shifters,
        wage_variance_param=wage_variance_param,
        wage_loc_groups=wage_loc_groups,
        couples_interaction_coef=couples_interaction_coef,
        all_param_names=all_param_names,
        initial_values=initial_values,
        bounds=bounds_dict,
        fixed_params=fixed_params,
        gender_split=gender_split,
        expression_constraints_enabled=expr_constraints_enabled,
        expression_constraints_default_mode=expr_constraints_default_mode,
        expression_constraints_default_weight=expr_constraints_default_weight,
        expression_constraints=expr_constraints,
        reporting=reporting_config,
        opt_method=opt_method,
        opt_analytical_gradient=opt_analytical_gradient,
        opt_max_iterations=opt_max_iterations,
        opt_tolerance=opt_tolerance,
        opt_gradient_tolerance=opt_gradient_tolerance,
        opt_display_convergence=opt_display,
        opt_iprint=opt_iprint,
        grad_verify_enabled=grad_verify.get('enabled', False),
        grad_verify_method=grad_verify.get('method', 'central'),
        grad_verify_epsilon=float(grad_verify.get('epsilon', 1e-7)),
        grad_verify_tolerance=float(grad_verify.get('tolerance', 1e-4)),
        grad_verify_at_init=grad_verify.get('check_at_init', True),
        grad_verify_random_points=int(grad_verify.get('check_random_points', 0)),
        grad_verify_seed=int(grad_verify.get('random_seed', 42)),
        grad_verify_verbose=grad_verify.get('verbose', False),
        # AC2013 settings
        model_version=model_version,
        model_family=model_family,
        market_opportunity_tier=market_opportunity_tier,
        market_opportunity_offer_only_vars=market_opportunity_offer_only_vars,
        market_opportunity_center_within_choice_set=market_opportunity_center_within_choice_set,
        market_opportunity_center_weights=market_opportunity_center_weights,
        market_opportunity_extra_dimension=market_opportunity_extra_dimension,
        market_opportunity_enforce_job_varying=market_opportunity_enforce_job_varying,
        market_opportunity_variable_scales=market_opportunity_variable_scales,
        pool_consumption_across_groups=pool_consumption_across_groups,
        utility_consumption_theta_singles_shared=utility_consumption_theta_singles_shared,
        utility_consumption_theta_couples_fixed=utility_consumption_theta_couples_fixed,
        utility_consumption_coef_fixed=utility_consumption_coef_fixed,
        ac2013_use_log_age=(model_version == "AC2013"),
        ac2013_children_age_groups=(model_version == "AC2013"),
        ac2013_experience_in_wage=(model_version == "AC2013"),
        ac2013_couples_cross_leisure=(model_version == "AC2013" and 'alpha_ll' in all_param_names),
        ac2013_couples_mu_0=(model_version == "AC2013" and 'mu_0' in all_param_names),
        # NEW: Occupation choice settings
        occupation_choice=occupation_choice,
        occupation_preferences=occupation_preferences,
        occupation_specific_hours=occupation_specific_hours,
        occupation_hour_configs=occupation_hour_configs,
        occupation_specific_wages=occupation_specific_wages,
        occupation_wage_configs=occupation_wage_configs,
        occupation_availability=occupation_availability,
        sampling_method=sampling_method,
        sampling_n_alternatives_per_occ=sampling_n_alternatives_per_occ,
        sampling_total_alternatives=sampling_total_alternatives,
        sampling_stratified_by_occ=sampling_stratified_by_occ,
    )


def _parse_expression_constraints(
    opt_config: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[bool, str, float, List[Dict[str, Any]]]:
    """
    Parse and validate optimization.expression_constraints block.

    Supported schema:
      optimization:
        expression_constraints:
          enabled: true
          default_mode: soft  # soft | hard
          default_weight: 1e4
          constraints:
            - name: muc_sm_positive
              expression: muc  # muc | mul | dmuc_dc | dmul_dl | param_diff
              group: singles_male
              at: {consumption: 1.0, leisure: 1.0, age_norm: 0.0}
              lower: 0.0
              mode: soft
              weight: 5e3

    A shorthand list under expression_constraints is also accepted.
    """
    expr_cfg = opt_config.get("expression_constraints", None)
    if expr_cfg is None:
        return False, "soft", 1e4, []

    if isinstance(expr_cfg, list):
        enabled = len(expr_cfg) > 0
        default_mode = "soft"
        default_weight = 1e4
        raw_constraints = expr_cfg
    elif isinstance(expr_cfg, dict):
        raw_constraints = expr_cfg.get("constraints", [])
        enabled = bool(expr_cfg.get("enabled", bool(raw_constraints)))
        default_mode = str(expr_cfg.get("default_mode", "soft")).strip().lower()
        default_weight = float(expr_cfg.get("default_weight", 1e4))
    else:
        raise ValueError(
            "optimization.expression_constraints must be a dict or a list of constraints."
        )

    if default_mode not in {"soft", "hard"}:
        raise ValueError(
            "optimization.expression_constraints.default_mode must be 'soft' or 'hard'."
        )
    if default_weight <= 0:
        raise ValueError(
            "optimization.expression_constraints.default_weight must be > 0."
        )

    if not isinstance(raw_constraints, list):
        raise ValueError(
            "optimization.expression_constraints.constraints must be a list."
        )

    valid_exprs = {"muc", "mul", "dmuc_dc", "dmul_dl", "param_diff"}
    valid_groups = {
        "singles_male",
        "singles_female",
        "couples_male",
        "couples_female",
        "couples_household",
        "couples",
        "global",
    }
    parsed_constraints: List[Dict[str, Any]] = []

    for i, raw in enumerate(raw_constraints):
        if not isinstance(raw, dict):
            raise ValueError(
                f"expression_constraints[{i}] must be a mapping, got {type(raw)}"
            )

        expr_name = str(raw.get("expression", "")).strip().lower()
        if expr_name not in valid_exprs:
            raise ValueError(
                f"expression_constraints[{i}].expression must be one of "
                f"{sorted(valid_exprs)}, got '{raw.get('expression')}'"
            )

        group = str(raw.get("group", "")).strip().lower()
        if group not in valid_groups:
            raise ValueError(
                f"expression_constraints[{i}].group must be one of "
                f"{sorted(valid_groups)}, got '{raw.get('group')}'"
            )
        if expr_name in {"mul", "dmul_dl"} and group in {"couples", "couples_household"}:
            raise ValueError(
                f"expression_constraints[{i}] with expression='{expr_name}' must use "
                "group 'couples_male' or 'couples_female' (not household/alias)."
            )

        at = raw.get("at", {}) or {}
        if not isinstance(at, dict):
            raise ValueError(f"expression_constraints[{i}].at must be a mapping.")

        at_clean: Dict[str, float] = {}
        for key, value in at.items():
            try:
                at_clean[str(key)] = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"expression_constraints[{i}].at['{key}'] must be numeric."
                )

        if expr_name != "param_diff":
            for pos_key in ("consumption", "leisure", "leisure_male", "leisure_female", "leisure_partner"):
                if pos_key in at_clean and at_clean[pos_key] <= 0:
                    raise ValueError(
                        f"expression_constraints[{i}].at['{pos_key}'] must be > 0."
                    )

        lhs_param = raw.get("lhs_param")
        rhs_param = raw.get("rhs_param")
        if expr_name == "param_diff":
            if not lhs_param or not str(lhs_param).strip():
                raise ValueError(
                    f"expression_constraints[{i}] with expression='param_diff' "
                    "must provide non-empty 'lhs_param'."
                )
            lhs_param = str(lhs_param).strip()
            rhs_param = str(rhs_param).strip() if rhs_param is not None else None
        else:
            lhs_param = None
            rhs_param = None

        lower = raw.get("lower", None)
        upper = raw.get("upper", None)
        if lower is None and upper is None:
            raise ValueError(
                f"expression_constraints[{i}] must define at least one of 'lower' or 'upper'."
            )
        lower_val = float(lower) if lower is not None else None
        upper_val = float(upper) if upper is not None else None
        if lower_val is not None and upper_val is not None and lower_val > upper_val:
            raise ValueError(
                f"expression_constraints[{i}] has lower > upper ({lower_val} > {upper_val})."
            )

        mode = str(raw.get("mode", default_mode)).strip().lower()
        if mode not in {"soft", "hard"}:
            raise ValueError(
                f"expression_constraints[{i}].mode must be 'soft' or 'hard'."
            )

        weight = raw.get("weight", default_weight)
        weight_val = float(weight)
        if weight_val <= 0:
            raise ValueError(
                f"expression_constraints[{i}].weight must be > 0."
            )

        name = str(raw.get("name", f"{expr_name}_{group}_{i}")).strip()
        if not name:
            name = f"{expr_name}_{group}_{i}"

        parsed_constraints.append({
            "name": name,
            "expression": expr_name,
            "group": group,
            "at": at_clean,
            "lhs_param": lhs_param,
            "rhs_param": rhs_param,
            "lower": lower_val,
            "upper": upper_val,
            "mode": mode,
            "weight": weight_val,
        })

    if enabled and not parsed_constraints:
        logger.warning(
            "optimization.expression_constraints.enabled=true but no constraints were provided."
        )

    return enabled, default_mode, default_weight, parsed_constraints


def _normalize_string_list(value: Any, field_name: str) -> List[str]:
    """Normalize a YAML scalar/list field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string or list of strings.")
    out: List[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{idx}] must be a string.")
        token = item.strip()
        if token:
            out.append(token)
    return out


def _normalize_interaction_list(value: Any) -> List[str]:
    """Normalize interaction specification to a list."""
    return _normalize_string_list(value, "market_opportunity.shifters[*].interaction")


def _is_alt_varying_market_var(var_name: str) -> bool:
    """
    Heuristic check whether a variable name is alternative-varying in job-choice data.

    Supports both raw categorical columns (hours_bin, wage_bin, isco1)
    and derived dummies (hours_bin_1, wage_bin_3, isco1_9, etc.).
    """
    name = str(var_name).strip()
    if not name:
        return False
    if name in {
        "working",
        "working_pt1",
        "working_pt2",
        "working_ft",
        "hours_bin",
        "wage_bin",
        "isco1",
        "loc4",
        "job_id",
    }:
        return True
    return bool(re.match(r"^(hours_bin|wage_bin|isco1|loc4|job_id)_[A-Za-z0-9-]+$", name))


def _build_market_shifters_from_tier(market_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build market_opportunity.shifters from a tier declaration (M0-M4).

    This keeps YAML concise for the identification ladder while preserving
    backward compatibility with explicit shifter lists.
    """
    tier = str(market_config.get("tier", "")).strip().upper()
    if tier not in {"M0", "M1", "M2", "M3", "M4"}:
        raise ValueError("market_opportunity.tier must be one of M0, M1, M2, M3, M4.")

    applies_to = str(market_config.get("applies_to", "both")).strip().lower() or "both"
    if applies_to not in {"male", "female", "both", "household"}:
        raise ValueError("market_opportunity.applies_to must be male/female/both/household.")

    employment_var = str(market_config.get("employment_indicator", "working")).strip() or "working"
    gsur_var = str(market_config.get("gsur_variable", "gsur")).strip() or "gsur"

    coef_working = str(market_config.get("coef_working", "beta_offer_working")).strip() or "beta_offer_working"
    coef_gsur = str(market_config.get("coef_gsur", "beta_offer_gsur")).strip() or "beta_offer_gsur"
    coef_gsur_occ_prefix = str(
        market_config.get("coef_gsur_occ_prefix", "beta_offer_gsur_isco1_")
    ).strip() or "beta_offer_gsur_isco1_"

    shifters: List[Dict[str, Any]] = [
        {
            "variable": employment_var,
            "coefficient": coef_working,
            "applies_to": applies_to,
        }
    ]

    if tier in {"M1", "M2", "M3", "M4"}:
        shifters.append(
            {
                "variable": gsur_var,
                "coefficient": coef_gsur,
                "interaction": [employment_var],
                "applies_to": applies_to,
            }
        )

    if tier in {"M2", "M3", "M4"}:
        occ_var = str(market_config.get("occupation_variable", "isco1")).strip() or "isco1"
        occ_categories = market_config.get("occupation_categories", list(range(10)))
        if not isinstance(occ_categories, list) or not occ_categories:
            raise ValueError("market_opportunity.occupation_categories must be a non-empty list.")
        occ_base = market_config.get("occupation_base", occ_categories[0])
        for occ in occ_categories:
            if str(occ) == str(occ_base):
                continue
            shifters.append(
                {
                    "variable": f"{occ_var}_{occ}",
                    "coefficient": f"{coef_gsur_occ_prefix}{occ}",
                    "interaction": [employment_var, gsur_var],
                    "applies_to": applies_to,
                }
            )

    if tier in {"M3", "M4"}:
        extra_dim = market_config.get("extra_dimension", None)
        if extra_dim is None:
            raise ValueError(
                "market_opportunity.extra_dimension is required for tier M3/M4 "
                "(hours_bin or wage_bin)."
            )
        extra_dim = str(extra_dim).strip().lower()
        if extra_dim not in {"hours_bin", "wage_bin"}:
            raise ValueError(
                "market_opportunity.extra_dimension must be 'hours_bin' or 'wage_bin'."
            )

        categories_key = f"{extra_dim}_categories"
        default_categories = [0, 1, 2, 3] if extra_dim == "hours_bin" else list(range(10))
        extra_categories = market_config.get(categories_key, default_categories)
        if not isinstance(extra_categories, list) or not extra_categories:
            raise ValueError(f"market_opportunity.{categories_key} must be a non-empty list.")

        extra_base = market_config.get(f"{extra_dim}_base", extra_categories[0])
        coef_extra_prefix = str(
            market_config.get("coef_gsur_extra_prefix", f"beta_offer_gsur_{extra_dim}_")
        ).strip() or f"beta_offer_gsur_{extra_dim}_"

        for cat in extra_categories:
            if str(cat) == str(extra_base):
                continue
            shifters.append(
                {
                    "variable": f"{extra_dim}_{cat}",
                    "coefficient": f"{coef_extra_prefix}{cat}",
                    "interaction": [employment_var, gsur_var],
                    "applies_to": applies_to,
                }
            )

    return shifters


def _validate_market_opportunity_configuration(
    utility_leisure_shifters: List[Dict[str, Any]],
    market_opportunity_shifters: List[Dict[str, Any]],
    offer_only_vars: List[str],
    enforce_job_varying: bool,
) -> None:
    """
    Validate exclusion restrictions and job-varying opportunity logic.
    """
    pref_vars = {
        str(shifter.get("variable", "")).strip()
        for shifter in utility_leisure_shifters
        if isinstance(shifter, dict) and str(shifter.get("variable", "")).strip()
    }
    if "gsur" in pref_vars:
        raise ValueError(
            "Identification restriction violated: 'gsur' cannot appear in utility.leisure.shifters. "
            "Use gsur only in market_opportunity."
        )

    overlap = sorted(set(offer_only_vars).intersection(pref_vars))
    if overlap:
        raise ValueError(
            "Identification restriction violated: offer-only variables appear in preferences: "
            + ", ".join(overlap)
        )

    if not enforce_job_varying:
        return

    for idx, shifter in enumerate(market_opportunity_shifters):
        if not isinstance(shifter, dict):
            continue
        var_name = str(shifter.get("variable", "")).strip()
        interaction_terms = _normalize_interaction_list(shifter.get("interaction", None))

        if _is_alt_varying_market_var(var_name):
            continue
        if any(_is_alt_varying_market_var(term) for term in interaction_terms):
            continue

        raise ValueError(
            f"market_opportunity.shifters[{idx}] is not job-varying and would cancel in conditional logit: "
            f"variable='{var_name}', interaction={interaction_terms}. "
            "Add an interaction with an alternative-varying job attribute or employment indicator."
        )


def _build_parameter_list(
    utility_form: str,
    utility_consumption_coef: str,
    utility_consumption_theta: Optional[str],
    utility_leisure_intercept: str,
    utility_leisure_theta: Optional[str],
    utility_leisure_shifters: List[Dict[str, Any]],
    utility_consumption_leisure_interaction_coef: Optional[str],
    hours_shifters: List[Dict[str, Any]],
    market_opportunity_shifters: Optional[List[Dict[str, Any]]],
    wage_spec: str,
    wage_form: str,
    wage_mean_shifters: List[Dict[str, Any]],
    wage_variance_param: Optional[str],
    wage_loc_groups: Optional[List[Dict[str, Any]]],
    couples_interaction_coef: Optional[str],
    # NEW: Occupation choice parameters
    occupation_choice: bool = False,
    occupation_preferences: Optional[List[Dict[str, Any]]] = None,
    occupation_specific_hours: bool = False,
    occupation_hour_configs: Optional[List[Dict[str, Any]]] = None,
    occupation_specific_wages: bool = False,
    occupation_wage_configs: Optional[List[Dict[str, Any]]] = None,
    occupation_availability: Optional[List[Dict[str, Any]]] = None,
    pool_consumption: bool = False,
    singles_shared_consumption_theta: Optional[str] = None,
    couples_fixed_theta: Optional[float] = None,
    consumption_coef_fixed: Optional[float] = None,
) -> List[str]:
    """
    Build ordered list of all parameter names.

    Parameter order convention:
    1. Preference parameters (leisure shifters, consumption coef, Box-Cox exponents)
    2. Hours opportunity parameters
    3. Wage opportunity parameters
    4. Consumption-leisure interaction (if applicable)
    5. Couples interaction (if applicable)

    This matches the order in the old script for backward compatibility.

    Returns
    -------
    list of str
        Ordered parameter names
    """
    params = []

    # ==========================================================================

    # FULLY SEPARATE 4-GROUP ARCHITECTURE
    # ==========================================================================

    # We have 4 distinct groups with their own preference parameters:
    # 1. Singles Male (_sm suffix)
    # 2. Singles Female (_sf suffix)
    # 3. Couples Male (_m suffix)
    # 4. Couples Female (_f suffix)
    #
    # SHARED parameters (all groups):
    # - Hours opportunity (beta_work, beta_pt1, beta_pt2, beta_ft, beta_gsur, beta_work_educL, beta_work_educH)
    # - Wage opportunity (beta_w0, beta_w_educL, beta_w_educH, beta_pexp, beta_pexp2, sigma)
    #
    # COUPLES ONLY parameters:
    # - Household consumption (beta_c, theta_c)
    # - Interaction term (beta_interact)
    # ==========================================================================

    # GROUP 1: Singles Male - Leisure preferences (_sm suffix)
    singles_male_params = [
        f"{utility_leisure_intercept}_sm",  # beta_l0_sm
    ]
    for shifter in utility_leisure_shifters:
        # Skip n_children for males (only for females)
        if shifter.get("gender_specific") and shifter["variable"] == "n_children":
            continue
        singles_male_params.append(f"{shifter['coefficient']}_sm")

    if not pool_consumption and consumption_coef_fixed is None:
        singles_male_params.append(f"{utility_consumption_coef}_sm")  # beta_c_sm (skipped if fixed)
    if utility_consumption_leisure_interaction_coef:
        singles_male_params.append(f"{utility_consumption_leisure_interaction_coef}_sm")

    if utility_form == "box_cox":
        if utility_leisure_theta:
            singles_male_params.append(f"{utility_leisure_theta}_sm")  # theta_l_sm
        # Singles-shared theta_c (M0a-clean) suppresses the per-singles _sm copy;
        # the shared name is added once below, after both singles blocks.
        if (utility_consumption_theta and not pool_consumption
                and not singles_shared_consumption_theta):
            singles_male_params.append(f"{utility_consumption_theta}_sm")  # theta_c_sm

    params.extend(singles_male_params)

    # GROUP 2: Singles Female - Leisure preferences (_sf suffix)
    singles_female_params = [
        f"{utility_leisure_intercept}_sf",  # beta_l0_sf
    ]
    for shifter in utility_leisure_shifters:
        singles_female_params.append(f"{shifter['coefficient']}_sf")

    if not pool_consumption and consumption_coef_fixed is None:
        singles_female_params.append(f"{utility_consumption_coef}_sf")  # beta_c_sf (skipped if fixed)
    if utility_consumption_leisure_interaction_coef:
        singles_female_params.append(f"{utility_consumption_leisure_interaction_coef}_sf")

    if utility_form == "box_cox":
        if utility_leisure_theta:
            singles_female_params.append(f"{utility_leisure_theta}_sf")  # theta_l_sf
        # Singles-shared theta_c (M0a-clean) suppresses the per-singles _sf copy.
        if (utility_consumption_theta and not pool_consumption
                and not singles_shared_consumption_theta):
            singles_female_params.append(f"{utility_consumption_theta}_sf")  # theta_c_sf

    params.extend(singles_female_params)

    # M0a-clean: a single shared singles consumption Box-Cox exponent, added
    # once after both singles blocks. Mutually exclusive with pool_consumption
    # (which would have folded singles into couples already).
    if (utility_consumption_theta and singles_shared_consumption_theta
            and utility_form == "box_cox" and not pool_consumption):
        params.append(singles_shared_consumption_theta)

    # GROUP 3: Couples Male - Leisure preferences (_m suffix)
    couples_male_params = [
        f"{utility_leisure_intercept}_m",  # beta_l0_m
    ]
    for shifter in utility_leisure_shifters:
        # Skip n_children for males (only for females)
        if shifter.get("gender_specific") and shifter["variable"] == "n_children":
            continue
        couples_male_params.append(f"{shifter['coefficient']}_m")

    if utility_form == "box_cox" and utility_leisure_theta:
        couples_male_params.append(f"{utility_leisure_theta}_m")  # theta_l_m
    if utility_consumption_leisure_interaction_coef:
        couples_male_params.append(f"{utility_consumption_leisure_interaction_coef}_m")

    params.extend(couples_male_params)

    # GROUP 4: Couples Female - Leisure preferences (_f suffix)
    couples_female_params = [
        f"{utility_leisure_intercept}_f",  # beta_l0_f
    ]
    for shifter in utility_leisure_shifters:
        couples_female_params.append(f"{shifter['coefficient']}_f")

    if utility_form == "box_cox" and utility_leisure_theta:
        couples_female_params.append(f"{utility_leisure_theta}_f")  # theta_l_f
    if utility_consumption_leisure_interaction_coef:
        couples_female_params.append(f"{utility_consumption_leisure_interaction_coef}_f")

    params.extend(couples_female_params)

    # COUPLES HOUSEHOLD: Consumption (shared for couples, no suffix)
    # Skip beta_c when consumption_coef_fixed is set — it's the scale-normalisation
    # numéraire (compile-time constant), not an estimated parameter.
    if consumption_coef_fixed is None:
        params.append(utility_consumption_coef)  # beta_c (skipped if fixed)
    # Skip theta_c when couples_fixed_theta is set — it's a compile-time constant,
    # not an estimated parameter.
    if utility_form == "box_cox" and utility_consumption_theta and couples_fixed_theta is None:
        params.append(utility_consumption_theta)  # theta_c

    # SHARED OPPORTUNITY: Hours parameters (all groups)
    for shifter in hours_shifters:
        params.append(shifter["coefficient"])

    # SHARED OPPORTUNITY: Job/market parameters (all groups)
    if market_opportunity_shifters:
        for shifter in market_opportunity_shifters:
            if "coefficient" in shifter:
                params.append(shifter["coefficient"])

    # SHARED OPPORTUNITY: Wage parameters (all groups)
    if wage_spec == "vw":
        # Mincer equation parameters
        for shifter in wage_mean_shifters:
            params.append(shifter["coefficient"])

        if wage_variance_param:
            params.append(wage_variance_param)  # sigma

    elif wage_spec == "vw_occupation":
        # Occupation-specific Mincer equations (NEW: occupation choice)
        if occupation_wage_configs:
            for occ_config in occupation_wage_configs:
                # For each occupation, add gender-specific wage parameters
                for gender_suffix in ["_sm", "_sf"]:
                    params.append(f"{occ_config['intercept']}{gender_suffix}")
                    params.append(f"{occ_config['experience']}{gender_suffix}")
                    params.append(f"{occ_config['experience_squared']}{gender_suffix}")
                    params.append(f"{occ_config['education']}{gender_suffix}")
                    params.append(f"{occ_config['variance']}{gender_suffix}")

    elif wage_spec == "loc_empirical":
        # LOC-specific intercepts
        for group in wage_loc_groups:
            params.append(group["intercept"])

        # LOC-specific sigmas
        for group in wage_loc_groups:
            params.append(group["sigma"])

        # Common shifters
        for shifter in wage_mean_shifters:
            params.append(shifter["coefficient"])

    # OCCUPATION CHOICE: Occupation preferences (NEW)
    if occupation_choice and occupation_preferences:
        for occ_pref in occupation_preferences:
            # Base occupation preference for each gender group
            for gender_suffix in ["_sm", "_sf", "_m", "_f"]:
                params.append(f"{occ_pref['coefficient']}{gender_suffix}")

            # Occupation-demographic interactions
            if "interactions" in occ_pref:
                for interaction in occ_pref["interactions"]:
                    for gender_suffix in ["_sm", "_sf", "_m", "_f"]:
                        params.append(f"{interaction['coefficient']}{gender_suffix}")

    # OCCUPATION CHOICE: Occupation-specific hours (NEW)
    if occupation_specific_hours and occupation_hour_configs:
        for occ_hour in occupation_hour_configs:
            # Part-time and full-time clustering parameters for each occupation
            params.append(occ_hour["part_time_peak"])
            params.append(occ_hour["full_time_peak"])

    # OCCUPATION CHOICE: Occupation availability (NEW)
    if occupation_choice and occupation_availability:
        for occ_avail in occupation_availability:
            # Only add parameter if not None (reference category has None)
            if occ_avail.get("parameter") is not None:
                # Gender-specific availability
                for gender_suffix in ["_sm", "_sf"]:
                    params.append(f"{occ_avail['parameter']}{gender_suffix}")

    # COUPLES ONLY: Interaction term
    if couples_interaction_coef:
        params.append(couples_interaction_coef)

    # Check for duplicates
    if len(params) != len(set(params)):
        duplicates = [p for p in params if params.count(p) > 1]
        raise ValueError(f"Duplicate parameter names found: {set(duplicates)}")

    return params


def _extract_ac2013_parameters(config: Dict) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """
    Extract initial values and bounds from AC2013 nested YAML structure.
    
    AC2013 format uses nested sections like singles/preference/consumption with:
      param_name:
        init: value
        bounds: [lower, upper]
        description: "..."
    
    Parameters
    ----------
    config : dict
        Full YAML config
        
    Returns
    -------
    initial_values : dict
        Parameter name -> initial value
    bounds : dict
        Parameter name -> (lower, upper)
    """
    initial_values = {}
    bounds = {}
    
    def extract_from_section(section: Dict, prefix: str = ""):
        """Recursively extract parameters from nested sections."""
        if not isinstance(section, dict):
            return
        for key, value in section.items():
            if isinstance(value, dict):
                if 'init' in value:
                    # This is a parameter definition
                    param_name = key
                    initial_values[param_name] = float(value['init'])
                    if 'bounds' in value:
                        b = value['bounds']
                        bounds[param_name] = (float(b[0]), float(b[1]))
                else:
                    # Nested section - recurse
                    extract_from_section(value, key)
    
    # Look in singles and couples sections
    for section_name in ['singles', 'couples']:
        section = config.get(section_name, {})
        extract_from_section(section)
    
    # Also look in top-level parameters section if present
    if 'parameters' in config:
        extract_from_section(config['parameters'])
    
    return initial_values, bounds


def load_custom_initial_values(csv_path: Path) -> Dict[str, float]:
    """
    Load custom initial values from CSV or JSON file.

    CSV format:
        parameter_name,value
        beta_l0,1.0
        beta_c,0.5
        ...

    JSON format:
        {
            "param_names": ["beta_l0", "beta_c", ...],
            "theta": [1.0, 0.5, ...]
        }

    Parameters
    ----------
    csv_path : Path
        Path to CSV or JSON file

    Returns
    -------
    dict
        Dictionary mapping parameter names to initial values
    """
    import pandas as pd
    import json

    if not csv_path.exists():
        raise FileNotFoundError(f"Initial values file not found: {csv_path}")    # Check if it's a JSON file
    if csv_path.suffix.lower() == '.json':
        with open(csv_path, 'r') as f:
            data = json.load(f)

        # NEW FORMAT: Check for 'results' key (enhanced estimation output)
        if "results" in data:
            init_dict = {}
            # Collect all parameters from all groups (support joint, singles_male, etc.)
            for group_name, group_data in data['results'].items():
                if isinstance(group_data, dict) and 'parameters' in group_data:
                    params = group_data.get('parameters', {})
                    init_dict.update(params)
            return init_dict

        # OLD FORMAT: Check for param_names and theta arrays
        elif "param_names" in data and "theta" in data:
            # Strip hierarchical prefixes like 'sm.pref.' from old format
            clean_names = []
            for name in data["param_names"]:
                # Remove prefixes: sm.pref.beta_l0 → beta_l0
                if '.' in name:
                    clean_name = name.split('.')[-1]
                else:
                    clean_name = name
                clean_names.append(clean_name)
            return dict(zip(clean_names, data["theta"]))

        else:
            raise ValueError("JSON must have either 'results' dict or 'param_names'+'theta' arrays")
    
    # Otherwise treat as CSV
    df = pd.read_csv(csv_path)

    # Support both 'parameter_name' and 'parameter' column names
    param_col = None
    if "parameter_name" in df.columns:
        param_col = "parameter_name"
    elif "parameter" in df.columns:
        param_col = "parameter"
    else:
        raise ValueError("CSV must have column 'parameter_name' or 'parameter'")

    if "value" not in df.columns:
        raise ValueError("CSV must have column 'value'")

    return dict(zip(df[param_col], df["value"]))


def find_latest_results(
    search_dirs: List[Path],
    results_filename: str = "estimation_results.json"
) -> Optional[Path]:
    """
    Find the most recent estimation results file across multiple directories.
    
    Searches in the specified directories and their subdirectories for
    results files, returning the path to the most recently modified one.
    
    Parameters
    ----------
    search_dirs : List[Path]
        Directories to search for results files
    results_filename : str
        Name of the results file to look for (default: estimation_results.json)
        
    Returns
    -------
    Optional[Path]
        Path to the most recent results file, or None if not found
    """
    import os
    
    logger = logging.getLogger(__name__)
    candidates = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
            
        # Search recursively for results files
        for root, dirs, files in os.walk(search_dir):
            if results_filename in files:
                results_path = Path(root) / results_filename
                mtime = results_path.stat().st_mtime
                candidates.append((results_path, mtime))
                
    if not candidates:
        logger.info("No previous estimation results found")
        return None
        
    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_path = candidates[0][0]
    
    logger.info(f"Found {len(candidates)} previous results file(s)")
    logger.info(f"Latest: {latest_path}")
    
    return latest_path


def load_warm_start_values(
    spec: 'EstimationSpec',
    results_path: Optional[Path] = None,
    search_dirs: Optional[List[Path]] = None,
    default_value: float = 0.0
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Load initial values from previous results with fallback to defaults.
    
    For parameters that exist in both the current specification and the
    previous results, uses the estimated values. For new parameters,
    uses the default value from the spec or the provided default_value.
    
    Parameters
    ----------
    spec : EstimationSpec
        Current specification with parameter names
    results_path : Optional[Path]
        Explicit path to results JSON file. If None, auto-finds latest.
    search_dirs : Optional[List[Path]]
        Directories to search if results_path is None
    default_value : float
        Default value for new parameters not in previous results (default: 0.0)
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, str]]
        - Initial values vector
        - Dictionary mapping parameter names to their source ('previous', 'spec_default', 'fallback_default')
    """
    import json
    
    logger = logging.getLogger(__name__)
    
    # Find results file if not explicitly provided
    if results_path is None and search_dirs is not None:
        results_path = find_latest_results(search_dirs)
    
    # Load previous parameters if available
    prev_params = {}
    if results_path is not None and results_path.exists():
        try:
            prev_params = load_custom_initial_values(results_path)
            logger.info(f"Loaded {len(prev_params)} parameters from: {results_path}")
        except Exception as e:
            logger.warning(f"Failed to load previous results: {e}")
            prev_params = {}
    
    # Build initial values vector
    theta_init = np.zeros(len(spec.all_param_names))
    sources = {}
    
    n_from_prev = 0
    n_from_spec = 0
    n_from_default = 0
    
    for i, param_name in enumerate(spec.all_param_names):
        if param_name in prev_params:
            # Use value from previous estimation
            theta_init[i] = prev_params[param_name]
            sources[param_name] = 'previous'
            n_from_prev += 1
        elif param_name in spec.initial_values and spec.initial_values[param_name] != 0.0:
            # Use spec default (if non-zero, meaning it was explicitly set)
            theta_init[i] = spec.initial_values[param_name]
            sources[param_name] = 'spec_default'
            n_from_spec += 1
        else:
            # Use fallback default
            theta_init[i] = default_value
            sources[param_name] = 'fallback_default'
            n_from_default += 1
    
    logger.info(f"Initial values: {n_from_prev} from previous, "
                f"{n_from_spec} from spec, {n_from_default} from fallback default ({default_value})")
    
    # Log which parameters are new
    if n_from_default > 0 or n_from_spec > 0:
        new_params = [p for p, s in sources.items() if s != 'previous']
        if new_params and len(new_params) <= 20:
            logger.info(f"New/default parameters: {new_params}")
        elif new_params:
            logger.info(f"New/default parameters: {new_params[:10]} ... and {len(new_params)-10} more")
    
    return theta_init, sources


# ==============================================================================
# End of estimation_spec_parser.py
# ==============================================================================
