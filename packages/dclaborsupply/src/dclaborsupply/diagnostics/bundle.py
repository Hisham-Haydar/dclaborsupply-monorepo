"""
==============================================================================
RURO Post-Estimation Diagnostics Bundle
==============================================================================

A specification-agnostic, model-aware, solver-aware diagnostics object that
both the styled HTML report and the LLM Markdown summary consume.

Design goals
------------
* One normalized representation. HTML and Markdown render the same numbers.
* Sections are dynamic: each section carries ``available`` + ``unavailable_reason``
  so the renderer can either render the section or say *why it is missing*.
* Metric registry: every metric carries label, category, source, applicability,
  interpretation note, display precision, warning threshold, profile membership.
* Reorganized fit-statistics:
    A. Core likelihood and sample statistics
    B. Null-model and pseudo-R² diagnostics
    C. Bound / fixed-parameter diagnostics
    D. Economic sanity diagnostics  (NOT model-fit; surfaced separately)
* Profiles: ``decision`` / ``standard`` / ``full`` / ``technical`` filter which
  metrics and sections are rendered.

Lift boundary (migration matrix Wave 2.6)
-----------------------------------------
Lifted from MNL/scripts/enhanced/diagnostics_bundle.py: ONLY the model-generic
diagnostics layer — the bundle data structures, the metric registry, the
solver-family classification, the finite-metric assembly
(``build_diagnostics_bundle``), and the generic CONOPT text parsers. The
HTML/Markdown renderers (``render_*``), the render-profile gating
(``section_is_visible``), and the artifact writer (``write_bundle_artifacts``,
file I/O) were NOT lifted — they are app/report-layer concerns and stay deferred.
No EUROMOD / France / Java / old-repo dependency; imports are stdlib + numpy, so
``import dclaborsupply.diagnostics`` stays light (no jax/gamspy/java).
==============================================================================
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


# ==============================================================================
# Solver family classification (specification-agnostic)
# ==============================================================================

SOLVER_FAMILY_CONOPT: str = "conopt"
SOLVER_FAMILY_IPOPT: str = "ipopt"
SOLVER_FAMILY_KNITRO: str = "knitro"
SOLVER_FAMILY_BFGS: str = "bfgs"
SOLVER_FAMILY_TRUST_CONSTR: str = "trust-constr"
SOLVER_FAMILY_OTHER: str = "other"
SOLVER_FAMILY_UNKNOWN: str = "unknown"

# Fields that only make sense for CONOPT/GAMS runs.
_CONOPT_ONLY_FIELDS: Tuple[str, ...] = (
    "rgmax", "model_status", "equations", "variables",
    "nonzeros", "max_infeasibility", "generation_time_s",
    "solve_time_s",
)


def classify_solver_family(name: Any) -> str:
    """Best-effort solver-family classification from a solver name string.

    Recognizes: CONOPT/GAMS family, IPOPT, KNITRO, BFGS/L-BFGS-B/SciPy,
    trust-constr; falls back to 'other' / 'unknown'.
    """
    if name is None:
        return SOLVER_FAMILY_UNKNOWN
    s = str(name).strip().lower()
    if not s:
        return SOLVER_FAMILY_UNKNOWN
    if "conopt" in s or "gams" in s:
        return SOLVER_FAMILY_CONOPT
    if "ipopt" in s:
        return SOLVER_FAMILY_IPOPT
    if "knitro" in s:
        return SOLVER_FAMILY_KNITRO
    if "trust" in s and "constr" in s:
        return SOLVER_FAMILY_TRUST_CONSTR
    if "bfgs" in s or "l-bfgs" in s or "scipy" in s:
        return SOLVER_FAMILY_BFGS
    return SOLVER_FAMILY_OTHER


# ==============================================================================
# CONOPT iteration-log parser
# ==============================================================================

# Matches CONOPT iteration headers in the solver log / listing:
#   "Iter Phase   Ninf   Infeasibility   RGmax      NSB   Step  InItr MX OK"
#   "Iter Phase   Ninf     Objective     RGmax      NSB   Step  InItr MX OK"
_CONOPT_ITER_HEADER_RE = re.compile(
    r"^\s*Iter\s+Phase\s+Ninf\s+(?P<col3>\S+(?:\s+\S+)?)\s+RGmax\b",
    re.IGNORECASE,
)


def parse_conopt_rgmax_from_text(text: str) -> Optional[float]:
    """Extract the *terminal* CONOPT RGmax value from a solver log / listing text.

    CONOPT prints an iteration table whose 5th column is RGmax (scientific
    notation, e.g. ``5.4E-08``). This function scans the text for CONOPT
    iteration headers and returns the LAST RGmax value seen.

    Returns ``None`` if the text contains no CONOPT iteration table, or if
    every row in such a table fails to parse.

    Robust to:
      * "Infeasibility" vs "Objective" header column;
      * multiple blocks (it returns the LAST RGmax value across all blocks);
      * mid-row truncations (e.g. ``14   4   -1.9E+04 5.4E-08`` with no NSB column).
    """
    if not isinstance(text, str) or "Iter Phase" not in text:
        return None
    rgmax: Optional[float] = None
    in_block = False
    # Number-row pattern: starts with an integer iter index, has 4+ tokens.
    # We capture the 5th numeric token (RGmax) which may appear in scientific notation.
    row_re = re.compile(
        r"^\s*(\d+)\s+\d+\s+(?:\d+\s+)?(?P<col3>[\-+]?[\d.]+E?[\-+]?\d*)\s+(?P<rgmax>[\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?)\b"
    )
    for raw in text.splitlines():
        line = raw.rstrip()
        if _CONOPT_ITER_HEADER_RE.match(line):
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        # Stop the block when we hit a non-iteration line that begins with non-digit
        if not stripped[:1].isdigit():
            in_block = False
            continue
        m = row_re.match(line)
        if m:
            try:
                rgmax = float(m.group("rgmax"))
            except (TypeError, ValueError):
                pass
    return rgmax


def parse_conopt_termination_text(text: str) -> Optional[str]:
    """Return the CONOPT termination sentence if present, else None."""
    if not isinstance(text, str):
        return None
    for keyword in (
        "Optimal solution. Reduced gradient less than tolerance",
        "Locally optimal solution",
        "Feasible solution. Value of objective",
        "Infeasible solution",
        "Iteration limit",
        "Time limit",
    ):
        m = re.search(r"\*\*\s*(" + re.escape(keyword) + r"[^\n]*)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


# CONOPT iteration row pattern (technical trace).
# Tokens in order: Iter Phase Ninf {Objective|Infeasibility} RGmax NSB Step InItr MX OK
# The terminal row in a block may be truncated (some trailing columns absent).
_CONOPT_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_CONOPT_ROW_RE = re.compile(
    r"^\s*(?P<iter>\d+)"
    r"\s+(?P<phase>\d+)"
    r"(?:\s+(?P<ninf>\d+))?"
    r"\s+(?P<col4>" + _CONOPT_NUM_RE + r")"
    r"(?:\s+(?P<rgmax>" + _CONOPT_NUM_RE + r"))?"
    r"(?:\s+(?P<nsb>\d+))?"
    r"(?:\s+(?P<step>" + _CONOPT_NUM_RE + r"))?"
    r"(?:\s+(?P<initr>\d+))?"
    r"(?:\s+(?P<mx>[FT]))?"
    r"(?:\s+(?P<ok>[FT]))?"
    r"\s*$"
)
_CONOPT_HEADER_OBJ_RE = re.compile(
    r"^\s*Iter\s+Phase\s+Ninf\s+(Objective|Infeasibility)\s+RGmax\b",
    re.IGNORECASE,
)


def parse_conopt_trace_from_text(text: str) -> Dict[str, Any]:
    """Parse the CONOPT iteration table(s) and warning lines from a solver
    log / listing text into a technical-trace dict.

    The returned dict is intended for an appendix-level technical view —
    not for the main solver section. Headline statistics (final RGmax,
    final objective, status, model status, termination text) are produced
    separately by ``parse_conopt_rgmax_from_text`` / the listing parser.

    Returns an empty dict if no CONOPT iteration table is detected. All
    fields below are optional and only populated when actually parsed.

    Trace fields (populated when present):
      * ``iteration_rows_parsed``  total number of CONOPT iteration rows
      * ``final_iteration``        last ``Iter`` index seen
      * ``final_objective``        last ``Objective``/``Infeasibility`` value
      * ``final_rgmax``            last ``RGmax`` value
      * ``final_ninf``             last ``Ninf`` value
      * ``final_nsb``              last ``NSB`` value
      * ``final_step``             last ``Step`` value
      * ``step_min, step_median``  across parsed iteration rows
      * ``ok_F_count, ok_T_count, ok_F_share, ok_T_share``
      * ``mx_F_count, mx_T_count, mx_T_share``
      * ``in_itr_max, in_itr_mean``
      * ``phase_counts``           {"4": n4, "3": n3, ...}
      * ``warnings``               dict of detected CONOPT warning categories
      * ``warning_lines``          up to 20 raw warning lines from the artifact
    """
    if not isinstance(text, str) or "Iter Phase" not in text:
        return {}

    rows: List[Dict[str, Any]] = []
    in_block = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if _CONOPT_HEADER_OBJ_RE.match(line):
            in_block = True
            continue
        if not in_block:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        # Stop the block when a non-iteration line appears
        if not stripped[:1].isdigit():
            in_block = False
            continue
        m = _CONOPT_ROW_RE.match(line)
        if not m:
            continue
        row: Dict[str, Any] = {}
        for key in ("iter", "phase", "ninf", "nsb", "initr"):
            v = m.group(key)
            if v is not None:
                try:
                    row[key] = int(v)
                except ValueError:
                    pass
        for key in ("col4", "rgmax", "step"):
            v = m.group(key)
            if v is not None:
                try:
                    row[key] = float(v)
                except ValueError:
                    pass
        for key in ("mx", "ok"):
            v = m.group(key)
            if v is not None:
                row[key] = v
        if row:
            rows.append(row)

    out: Dict[str, Any] = {}
    if rows:
        out["iteration_rows_parsed"] = len(rows)
        last = rows[-1]
        if "iter" in last:
            out["final_iteration"] = last["iter"]
        if "col4" in last:
            out["final_objective"] = last["col4"]
        if "rgmax" in last:
            out["final_rgmax"] = last["rgmax"]
        if "ninf" in last:
            out["final_ninf"] = last["ninf"]
        if "nsb" in last:
            out["final_nsb"] = last["nsb"]
        if "step" in last:
            out["final_step"] = last["step"]

        # Step statistics across all parsed rows
        steps = [r["step"] for r in rows if "step" in r]
        if steps:
            out["step_min"] = float(min(steps))
            out["step_median"] = float(np.median(steps))

        # OK / MX counts and shares
        ok_vals = [r.get("ok") for r in rows if r.get("ok") in ("F", "T")]
        if ok_vals:
            n = len(ok_vals)
            ok_t = sum(1 for v in ok_vals if v == "T")
            ok_f = n - ok_t
            out["ok_T_count"] = ok_t
            out["ok_F_count"] = ok_f
            out["ok_T_share"] = ok_t / n if n else None
            out["ok_F_share"] = ok_f / n if n else None
        mx_vals = [r.get("mx") for r in rows if r.get("mx") in ("F", "T")]
        if mx_vals:
            n = len(mx_vals)
            mx_t = sum(1 for v in mx_vals if v == "T")
            mx_f = n - mx_t
            out["mx_T_count"] = mx_t
            out["mx_F_count"] = mx_f
            out["mx_T_share"] = mx_t / n if n else None

        # InItr stats
        initrs = [r["initr"] for r in rows if "initr" in r]
        if initrs:
            out["in_itr_max"] = int(max(initrs))
            out["in_itr_mean"] = float(sum(initrs) / len(initrs))

        # Phase counts
        phase_counts: Dict[str, int] = {}
        for r in rows:
            p = r.get("phase")
            if p is None:
                continue
            phase_counts[str(p)] = phase_counts.get(str(p), 0) + 1
        if phase_counts:
            out["phase_counts"] = phase_counts

    # --- CONOPT warning detection (works even if no iteration table parsed) ---
    warning_categories = {
        "evaluation_errors": re.compile(
            r"EVALUATION\s+ERRORS\s+(\d+)\s+(\d+)", re.IGNORECASE),
        "domain_errors": re.compile(
            r"DOMAIN\s+ERRORS?\s+(\d+)", re.IGNORECASE),
    }
    warnings_out: Dict[str, Any] = {}
    for key, pat in warning_categories.items():
        m = pat.search(text)
        if m:
            try:
                # Sum the numeric groups so "EVALUATION ERRORS 0 0" → 0
                total = sum(int(g) for g in m.groups() if g and g.isdigit())
                warnings_out[key] = total
            except ValueError:
                warnings_out[key] = m.group(0).strip()

    # Free-text warning keywords (slow convergence, time limit, scaling, etc.)
    text_warnings: List[str] = []
    text_warning_patterns = (
        ("scaling", re.compile(r"^\s*\*+\s*WARNING.*scal", re.IGNORECASE | re.MULTILINE)),
        ("slow_convergence", re.compile(r"slow\s+convergence|stalled|small\s+step", re.IGNORECASE)),
        ("time_limit", re.compile(r"time\s+limit|resource\s+limit", re.IGNORECASE)),
        ("iteration_limit", re.compile(r"iteration\s+limit", re.IGNORECASE)),
        ("infeasibility", re.compile(r"infeasible\s+solution", re.IGNORECASE)),
    )
    for key, pat in text_warning_patterns:
        if pat.search(text):
            warnings_out[key] = True

    # Capture up to 20 raw lines containing the substring "warning" (case-insensitive)
    raw_warning_lines: List[str] = []
    for raw in text.splitlines():
        if "warning" in raw.lower() and len(raw.strip()) < 300:
            raw_warning_lines.append(raw.strip())
            if len(raw_warning_lines) >= 20:
                break
    if raw_warning_lines:
        warnings_out["warning_lines"] = raw_warning_lines

    if warnings_out:
        out["warnings"] = warnings_out

    return out


# ==============================================================================
# Profile vocabulary
# ==============================================================================

PROFILE_CHOICES: Tuple[str, ...] = ("decision", "standard", "full", "technical")
DEFAULT_PROFILE: str = "standard"


# ==============================================================================
# Metric registry
# ==============================================================================

@dataclass(frozen=True)
class MetricSpec:
    """One entry in the metric registry."""
    key: str
    label: str
    category: str           # 'core' | 'null' | 'bounds' | 'economic' | 'solver' | 'inference' | 'hessian' | 'gradient' | 'data'
    source: str             # human-readable source path (informational)
    applicability: str      # when this metric is meaningful (free text)
    interpretation: str
    precision: int = 4
    threshold: Optional[str] = None
    profiles: Tuple[str, ...] = PROFILE_CHOICES  # which profiles include it
    appendix_only: bool = False                  # if True, only shown in technical profile

    def for_profile(self, profile: str) -> bool:
        if profile not in self.profiles:
            return False
        if self.appendix_only and profile != "technical":
            return False
        return True


# Registry. Add metrics here; renderers consult this to know how to format and
# whether to include a given metric for a given profile.
METRIC_REGISTRY: Dict[str, MetricSpec] = {m.key: m for m in [
    # A. Core likelihood and sample
    MetricSpec("log_likelihood", "Log-likelihood",
               category="core", source="results_json.summary.joint_ll",
               applicability="Always available when estimation converged.",
               interpretation="Higher (less negative) is better. Comparable only across runs with identical sample, alternative set, and prior weights.",
               precision=4),
    MetricSpec("n_observations", "Observations (rows)",
               category="core", source="results_json.summary.n_obs_total",
               applicability="Always.",
               interpretation="Long-format alternative-level row count.",
               precision=0),
    MetricSpec("n_groups", "Choice sets / groups",
               category="core", source="results_json.summary.n_groups_total",
               applicability="Always.",
               interpretation="Number of decision units (households).",
               precision=0),
    MetricSpec("n_alts_per_set", "Alternatives per choice set",
               category="core", source="derived: n_observations / n_groups",
               applicability="When n_groups > 0.",
               interpretation="Average alternatives per decision unit (may not be integer if unbalanced).",
               precision=2),
    MetricSpec("n_free_parameters", "Free parameters",
               category="core", source="parsed.bounds + theta",
               applicability="Always.",
               interpretation="Parameters being estimated freely (not fixed and not at a tight bound range).",
               precision=0),
    MetricSpec("n_fixed_parameters", "Fixed parameters",
               category="core", source="parsed.bounds (lb == ub)",
               applicability="Always.",
               interpretation="Parameters with lb == ub or fixed by spec.",
               precision=0),
    MetricSpec("AIC", "AIC",
               category="core", source="-2*ll + 2*k",
               applicability="Meaningful for nested or same-sample comparisons.",
               interpretation="Lower is better. Comparable only between models on the same data with the same null structure.",
               precision=4),
    MetricSpec("BIC", "BIC",
               category="core", source="-2*ll + log(n)*k",
               applicability="Meaningful for nested or same-sample comparisons.",
               interpretation="Lower is better. Penalises parameter count more than AIC.",
               precision=4),
    MetricSpec("AIC_per_obs", "AIC / n_obs",
               category="core", source="AIC / n_observations",
               applicability="When n_observations > 0.",
               interpretation="Per-observation AIC for cross-sample comparison sanity check.",
               precision=6),

    # B. Null-model / pseudo-R²
    MetricSpec("ll_null_uniform", "ll_null (uniform)",
               category="null", source="compute_null_log_likelihood",
               applicability="Requires data parquet to be readable.",
               interpretation="Log-likelihood of the uniform-choice null model.",
               precision=4),
    MetricSpec("ll_null_prior_corrected", "ll_null (prior-corrected)",
               category="null", source="compute_null_log_likelihood_prior_corrected",
               applicability="Requires data parquet with proposal/prior weights.",
               interpretation="Recommended null for sampled-alternative / job-choice models.",
               precision=4),
    MetricSpec("rho_squared_uniform", "ρ² (McFadden, uniform null)",
               category="null", source="1 - ll/ll_null_uniform",
               applicability="When ll_null_uniform is available and non-zero.",
               interpretation="0.2–0.4 is typically a 'good' fit for MNL. Comparable only across models with same uniform null structure.",
               precision=4),
    MetricSpec("rho_squared_prior_corrected", "ρ² (prior-corrected null)",
               category="null", source="1 - ll/ll_null_prior_corrected",
               applicability="When ll_null_prior_corrected is available and non-zero.",
               interpretation="The right pseudo-R² for sampled-alternative or job-choice models.",
               precision=4),
    MetricSpec("rho_squared_adj_uniform", "Adj. ρ² (uniform)",
               category="null", source="1 - (ll-k)/ll_null_uniform",
               applicability="When ll_null_uniform is available and non-zero.",
               interpretation="Penalises additional parameters.",
               precision=4),
    MetricSpec("rho_squared_adj_prior_corrected", "Adj. ρ² (prior-corrected)",
               category="null", source="1 - (ll-k)/ll_null_prior_corrected",
               applicability="When ll_null_prior_corrected is available and non-zero.",
               interpretation="Penalised pseudo-R² against the correct null.",
               precision=4),

    # C. Bound / fixed parameters
    MetricSpec("n_parameters", "Parameters (total)",
               category="bounds", source="parsed.param_names",
               applicability="Always.",
               interpretation="Total parameters in the specification.",
               precision=0),
    MetricSpec("n_parameters_with_bounds", "Parameters with bounds",
               category="bounds", source="parsed.bounds",
               applicability="When bounds are loaded.",
               interpretation="Parameters with at least one finite bound.",
               precision=0),
    MetricSpec("n_at_lower_bound", "At lower bound",
               category="bounds", source="abs(theta - lb) < tol",
               applicability="When bounds are present.",
               interpretation="0 is preferred. Non-zero usually indicates a binding economic constraint or a misspecified bound.",
               precision=0, threshold="warn if > 0"),
    MetricSpec("n_at_upper_bound", "At upper bound",
               category="bounds", source="abs(theta - ub) < tol",
               applicability="When bounds are present.",
               interpretation="0 is preferred. Non-zero may indicate truncation by spec.",
               precision=0, threshold="warn if > 0"),

    # D. Economic sanity
    MetricSpec("negative_muc_count", "Households with MUC < 0",
               category="economic", source="mu_results.totals",
               applicability="When MU diagnostics were computed.",
               interpretation="Count of households where marginal utility of consumption is non-positive at the chosen alternative.",
               precision=0, threshold="warn if > 0"),
    MetricSpec("negative_muc_pct", "% households with MUC < 0",
               category="economic", source="mu_results.totals",
               applicability="When MU diagnostics were computed.",
               interpretation="Share of households violating MUC > 0.",
               precision=2, threshold="warn if > 1%"),
    MetricSpec("negative_mul_count", "Households with MUL < 0",
               category="economic", source="mu_results.totals",
               applicability="When MU diagnostics were computed.",
               interpretation="Count of households where marginal utility of leisure is non-positive at the chosen alternative.",
               precision=0, threshold="warn if > 0"),
    MetricSpec("negative_mul_pct", "% households with MUL < 0",
               category="economic", source="mu_results.totals",
               applicability="When MU diagnostics were computed.",
               interpretation="Share of households violating MUL > 0.",
               precision=2, threshold="warn if > 1%"),

    # Solver
    MetricSpec("solver_name", "Solver",
               category="solver", source="results_json.metadata.opt_method or summary.solver",
               applicability="Always.",
               interpretation="Name of the optimization engine that produced the estimates.",
               precision=0),
    MetricSpec("solver_status", "Solver status",
               category="solver", source="results.<group>.message or listing file",
               applicability="When solver reports a status.",
               interpretation="'Optimal' / 'Normal Completion' indicate clean convergence.",
               precision=0),
    MetricSpec("model_status", "Model status",
               category="solver", source="listing file",
               applicability="CONOPT/GAMS only.",
               interpretation="GAMS model-status code, e.g. 'Locally Optimal'.",
               precision=0),
    MetricSpec("rgmax", "Reduced-gradient max (RGmax)",
               category="solver", source="solver.log / solver.lst (CONOPT)",
               applicability="CONOPT / GAMSPy runs only; requires solver log or listing.",
               interpretation="Solver-internal reduced-gradient norm at termination. DISTINCT from Python likelihood gradient at θ.",
               precision=6),

    # Gradient (Python score)
    MetricSpec("grad_inf_norm", "‖∇ log L‖∞ (Python score)",
               category="gradient", source="--gradient-diagnostics",
               applicability="When --gradient-diagnostics is supplied with --mnl-base and --spec-config.",
               interpretation="Infinity norm of the Python likelihood gradient at converged θ. NOT necessarily the solver reduced gradient when bounds or constraints are active.",
               precision=6),
    MetricSpec("grad_l2_norm", "‖∇ log L‖₂ (Python score)",
               category="gradient", source="--gradient-diagnostics",
               applicability="When --gradient-diagnostics is supplied.",
               interpretation="L2 norm of the Python likelihood gradient at converged θ.",
               precision=6),

    # Hessian
    MetricSpec("hessian_cond_number", "Hessian condition number",
               category="hessian", source="results_json.hessian_diagnostics or cluster-SE JSON",
               applicability="When Hessian is invertible.",
               interpretation="Large values (≫ 1e10) signal weak identification or near-singular Hessian.",
               precision=2, threshold="warn if > 1e10"),
    MetricSpec("hessian_n_negative_eigs", "Hessian negative eigenvalues",
               category="hessian", source="np.linalg.eigvalsh(H)",
               applicability="When Hessian is available.",
               interpretation="Should be 0 at a local optimum of the *negative* log-likelihood. >0 indicates saddle or non-optimal.",
               precision=0, threshold="warn if > 0"),
]}


def get_metric(key: str) -> Optional[MetricSpec]:
    return METRIC_REGISTRY.get(key)


# ==============================================================================
# DiagnosticsBundle dataclass
# ==============================================================================

@dataclass
class Section:
    """A bundle section with availability + reason."""
    available: bool = False
    unavailable_reason: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticsBundle:
    """Normalized diagnostics object consumed by HTML + Markdown renderers."""
    profile: str = DEFAULT_PROFILE

    estimation_metadata: Dict[str, Any] = field(default_factory=dict)
    data_metadata: Dict[str, Any] = field(default_factory=dict)
    spec_metadata: Dict[str, Any] = field(default_factory=dict)

    solver: Section = field(default_factory=Section)
    likelihood_fit_core: Section = field(default_factory=Section)
    null_model_fit: Section = field(default_factory=Section)
    bounds_diagnostics: Section = field(default_factory=Section)
    economic_sanity: Section = field(default_factory=Section)
    inference: Section = field(default_factory=Section)
    robust_se: Section = field(default_factory=Section)
    hessian: Section = field(default_factory=Section)
    gradient_score: Section = field(default_factory=Section)
    probability_fit: Section = field(default_factory=Section)

    reproducibility: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Builders
# ==============================================================================

def _safe_num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    f = _safe_num(x)
    return int(f) if f is not None else None


def _block_for_param(name: str, block_map: Optional[Dict[str, str]]) -> str:
    """Specification-agnostic parameter -> block classifier.

    Strategy:
    1) Exact match in ``block_map``;
    2) Strip group-suffix tokens (_sm/_sf/_m/_f/_cm/_cf) then exact match;
    3) Substring heuristics on the parameter name.
    """
    if not name:
        return "other"
    lookup = block_map or {}
    if name in lookup:
        return lookup[name]
    # Strip common group suffixes
    base = name
    for suf in ("_sm", "_sf", "_cm", "_cf", "_m", "_f"):
        if base.endswith(suf):
            stripped = base[: -len(suf)]
            if stripped in lookup:
                return lookup[stripped]
            base = stripped
            break
    n = name.lower()
    if "beta_offer_" in n:
        return "market_opportunity"
    if "beta_occ_" in n:
        return "occupation_opportunity"
    if any(t in n for t in ("beta_w", "sigma", "beta_pexp")):
        return "wage_opportunity"
    if any(t in n for t in ("beta_work", "beta_pt", "beta_ft", "beta_gsur",
                            "beta_e_", "beta_e ", "beta_h_")):
        return "employment_hours_opportunity"
    if n in ("beta_e",):
        return "employment_hours_opportunity"
    if any(t in n for t in ("beta_l", "beta_c", "theta_l", "theta_c", "beta_interact")):
        return "preference"
    return "other"


def build_diagnostics_bundle(
    *,
    profile: str,
    results_data: Dict[str, Any],
    parsed_params: Any,
    fit_stats: Dict[str, Any],
    bound_diagnostics: List[Dict[str, Any]],
    mu_results: Optional[Dict[str, Any]] = None,
    prob_diagnostics: Optional[Dict[str, Any]] = None,
    hessian_diagnostics: Optional[Dict[str, Any]] = None,
    cluster_se_data: Optional[Dict[str, Any]] = None,
    solver_diag: Optional[Dict[str, Any]] = None,
    gradient_diag: Optional[Dict[str, Any]] = None,
    repro_meta: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    block_map: Optional[Dict[str, str]] = None,
) -> DiagnosticsBundle:
    """Assemble a DiagnosticsBundle from already-computed pieces.

    All optional inputs degrade gracefully: missing pieces produce sections
    with ``available=False`` and a human-readable ``unavailable_reason``.
    """
    if profile not in PROFILE_CHOICES:
        LOGGER.warning("Unknown profile %r, falling back to %r", profile, DEFAULT_PROFILE)
        profile = DEFAULT_PROFILE

    bundle = DiagnosticsBundle(profile=profile)
    metadata = results_data.get("metadata", {}) or {}
    summary = results_data.get("summary", {}) or {}

    # --- estimation metadata
    bundle.estimation_metadata = {
        "specification": results_data.get("specification"),
        "wage_spec": results_data.get("wage_spec"),
        "group": metadata.get("group"),
        "opt_method": metadata.get("opt_method"),
        "analytical_gradient": metadata.get("analytical_gradient"),
        "command_line": results_data.get("command_line"),
        "timestamp": results_data.get("timestamp"),
    }

    # --- data metadata
    n_obs = _safe_int(summary.get("n_obs_total")) or _safe_int(fit_stats.get("n_observations"))
    n_groups = _safe_int(summary.get("n_groups_total")) or _safe_int(fit_stats.get("n_groups"))
    n_alts_per_set = None
    if n_obs and n_groups:
        n_alts_per_set = n_obs / n_groups
    bundle.data_metadata = {
        "n_observations": n_obs,
        "n_groups": n_groups,
        "n_alts_per_set": n_alts_per_set,
        "n_obs_long": _safe_int(fit_stats.get("n_obs_long")),
        "mnl_base": metadata.get("mnl_base"),
    }

    # --- spec metadata
    bundle.spec_metadata = {
        "spec_config": metadata.get("spec_config"),
        "n_parameters": _safe_int(fit_stats.get("n_parameters")) or len(getattr(parsed_params, "param_names", []) or []),
    }

    # --- bounds diagnostics (always available; values may be zero)
    n_params = bundle.spec_metadata["n_parameters"] or 0
    n_with_bounds = 0
    n_at_lower = 0
    n_at_upper = 0
    n_fixed = 0
    near_bound_list: List[Dict[str, Any]] = []
    bounds = getattr(parsed_params, "bounds", None)
    theta = getattr(parsed_params, "theta", None)
    names = getattr(parsed_params, "param_names", None) or []
    tol_bound = 1e-6
    tol_near = 1e-3
    if bounds is not None and theta is not None:
        for i, name in enumerate(names):
            if i >= len(bounds):
                continue
            lb, ub = bounds[i]
            if lb is not None or ub is not None:
                n_with_bounds += 1
            if lb is not None and ub is not None and abs(float(ub) - float(lb)) <= tol_bound:
                n_fixed += 1
                continue
            try:
                v = float(theta[i])
            except (TypeError, ValueError, IndexError):
                continue
            if lb is not None and abs(v - float(lb)) < tol_bound:
                n_at_lower += 1
                near_bound_list.append({"parameter": name, "estimate": v, "bound": float(lb), "side": "lower", "distance": 0.0})
            elif ub is not None and abs(v - float(ub)) < tol_bound:
                n_at_upper += 1
                near_bound_list.append({"parameter": name, "estimate": v, "bound": float(ub), "side": "upper", "distance": 0.0})
            elif lb is not None and (v - float(lb)) < tol_near:
                near_bound_list.append({"parameter": name, "estimate": v, "bound": float(lb), "side": "near_lower", "distance": v - float(lb)})
            elif ub is not None and (float(ub) - v) < tol_near:
                near_bound_list.append({"parameter": name, "estimate": v, "bound": float(ub), "side": "near_upper", "distance": float(ub) - v})

    n_free = max(0, (n_params or 0) - n_fixed)
    bundle.spec_metadata["n_free_parameters"] = n_free
    bundle.spec_metadata["n_fixed_parameters"] = n_fixed
    bundle.spec_metadata["n_parameters_with_bounds"] = n_with_bounds

    bundle.bounds_diagnostics = Section(
        available=True,
        data={
            "n_parameters": n_params,
            "n_free_parameters": n_free,
            "n_fixed_parameters": n_fixed,
            "n_parameters_with_bounds": n_with_bounds,
            "n_at_lower_bound": n_at_lower,
            "n_at_upper_bound": n_at_upper,
            "at_or_near_bounds": near_bound_list,
            "tol_at_bound": tol_bound,
            "tol_near_bound": tol_near,
        },
    )

    # --- A. core likelihood-fit
    ll = _safe_num(fit_stats.get("log_likelihood"))
    aic = _safe_num(fit_stats.get("AIC"))
    bic = _safe_num(fit_stats.get("BIC"))
    aic_per_obs = _safe_num(fit_stats.get("AIC_per_obs"))
    bundle.likelihood_fit_core = Section(
        available=ll is not None,
        unavailable_reason="" if ll is not None else "log_likelihood not present in fit_stats.",
        data={
            "log_likelihood": ll,
            "n_observations": n_obs,
            "n_groups": n_groups,
            "n_alts_per_set": n_alts_per_set,
            "n_free_parameters": n_free,
            "n_fixed_parameters": n_fixed,
            "AIC": aic,
            "BIC": bic,
            "AIC_per_obs": aic_per_obs,
        },
    )

    # --- B. null-model / pseudo-R²
    ll_null_uni = _safe_num(fit_stats.get("ll_null_uniform"))
    ll_null_prior = _safe_num(fit_stats.get("ll_null_prior_corrected"))
    rho2_uni = _safe_num(fit_stats.get("rho_squared_uniform"))
    rho2_prior = _safe_num(fit_stats.get("rho_squared_prior_corrected"))
    rho2_adj_uni = _safe_num(fit_stats.get("rho_squared_adj_uniform"))
    rho2_adj_prior = _safe_num(fit_stats.get("rho_squared_adj_prior_corrected"))
    null_available = any(v is not None for v in (ll_null_uni, ll_null_prior))
    bundle.null_model_fit = Section(
        available=null_available,
        unavailable_reason=("" if null_available else
                            "No null log-likelihood available. Supply --mnl-base so the script can read the parquet data and compute LL0."),
        data={
            "ll_null_uniform": ll_null_uni,
            "ll_null_prior_corrected": ll_null_prior,
            "rho_squared_uniform": rho2_uni,
            "rho_squared_prior_corrected": rho2_prior,
            "rho_squared_adj_uniform": rho2_adj_uni,
            "rho_squared_adj_prior_corrected": rho2_adj_prior,
            "note": ("ρ² values use McFadden's formulation 1 - LL/LL0. "
                     "For sampled-alternative / job-choice models the prior-corrected null is the right comparison; "
                     "the uniform null is kept for legacy comparability."),
        },
    )

    # --- D. economic sanity
    # The marginal-utility computation (compute_marginal_utilities_at_chosen)
    # writes totals under the keys n_negative_muc_total / pct_negative_muc_total
    # (and the _mul_ variants). Accept those AND the older negative_muc_count /
    # _pct aliases so the section populates whenever MUC was computed (it is
    # whenever --mnl-base is supplied). Previously the key names did not match
    # and the section falsely reported "not computed (requires --mnl-base)".
    econ_data = {}
    if mu_results and isinstance(mu_results, dict):
        totals = mu_results.get("totals", {}) or {}
        _alias = {
            "n_negative_muc_total": "negative_muc_count",
            "pct_negative_muc_total": "negative_muc_pct",
            "n_negative_mul_total": "negative_mul_count",
            "pct_negative_mul_total": "negative_mul_pct",
        }
        for src, dst in _alias.items():
            if src in totals:
                econ_data[dst] = totals.get(src)
        # also pass through any already-canonical keys
        for k in ("negative_muc_count", "negative_muc_pct",
                  "negative_mul_count", "negative_mul_pct",
                  "negative_mu_count", "negative_mu_pct",
                  "monotonicity_violations"):
            if k in totals and k not in econ_data:
                econ_data[k] = totals.get(k)
    bundle.economic_sanity = Section(
        available=bool(econ_data),
        unavailable_reason=("" if econ_data else
                            "Marginal-utility diagnostics not computed (requires --mnl-base)."),
        data=econ_data,
    )

    # --- Solver section
    solver_name = (
        (solver_diag or {}).get("solver_name")
        if isinstance(solver_diag, dict) else None
    )
    if not solver_name:
        solver_name = metadata.get("opt_method")
    solver_data = {
        "solver_name": solver_name,
        "objective_ll": ll,
        "wall_time_seconds": _safe_num(summary.get("total_walltime_seconds")),
    }
    # Pull per-group iteration / nfev / gradient info
    per_group_solver: List[Dict[str, Any]] = []
    results_groups = results_data.get("results", {}) or {}
    for gname, gdat in results_groups.items():
        if not isinstance(gdat, dict):
            continue
        per_group_solver.append({
            "group": gname,
            "success": gdat.get("success"),
            "message": gdat.get("message"),
            "n_iterations": gdat.get("n_iterations"),
            "n_function_evaluations": gdat.get("n_function_evaluations"),
            "gradient_norm_results_json": gdat.get("gradient_norm"),
            "final_ll": gdat.get("final_ll") or gdat.get("log_likelihood"),
            "walltime_seconds": gdat.get("walltime_seconds"),
        })
    if per_group_solver:
        solver_data["per_group"] = per_group_solver

    # Augment with parsed CONOPT/log artifacts if available
    listing = (solver_diag or {}).get("listing_diagnostics", {}) if isinstance(solver_diag, dict) else {}
    solver_log = (solver_diag or {}).get("solver_log_diagnostics", {}) if isinstance(solver_diag, dict) else {}
    rgmax = None
    if isinstance(listing, dict):
        for k in ("rgmax", "RGmax", "reduced_gradient_max", "max_reduced_gradient"):
            if k in listing:
                rgmax = _safe_num(listing[k])
                break
    if rgmax is None and isinstance(solver_log, dict):
        for k in ("rgmax", "RGmax", "reduced_gradient_max"):
            if k in solver_log:
                rgmax = _safe_num(solver_log[k])
                break
    if rgmax is not None:
        solver_data["rgmax"] = rgmax
    if isinstance(listing, dict):
        for k in ("solver_status", "model_status", "solve_time_s",
                  "equations", "variables", "nonzeros", "max_infeasibility"):
            if k in listing and k not in solver_data:
                solver_data[k] = listing[k]
    if isinstance(solver_log, dict):
        for k in ("termination_message", "log_objective"):
            if k in solver_log and k not in solver_data:
                solver_data[k] = solver_log[k]

    # Solver-family classification + CONOPT applicability notes.
    family = classify_solver_family(solver_data.get("solver_name"))
    # Override: if a CONOPT/GAMS listing or solver-log was parsed (i.e. the
    # caller actually supplied --listing-file / --solver-log from a GAMSPy
    # run), force family to CONOPT regardless of opt_method label. The
    # opt_method metadata field can be misleading because some GAMSPy
    # estimators record a fallback name like "L-BFGS-B" while still using
    # CONOPT under the hood.
    _listing_present = isinstance(listing, dict) and listing and not listing.get("_note")
    _log_present = isinstance(solver_log, dict) and solver_log and not solver_log.get("_note")
    if (_listing_present or _log_present) and family in (
        SOLVER_FAMILY_BFGS, SOLVER_FAMILY_UNKNOWN, SOLVER_FAMILY_OTHER,
    ):
        # Heuristic CONOPT signals in the parsed dicts
        if any(k in (listing or {}) for k in ("rgmax", "model_status", "equations", "variables")) \
                or any(k in (solver_log or {}) for k in ("rgmax", "termination_text")):
            family = SOLVER_FAMILY_CONOPT
    # Also honor metadata.solver_artifacts as a strong CONOPT signal
    sa = (metadata.get("solver_artifacts") or {}) if isinstance(metadata, dict) else {}
    if isinstance(sa, dict) and (sa.get("saved") or sa.get("solver_log") or sa.get("listing_file")):
        if family in (SOLVER_FAMILY_BFGS, SOLVER_FAMILY_UNKNOWN, SOLVER_FAMILY_OTHER):
            family = SOLVER_FAMILY_CONOPT
    solver_data["solver_family"] = family
    is_conopt = family == SOLVER_FAMILY_CONOPT

    # --- CONOPT technical trace (Phase 2.1) ---
    # Trace data is only attached when the solver is CONOPT/GAMS *and* the
    # listing or solver-log parser captured a trace. The trace lives in the
    # appendix-level field ``solver.data["conopt_trace"]`` so the main
    # solver section stays compact (status / model status / RGmax /
    # infeasibility / ninf / iterations / termination).
    trace: Dict[str, Any] = {}
    if is_conopt:
        for src in (listing, solver_log):
            if isinstance(src, dict):
                t = src.get("conopt_trace")
                if isinstance(t, dict) and t:
                    # Merge: prefer values from the listing file first
                    # (richer), then fall back to the solver log.
                    for k, v in t.items():
                        if k not in trace:
                            trace[k] = v
        if trace:
            solver_data["conopt_trace"] = trace

    # AGNOSTIC SOLVER REPORTING: show only the stats relevant to the solver
    # actually used. For a non-CONOPT solver (scipy / JAX BFGS family) we DO NOT
    # list CONOPT-specific fields (RGmax, model_status, equations, ...) as
    # "not applicable" — they are simply omitted, and the per-group convergence
    # table (iters / nfev / gradient_norm / walltime) is the relevant view.
    # Symmetrically, a CONOPT run shows RGmax etc. and not scipy-only fields.
    # (Kept as an internal marker for any programmatic consumer, but NOT
    # surfaced in the rendered report — renderers gate on this.)
    not_applicable_fields: List[str] = []
    if not is_conopt:
        for k in _CONOPT_ONLY_FIELDS:
            if k not in solver_data:
                not_applicable_fields.append(k)
        not_applicable_fields.append("conopt_trace")
    if not_applicable_fields:
        # store for programmatic use, but DO NOT set not_applicable_note so the
        # renderers omit the "Fields not applicable: ..." line entirely.
        solver_data["not_applicable_fields"] = not_applicable_fields

    solver_section_available = (
        solver_data.get("solver_name") is not None
        or bool(per_group_solver)
        or rgmax is not None
        or bool(listing and not listing.get("_note"))
        or bool(solver_log and not solver_log.get("_note"))
    )
    bundle.solver = Section(
        available=solver_section_available,
        unavailable_reason=("" if solver_section_available else
                            "No solver metadata available."),
        data=solver_data,
    )

    # --- Hessian section (enriched: scalars + eigenvalues + correlations + identification)
    hess_data: Dict[str, Any] = {}
    _hess_keys = (
        "condition_number", "n_negative_eigenvalues",
        "min_eigenvalue", "max_eigenvalue",
        "eigenvalues", "top_correlations",
        "poorly_identified_params", "eigenvector_diagnostics",
    )
    if hessian_diagnostics and isinstance(hessian_diagnostics, dict):
        for k in _hess_keys:
            if k in hessian_diagnostics:
                hess_data[k] = hessian_diagnostics[k]
    rhd = results_data.get("hessian_diagnostics")
    if isinstance(rhd, dict):
        for k in _hess_keys:
            if k not in hess_data and k in rhd:
                hess_data[k] = rhd[k]
    bundle.hessian = Section(
        available=bool(hess_data),
        unavailable_reason=("" if hess_data else
                            "Hessian diagnostics not present in results JSON or cluster-SE JSON."),
        data=hess_data,
    )

    # --- Robust SE / cluster section
    std_errors_data = results_data.get("standard_errors")
    nested_cluster_available = False
    if isinstance(std_errors_data, dict):
        se_clustered_vec = std_errors_data.get("se_clustered") or std_errors_data.get("se_robust")
        if isinstance(se_clustered_vec, (list, tuple)):
            nested_cluster_available = any(_safe_num(v) is not None for v in se_clustered_vec)
    robust_available = bool(cluster_se_data) or nested_cluster_available
    robust_data: Dict[str, Any] = {}
    if robust_available:
        checks = (cluster_se_data or {}).get("checks", {}) or {}
        robust_data = {
            "source_artifact": ("cluster-SE JSON" if cluster_se_data
                                else "results_json.standard_errors.se_clustered"),
            "T3_cluster_count": checks.get("T3_cluster_count"),
            "T4_se_positivity": checks.get("T4_se_positivity"),
            "T5_robust_vs_hessian": checks.get("T5_robust_vs_hessian"),
            "PE3_data_loaded": checks.get("PE3_data_loaded"),
        }
    bundle.robust_se = Section(
        available=robust_available,
        unavailable_reason=("" if robust_available else
                            "Cluster-robust SEs require --cluster-se-json. Hessian SE is the primary inference source in this report."),
        data=robust_data,
    )

    # --- Gradient/score (Python likelihood gradient)
    if gradient_diag and isinstance(gradient_diag, dict) and gradient_diag.get("available"):
        bundle.gradient_score = Section(
            available=True,
            data={
                "inf_norm": _safe_num(gradient_diag.get("inf_norm")),
                "l2_norm": _safe_num(gradient_diag.get("l2_norm")),
                "top10": gradient_diag.get("top10") or gradient_diag.get("top_components"),
                "label_note": (
                    "Python likelihood-gradient (score at converged θ) computed by central differences. "
                    "This is NOT necessarily the solver reduced gradient when bounds or constraints are active."
                ),
            },
        )
    else:
        bundle.gradient_score = Section(
            available=False,
            unavailable_reason="Pass --gradient-diagnostics with --mnl-base and --spec-config to compute the Python likelihood gradient.",
        )

    # --- Inference table (per parameter)
    inf_rows: List[Dict[str, Any]] = []
    se_full = _extract_se_array(results_data, parsed_params)
    t_full = _extract_t_array(results_data, parsed_params)
    cluster_param_map = _cluster_se_param_map(cluster_se_data, parsed_params=parsed_params)
    if not cluster_param_map:
        cluster_param_map = _cluster_se_param_map(std_errors_data, parsed_params=parsed_params)
    for i, name in enumerate(names):
        try:
            est = float(theta[i]) if theta is not None and i < len(theta) else None
        except (TypeError, ValueError):
            est = None
        lb = ub = None
        if bounds is not None and i < len(bounds):
            lb, ub = bounds[i]
        fixed = (lb is not None and ub is not None
                 and abs(float(ub) - float(lb)) <= tol_bound)
        at_lower = (est is not None and lb is not None
                    and abs(est - float(lb)) < tol_bound)
        at_upper = (est is not None and ub is not None
                    and abs(est - float(ub)) < tol_bound)
        se_h = se_full[i] if se_full is not None and i < len(se_full) else None
        t_h = t_full[i] if t_full is not None and i < len(t_full) else None
        cl = cluster_param_map.get(name, {}) if cluster_param_map else {}
        se_rob = cl.get("se_robust") if cl else None
        t_rob = cl.get("t_robust") if cl else None
        p_rob = cl.get("p_robust") if cl else None
        primary_se = "robust" if se_rob is not None else ("hessian" if se_h is not None else "none")
        inf_rows.append({
            "parameter": name,
            "block": _block_for_param(name, block_map),
            "estimate": est,
            "se_hessian": _safe_num(se_h),
            "t_hessian": _safe_num(t_h),
            "se_robust": _safe_num(se_rob),
            "t_robust": _safe_num(t_rob),
            "p_robust": _safe_num(p_rob),
            "fixed": fixed,
            "at_lower_bound": bool(at_lower),
            "at_upper_bound": bool(at_upper),
            "primary_se": primary_se,
        })
    bundle.inference = Section(
        available=bool(inf_rows),
        unavailable_reason="" if inf_rows else "No parameters to report.",
        data={
            "rows": inf_rows,
            "primary_se_for_run": "robust" if cluster_param_map else "hessian",
            "note": (
                "Primary SE is robust/cluster when cluster SEs are supplied "
                "by --cluster-se-json or embedded in results JSON; "
                "otherwise the Hessian (classical) SE is primary."
            ),
        },
    )

    # --- Probability fit (top-level summary only; full lists stay in CSVs)
    pf_data: Dict[str, Any] = {}
    if prob_diagnostics and isinstance(prob_diagnostics, dict):
        if "prob_sum_errors" in prob_diagnostics:
            pf_data["prob_sum_errors"] = prob_diagnostics.get("prob_sum_errors")
        if "p_chosen_dist" in prob_diagnostics:
            pf_data["p_chosen_dist"] = prob_diagnostics.get("p_chosen_dist")
        worst = (prob_diagnostics.get("worst_fit_households") or [])[:10]
        if worst:
            pf_data["worst_fit_households_top10"] = worst
    bundle.probability_fit = Section(
        available=bool(pf_data),
        unavailable_reason="" if pf_data else "Probability diagnostics require --mnl-base.",
        data=pf_data,
    )

    # --- Reproducibility
    if repro_meta and isinstance(repro_meta, dict):
        bundle.reproducibility = dict(repro_meta)

    # --- Warnings / limitations
    if n_at_lower:
        bundle.warnings.append(f"{n_at_lower} parameter(s) at lower bound.")
    if n_at_upper:
        bundle.warnings.append(f"{n_at_upper} parameter(s) at upper bound.")
    if hess_data.get("n_negative_eigenvalues") and int(hess_data["n_negative_eigenvalues"]) > 0:
        bundle.warnings.append(
            f"Hessian has {hess_data['n_negative_eigenvalues']} negative eigenvalue(s); "
            "estimates may be at a saddle point."
        )
    cond = hess_data.get("condition_number")
    if cond is not None and isinstance(cond, (int, float)) and cond > 1e10:
        bundle.warnings.append(
            f"Hessian condition number is {cond:.2e} — weak identification likely."
        )
    if not bundle.null_model_fit.available:
        bundle.limitations.append("Pseudo-R² unavailable: supply --mnl-base.")
    if not bundle.solver.data.get("rgmax"):
        bundle.limitations.append(
            "CONOPT RGmax unavailable: supply --solver-log and --listing-file from a "
            "GAMSPy run that saved solver artifacts."
        )
    if not bundle.robust_se.available:
        bundle.limitations.append(
            "Cluster-robust SEs unavailable: supply --cluster-se-json."
        )
    if not bundle.gradient_score.available:
        bundle.limitations.append(
            "Python likelihood gradient unavailable: supply --gradient-diagnostics with --mnl-base and --spec-config."
        )

    return bundle


def _extract_se_array(results_data: Dict[str, Any], parsed_params: Any) -> Optional[np.ndarray]:
    """Return SE array aligned to parsed_params.param_names, or None.

    Handles three layouts of ``standard_errors``:
      1. flat name->value dict   ({"beta_E": 0.1, ...})
      2. flat list/tuple aligned to theta
      3. nested {"se": [...], "t_values": [...]} (the enhanced-pipeline layout;
         the ``se`` list is theta-aligned).
      4. nested {"se_hessian": [...], "se_clustered": [...]} from the joint
         Step-4 bridge; the Hessian vector is used for this classical column.
    """
    se = results_data.get("standard_errors")
    if isinstance(se, dict):
        # nested joint-emitter layout with both SE flavours. The inference table
        # treats Hessian SE as the classical diagnostic column.
        if isinstance(se.get("se_hessian"), (list, tuple)):
            return np.array([_safe_num(v) for v in se["se_hessian"]], dtype=float)
        # nested layout: {"se": [...], "t_values": [...], "p_values": [...]}
        if isinstance(se.get("se"), (list, tuple)):
            return np.array([_safe_num(v) for v in se["se"]], dtype=float)
        # flat name->value dict
        names = getattr(parsed_params, "param_names", []) or []
        return np.array([_safe_num(se.get(n)) for n in names], dtype=float)
    if isinstance(se, (list, tuple)):
        return np.array([_safe_num(v) for v in se], dtype=float)
    return None


def _extract_t_array(results_data: Dict[str, Any], parsed_params: Any) -> Optional[np.ndarray]:
    """Return t-values aligned to parsed_params.param_names, or None.

    Mirrors _extract_se_array: supports the nested
    standard_errors={"t_values": [...]} layout in addition to a top-level
    ``t_values`` dict/list.
    """
    tv = results_data.get("t_values")
    if tv is None:
        se = results_data.get("standard_errors")
        if isinstance(se, dict) and isinstance(se.get("t_hessian"), (list, tuple)):
            return np.array([_safe_num(v) for v in se["t_hessian"]], dtype=float)
        if isinstance(se, dict) and isinstance(se.get("t_values"), (list, tuple)):
            return np.array([_safe_num(v) for v in se["t_values"]], dtype=float)
    if isinstance(tv, dict):
        names = getattr(parsed_params, "param_names", []) or []
        return np.array([_safe_num(tv.get(n)) for n in names], dtype=float)
    if isinstance(tv, (list, tuple)):
        return np.array([_safe_num(v) for v in tv], dtype=float)
    return None


def _cluster_se_param_map(
    cluster_se_data: Optional[Dict[str, Any]],
    parsed_params: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Map parameter name -> {se_robust, t_robust, p_robust} from cluster-SE JSON.

    Supports three known layouts:

    1. ``parameters``/``rows`` list of dicts (one row per parameter);
    2. ``parameters``/``rows`` dict keyed by parameter name;
    3. nested ``standard_errors.se_clustered`` parallel list from the joint
       Step-4 bridge;
    4. ``cluster_robust_se_artifacts.se_robust_vector`` parallel list to
       ``parsed_params.param_names`` (the layout used by
       ``cluster_robust_se.py``).
    """
    if not cluster_se_data:
        return {}
    out: Dict[str, Dict[str, Any]] = {}

    # Nested standard_errors layout emitted by the joint Step-4 bridge:
    # {"se_hessian": [...], "se_clustered": [...], "t_clustered": [...], ...}.
    # Map it to the same robust columns used by an external cluster-SE artifact.
    if parsed_params is not None and isinstance(cluster_se_data, dict):
        names = list(getattr(parsed_params, "param_names", []) or [])
        se_clustered_vec = (
            cluster_se_data.get("se_clustered")
            or cluster_se_data.get("se_robust")
            or cluster_se_data.get("robust_se")
        )
        if isinstance(se_clustered_vec, (list, tuple)) and names:
            t_clustered_vec = (
                cluster_se_data.get("t_clustered")
                or cluster_se_data.get("t_robust")
                or cluster_se_data.get("t_values")
                or []
            )
            p_clustered_vec = (
                cluster_se_data.get("p_clustered")
                or cluster_se_data.get("p_robust")
                or cluster_se_data.get("p_values")
                or []
            )
            for i, name in enumerate(names[:len(se_clustered_vec)]):
                se_r = _safe_num(se_clustered_vec[i])
                if se_r is None:
                    continue
                t_r = _safe_num(t_clustered_vec[i]) if i < len(t_clustered_vec) else None
                p_r = _safe_num(p_clustered_vec[i]) if i < len(p_clustered_vec) else None
                out[str(name)] = {
                    "se_robust": se_r,
                    "t_robust": t_r,
                    "p_robust": p_r,
                }
            if out:
                return out

    rows = cluster_se_data.get("parameters") or cluster_se_data.get("rows")
    if isinstance(rows, list):
        for r in rows:
            if not isinstance(r, dict):
                continue
            name = r.get("param") or r.get("parameter") or r.get("name")
            if not name:
                continue
            out[str(name)] = {
                "se_robust": r.get("se_robust") or r.get("robust_se"),
                "t_robust": r.get("t_robust") or r.get("t_ratio_robust"),
                "p_robust": r.get("p_robust") or r.get("p_value_robust"),
            }
    elif isinstance(rows, dict):
        for name, r in rows.items():
            if isinstance(r, dict):
                out[str(name)] = {
                    "se_robust": r.get("se_robust") or r.get("robust_se"),
                    "t_robust": r.get("t_robust") or r.get("t_ratio_robust"),
                    "p_robust": r.get("p_robust") or r.get("p_value_robust"),
                }

    # Parallel-vector layout (cluster_robust_se.py output)
    if not out and parsed_params is not None:
        names = list(getattr(parsed_params, "param_names", []) or [])
        arts = cluster_se_data.get("cluster_robust_se_artifacts") or {}
        if isinstance(arts, dict):
            se_robust_vec = arts.get("se_robust_vector") or []
            theta_vec = arts.get("converged_theta") or []
            if names and isinstance(se_robust_vec, list) and len(se_robust_vec) == len(names):
                for i, name in enumerate(names):
                    se_r = se_robust_vec[i] if i < len(se_robust_vec) else None
                    theta_i = theta_vec[i] if i < len(theta_vec) else None
                    t_r = None
                    try:
                        if se_r and theta_i is not None and float(se_r) > 0:
                            t_r = float(theta_i) / float(se_r)
                    except (TypeError, ValueError):
                        t_r = None
                    out[str(name)] = {
                        "se_robust": se_r,
                        "t_robust": t_r,
                        "p_robust": None,
                    }
    return out


# ==============================================================================
# Profile filtering
# ==============================================================================
