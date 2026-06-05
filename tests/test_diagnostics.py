"""Wave 2.6 tests for the model-generic diagnostics bundle (self-contained).

Covers: solver-family classification, the generic CONOPT RGmax parser, a synthetic
diagnostics bundle with finite core metrics, graceful degradation of missing
optional sections, and the light import. No FR/EUROMOD/Java/old-repo deps.
"""
import math
import subprocess
import sys
from types import SimpleNamespace

from dclaborsupply.diagnostics import (
    build_diagnostics_bundle,
    classify_solver_family,
    parse_conopt_rgmax_from_text,
)


# --- solver-family classification ---------------------------------------------
def test_classify_solver_family():
    assert classify_solver_family("CONOPT") == "conopt"
    assert classify_solver_family("GAMS/CONOPT4") == "conopt"
    assert classify_solver_family("L-BFGS-B") == "bfgs"
    assert classify_solver_family("scipy") == "bfgs"
    assert classify_solver_family("IPOPT") == "ipopt"
    assert classify_solver_family("KNITRO") == "knitro"
    assert classify_solver_family("trust-constr") == "trust-constr"
    assert classify_solver_family("some-weird-solver") == "other"
    assert classify_solver_family(None) == "unknown"
    assert classify_solver_family("") == "unknown"


# --- CONOPT RGmax parser ------------------------------------------------------
def test_parse_conopt_rgmax_terminal_value():
    log = (
        "Reading the model...\n"
        "   Iter Phase Ninf   Infeasibility    RGmax        NSB   Step InItr MX OK\n"
        "      1   0    2     1.5000E+03     5.4000E-02      3   1.0E0    0  T  T\n"
        "      2   0    0     1.9000E+04     5.4000E-08      3   1.0E0    0  T  T\n"
        "\n** Optimal solution found.\n"
    )
    rg = parse_conopt_rgmax_from_text(log)
    assert rg is not None
    assert rg == 5.4e-08  # terminal (last) RGmax across the block


def test_parse_conopt_rgmax_none_when_no_table():
    assert parse_conopt_rgmax_from_text("no conopt iteration table here") is None
    assert parse_conopt_rgmax_from_text("") is None


# --- synthetic diagnostics bundle ---------------------------------------------
def _payload():
    pp = SimpleNamespace(
        param_names=["a", "b", "c"],
        theta=[0.5, 1.0, 1e-6],
        bounds=[(None, None), (0.0, 2.0), (1e-6, 50.0)],  # c sits at its lower bound
    )
    n_obs, n_groups, k = 1000, 100, 3
    ll = -1234.5
    fit_stats = {
        "log_likelihood": ll,
        "n_observations": n_obs,
        "n_groups": n_groups,
        "n_parameters": k,
        "AIC": 2 * k - 2 * ll,
        "BIC": k * math.log(n_obs) - 2 * ll,
        "AIC_per_obs": (2 * k - 2 * ll) / n_obs,
    }
    results_data = {
        "specification": "synthetic",
        "wage_spec": "fw",
        "metadata": {"opt_method": "L-BFGS-B", "group": "test"},
        "summary": {"n_obs_total": n_obs, "n_groups_total": n_groups,
                    "total_walltime_seconds": 12.3},
        "results": {"g1": {"success": True, "n_iterations": 50, "gradient_norm": 1e-6}},
    }
    return pp, fit_stats, results_data


def test_build_bundle_finite_core_metrics():
    pp, fit_stats, results_data = _payload()
    bundle = build_diagnostics_bundle(
        profile="standard", results_data=results_data, parsed_params=pp,
        fit_stats=fit_stats, bound_diagnostics=[],
    )
    fit = bundle.likelihood_fit_core
    assert fit.available
    d = fit.data
    assert math.isfinite(d["log_likelihood"]) and d["log_likelihood"] == -1234.5
    assert d["n_observations"] == 1000 and d["n_groups"] == 100
    assert math.isfinite(d["AIC"]) and math.isfinite(d["BIC"])
    assert abs(d["n_alts_per_set"] - 10.0) < 1e-9

    b = bundle.bounds_diagnostics.data
    assert b["n_parameters"] == 3
    assert b["n_parameters_with_bounds"] == 2     # b and c
    assert b["n_at_lower_bound"] == 1             # c at its lower bound
    assert b["n_at_upper_bound"] == 0

    assert bundle.solver.available
    assert bundle.solver.data["solver_family"] == "bfgs"

    assert bundle.to_dict()["profile"] == "standard"   # serializable data structure


def test_missing_optional_sections_degrade_gracefully():
    pp, fit_stats, results_data = _payload()
    # no ll_null in fit_stats, no mu_results -> those sections must degrade, not error
    bundle = build_diagnostics_bundle(
        profile="standard", results_data=results_data, parsed_params=pp,
        fit_stats=fit_stats, bound_diagnostics=[],
        mu_results=None, hessian_diagnostics=None, cluster_se_data=None,
        prob_diagnostics=None, gradient_diag=None,
    )
    for section in (bundle.null_model_fit, bundle.economic_sanity):
        assert section.available is False
        assert isinstance(section.unavailable_reason, str) and section.unavailable_reason


# --- light import (fresh process) ---------------------------------------------
def test_diagnostics_import_is_light():
    code = (
        "import sys, dclaborsupply.diagnostics\n"
        "for m in ('jax', 'gamspy', 'jpype', 'java', 'scipy', 'numba', 'pandas'):\n"
        "    assert m not in sys.modules, m + ' imported at diagnostics import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
