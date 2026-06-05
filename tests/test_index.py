"""Wave 1.4 tests for the compute_index dispatcher (self-contained; no jax, no MNL data).

The certified-data gates (ruro=True == 238504.636097; NumPy==JAX) run as a
validation script with MNL read-only helpers, not as committed tests (they require
the research-repo parquets). Here we cover the self-contained contract: data
validation, fixed_params folding, ruro=False MNL nulling with per-term zero
assertions, param-binding, and the jax-free light import.
"""
import subprocess
import sys

import numpy as np
import pytest

from dclaborsupply import EstimationSpec
from dclaborsupply.likelihood import index as idx
from dclaborsupply.likelihood._numpy_primitives import PrecomputedDataSingles
from dclaborsupply.likelihood.engine_numpy import compute_likelihood_singles


# --- a tiny self-contained spec (Box-Cox, no shifters/opportunity) ------------
_MIN_YAML = (
    "specification:\n  name: synth\n  wage_spec: fw\n"
    "utility:\n  functional_form: box_cox\n"
    "  consumption:\n    coefficient: beta_c\n    box_cox_exponent: theta_c\n"
    "  leisure:\n    intercept: beta_l0\n    box_cox_exponent: theta_l\n"
    "initial_values:\n"
    "  beta_l0_sm: 1.0\n  beta_c_sm: 1.0\n  theta_l_sm: 0.5\n  theta_c_sm: 0.5\n"
    "  beta_l0_sf: 1.0\n  beta_c_sf: 1.0\n  theta_l_sf: 0.5\n  theta_c_sf: 0.5\n"
    "  beta_l0_m: 1.0\n  theta_l_m: 0.5\n  beta_l0_f: 1.0\n  theta_l_f: 0.5\n"
    "  beta_c: 1.0\n  theta_c: 0.5\n"
)


def _spec(tmp_path, extra=""):
    p = tmp_path / "synth.yaml"
    p.write_text(_MIN_YAML + extra, encoding="utf-8")
    return EstimationSpec.from_yaml(p)


def _synth_singles(n_groups=3, n_alts=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * n_alts
    z = lambda: np.zeros(n)
    consumption = 1.0 + rng.random(n)
    leisure = 1.0 + rng.random(n)
    starts = np.arange(0, n, n_alts)
    return PrecomputedDataSingles(
        consumption=consumption, leisure=leisure,
        log_c=np.log(consumption), log_l=np.log(leisure),
        age_norm=z(), age_norm2=z(), n_children=z(), educL=z(), educM=z(), educH=z(),
        working=z(), working_pt1=z(), working_pt2=z(), working_ft=z(), working_lh=z(), gsur=z(),
        female=z(), in_couple=z(), drgn1=z(),
        reg2=z(), reg3=z(), reg4=z(), reg5=z(), reg6=z(), reg7=z(), reg8=z(),
        drgur=z(), drgmd=z(), drgru=z(),
        year_2015_indicator=z(), year_2017_indicator=z(),
        log_wage=None, pexp_years=None, pexp_years2=None,
        loc4=None, loc4_1=None, loc4_2=None, loc4_3=None, loc4_4=None,
        prior=1.0 + rng.random(n), c_scale=1.0, l_scale=1.0,
        group_ids=np.arange(n_groups), group_starts=starts, group_ends=starts + n_alts,
        n_groups=n_groups, n_obs=n,
        actual_choice=z(), cluster_ids=np.arange(n_groups), is_male=True,
    )


def _theta(spec):
    return spec.get_initial_vector()


# --- data contract ------------------------------------------------------------
def test_data_contract_validation(tmp_path):
    spec = _spec(tmp_path)
    th = _theta(spec)
    with pytest.raises(TypeError):
        idx.compute_index(spec, object(), th, ruro=False)
    with pytest.raises(ValueError):
        idx.compute_index(spec, (1, 2), th, ruro=False)          # wrong length
    with pytest.raises(ValueError):
        idx.compute_index(spec, {"singles_male": None}, th, ruro=False)  # missing keys


# --- fixed_params boundary fold ----------------------------------------------
def test_fold_fixed_params_appends_pinned(tmp_path):
    spec = _spec(tmp_path, extra="fixed_params:\n  theta_l_m: -0.8\n")
    assert "theta_l_m" not in spec.all_param_names      # pinned -> out of free vector
    assert spec.fixed_params == {"theta_l_m": -0.8}
    th = _theta(spec)
    spec2, th2 = idx._fold_fixed_params(spec, th)
    assert "theta_l_m" in spec2.all_param_names         # folded back in
    assert spec2.fixed_params == {}
    assert th2[spec2.all_param_names.index("theta_l_m")] == -0.8
    assert len(th2) == len(th) + 1
    # caller's spec untouched
    assert "theta_l_m" not in spec.all_param_names and spec.fixed_params == {"theta_l_m": -0.8}


# --- ruro=False: standard MNL, opportunity + correction nulled to zero --------
def test_ruro_false_nulls_all_opportunity_terms(tmp_path):
    spec = _spec(tmp_path)
    data = _synth_singles()
    th = _theta(spec)

    # per-term zero verification via the NumPy engine's components on the RUM view
    spec_rum, (d_sm, _, _) = idx.build_rum_view(spec, (data, None, None))
    comp = compute_likelihood_singles(th, d_sm, spec_rum, return_components=True)
    assert np.all(comp["log_h"] == 0.0)
    assert np.all(comp["log_w"] == 0.0)
    assert np.all(comp["log_market"] == 0.0)
    assert np.allclose(comp["V"], comp["u"], atol=0, rtol=0)   # V == u  => log_prior term == 0
    assert np.all(d_sm.prior == 1.0)                            # IS correction nulled
    # caller's data untouched
    assert not np.all(data.prior == 1.0)

    # compute_index ruro=False returns a finite negLL (= -ll, ll<0)
    negll = idx.compute_index(spec, (data, None, None), th, ruro=False, backend="numpy")
    assert np.isfinite(negll)
    assert negll > 0.0 and (-negll) < 0.0                       # negLL = -ll, ll = sum(logP) < 0


def test_ruro_false_does_not_mutate_caller_spec(tmp_path):
    spec = _spec(tmp_path)
    data = _synth_singles()
    before = (spec.wage_spec, list(spec.hours_shifters), list(spec.market_opportunity_shifters))
    idx.compute_index(spec, (data, None, None), _theta(spec), ruro=False, backend="numpy")
    after = (spec.wage_spec, list(spec.hours_shifters), list(spec.market_opportunity_shifters))
    assert before == after


# --- param-binding: every free param resolves, pinned disjoint, no drops ------
def test_param_binding_no_silent_drops(tmp_path):
    spec = _spec(tmp_path, extra="fixed_params:\n  theta_l_m: -0.8\n")
    names = list(spec.all_param_names)
    fixed = dict(spec.fixed_params)
    assert len(names) == len(set(names))                         # no duplicates
    assert set(fixed).isdisjoint(names)                          # pinned not also free
    for i, n in enumerate(names):
        assert spec.get_param_index(n) == i                      # round-trips
    th = _theta(spec)
    assert len(th) == len(names)                                 # theta covers all free
    # folded vector covers free + pinned with nothing dropped
    _, th2 = idx._fold_fixed_params(spec, th)
    assert len(th2) == len(names) + len(fixed)


# --- light import: importing the dispatcher pulls in no jax (fresh process) ---
def test_index_import_is_jax_free():
    code = (
        "import sys, dclaborsupply.likelihood.index\n"
        "assert 'jax' not in sys.modules, 'jax imported!'\n"
        "assert 'gamspy' not in sys.modules, 'gamspy imported!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
