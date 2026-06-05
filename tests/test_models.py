"""Wave 3.3 tests for the thin front-end models (RUMModel/RUROModel/Result).

The estimator path is exercised on a small self-contained synthetic singles dataset
(no df->PrecomputedData conversion): a singles-only joint with the couples params
(and sigma) pinned via fixed_params, so the 8 free singles params are identified by
sm+sf. The SE/JSON/partition behaviour is tested without the engine.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from dclaborsupply.models import (
    RUMModel, RUROModel, Result, classify_param_block, _partition_blocks,
)
from dclaborsupply.spec.parser import EstimationSpec

_SPEC_YAML = (
    "specification:\n  name: synth_joint\n  wage_spec: \"vw\"\n"
    "utility:\n  functional_form: box_cox\n"
    "  consumption:\n    coefficient: beta_c\n    box_cox_exponent: theta_c\n"
    "  leisure:\n    intercept: beta_l0\n    box_cox_exponent: theta_l\n"
    "wage_opportunity:\n  specification: \"log_normal\"\n  variance:\n    parameter: \"sigma\"\n"
    "initial_values:\n"
    "  beta_l0_sm: 0.8\n  beta_c_sm: 0.8\n  theta_l_sm: 0.4\n  theta_c_sm: 0.4\n"
    "  beta_l0_sf: 0.8\n  beta_c_sf: 0.8\n  theta_l_sf: 0.4\n  theta_c_sf: 0.4\n"
    "  beta_l0_m: 1.0\n  theta_l_m: 0.5\n  beta_l0_f: 1.0\n  theta_l_f: 0.5\n"
    "  beta_c: 1.0\n  theta_c: 0.5\n  sigma: 0.5\n"
    "fixed_params:\n  beta_l0_m: 1.0\n  theta_l_m: 0.5\n  beta_l0_f: 1.0\n  theta_l_f: 0.5\n"
    "  beta_c: 1.0\n  theta_c: 0.5\n  sigma: 0.5\n"
)


def _spec(tmp_path):
    p = tmp_path / "synth.yaml"
    p.write_text(_SPEC_YAML, encoding="utf-8")
    return EstimationSpec.from_yaml(p)


def _synth_singles(seed, is_male, ng=800, na=6):
    from dclaborsupply.likelihood._numpy_primitives import PrecomputedDataSingles
    rng = np.random.default_rng(seed)
    n = ng * na
    z = lambda: np.zeros(n)  # noqa: E731
    c = 0.5 + 4.5 * rng.random(n)
    l = 0.5 + 4.5 * rng.random(n)
    st = np.arange(0, n, na)
    return PrecomputedDataSingles(
        consumption=c, leisure=l, log_c=np.log(c), log_l=np.log(l),
        age_norm=z(), age_norm2=z(), n_children=z(), educL=z(), educM=z(), educH=z(),
        working=z(), working_pt1=z(), working_pt2=z(), working_ft=z(), working_lh=z(), gsur=z(),
        female=z(), in_couple=z(), drgn1=z(),
        reg2=z(), reg3=z(), reg4=z(), reg5=z(), reg6=z(), reg7=z(), reg8=z(),
        drgur=z(), drgmd=z(), drgru=z(),
        year_2015_indicator=z(), year_2017_indicator=z(),
        log_wage=1.0 + rng.random(n), pexp_years=None, pexp_years2=None,
        loc4=None, loc4_1=None, loc4_2=None, loc4_3=None, loc4_4=None,
        prior=1.0 + rng.random(n), c_scale=1.0, l_scale=1.0,
        group_ids=np.arange(ng), group_starts=st, group_ends=st + na, n_groups=ng, n_obs=n,
        actual_choice=z(), cluster_ids=np.arange(ng), is_male=is_male,
    )


# --- (c) block partition ------------------------------------------------------
def test_block_partition_assigns_each_param_once():
    names = ["beta_l0_sm", "theta_c_singles", "beta_c", "beta_l_age_f",  # preference
             "beta_h_pt1", "beta_h_ft",                                   # hours
             "beta_w0", "beta_w_pexp", "sigma",                           # wage
             "beta_E", "beta_E_gsur", "beta_E_drgn2", "beta_E_y2017",     # market
             "beta_occ_2_sm", "beta_occ_3_cf"]                            # occupation
    blocks = _partition_blocks(names, range(len(names)))
    flat = [n for b in blocks for n in blocks[b]]
    assert sorted(flat) == sorted(names)          # every param assigned exactly once
    assert len(flat) == len(set(flat))            # no duplicates
    assert set(blocks) == {"preference", "hours", "wage", "market", "occupation"}
    assert classify_param_block("beta_occ_2_sm") == "occupation"
    assert classify_param_block("sigma") == "wage"
    assert classify_param_block("beta_E_drgn2") == "market"
    assert classify_param_block("theta_l_sm") == "preference"


# --- (a) RUM synthetic fit ----------------------------------------------------
def test_rum_fit_synthetic(tmp_path):
    pytest.importorskip("jax")
    spec = _spec(tmp_path)
    sm, sf = _synth_singles(1, True), _synth_singles(2, False)
    r = RUMModel.from_spec(spec).fit((sm, sf, None), compute_se=True)
    assert np.isfinite(r.convergence["neg_ll"])
    assert r.convergence["success"] or isinstance(r.convergence["message"], str)
    assert len(r.params) == 8 and len(r.theta) == 8
    # blocks partition the fitted free params exactly once
    flat = [n for b in r.blocks for n in r.blocks[b]]
    assert sorted(flat) == sorted(r.param_names)
    # compute_se cached an exact-JAX Hessian SE
    assert r.se_hessian is not None and len(r.se("hessian")) == 8
    assert "hessian_min_eig" in r.diagnostics


# --- (b) RURO synthetic recovery (Wave-2.5 portable path) ---------------------
def test_ruro_recovery_synthetic(tmp_path):
    pytest.importorskip("jax")
    spec = _spec(tmp_path)
    sm, sf = _synth_singles(1, True), _synth_singles(2, False)
    r = RUROModel.from_spec(spec).recover_synthetic((sm, sf, None), seed=3, band=1.0, perturb=0.05)
    assert r.diagnostics["hessian_pd"] is True
    assert r.diagnostics["hessian_min_eig"] > 0          # PD identification gate
    assert r.diagnostics["within_band"] is True          # all 8 within band 1.0
    assert r.diagnostics["max_dev"] < 1.0


# --- (d) SE access: hessian cached/compute, cluster cached/compute/raise ------
def test_se_hessian_cached_and_missing():
    r = Result(param_names=["a", "b"], se_hessian=[0.1, 0.2])
    assert r.se("hessian") == {"a": 0.1, "b": 0.2}      # cached
    empty = Result(param_names=["a", "b"])
    with pytest.raises(ValueError):                      # no cache, no objective
        empty.se("hessian")


def test_se_cluster_cached_compute_and_raise():
    names = ["a", "b", "c"]
    # cached
    assert Result(param_names=names, se_cluster=[1.0, 2.0, 3.0]).se("cluster") == {
        "a": 1.0, "b": 2.0, "c": 3.0}
    # absent -> documented ValueError
    with pytest.raises(ValueError, match="scores_all"):
        Result(param_names=names).se("cluster")
    # compute path (pure numpy; no jax): synthetic PD hessian + per-cluster scores
    r = Result(param_names=names)
    H = np.array([[2.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 3.0]])
    scores = np.random.default_rng(0).standard_normal((40, 3))
    r.attach_cluster_inputs(scores, np.arange(40), hessian=H)
    out = r.se("cluster")
    assert set(out) == set(names) and all(np.isfinite(v) for v in out.values())


# --- (e) JSON round-trip ------------------------------------------------------
def test_json_roundtrip():
    r = Result(
        params={"a": 1.0, "b": -2.0}, theta=[1.0, -2.0], param_names=["a", "b"],
        blocks={"preference": {"a": 1.0}, "wage": {"b": -2.0}},
        convergence={"success": True, "status": 0, "neg_ll": 12.5},
        metadata={"model": "RUM", "n_free": 2},
        diagnostics={"hessian_min_eig": 0.5},
        se_hessian=[0.1, 0.2], se_cluster=None,
    )
    s = r.to_json()
    assert isinstance(s, str)
    json.loads(s)  # valid JSON
    r2 = Result.from_json(s)
    assert r2.params == r.params and r2.blocks == r.blocks
    assert r2.convergence == r.convergence and r2.se_hessian == r.se_hessian
    assert r2.se("hessian") == {"a": 0.1, "b": 0.2}   # cached SE survives round-trip


# --- (f) light import ---------------------------------------------------------
def test_models_import_is_light():
    code = (
        "import sys, dclaborsupply\n"
        "import dclaborsupply.models\n"
        "for m in ('jax', 'scipy', 'gamspy', 'numba'):\n"
        "    assert m not in sys.modules, m + ' imported at dclaborsupply import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
