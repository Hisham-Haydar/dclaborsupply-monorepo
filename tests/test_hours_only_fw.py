"""Hours-only fixed-wage (wage_spec='fw') dimension-dropping tests.

Verifies the JAX fixed-wage branch (no wage/log_wage/sigma access), NumPy<->JAX parity,
occupation omission (no loc4 needed), the truthful fit backend guard, and the CLI no
longer advertising NumPy estimation. Synthetic fixtures only; no external data.
"""
import dataclasses as dc
import textwrap

import numpy as np
import pytest

from dclaborsupply.spec.parser import EstimationSpec
from dclaborsupply.likelihood._numpy_primitives import (
    PrecomputedDataSingles, PrecomputedDataCouples,
)

_IV = "\n".join(f"  {n}: {v}" for n, v in [
    ("beta_l0_sm", 1.0), ("beta_l_age_sm", 0.1), ("beta_l_age2_sm", 0.0), ("theta_l_sm", -1.0),
    ("beta_l0_sf", 1.0), ("beta_l_age_sf", 0.1), ("beta_l_age2_sf", 0.0), ("beta_l_nkids_sf", 0.2), ("theta_l_sf", -1.0),
    ("theta_c_singles", -1.0),
    ("beta_l0_m", 1.0), ("beta_l_age_m", 0.1), ("beta_l_age2_m", 0.0), ("theta_l_m", -1.0),
    ("beta_l0_f", 1.0), ("beta_l_age_f", 0.1), ("beta_l_age2_f", 0.0), ("beta_l_nkids_f", 0.2), ("theta_l_f", -1.0),
    ("beta_E", 0.3), ("beta_h_pt1", -0.2), ("beta_h_pt2", 0.1), ("beta_h_ft", 0.4), ("beta_h_lh", -0.1)])

_YAML = textwrap.dedent(f'''
specification: {{name: hours_only_fw, description: "hours-only fixed-wage", wage_spec: "fw", model_family: "regular"}}
utility:
  functional_form: "box_cox"
  consumption: {{coefficient: beta_c, fixed_value: 1.0, box_cox_exponent: theta_c, singles_box_cox_exponent: theta_c_singles, couples_fixed_box_cox_exponent: 0.0, box_cox_bounds: [-8.0, 0.95]}}
  leisure:
    intercept: beta_l0
    box_cox_exponent: theta_l
    box_cox_bounds: [-8.0, 0.95]
    shifters:
      - {{variable: age_norm, coefficient: beta_l_age}}
      - {{variable: age_norm2, coefficient: beta_l_age2}}
      - {{variable: n_children, coefficient: beta_l_nkids, gender_specific: true}}
hours_opportunity:
  shifters:
      - {{variable: working, coefficient: beta_E}}
      - {{variable: working_pt1, coefficient: beta_h_pt1}}
      - {{variable: working_pt2, coefficient: beta_h_pt2}}
      - {{variable: working_ft, coefficient: beta_h_ft}}
      - {{variable: working_lh, coefficient: beta_h_lh}}
initial_values:
{_IV}
optimization:
  bounds: {{theta_l_sm: [-8.0,0.95], theta_l_sf: [-8.0,0.95], theta_l_m: [-8.0,0.95], theta_l_f: [-8.0,0.95], theta_c_singles: [-8.0,0.95]}}
''')


def _spec(tmp_path):
    p = tmp_path / "hours_only_fw.yaml"
    p.write_text(_YAML)
    return EstimationSpec.from_yaml(p)


def _singles(is_male, n_groups=2, n_alts=3):
    n = n_groups * n_alts
    hours = np.tile(np.array([0.0, 20.0, 40.0])[:n_alts], n_groups)
    band = lambda lo, hi: ((hours >= lo) & (hours <= hi)).astype(float)
    cons = np.full(n, 1.3); leis = np.linspace(1.0, 1.6, n)
    ac = np.zeros(n)
    for g in range(n_groups):
        ac[g * n_alts] = 1.0   # chosen-first
    vals = dict(
        consumption=cons, leisure=leis, log_c=np.log(cons), log_l=np.log(leis),
        age_norm=np.full(n, 0.3), age_norm2=np.full(n, 0.09), n_children=np.full(n, 1.0),
        educL=np.zeros(n), educM=np.zeros(n), educH=np.zeros(n),
        working=(hours > 0).astype(float), working_pt1=band(18.5, 21.5), working_pt2=band(29.5, 30.5),
        working_ft=band(37.5, 40.5), working_lh=np.zeros(n),
        gsur=np.zeros(n), female=(np.zeros(n) if is_male else np.ones(n)), in_couple=np.zeros(n),
        drgn1=np.zeros(n), reg2=np.zeros(n), reg3=np.zeros(n), reg4=np.zeros(n), reg5=np.zeros(n),
        reg6=np.zeros(n), reg7=np.zeros(n), reg8=np.zeros(n),
        drgur=np.zeros(n), drgmd=np.zeros(n), drgru=np.zeros(n),
        year_2015_indicator=np.zeros(n), year_2017_indicator=np.zeros(n),
        log_wage=None, pexp_years=None, pexp_years2=None,
        loc4=None, loc4_1=None, loc4_2=None, loc4_3=None, loc4_4=None,
        prior=np.full(n, 1.0), c_scale=1.3, l_scale=1.0,
        group_ids=np.arange(n_groups), group_starts=np.arange(n_groups) * n_alts,
        group_ends=(np.arange(n_groups) + 1) * n_alts, n_groups=n_groups, n_obs=n,
        actual_choice=ac, cluster_ids=np.arange(n_groups), is_male=is_male)
    return PrecomputedDataSingles(**{f.name: vals[f.name] for f in dc.fields(PrecomputedDataSingles)})


def _couples(n_groups=2, n_alts=4):
    n = n_groups * n_alts
    hm = np.tile(np.array([0.0, 40.0, 20.0, 40.0])[:n_alts], n_groups)
    hf = np.tile(np.array([0.0, 0.0, 20.0, 30.0])[:n_alts], n_groups)
    band = lambda h, lo, hi: ((h >= lo) & (h <= hi)).astype(float)
    cons = np.full(n, 2.0); lm = np.linspace(1.0, 1.6, n); lf = np.linspace(1.1, 1.5, n)
    ac = np.zeros(n)
    for g in range(n_groups):
        ac[g * n_alts] = 1.0
    z = np.zeros(n)
    vals = dict(
        consumption=cons, log_c=np.log(cons),
        leisure_male=lm, log_l_male=np.log(lm), leisure_female=lf, log_l_female=np.log(lf),
        age_norm_male=np.full(n, 0.2), age_norm2_male=np.full(n, 0.04),
        educL_male=z, educM_male=z, educH_male=z,
        age_norm_female=np.full(n, 0.3), age_norm2_female=np.full(n, 0.09), n_children=np.full(n, 1.0),
        educL_female=z, educM_female=z, educH_female=z,
        working_male=(hm > 0).astype(float), working_pt1_male=band(hm, 18.5, 21.5),
        working_pt2_male=band(hm, 29.5, 30.5), working_ft_male=band(hm, 37.5, 40.5), working_lh_male=z, gsur_male=z,
        working_female=(hf > 0).astype(float), working_pt1_female=band(hf, 18.5, 21.5),
        working_pt2_female=band(hf, 29.5, 30.5), working_ft_female=band(hf, 37.5, 40.5), working_lh_female=z, gsur_female=z,
        female_male=z, female_female=np.ones(n), in_couple_male=np.ones(n), in_couple_female=np.ones(n),
        drgn1_male=z, drgn1_female=z, reg2=z, reg3=z, reg4=z, reg5=z, reg6=z, reg7=z, reg8=z,
        drgur=z, drgmd=z, drgru=z, year_2015_indicator=z, year_2017_indicator=z,
        log_wage_male=None, pexp_years_male=None, pexp_years2_male=None,
        loc4_male=None, loc4_1_male=None, loc4_2_male=None, loc4_3_male=None, loc4_4_male=None,
        log_wage_female=None, pexp_years_female=None, pexp_years2_female=None,
        loc4_female=None, loc4_1_female=None, loc4_2_female=None, loc4_3_female=None, loc4_4_female=None,
        prior=np.full(n, 1.0), c_scale=2.0, l_scale=1.0,
        group_ids=np.arange(n_groups), group_starts=np.arange(n_groups) * n_alts,
        group_ends=(np.arange(n_groups) + 1) * n_alts, n_groups=n_groups, n_obs=n,
        actual_choice=ac, cluster_ids=np.arange(n_groups))
    return PrecomputedDataCouples(**{f.name: vals[f.name] for f in dc.fields(PrecomputedDataCouples)})


# === parser ===============================================================
def test_parser_hours_only_fw_no_wage_occ_market_params(tmp_path):
    spec = _spec(tmp_path)
    names = list(spec.all_param_names)
    assert len(names) == 24
    assert not any(n.startswith("beta_w") for n in names)
    assert "sigma" not in names
    assert not any(n.startswith("beta_occ") for n in names)
    assert not any(n.startswith("beta_E_") for n in names)   # market beta_E_<...> (beta_E itself is hours)
    assert spec.wage_mean_shifters == [] and spec.wage_variance_param is None
    assert spec.market_opportunity_shifters == []


# === JAX runs without wage/log_wage/sigma + no wage binding ===============
def test_jax_singles_and_couples_run_fixed_wage(tmp_path):
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dclaborsupply.likelihood.engine_jax import build_jax_singles_ll, build_jax_couples_ll
    spec = _spec(tmp_path)
    theta = jnp.asarray(np.asarray(spec.get_initial_vector(), float))
    sm, sf, cou = _singles(True), _singles(False), _couples()
    assert sm.log_wage is None and cou.log_wage_male is None   # no wage data present
    fm, _ = build_jax_singles_ll(sm, spec, is_male=True)
    ff, _ = build_jax_singles_ll(sf, spec, is_male=False)
    fc, _ = build_jax_couples_ll(cou, spec)
    for f in (fm, ff, fc):
        assert np.isfinite(float(f(theta)))


def test_jax_fw_objective_has_no_wage_parameter_binding(tmp_path):
    pytest.importorskip("jax")
    # the JAX objective binds only the 24 non-wage params (no beta_w*/sigma to bind).
    from dclaborsupply.gates.param_binding import check_param_binding
    from dclaborsupply.models import _build_objective
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    spec = _spec(tmp_path)
    obj = _build_objective(spec, _singles(True), _singles(False), _couples(), use_actual_choice=False)
    res = check_param_binding(spec, lambda th: float(obj(jnp.asarray(th, dtype=jnp.float64))))
    assert res["n_free"] == 24 and not res["duplicates"] and res["index_round_trip"]
    assert not any(n.startswith("beta_w") or n == "sigma" for n in res["bound"] + res["not_bound"])
    assert res["ll0"] is not None and np.isfinite(res["ll0"])


# === NumPy <-> JAX parity =================================================
def test_numpy_jax_parity_hours_only(tmp_path):
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    from dclaborsupply.likelihood.index import compute_index
    spec = _spec(tmp_path)
    data = (_singles(True), _singles(False), _couples())
    theta = np.asarray(spec.get_initial_vector(), float)
    nll_np = compute_index(spec, data, theta, ruro=True, backend="numpy")
    nll_jax = compute_index(spec, data, theta, ruro=True, backend="jax")
    assert np.isfinite(nll_np) and np.isfinite(nll_jax)
    assert abs(nll_np - nll_jax) <= 1e-9


# === occupation omission needs no loc4 ====================================
def test_occupation_omission_needs_no_loc4(tmp_path):
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from dclaborsupply.likelihood.engine_jax import build_jax_singles_ll
    spec = _spec(tmp_path)
    sm = _singles(True)
    assert sm.loc4 is None and sm.loc4_2 is None   # no occupation arrays
    f, _ = build_jax_singles_ll(sm, spec, is_male=True)
    assert np.isfinite(float(f(jnp.asarray(np.asarray(spec.get_initial_vector(), float)))))


# === truthful fit backend =================================================
def test_fit_rejects_numpy_backend(tmp_path):
    from dclaborsupply.models import RUMModel, RUROModel
    spec = _spec(tmp_path)
    data = (_singles(True), _singles(False), _couples())
    for cls in (RUMModel, RUROModel):
        with pytest.raises(NotImplementedError, match="backend='jax'"):
            cls.from_spec(spec).fit(data, backend="numpy", compute_se=False)


def test_cli_estimate_backend_excludes_numpy():
    from dclaborsupply.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):   # argparse rejects invalid choice
        parser.parse_args(["estimate", "--config", "x", "--out", "y", "--backend", "numpy"])
    ns = parser.parse_args(["estimate", "--config", "x", "--out", "y", "--backend", "jax"])
    assert ns.backend == "jax"
