"""Parser subgate for the minimal DE 2017 estimation spec (config-only).

Verifies the UNCHANGED core parser accepts the DE spec authored purely by
configuration, with the expected 34 free parameters and the FR market/region/year
coefficients removed. No scratch/data dependency: the committed config is located
relative to this test file.
"""
from pathlib import Path

import pytest

from dclaborsupply.spec.parser import EstimationSpec

# packages/dclaborsupply_app/configs/de_2017_minimal.yaml
DE_SPEC = Path(__file__).parents[4] / "configs" / "de_2017_minimal.yaml"

# Expected ordered free-parameter list (34): preference 19 + hours 5 + occ 6 + wage 4.
EXPECTED_NAMES = [
    # preference (19)
    "beta_l0_sm", "beta_l_age_sm", "beta_l_age2_sm", "theta_l_sm",
    "beta_l0_sf", "beta_l_age_sf", "beta_l_age2_sf", "beta_l_nkids_sf", "theta_l_sf",
    "theta_c_singles",
    "beta_l0_m", "beta_l_age_m", "beta_l_age2_m", "theta_l_m",
    "beta_l0_f", "beta_l_age_f", "beta_l_age2_f", "beta_l_nkids_f", "theta_l_f",
    # hours (5)
    "beta_E", "beta_h_pt1", "beta_h_pt2", "beta_h_ft", "beta_h_lh",
    # occupation (6) — appended via occupation_opportunity -> market_opportunity_shifters
    "beta_occ_2_m", "beta_occ_3_m", "beta_occ_4_m",
    "beta_occ_2_f", "beta_occ_3_f", "beta_occ_4_f",
    # wage (4)
    "beta_w0", "beta_w_educL", "beta_w_educH", "sigma",
]

# The 12 FR market/region/year coefficients that must NOT appear in the DE spec.
REMOVED_FR_COEFFS = [
    "beta_E_gsur",
    "beta_E_drgn2", "beta_E_drgn3", "beta_E_drgn4", "beta_E_drgn5",
    "beta_E_drgn6", "beta_E_drgn7", "beta_E_drgn8",
    "beta_E_drgur", "beta_E_drgmd",
    "beta_E_y2015", "beta_E_y2017",
]


@pytest.fixture(scope="module")
def de_spec() -> EstimationSpec:
    assert DE_SPEC.exists(), f"DE spec not found at {DE_SPEC}"
    return EstimationSpec.from_yaml(DE_SPEC)  # GATE (a): parses with unchanged core


def test_de_spec_parses_to_34_ordered_params(de_spec):
    # GATE (b): exact ordered free-parameter list of length 34.
    assert de_spec.all_param_names == EXPECTED_NAMES
    assert len(de_spec.all_param_names) == 34
    # block decomposition 19 + 5 + 6 + 4
    names = set(de_spec.all_param_names)
    pref = {n for n in names if n.startswith(
        ("beta_l0", "beta_l_age", "beta_l_nkids", "theta_l", "theta_c"))}
    hours = {"beta_E", "beta_h_pt1", "beta_h_pt2", "beta_h_ft", "beta_h_lh"}
    occ = {n for n in names if n.startswith("beta_occ_")}
    wage = {"beta_w0", "beta_w_educL", "beta_w_educH", "sigma"}
    assert len(pref) == 19 and len(occ) == 6
    assert hours <= names and wage <= names
    # every free parameter has an initial value (parser would have raised otherwise)
    assert all(n in de_spec.initial_values for n in de_spec.all_param_names)


def test_removed_fr_market_region_year_coeffs_absent(de_spec):
    # GATE (c): all 12 FR market/region/year coefficients are gone...
    present = [c for c in REMOVED_FR_COEFFS if c in de_spec.all_param_names]
    assert present == [], f"unexpected FR market/region/year coeffs present: {present}"
    # ...and so are the dropped wage experience terms.
    assert "beta_w_pexp" not in de_spec.all_param_names
    assert "beta_w_pexp2" not in de_spec.all_param_names
    # none of them lingers in bounds/initials either
    for c in REMOVED_FR_COEFFS + ["beta_w_pexp", "beta_w_pexp2"]:
        assert c not in de_spec.bounds
        assert c not in de_spec.initial_values


def test_theta_l_m_free_and_structural_fixes(de_spec):
    # GATE (d): theta_l_m FREE (FR pin removed); beta_c / couples theta_c / beta_ll fixed-or-absent.
    assert "theta_l_m" in de_spec.all_param_names
    assert "theta_l_m" not in de_spec.fixed_params
    assert de_spec.fixed_params == {}
    # beta_c is the fixed scale numéraire (not estimated)
    assert "beta_c" not in de_spec.all_param_names
    assert de_spec.utility_consumption_coef_fixed == 1.0
    # couples theta_c fixed at 0.0 (not estimated); singles share theta_c_singles
    assert "theta_c" not in de_spec.all_param_names
    assert de_spec.utility_consumption_theta_couples_fixed == 0.0
    assert "theta_c_singles" in de_spec.all_param_names
    # beta_ll omitted (no couples leisure_interaction)
    assert "beta_ll" not in de_spec.all_param_names
    assert de_spec.couples_interaction_coef is None
