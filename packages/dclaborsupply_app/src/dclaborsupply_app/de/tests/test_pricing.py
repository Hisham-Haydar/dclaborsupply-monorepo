"""DE earnings-mutation policy unit tests (validated DE pricing rule)."""
import numpy as np

from dclaborsupply_app.de import de_earnings_policy

WPM = 52.0 / 12.0


def test_de_policy_mutates_only_lhw_yem_yemse():
    member = {"yse": 0.0, "bun": 120.0, "bsa": 50.0, "yivwg": 20.0, "lhw": 40}
    out = de_earnings_policy(member, hours=30.0, wage=25.0)
    assert set(out) == {"lhw", "yem", "yemse"}          # nothing else mutated (bun/bsa/yivwg untouched)
    assert out["lhw"] == 30.0
    assert np.isclose(out["yem"], 25.0 * 30.0 * WPM)
    assert np.isclose(out["yemse"], out["yem"] + 0.0)


def test_de_policy_yemse_includes_yse():
    member = {"yse": 300.0}
    out = de_earnings_policy(member, hours=20.0, wage=15.0)
    assert np.isclose(out["yemse"], out["yem"] + 300.0)


def test_de_policy_non_employment_zeroes_earnings():
    out = de_earnings_policy({"yse": 0.0}, hours=0.0, wage=0.0)
    assert out["lhw"] == 0.0 and out["yem"] == 0.0 and out["yemse"] == 0.0


def test_de_policy_missing_yse_defaults_zero():
    out = de_earnings_policy({}, hours=10.0, wage=12.0)  # no 'yse' key
    assert np.isclose(out["yemse"], out["yem"])
