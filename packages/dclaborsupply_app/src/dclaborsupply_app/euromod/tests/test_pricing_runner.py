"""Tests for the connector-injected EUROMOD pricing runner.

(a) Fake-connector gates: policy swap, complete-household replication, ID remap /
    reverse-map, no cross-household refs, stable-key survival, member rows not collapsed,
    warning/error capture, raw negative ils_dispy retained, separate tax-unit sum, no floor.
(b) Lazy import: importing the package pulls no euromod/pythonnet/clr; the real connector
    raises a documented ImportError when euromod is absent.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from dclaborsupply_app.euromod import (
    EuromodPricingRunner, PricingConnectorResult, EuromodConnector,
)
from dclaborsupply_app.de import de_earnings_policy

WPM = 52.0 / 12.0


# --- fakes -----------------------------------------------------------------
class FakeConnector:
    """Deterministic stand-in for EUROMOD: ils_dispy = 0.65*yem + bun + bsa - 200
    (negative for zero-income members), passes rows through 1:1, emits messages."""
    def __init__(self, warnings=None, errors=None):
        self._w = list(warnings or []); self._e = list(errors or [])
        self.seen_cols = None

    def run(self, data, *, country, system, dataset):
        self.seen_cols = list(data.columns)
        out = data.copy()
        yem = pd.to_numeric(out["yem"], errors="coerce").fillna(0.0)
        bun = pd.to_numeric(out.get("bun", 0.0), errors="coerce").fillna(0.0)
        bsa = pd.to_numeric(out.get("bsa", 0.0), errors="coerce").fillna(0.0)
        out["ils_dispy"] = 0.65 * yem + bun + bsa - 200.0
        return PricingConnectorResult(output=out, warnings=list(self._w), errors=list(self._e))


def fake_policy(member, *, hours, wage, weeks_per_month):
    # deliberately different from DE: no weeks_per_month, ignores yse
    return {"lhw": hours, "yem": wage * hours, "yemse": wage * hours}


def _fixture():
    """Baseline (complete households incl. children) + alternatives table."""
    base = pd.DataFrame([
        # single hh 1 + child
        dict(idhh=1, idperson=101, idpartner=0, idfather=0, idmother=0, idorighh=1, idorigperson=101, dgn=0, yem=3000.0, yse=0.0, bun=0.0, bsa=0.0, lhw=40, yivwg=20.0, yemse=3000.0),
        dict(idhh=1, idperson=102, idpartner=0, idfather=0, idmother=101, idorighh=1, idorigperson=102, dgn=1, yem=0.0, yse=0.0, bun=0.0, bsa=0.0, lhw=0, yivwg=0.0, yemse=0.0),
        # single hh 2 (no child)
        dict(idhh=2, idperson=201, idpartner=0, idfather=0, idmother=0, idorighh=2, idorigperson=201, dgn=1, yem=2500.0, yse=200.0, bun=0.0, bsa=0.0, lhw=38, yivwg=25.0, yemse=2700.0),
        # couple hh 3 + child
        dict(idhh=3, idperson=301, idpartner=302, idfather=0, idmother=0, idorighh=3, idorigperson=301, dgn=1, yem=4000.0, yse=0.0, bun=0.0, bsa=0.0, lhw=50, yivwg=19.0, yemse=4000.0),
        dict(idhh=3, idperson=302, idpartner=301, idfather=0, idmother=0, idorighh=3, idorigperson=302, dgn=0, yem=1000.0, yse=0.0, bun=0.0, bsa=0.0, lhw=15, yivwg=15.0, yemse=1000.0),
        dict(idhh=3, idperson=303, idpartner=0, idfather=301, idmother=302, idorighh=3, idorigperson=303, dgn=1, yem=0.0, yse=0.0, bun=0.0, bsa=0.0, lhw=0, yivwg=0.0, yemse=0.0),
        # couple hh 4
        dict(idhh=4, idperson=401, idpartner=402, idfather=0, idmother=0, idorighh=4, idorigperson=401, dgn=1, yem=3500.0, yse=0.0, bun=0.0, bsa=0.0, lhw=45, yivwg=18.0, yemse=3500.0),
        dict(idhh=4, idperson=402, idpartner=401, idfather=0, idmother=0, idorighh=4, idorigperson=402, dgn=0, yem=1200.0, yse=0.0, bun=0.0, bsa=0.0, lhw=20, yivwg=12.0, yemse=1200.0),
    ])
    # expected-decider baseline flag (deciders 1; children 0)
    base["ruro_decider"] = base["idperson"].isin({101, 201, 301, 302, 401, 402}).astype(int)
    alts = pd.DataFrame([
        dict(source_idhh=1, alt="chosen", decider_idperson=101, hours=40, wage=20.0),
        dict(source_idhh=1, alt="sim1", decider_idperson=101, hours=20, wage=30.0),
        dict(source_idhh=2, alt="chosen", decider_idperson=201, hours=38, wage=25.0),
        dict(source_idhh=2, alt="sim1", decider_idperson=201, hours=0, wage=0.0),
        dict(source_idhh=3, alt="chosen", decider_idperson=301, hours=50, wage=19.0),
        dict(source_idhh=3, alt="chosen", decider_idperson=302, hours=15, wage=15.0),
        dict(source_idhh=3, alt="sim1", decider_idperson=301, hours=35, wage=40.0),
        dict(source_idhh=3, alt="sim1", decider_idperson=302, hours=30, wage=22.0),
        dict(source_idhh=4, alt="chosen", decider_idperson=401, hours=45, wage=18.0),
        dict(source_idhh=4, alt="chosen", decider_idperson=402, hours=20, wage=12.0),
        dict(source_idhh=4, alt="sim1", decider_idperson=401, hours=60, wage=25.0),
        dict(source_idhh=4, alt="sim1", decider_idperson=402, hours=10, wage=14.0),
    ])
    return base, alts


def _run(policy=de_earnings_policy, connector=None, **kw):
    base, alts = _fixture()
    conn = connector or FakeConnector()
    runner = EuromodPricingRunner(conn, policy)
    res = runner.price(alts, base, country="DE", system="DE_2016", dataset="DE_2017_a2",
                       alt_key_cols=["alt"], data_year=2017, **kw)
    return res, conn, base, alts


# === (a) fake-connector gates ============================================
def test_complete_household_replication_and_not_collapsed():
    res, conn, base, alts = _run()
    out = res.output
    # 8 alternatives (groups of source_idhh+alt); members not collapsed
    n_alts = alts[["source_idhh", "alt"]].drop_duplicates().shape[0]
    assert out["idhh"].nunique() == n_alts == 8
    # member rows preserved: each synthetic hh has its source household's member count
    sizes = base.groupby("idhh").size().to_dict()
    for new_hh, g in out.groupby("idhh"):
        src = int(g["source_idhh"].iloc[0])
        assert len(g) == sizes[src]
    # 16 member-rows total: hh1 2 alts x2 + hh2 2 alts x1 + hh3 2 alts x3 + hh4 2 alts x2
    assert len(out) == 16
    # child rows survive (hh1 child, hh3 child) across alternatives
    assert (out["source_idperson"] == 102).sum() == 2
    assert (out["source_idperson"] == 303).sum() == 2


def test_injected_policy_swap_changes_mutation():
    res_de, *_ = _run(policy=de_earnings_policy)
    res_fk, *_ = _run(policy=fake_policy)
    # decider 101 chosen: hours 40, wage 20. DE yem = 20*40*52/12; fake yem = 20*40.
    def yem_of(res, src, alt, pid):
        o = res.output
        m = o[(o.source_idhh == src) & (o.alt == alt) & (o.source_idperson == pid)]
        return float(m["yem"].iloc[0])
    assert np.isclose(yem_of(res_de, 1, "chosen", 101), 20 * 40 * WPM)
    assert np.isclose(yem_of(res_fk, 1, "chosen", 101), 20 * 40)
    assert not np.isclose(yem_of(res_de, 1, "chosen", 101), yem_of(res_fk, 1, "chosen", 101))


def test_id_remap_and_reverse_mapping():
    res, *_ = _run()
    out, rev = res.output, res.reverse_mapping
    assert out["idperson"].is_unique               # globally unique synthetic persons
    assert (out["idhh"] >= 900_000_000).all()       # synthetic range
    assert rev["new_idperson"].is_unique
    # reverse mapping covers every output member and round-trips (synthetic -> source)
    merged = out.merge(rev, left_on=["idhh", "idperson"],
                       right_on=["new_idhh", "new_idperson"], suffixes=("", "_rev"))
    assert len(merged) == len(out)
    assert (merged["source_idperson"] == merged["source_idperson_rev"]).all()
    # idorigperson was remapped to the synthetic person id
    assert (out["idorigperson"] == out["idperson"]).all()


def test_no_cross_household_relationship_references():
    res, *_ = _run()
    out = res.output
    for _, g in out.groupby("idhh"):
        pids = set(g["idperson"].astype(int))
        for rc in ("idpartner", "idfather", "idmother"):
            ref = pd.to_numeric(g[rc], errors="coerce").fillna(0).astype(int)
            assert int(((ref != 0) & (~ref.isin(pids))).sum()) == 0
    # the child's mother ref in hh1 points to the SAME alternative's remapped decider
    hh1 = out[(out.source_idhh == 1) & (out.alt == "chosen")]
    child = hh1[hh1.source_idperson == 102].iloc[0]
    mother = hh1[hh1.source_idperson == 101].iloc[0]
    assert int(child["idmother"]) == int(mother["idperson"])


def test_stable_keys_and_provenance_survive():
    res, *_ = _run()
    out = res.output
    for col in ("alt", "source_idhh", "source_idorighh", "source_idperson",
                "ruro_decider", "dgn", "data_year"):
        assert col in out.columns
    assert set(out["alt"]) == {"chosen", "sim1"}
    assert (out["data_year"] == 2017).all()
    # ruro_decider correct: deciders flagged, children not
    assert int(out.loc[out.source_idperson == 101, "ruro_decider"].iloc[0]) == 1
    assert int(out.loc[out.source_idperson == 102, "ruro_decider"].iloc[0]) == 0
    assert int(out.loc[out.source_idperson == 301, "ruro_decider"].iloc[0]) == 1


def test_warning_error_capture():
    conn = FakeConnector(warnings=["Warning: parts of x do not sum"], errors=["fatal-ish note"])
    res, *_ = _run(connector=conn)
    assert res.warnings == ["Warning: parts of x do not sum"]
    assert res.errors == ["fatal-ish note"]


def test_raw_negative_dispy_retained_and_no_floor():
    res, *_ = _run()
    out = res.output
    # children (yem=0) -> 0.65*0 + 0 + 0 - 200 = -200 retained (no floor/clip)
    assert (out["ils_dispy"] < 0).any()
    assert np.isclose(out.loc[out.source_idperson == 102, "ils_dispy"].iloc[0], -200.0)
    # no floored/normalized/relabelled columns produced by the runner
    for forbidden in ("consumption", "ils_dispy_real", "c_norm", "c_scale"):
        assert forbidden not in out.columns
        assert forbidden not in res.taxunit_totals.columns


def test_taxunit_sum_separate_and_correct():
    res, *_ = _run()
    tu = res.taxunit_totals
    assert "ils_dispy_taxunit_sum" in tu.columns
    assert set(tu.columns) >= {"source_idhh", "alt", "ils_dispy_taxunit_sum"}
    # one row per alternative (8); matches manual per-(source_idhh,alt) sum of raw ils_dispy
    assert len(tu) == 8
    manual = res.output.groupby(["source_idhh", "alt"])["ils_dispy"].sum().reset_index()
    chk = tu.merge(manual, on=["source_idhh", "alt"])
    assert np.allclose(chk["ils_dispy_taxunit_sum"], chk["ils_dispy"])


def test_provenance_not_sent_to_connector():
    res, conn, *_ = _run()
    # the connector received raw EUROMOD-schema columns only, not provenance/alt keys
    for forbidden in ("alt", "source_idhh", "source_idperson", "source_idorighh", "ruro_decider"):
        assert forbidden not in conn.seen_cols


# === validation / failure modes ==========================================
def test_duplicate_alternative_keys_raise():
    base, alts = _fixture()
    dup = pd.concat([alts, alts.iloc[[0]]], ignore_index=True)  # repeat (1, chosen, 101)
    with pytest.raises(ValueError, match="duplicate"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            dup, base, alt_key_cols=["alt"])


def test_missing_decision_maker_raises():
    base, alts = _fixture()
    bad = alts.copy(); bad.loc[0, "decider_idperson"] = 999
    with pytest.raises(ValueError, match="missing decision-maker"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            bad, base, alt_key_cols=["alt"])


def test_unresolved_relationship_reference_raises():
    base, alts = _fixture()
    base = base.copy(); base.loc[base.idperson == 102, "idmother"] = 88888  # outside hh
    with pytest.raises(ValueError, match="unresolved relationship"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            alts, base, alt_key_cols=["alt"])


def test_incomplete_household_restoration_raises():
    base, alts = _fixture()
    base = base[base.idhh != 2]  # drop hh2 baseline while alts still reference it
    with pytest.raises(ValueError, match="incomplete household restoration"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            alts, base, alt_key_cols=["alt"])


def test_unsafe_id_range_raises():
    base, alts = _fixture()
    with pytest.raises(ValueError, match="unsafe ids|person_mult"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy, person_mult=2).build_inputs(
            alts, base, alt_key_cols=["alt"])  # households have 3 members > 2


# === (1) expected-decider completeness ===================================
def test_complete_singles_and_couples_pass():
    res, *_ = _run()   # full fixture deciders match the flagged baseline -> no raise
    assert len(res.output) == 16


def test_couple_alternative_missing_female_raises():
    base, alts = _fixture()
    # drop the female (302) decider row from couple hh3 "chosen"
    bad = alts.drop(alts[(alts.source_idhh == 3) & (alts.alt == "chosen") &
                         (alts.decider_idperson == 302)].index)
    with pytest.raises(ValueError, match="decider-set mismatch"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            bad, base, alt_key_cols=["alt"])


def test_unexpected_extra_decider_raises():
    base, alts = _fixture()
    # add the child (303, not flagged) as a decider of couple hh3 "chosen"
    extra = pd.DataFrame([dict(source_idhh=3, alt="chosen", decider_idperson=303, hours=10, wage=10.0)])
    bad = pd.concat([alts, extra], ignore_index=True)
    with pytest.raises(ValueError, match="decider-set mismatch"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            bad, base, alt_key_cols=["alt"])


# === (2) connector-output one-to-one validation ==========================
class _DropConnector(FakeConnector):
    def run(self, data, *, country, system, dataset):
        r = super().run(data, country=country, system=system, dataset=dataset)
        r.output = r.output.iloc[1:].copy()              # drop one member
        return r


class _DupConnector(FakeConnector):
    def run(self, data, *, country, system, dataset):
        r = super().run(data, country=country, system=system, dataset=dataset)
        r.output = pd.concat([r.output, r.output.iloc[[0]]], ignore_index=True)  # duplicate
        return r


class _AddConnector(FakeConnector):
    def run(self, data, *, country, system, dataset):
        r = super().run(data, country=country, system=system, dataset=dataset)
        extra = r.output.iloc[[0]].copy()
        extra["idperson"] = extra["idperson"].astype("int64") + 777_777  # not an expected member
        r.output = pd.concat([r.output, extra], ignore_index=True)
        return r


def test_connector_dropping_member_raises():
    with pytest.raises(ValueError, match="missing .* expected member"):
        _run(connector=_DropConnector())


def test_connector_duplicating_member_raises():
    with pytest.raises(ValueError, match="duplicate synthetic"):
        _run(connector=_DupConnector())


def test_connector_adding_member_raises():
    with pytest.raises(ValueError, match="unexpected member"):
        _run(connector=_AddConnector())


# === (3) protected identity columns ======================================
def test_policy_overriding_identity_column_raises():
    def evil_policy(member, *, hours, wage, weeks_per_month):
        return {"idhh": 5, "lhw": hours}   # idhh is protected
    base, alts = _fixture()
    with pytest.raises(ValueError, match="protected"):
        EuromodPricingRunner(FakeConnector(), evil_policy).build_inputs(
            alts, base, alt_key_cols=["alt"])


# === (4) reverse mapping: source_idorighh ================================
def test_reverse_mapping_has_source_idorighh_roundtrip():
    res, *_ = _run()
    rev = res.reverse_mapping
    assert "source_idorighh" in rev.columns
    base, _ = _fixture()
    orighh_by_person = dict(zip(base["idperson"].astype(int), base["idorighh"].astype(int)))
    for _, r in rev.iterrows():
        assert int(r["source_idorighh"]) == orighh_by_person[int(r["source_idperson"])]


def test_source_idorighh_household_consistency_validated():
    base, alts = _fixture()
    base = base.copy()
    base.loc[base.idperson == 102, "idorighh"] = 999_999   # break consistency within hh1
    with pytest.raises(ValueError, match="not household-consistent"):
        EuromodPricingRunner(FakeConnector(), de_earnings_policy).build_inputs(
            alts, base, alt_key_cols=["alt"])


# === (b) lazy import + ImportError =======================================
def test_package_import_is_light():
    code = (
        "import sys, dclaborsupply_app.euromod\n"
        "for m in ('euromod','pythonnet','clr','jpype','jnius'):\n"
        "    assert m not in sys.modules, m + ' imported at dclaborsupply_app.euromod import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_real_connector_raises_documented_importerror(monkeypatch):
    # simulate euromod absent: None in sys.modules makes `import euromod` raise ImportError
    monkeypatch.setitem(sys.modules, "euromod", None)
    df = pd.DataFrame({"idhh": [1], "idperson": [1], "yem": [0.0]})
    with pytest.raises(ImportError, match="euromod"):
        EuromodConnector("nonexistent-root").run(df, country="DE", system="DE_2016", dataset="DE_2017_a2")
