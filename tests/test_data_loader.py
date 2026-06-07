"""Committed, self-contained tests for the package-native, spec-aware engine-ready loader.

Synthetic fixtures only. Covers shapes, chosen-first column-0 ordering, the two
hours_band_policy modes, spec-driven required-variable enforcement, strict no-silent-
coercion, structural validation, DataFrame + parquet-path input, and the light-import
boundary. External FR/DE reproduction gates are NOT committed here.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from dclaborsupply.data import load_singles, load_couples, load_engine_ready_stem


class _Spec:
    """Minimal spec exposing only what the loader inspects (shifter lists + wage_spec)."""
    def __init__(self, *, leisure=None, hours=None, market=None, wage=None, wage_spec="vw"):
        self.wage_spec = wage_spec
        self.utility_leisure_shifters = leisure if leisure is not None else [
            {"variable": "age_norm", "coefficient": "b_age"},
            {"variable": "age_norm2", "coefficient": "b_age2"},
            {"variable": "n_children", "coefficient": "b_nkids", "gender_specific": True}]
        self.hours_shifters = hours if hours is not None else [
            {"variable": b, "coefficient": "h"} for b in
            ("working", "working_pt1", "working_pt2", "working_ft", "working_lh")]
        self.market_opportunity_shifters = market if market is not None else [
            {"variable": v} for v in ("loc4_2", "loc4_3", "loc4_4")]   # applies_to default "both"
        self.wage_mean_shifters = wage if wage is not None else [
            {"variable": "intercept"}, {"variable": "educL"}, {"variable": "educH"}]


_SPEC = _Spec()
_META = {
    "normalization": {
        "singles": {"c_scale": 100.0, "l_scale": 10.0},
        "couples": {"c_scale": 200.0, "l_male_scale": 10.0, "l_female_scale": 12.0},
    },
    "cluster_key": {"cluster_id_col": "cluster_id", "source_col": "idorighh"},
}

_PT1, _PT2, _FT = (18.5, 21.5), (29.5, 30.5), (37.5, 40.5)


def _bands(h):
    h = np.asarray(h, float)
    return {"working": (h > 0).astype(float),
            "working_pt1": ((h >= _PT1[0]) & (h <= _PT1[1])).astype(float),
            "working_pt2": ((h >= _PT2[0]) & (h <= _PT2[1])).astype(float),
            "working_ft": ((h >= _FT[0]) & (h <= _FT[1])).astype(float)}


def _singles_df(dgn, n_hh=2, n_alts=3, idhh0=1):
    rows = []
    for hh in range(n_hh):
        idhh = idhh0 + hh
        for a in range(n_alts):
            hours = 0.0 if a == 0 else (20.0 if a == 1 else 40.0)
            b = _bands([hours])
            rows.append(dict(
                idhh=idhh, idorighh=1000 + idhh, cluster_id=1000 + idhh, dgn=dgn,
                is_chosen=(1 if a == 0 else 0),
                c_norm=1.0 + 0.1 * a, l_norm=1.0 + 0.05 * a, hours=hours, prior=(1.0 if a == 0 else 0.5),
                working=b["working"][0], working_pt1=b["working_pt1"][0],
                working_pt2=b["working_pt2"][0], working_ft=b["working_ft"][0], working_lh=0.0,
                age_norm=0.3, age_norm2=0.09, educL=0.0, educM=1.0, educH=0.0,
                n_children=2.0, loc4=(-1 if hours == 0 else 2), wage=(0.0 if hours == 0 else 15.0)))
    return pd.DataFrame(rows)


def _couples_df(n_hh=2, n_alts=4, idhh0=1):
    rows = []
    for hh in range(n_hh):
        idhh = idhh0 + hh
        for a in range(n_alts):
            hm = 0.0 if a == 0 else 40.0
            hf = 0.0 if a in (0, 1) else 20.0
            bm, bf = _bands([hm]), _bands([hf])
            rows.append(dict(
                idhh=idhh, idorighh=2000 + idhh, cluster_id=2000 + idhh, dgn=-1,
                is_chosen=(1 if a == 0 else 0),
                c_norm=1.0 + 0.1 * a, l_norm_male=1.0 + 0.05 * a, l_norm_female=1.0 + 0.04 * a,
                hours_male=hm, hours_female=hf, prior=(1.0 if a == 0 else 0.5), n_children=1.0,
                working_male=bm["working"][0], working_pt1_male=bm["working_pt1"][0],
                working_pt2_male=bm["working_pt2"][0], working_ft_male=bm["working_ft"][0], working_lh_male=0.0,
                working_female=bf["working"][0], working_pt1_female=bf["working_pt1"][0],
                working_pt2_female=bf["working_pt2"][0], working_ft_female=bf["working_ft"][0], working_lh_female=0.0,
                age_norm_male=0.2, age_norm2_male=0.04, educL_male=0.0, educM_male=1.0, educH_male=0.0,
                age_norm_female=0.3, age_norm2_female=0.09, educL_female=1.0, educM_female=0.0, educH_female=0.0,
                loc4_male=(-1 if hm == 0 else 4), loc4_female=(-1 if hf == 0 else 2),
                wage_male=(0.0 if hm == 0 else 18.0), wage_female=(0.0 if hf == 0 else 12.0)))
    return pd.DataFrame(rows)


# === shapes + chosen-first ================================================
def test_singles_shapes_and_chosen_first():
    dm = load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META)
    assert dm.n_groups == 2 and dm.n_obs == 6 and dm.n_obs // dm.n_groups == 3 and dm.is_male is True
    assert np.all(dm.actual_choice[dm.group_starts] == 1)
    assert all(dm.actual_choice[s:e].sum() == 1 for s, e in zip(dm.group_starts, dm.group_ends))
    assert np.allclose(dm.consumption[:3], [1.0, 1.1, 1.2])
    assert np.allclose(dm.working[:3], [0, 1, 1]) and np.allclose(dm.working_ft[:3], [0, 0, 1])
    assert np.all(dm.n_children == 0) and np.all(dm.female == 0)


def test_singles_female_keeps_n_children():
    df = load_singles(_singles_df(dgn=0), _SPEC, is_male=False, metadata=_META)
    assert np.all(df.n_children == 2.0) and np.all(df.female == 1) and df.is_male is False


def test_couples_shapes():
    c = load_couples(_couples_df(), _SPEC, metadata=_META)
    assert c.n_groups == 2 and c.n_obs == 8 and c.n_obs // c.n_groups == 4
    assert np.all(c.actual_choice[c.group_starts] == 1)
    assert np.allclose(c.working_male[:4], [0, 1, 1, 1]) and np.allclose(c.working_female[:4], [0, 0, 1, 1])
    assert np.all(c.female_male == 0) and np.all(c.female_female == 1) and c.loc4_2_male is not None


# === hours_band_policy: assembled (read) vs legacy_certified (re-derive) ==
def test_hours_band_policy_assembled_vs_legacy():
    df = _singles_df(dgn=1)
    # a sim row at hours=21.0 lies in (20.5, 21.5]; set the ASSEMBLED column to 0 there.
    df.loc[(df.idhh == 1) & (df.hours == 20.0), "hours"] = 21.0
    df.loc[(df.idhh == 1) & (df.hours == 21.0), "working_pt1"] = 0.0   # deliberately != re-derive
    a = load_singles(df, _SPEC, is_male=True, metadata=_META, hours_band_policy="assembled")
    leg = load_singles(df, _SPEC, is_male=True, metadata=_META, hours_band_policy="legacy_certified")
    # assembled reads the (zeroed) column; legacy re-derives 1 from hours=21.0 (<=21.5)
    idx = int(np.flatnonzero(np.isclose(leg.working_pt1, 1))[0])
    assert a.working_pt1[idx] == 0.0 and leg.working_pt1[idx] == 1.0


def test_invalid_policy_raises():
    with pytest.raises(ValueError, match="hours_band_policy"):
        load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META, hours_band_policy="nope")


# === spec-driven required variables =======================================
def test_missing_required_shifter_raises():
    df = _singles_df(dgn=1).drop(columns=["educL"])   # educL is a spec wage shifter
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_missing_required_band_assembled_raises():
    df = _singles_df(dgn=1).drop(columns=["working_ft"])
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(df, _SPEC, is_male=True, metadata=_META, hours_band_policy="assembled")


def test_missing_hours_legacy_raises():
    df = _singles_df(dgn=1).drop(columns=["hours"])
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(df, _SPEC, is_male=True, metadata=_META, hours_band_policy="legacy_certified")


def test_wage_required_when_vw():
    df = _singles_df(dgn=1).drop(columns=["wage"])
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


# === strict, no silent coercion ===========================================
def test_present_required_nan_raises():
    df = _singles_df(dgn=1)
    df.loc[df.index[1], "age_norm"] = np.nan
    with pytest.raises(ValueError, match="NaN / non-numeric / non-finite"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_present_optional_nan_also_raises():
    # educM is PRESENT but NOT spec-referenced; a NaN must still raise (no silent default).
    df = _singles_df(dgn=1)
    df.loc[df.index[0], "educM"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


# === structural validation ================================================
def test_non_constant_alternatives_raises():
    df = _singles_df(dgn=1).drop(index=5)
    with pytest.raises(ValueError, match="non-constant alternatives"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_chosen_not_first_raises():
    df = _singles_df(dgn=1)
    df.loc[df.idhh == 1, "is_chosen"] = [0, 1, 0]
    with pytest.raises(ValueError, match="not first"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_multiple_chosen_raises():
    df = _singles_df(dgn=1)
    df.loc[df.idhh == 1, "is_chosen"] = [1, 1, 0]
    with pytest.raises(ValueError, match="exactly one chosen"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_dgn_mismatch_raises():
    with pytest.raises(ValueError, match="dgn must all equal 1"):
        load_singles(_singles_df(dgn=0), _SPEC, is_male=True, metadata=_META)


def test_cluster_not_constant_within_group_raises():
    df = _singles_df(dgn=1)
    df.loc[df.index[1], "cluster_id"] = 999999   # breaks within-group constancy
    with pytest.raises(ValueError, match="cluster id .* not constant within"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_metadata_scale_must_be_positive_finite():
    bad = {**_META, "normalization": {**_META["normalization"], "singles": {"c_scale": 0.0, "l_scale": 10.0}}}
    with pytest.raises(ValueError, match="must be finite and > 0"):
        load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=bad)


def test_spec_required():
    with pytest.raises(ValueError, match="spec is required"):
        load_singles(_singles_df(dgn=1), None, is_male=True, metadata=_META)


# === DataFrame + parquet-path + stem ======================================
def test_dataframe_input_default():
    assert load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META).n_obs == 6


def test_parquet_path_input(tmp_path):
    pytest.importorskip("pyarrow")
    p = tmp_path / "s.parquet"
    _singles_df(dgn=1).to_parquet(p)
    a = load_singles(p, _SPEC, is_male=True, metadata=_META)
    b = load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META)
    assert np.allclose(a.consumption, b.consumption)


def test_load_engine_ready_stem(tmp_path):
    pytest.importorskip("pyarrow")
    import json
    stem = tmp_path / "synth"
    pd.concat([_singles_df(dgn=1, idhh0=1), _singles_df(dgn=0, idhh0=10)], ignore_index=True) \
        .to_parquet(f"{stem}__singles.parquet")
    _couples_df().to_parquet(f"{stem}__couples.parquet")
    (tmp_path / "synth__mnlmeta.json").write_text(json.dumps(_META))
    sm, sf, cou = load_engine_ready_stem(stem, _SPEC)
    assert sm.is_male is True and sf.is_male is False and cou.n_groups == 2


def test_stem_year_tags_unavailable_raises(tmp_path):
    pytest.importorskip("pyarrow")
    import json
    stem = tmp_path / "synth"
    _singles_df(dgn=1).to_parquet(f"{stem}__singles.parquet")
    _couples_df().to_parquet(f"{stem}__couples.parquet")
    (tmp_path / "synth__mnlmeta.json").write_text(json.dumps(_META))
    with pytest.raises(ValueError, match="year_tag.* absent"):
        load_engine_ready_stem(stem, _SPEC, year_tags=[1])


def test_stem_year_tags_empty_raises(tmp_path):
    pytest.importorskip("pyarrow")
    import json
    stem = tmp_path / "synth"
    s = pd.concat([_singles_df(dgn=1, idhh0=1), _singles_df(dgn=0, idhh0=10)], ignore_index=True)
    s["year_tag"] = 2
    c = _couples_df(); c["year_tag"] = 2
    s.to_parquet(f"{stem}__singles.parquet"); c.to_parquet(f"{stem}__couples.parquet")
    (tmp_path / "synth__mnlmeta.json").write_text(json.dumps(_META))
    with pytest.raises(ValueError, match="produced empty"):
        load_engine_ready_stem(stem, _SPEC, year_tags=[99])


# === gap 1: applies_to routing mirrors the engine =========================
def test_singles_applies_to_skips_incompatible_route():
    # a market var routed to males only must be required for is_male=True, skipped for female.
    spec = _Spec(market=[{"variable": "drgur", "applies_to": "male"}])
    df = _singles_df(dgn=1).drop(columns=["drgur"]) if "drgur" in _singles_df(dgn=1) else _singles_df(dgn=1)
    # male route requires drgur -> missing raises
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(df, spec, is_male=True, metadata=_META)
    # female route skips it -> loads fine even though drgur is absent
    f = load_singles(_singles_df(dgn=0), spec, is_male=False, metadata=_META)
    assert f.n_groups == 2


def test_couples_male_only_occupation_requires_only_male_loc4():
    spec = _Spec(market=[{"variable": v, "applies_to": "male"} for v in ("loc4_2", "loc4_3", "loc4_4")])
    df = _couples_df().drop(columns=["loc4_female"])   # female loc4 absent
    c = load_couples(df, spec, metadata=_META)          # male-only route -> female not required
    assert c.loc4_2_male is not None and c.loc4_2_female is None


def test_couples_household_route_uses_unsuffixed():
    # year_2015_indicator routed household -> requires the unsuffixed column (not _male/_female)
    spec = _Spec(market=[{"variable": "year_2015_indicator", "applies_to": "household"}])
    df = _couples_df(); df["year_2015_indicator"] = 1.0
    c = load_couples(df, spec, metadata=_META)
    assert np.all(c.year_2015_indicator == 1.0)
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_couples(_couples_df(), spec, metadata=_META)   # unsuffixed column absent


# === gap 2: no silent clipping ============================================
@pytest.mark.parametrize("col,val", [("c_norm", 0.0), ("c_norm", -1.0), ("l_norm", 0.0), ("prior", 0.0)])
def test_nonpositive_core_raises(col, val):
    df = _singles_df(dgn=1)
    df.loc[df.index[1], col] = val
    with pytest.raises(ValueError, match="strictly positive"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_negative_wage_raises():
    df = _singles_df(dgn=1)
    df.loc[df.index[2], "wage"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_zero_wage_when_working_raises():
    df = _singles_df(dgn=1)
    # row at hours=40 (working after assembled bands) but wage forced to 0
    df.loc[(df.idhh == 1) & (df.hours == 40.0), "wage"] = 0.0
    with pytest.raises(ValueError, match="> 0 where the working flag is 1"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_zero_nonworker_wage_preserved():
    # chosen row hours=0 (non-working) keeps wage=0 -> log_wage=log(EPS), finite, no raise
    dm = load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META)
    assert dm.log_wage is not None and np.all(np.isfinite(dm.log_wage))
    nonwork = dm.working == 0
    assert np.all(dm.log_wage[nonwork] < -20)   # log(EPS)=~ -27.6


# === gap 3: pooled group_ids genuinely unique =============================
def test_pooled_group_ids_unique_across_years():
    a = _singles_df(dgn=1, n_hh=2, idhh0=1)
    b = _singles_df(dgn=1, n_hh=2, idhh0=1)   # SAME idhh values, different year
    a["year_tag"] = 1
    b["year_tag"] = 2
    dm = load_singles(pd.concat([a, b], ignore_index=True), _SPEC, is_male=True, metadata=_META)
    assert dm.n_groups == 4                       # 2 idhh x 2 years
    assert len(set(dm.group_ids.tolist())) == 4   # genuinely unique despite repeated idhh


# === gap 4: region all-or-drgn1; partial raises ===========================
def test_partial_reg_nuts1_raises():
    df = _singles_df(dgn=1)
    df["reg_nuts1_2"] = 0.0   # only one of seven, no drgn1
    with pytest.raises(ValueError, match="partial region dummy set"):
        load_singles(df, _SPEC, is_male=True, metadata=_META)


def test_region_required_all_or_drgn1():
    spec = _Spec(market=[{"variable": f"reg{k}", "applies_to": "household"} for k in range(2, 9)])
    # all seven reg_nuts1 present -> ok
    df = _singles_df(dgn=1)
    for k in range(2, 9):
        df[f"reg_nuts1_{k}"] = 0.0
    assert load_singles(df, spec, is_male=True, metadata=_META).n_groups == 2
    # drgn1 fallback -> ok
    df2 = _singles_df(dgn=1); df2["drgn1"] = 1.0
    assert load_singles(df2, spec, is_male=True, metadata=_META).n_groups == 2
    # neither -> missing
    with pytest.raises(ValueError, match="ALL reg_nuts1_2..8 OR drgn1"):
        load_singles(_singles_df(dgn=1), spec, is_male=True, metadata=_META)


# === gap 5: couples l_female_scale validated ==============================
@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_couples_l_female_scale_validated(bad):
    meta = {**_META, "normalization": {**_META["normalization"],
            "couples": {"c_scale": 200.0, "l_male_scale": 10.0, "l_female_scale": bad}}}
    with pytest.raises(ValueError, match="l_female_scale.* must be finite and > 0"):
        load_couples(_couples_df(), _SPEC, metadata=meta)


# === load-to-engine binding: custom parser-approved variables ============
def test_custom_hours_var_attached_singles():
    # hours_bin_1 is a spec-required custom var present in the DataFrame: it must be
    # attached to the returned object under the exact engine attribute name (not dropped).
    spec = _Spec(hours=[{"variable": b, "coefficient": "h"} for b in
                        ("working", "working_pt1", "working_pt2", "working_ft", "working_lh")]
                 + [{"variable": "hours_bin_1", "coefficient": "h_b1"}])
    df = _singles_df(dgn=1)
    df["hours_bin_1"] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    dm = load_singles(df, spec, is_male=True, metadata=_META)
    assert getattr(dm, "hours_bin_1", None) is not None
    assert np.allclose(dm.hours_bin_1, df["hours_bin_1"].to_numpy())


def test_custom_market_var_attached_couples_male_only():
    # isco1_1 routed to males only -> only isco1_1_male is required & attached.
    spec = _Spec(market=[{"variable": v, "applies_to": "male"} for v in ("loc4_2", "loc4_3", "loc4_4")]
                 + [{"variable": "isco1_1", "applies_to": "male"}])
    df = _couples_df()
    df["isco1_1_male"] = 1.0
    c = load_couples(df, spec, metadata=_META)
    assert getattr(c, "isco1_1_male", None) is not None and np.allclose(c.isco1_1_male, 1.0)
    assert getattr(c, "isco1_1_female", None) is None   # female route inactive


def test_custom_var_missing_column_rejected_not_dropped():
    spec = _Spec(hours=[{"variable": b, "coefficient": "h"} for b in
                        ("working", "working_pt1", "working_pt2", "working_ft", "working_lh")]
                 + [{"variable": "isco1_1", "coefficient": "h_i"}])
    with pytest.raises(ValueError, match="missing spec-required variable"):
        load_singles(_singles_df(dgn=1), spec, is_male=True, metadata=_META)   # explicit reject, never silent


def test_binding_audit_all_active_vars_bound():
    # every active engine attribute resolves to a non-None array on the returned object.
    dm = load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META)
    for name in ("age_norm", "age_norm2", "working", "working_pt1", "working_ft",
                 "working_lh", "loc4_2", "loc4_3", "loc4_4", "educL", "educH"):
        assert getattr(dm, name, None) is not None, name
    c = load_couples(_couples_df(), _SPEC, metadata=_META)
    for name in ("age_norm_male", "age_norm_female", "working_pt1_male", "working_pt1_female",
                 "loc4_2_male", "loc4_2_female", "n_children"):
        assert getattr(c, name, None) is not None, name


# === n_children gender_specific rule (mirror engine exactly) =============
def test_n_children_gender_specific_zeroed_for_male():
    dm = load_singles(_singles_df(dgn=1), _SPEC, is_male=True, metadata=_META)  # default gs=True
    assert np.all(dm.n_children == 0)


def test_n_children_not_gender_specific_loaded_for_male():
    spec = _Spec(leisure=[{"variable": "age_norm", "coefficient": "b_age"},
                          {"variable": "age_norm2", "coefficient": "b_age2"},
                          {"variable": "n_children", "coefficient": "b_nkids"}])  # NOT gender_specific
    dm = load_singles(_singles_df(dgn=1), spec, is_male=True, metadata=_META)
    assert np.all(dm.n_children == 2.0)   # male values loaded, not forced to zero


# === light-import boundary ================================================
def test_light_import_no_jax_gamspy_java():
    code = textwrap.dedent("""
        import sys
        import dclaborsupply.data            # noqa
        import dclaborsupply.data.loader     # noqa
        bad = [m for m in sys.modules if m.split('.')[0] in
               ('jax','jaxlib','gamspy','jpype','java','jpype1')]
        assert not bad, f"heavy import leaked: {bad}"
        print("OK")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, f"stdout={out.stdout}\nstderr={out.stderr}"
    assert "OK" in out.stdout
