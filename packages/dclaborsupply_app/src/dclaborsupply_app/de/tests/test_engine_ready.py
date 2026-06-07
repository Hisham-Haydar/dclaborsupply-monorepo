"""Tests for DE engine-ready assembly (gates a-f).

Synthetic priced rows + pre-pricing features. Verifies the aggregation asymmetry,
the 1.0 floor (flag/count, not exclusion), cluster_id restoration from source_idorighh,
loc4 one-hots, spec-referenced columns, and the Wave-0.1 prior invariant.
"""
import numpy as np
import pandas as pd
import pytest

from dclaborsupply_app.de import (
    assemble, assemble_singles, assemble_couples, loc4_one_hots,
)

LOGQ = ["log_q_E", "log_q_H", "log_q_W", "log_q_Occ"]


def _singles():
    # priced per-person (decider + child); singles consumption must use DECIDER only
    priced = pd.DataFrame([
        dict(source_idhh=1, source_idorighh=11, alt="chosen", idperson=101, ruro_decider=1, ils_dispy=2000.0),
        dict(source_idhh=1, source_idorighh=11, alt="chosen", idperson=102, ruro_decider=0, ils_dispy=50.0),
        dict(source_idhh=1, source_idorighh=11, alt="sim1", idperson=101, ruro_decider=1, ils_dispy=2500.0),
        dict(source_idhh=1, source_idorighh=11, alt="sim1", idperson=102, ruro_decider=0, ils_dispy=50.0),
        dict(source_idhh=2, source_idorighh=22, alt="chosen", idperson=201, ruro_decider=1, ils_dispy=0.5),
        dict(source_idhh=2, source_idorighh=22, alt="sim1", idperson=201, ruro_decider=1, ils_dispy=1800.0),
    ])
    def feat(hh, orig, alt, pid, dgn, hours, loc4, ischosen, lq):
        return dict(source_idhh=hh, source_idorighh=orig, alt=alt, idperson=pid, ruro_decider=1, dgn=dgn,
                    hours=hours, lhw=hours, wage=(20.0 if hours > 0 else 0.0), loc4=loc4, age_norm=0.5, age_norm2=0.25, n_children=1,
                    educL=0, educM=1, educH=0, working=int(hours > 0),
                    working_pt1=int(18.5 <= hours <= 20.5), working_pt2=int(29.5 <= hours <= 30.5),
                    working_ft=int(37.5 <= hours <= 40.5), working_lh=int(44.5 <= hours <= 70),
                    is_chosen=ischosen, log_q_E=lq,
                    log_q_H=(lq if hours > 0 else 0.0), log_q_W=(lq if hours > 0 else 0.0),
                    log_q_Occ=0.0,
                    log_prior=(lq + (2 * lq if hours > 0 else 0.0)),
                    prior=float(np.exp(lq + (2 * lq if hours > 0 else 0.0))))
    features = pd.DataFrame([
        feat(1, 11, "chosen", 101, 0, 40, 2, 1, 0.0),
        feat(1, 11, "sim1", 101, 0, 20, 2, 0, -2.0),
        feat(2, 22, "chosen", 201, 1, 0, -1, 1, 0.0),      # non-employment chosen -> loc4=-1
        feat(2, 22, "sim1", 201, 1, 35, 3, 0, -1.5),
    ])
    return priced, features


def _couples():
    priced = pd.DataFrame([
        dict(source_idhh=3, source_idorighh=33, alt="chosen", idperson=301, ruro_decider=1, ils_dispy=1800.0),
        dict(source_idhh=3, source_idorighh=33, alt="chosen", idperson=302, ruro_decider=1, ils_dispy=1000.0),
        dict(source_idhh=3, source_idorighh=33, alt="chosen", idperson=303, ruro_decider=0, ils_dispy=30.0),  # nonzero child
        dict(source_idhh=3, source_idorighh=33, alt="sim", idperson=301, ruro_decider=1, ils_dispy=2200.0),
        dict(source_idhh=3, source_idorighh=33, alt="sim", idperson=302, ruro_decider=1, ils_dispy=1200.0),
        dict(source_idhh=3, source_idorighh=33, alt="sim", idperson=303, ruro_decider=0, ils_dispy=30.0),
        dict(source_idhh=4, source_idorighh=44, alt="chosen", idperson=401, ruro_decider=1, ils_dispy=1500.0),
        dict(source_idhh=4, source_idorighh=44, alt="chosen", idperson=402, ruro_decider=1, ils_dispy=800.0),
        dict(source_idhh=4, source_idorighh=44, alt="sim", idperson=401, ruro_decider=1, ils_dispy=0.4),  # sub-floor sum w/ female
        dict(source_idhh=4, source_idorighh=44, alt="sim", idperson=402, ruro_decider=1, ils_dispy=0.3),
    ])
    def feat(hh, orig, alt, pid, dgn, hours, loc4, ischosen, lq):
        return dict(source_idhh=hh, source_idorighh=orig, alt=alt, idperson=pid, ruro_decider=1, dgn=dgn,
                    hours=hours, lhw=hours, wage=(20.0 if hours > 0 else 0.0), loc4=loc4, age_norm=0.3, age_norm2=0.09, n_children=2,
                    educL=0, educM=1, educH=0, working=int(hours > 0),
                    working_pt1=int(18.5 <= hours <= 20.5), working_pt2=int(29.5 <= hours <= 30.5),
                    working_ft=int(37.5 <= hours <= 40.5), working_lh=int(44.5 <= hours <= 70),
                    is_chosen=ischosen, log_q_E=lq,
                    log_q_H=(lq if hours > 0 else 0.0), log_q_W=(lq if hours > 0 else 0.0),
                    log_q_Occ=0.0,
                    log_prior=(lq + (2 * lq if hours > 0 else 0.0)),
                    prior=float(np.exp(lq + (2 * lq if hours > 0 else 0.0))))
    features = pd.DataFrame([
        feat(3, 33, "chosen", 301, 1, 50, 4, 1, 0.0), feat(3, 33, "chosen", 302, 0, 15, 2, 1, 0.0),
        feat(3, 33, "sim", 301, 1, 35, 4, 0, -2.0), feat(3, 33, "sim", 302, 0, 30, 2, 0, -1.0),
        feat(4, 44, "chosen", 401, 1, 45, 1, 1, 0.0), feat(4, 44, "chosen", 402, 0, 20, 3, 1, 0.0),
        feat(4, 44, "sim", 401, 1, 0, -1, 0, -3.0), feat(4, 44, "sim", 402, 0, 0, -1, 0, -3.0),
    ])
    return priced, features


# === (a) aggregation asymmetry ===========================================
def test_singles_consumption_is_decider_only():
    p, f = _singles()
    s = assemble_singles(p, f)
    row = s[(s.source_idhh == 1) & (s.alt == "chosen")].iloc[0]
    # decider 2000, child 50 -> singles consumption EXCLUDES child
    assert row["consumption_raw"] == 2000.0 and row["consumption"] == 2000.0


def test_couples_consumption_is_taxunit_sum_incl_child():
    p, f = _couples()
    c = assemble_couples(p, f)
    row = c[(c.source_idhh == 3) & (c.alt == "chosen")].iloc[0]
    # 1800 + 1000 + 30 (child) = 2830
    assert row["consumption_raw"] == 2830.0 and row["consumption"] == 2830.0


# === (b) floor ===========================================================
def test_floor_applied_flagged_not_excluded():
    p, f = _singles()
    s = assemble_singles(p, f)
    r = s[(s.source_idhh == 2) & (s.alt == "chosen")].iloc[0]   # raw 0.5 -> floor 1.0
    assert r["consumption_raw"] == 0.5 and r["consumption"] == 1.0 and bool(r["consumption_floored"])
    assert len(s) == 4   # no rows dropped (key count invariant)
    pc, fc = _couples()
    res = assemble(p, f, pc, fc)
    fr = res.floor_report
    assert fr["singles"]["floored"] == 1 and fr["singles"]["chosen_floored"] == 1
    # couples hh4 sim: 0.4+0.3=0.7 -> floored
    assert fr["couples"]["floored"] == 1 and fr["couples"]["chosen_floored"] == 0
    assert len(res.couples) == 4   # 2 couples x 2 alts, none dropped


# === (c) cluster identity ================================================
def test_cluster_id_from_source_idorighh_consistent():
    p, f = _singles()
    s = assemble_singles(p, f)
    assert (s.loc[s.source_idhh == 1, "cluster_id"] == 11).all()
    assert (s.loc[s.source_idhh == 2, "cluster_id"] == 22).all()
    assert (s["cluster_id"] != s["source_idhh"]).any()  # real household, not synthetic
    pc, fc = _couples()
    c = assemble_couples(pc, fc)
    assert (c.loc[c.source_idhh == 3, "cluster_id"] == 33).all()


def test_cluster_inconsistent_source_idorighh_raises():
    p, f = _singles()
    f = f.copy()
    f.loc[(f.source_idhh == 1) & (f.alt == "sim1"), "source_idorighh"] = 999  # break consistency
    with pytest.raises(ValueError, match="not household-consistent"):
        assemble_singles(p, f)


# === (d) loc4 one-hots ===================================================
def test_loc4_one_hots_helper():
    oh = loc4_one_hots(pd.Series([1, 2, 3, 4, -1, -2]))
    assert list(oh["loc4_1"]) == [1, 0, 0, 0, 0, 0]
    assert list(oh["loc4_2"]) == [0, 1, 0, 0, 0, 0]
    assert list(oh["loc4_4"]) == [0, 0, 0, 1, 0, 0]


def test_loc4_one_hots_in_singles():
    p, f = _singles()
    s = assemble_singles(p, f)
    work = s[s["working"] == 1]
    # working rows: exactly one of loc4_1..4 set (incl. reference loc4_1)
    onehot_sum = work[["loc4_1", "loc4_2", "loc4_3", "loc4_4"]].sum(axis=1)
    assert (onehot_sum == 1).all()
    nonwork = s[s["working"] == 0]   # loc4 == -1 -> all zero
    assert (nonwork[["loc4_1", "loc4_2", "loc4_3", "loc4_4"]].sum(axis=1) == 0).all()


# === (e) spec-referenced columns + prior invariant ========================
def test_spec_columns_present_singles_and_couples():
    p, f = _singles(); pc, fc = _couples()
    res = assemble(p, f, pc, fc)
    s, c = res.singles, res.couples
    for col in ("age_norm", "age_norm2", "n_children", "educL", "educM", "educH",
                "working", "working_pt1", "working_pt2", "working_ft", "working_lh",
                "loc4_2", "loc4_3", "loc4_4", "consumption", "leisure",
                "log_q_E", "log_q_H", "log_q_W", "log_q_Occ", "log_prior", "prior", "cluster_id"):
        assert col in s.columns, f"singles missing {col}"
    for col in ("age_norm_male", "age_norm_female", "working_male", "working_female",
                "loc4_2_male", "loc4_2_female", "consumption", "leisure_male", "leisure_female",
                "log_q_E_male", "log_q_E_female", "log_prior", "prior", "cluster_id"):
        assert col in c.columns, f"couples missing {col}"
    # income-basis metadata recorded
    assert res.metadata["income_source"] == "ils_dispy" and res.metadata["price_factor"] == 1.0


def test_logq_logprior_preserved_prior_canonicalized():
    # log_q_* and log_prior are carried UNCHANGED; prior is NOT preserved -> it is
    # canonicalized from log_prior (matching the certified harmonizer). Do not claim
    # prior is preserved.
    p, f = _singles()
    s = assemble_singles(p, f)
    merged = s.merge(f, on=["source_idhh", "alt"], suffixes=("", "_in"))
    for col in LOGQ + ["log_prior"]:
        assert np.allclose(merged[col], merged[f"{col}_in"]), f"{col} changed by assembly"
    exp = np.clip(np.exp(np.clip(s["log_prior"].to_numpy(float), -700, 700)), 1e-16, None)
    assert np.allclose(s["prior"].to_numpy(float), exp)   # prior == canonical(log_prior)
    # couples: per-partner log_q unchanged; joint log_prior = male + female
    pc, fc = _couples()
    c = assemble_couples(pc, fc)
    male = fc[fc.dgn == 1].set_index(["source_idhh", "alt"])
    row = c[(c.source_idhh == 3) & (c.alt == "sim")].iloc[0]
    assert np.isclose(row["log_q_E_male"], male.loc[(3, "sim"), "log_q_E"])
    assert np.isclose(row["log_prior"], -2.0 * 3 + -1.0 * 3)  # male log_prior + female log_prior


# === one-to-one join + widening ==========================================
def test_join_rejects_missing_key():
    p, f = _singles()
    f2 = f[~((f.source_idhh == 2) & (f.alt == "sim1"))]   # drop one alt from features
    with pytest.raises(ValueError, match="alt-key set mismatch"):
        assemble_singles(p, f2)


def test_couples_widened_one_row_per_joint_alt():
    pc, fc = _couples()
    c = assemble_couples(pc, fc)
    assert len(c) == c[["source_idhh", "alt"]].drop_duplicates().shape[0] == 4
    r = c[(c.source_idhh == 3) & (c.alt == "chosen")].iloc[0]
    assert r["hours_male"] == 50 and r["hours_female"] == 15   # explicit male/female fields
    assert r["is_chosen"] == 1


# === new repair gates ====================================================
def test_couples_missing_female_alternative_raises():
    pc, fc = _couples()
    bad = fc.drop(fc[(fc.source_idhh == 3) & (fc.alt == "sim") & (fc.dgn == 0)].index)  # female-only missing
    with pytest.raises(ValueError, match="exactly one dgn=1"):
        assemble_couples(pc, bad)


def test_couples_extra_male_alternative_raises():
    pc, fc = _couples()
    extra = fc[(fc.source_idhh == 3) & (fc.alt == "sim") & (fc.dgn == 1)].copy()
    extra["alt"] = "extra_male_only"
    with pytest.raises(ValueError, match="exactly one dgn=1"):
        assemble_couples(pc, pd.concat([fc, extra], ignore_index=True))


def test_output_has_idhh_idorighh_and_wage():
    p, f = _singles(); pc, fc = _couples()
    s = assemble_singles(p, f)
    c = assemble_couples(pc, fc)
    for col in ("idhh", "idorighh", "wage", "log_wage", "source_idhh", "source_idorighh"):
        assert col in s.columns, f"singles missing {col}"
    assert (s["idhh"] == s["source_idhh"]).all() and (s["idorighh"] == s["source_idorighh"]).all()
    for col in ("idhh", "idorighh", "wage_male", "wage_female", "source_idhh", "source_idorighh"):
        assert col in c.columns, f"couples missing {col}"
    assert (c["idhh"] == c["source_idhh"]).all()


def test_couples_separate_leisure_scales():
    pc, fc = _couples()
    c = assemble_couples(pc, fc)
    assert "l_male_scale" in c.columns and "l_female_scale" in c.columns
    assert "l_scale" not in c.columns   # singles-only name not used for couples


def test_metadata_normalization_and_ndraws_structure():
    p, f = _singles(); pc, fc = _couples()
    m = assemble(p, f, pc, fc).metadata
    assert set(m["normalization"]["singles"]) == {"c_scale", "l_scale", "n_chosen"}
    assert set(m["normalization"]["couples"]) == {"c_scale", "l_male_scale", "l_female_scale", "n_chosen"}
    assert set(m["n_draws"]) == {"singles", "couples"}
    assert set(m["row_counts"]) == {"singles", "couples", "total"}
    assert m["normalization"]["singles"]["n_chosen"] == 2   # 2 single households
    assert m["n_draws"]["singles"] == 2                      # chosen + sim1


def test_missing_or_nonfinite_feature_raises():
    p, f = _singles()
    f2 = f.copy(); f2.loc[0, "age_norm"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        assemble_singles(p, f2)
    f3 = f.drop(columns=["wage"])
    with pytest.raises(ValueError, match="missing required columns"):
        assemble_singles(p, f3)


# === gap 1: NaN income rejected on EVERY priced row before aggregation ====
def test_singles_nan_income_raises():
    p, f = _singles()
    p2 = p.copy(); p2.loc[1, "ils_dispy"] = np.nan   # a (non-decider) child row
    with pytest.raises(ValueError, match="non-finite"):
        assemble_singles(p2, f)


def test_couples_nan_income_raises():
    pc, fc = _couples()
    pc2 = pc.copy(); pc2.loc[2, "ils_dispy"] = np.nan   # the nonzero child row
    with pytest.raises(ValueError, match="non-finite"):
        assemble_couples(pc2, fc)


# === gap 2: Wave-0.1 contract after state recomputation ===================
def test_singles_wave01_logprior_violation_raises():
    p, f = _singles()
    f2 = f.copy()
    f2.loc[(f2.source_idhh == 1) & (f2.alt == "sim1"), "log_prior"] = 123.0  # break identity
    with pytest.raises(ValueError, match=r"log_prior != log_q_E"):
        assemble_singles(p, f2)


def test_couples_wave01_partner_violation_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2.loc[(fc2.source_idhh == 3) & (fc2.alt == "sim") & (fc2.dgn == 1), "log_prior"] = 77.0
    with pytest.raises(ValueError, match=r"log_prior_male != "):
        assemble_couples(pc, fc2)


def test_prior_is_canonical_from_log_prior():
    p, f = _singles()
    s = assemble_singles(p, f)
    exp = np.clip(np.exp(np.clip(s["log_prior"].to_numpy(float), -700, 700)), 1e-16, None)
    assert np.allclose(s["prior"].to_numpy(float), exp) and (s["prior"] > 0).all()


# === gap 3: is_chosen strictly binary ====================================
def test_is_chosen_half_half_raises():
    p, f = _singles()
    f2 = f.copy()
    f2["is_chosen"] = f2["is_chosen"].astype(float)
    f2.loc[f2.source_idhh == 2, "is_chosen"] = 0.5   # [0.5,0.5] -> sums to 1 but not binary
    with pytest.raises(ValueError, match="is_chosen must be binary"):
        assemble_singles(p, f2)


# === wage > 0 when working ===============================================
def test_wage_zero_when_working_raises():
    p, f = _singles()
    f2 = f.copy()
    # hh1 chosen: hours=40 (working after recompute) but force wage=0 -> reject
    f2.loc[(f2.source_idhh == 1) & (f2.alt == "chosen"), "wage"] = 0.0
    with pytest.raises(ValueError, match=r"wage must be > 0 when working"):
        assemble_singles(p, f2)


# === strict tolerance: a 1e-6 identity error must be REJECTED =============
def test_wave01_rejects_1e6_identity_error():
    p, f = _singles()
    f2 = f.copy()
    f2["log_prior"] = f2["log_prior"].astype(float)
    f2.loc[(f2.source_idhh == 1) & (f2.alt == "sim1"), "log_prior"] += 1e-6  # within default rtol, > atol
    with pytest.raises(ValueError, match=r"log_prior != log_q_E"):
        assemble_singles(p, f2)


# === explicit component-zero on non-working rows =========================
def test_singles_nonworking_nonzero_logq_raises():
    p, f = _singles()
    f2 = f.copy()
    f2["log_q_H"] = f2["log_q_H"].astype(float)
    f2.loc[(f2.source_idhh == 2) & (f2.alt == "chosen"), "log_q_H"] = 0.5  # hours==0 -> non-working
    with pytest.raises(ValueError, match=r"singles: .* log_q_H != 0.0"):
        assemble_singles(p, f2)


def test_couples_male_nonworking_nonzero_logq_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2["log_q_W"] = fc2["log_q_W"].astype(float)
    fc2.loc[(fc2.source_idhh == 4) & (fc2.alt == "sim") & (fc2.dgn == 1), "log_q_W"] = 0.7  # male non-working
    with pytest.raises(ValueError, match=r"couples male: .* log_q_W_male != 0.0"):
        assemble_couples(pc, fc2)


def test_couples_female_nonworking_nonzero_logq_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2["log_q_Occ"] = fc2["log_q_Occ"].astype(float)
    fc2.loc[(fc2.source_idhh == 4) & (fc2.alt == "sim") & (fc2.dgn == 0), "log_q_Occ"] = 0.9  # female non-working
    with pytest.raises(ValueError, match=r"couples female: .* log_q_Occ_female != 0.0"):
        assemble_couples(pc, fc2)


# === price_factor finite and strictly positive ===========================
@pytest.mark.parametrize("pf", [0.0, -1.0, float("inf"), float("nan")])
def test_price_factor_must_be_finite_positive(pf):
    p, f = _singles()
    with pytest.raises(ValueError, match="price_factor must be finite"):
        assemble_singles(p, f, price_factor=pf)


# === gap 1: couples decider structure (exactly one dgn=1 + one dgn=0) =====
def test_couples_unexpected_dgn_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2.loc[(fc2.source_idhh == 3) & (fc2.alt == "chosen") & (fc2.dgn == 0), "dgn"] = 2  # invalid dgn
    with pytest.raises(ValueError, match=r"dgn must be in \{0,1\}"):
        assemble_couples(pc, fc2)


def test_couples_extra_decider_row_raises():
    pc, fc = _couples()
    extra = fc[(fc.source_idhh == 3) & (fc.alt == "chosen") & (fc.dgn == 1)].copy()  # duplicate male
    with pytest.raises(ValueError, match="exactly one dgn=1"):
        assemble_couples(pc, pd.concat([fc, extra], ignore_index=True))


# === gap 2: partner-shared fields must agree ==============================
def test_couples_partner_chosen_mismatch_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2.loc[(fc2.source_idhh == 3) & (fc2.alt == "chosen") & (fc2.dgn == 0), "is_chosen"] = 0  # male=1, female=0
    with pytest.raises(ValueError, match=r"partner-shared field 'is_chosen' differs"):
        assemble_couples(pc, fc2)


def test_couples_partner_provenance_mismatch_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    fc2.loc[(fc2.source_idhh == 3) & (fc2.alt == "chosen") & (fc2.dgn == 0), "source_idorighh"] = 99999
    with pytest.raises(ValueError, match=r"partner-shared field 'source_idorighh' differs"):
        assemble_couples(pc, fc2)


# === gap 3: invalid working loc4 (occupation-state contract) ==============
def test_invalid_working_loc4_raises():
    p, f = _singles()
    f2 = f.copy()
    # hh1 chosen: hours=40 (working) but loc4=-2 -> working must have loc4 in {1,2,3,4}
    f2.loc[(f2.source_idhh == 1) & (f2.alt == "chosen"), "loc4"] = -2
    with pytest.raises(ValueError, match=r"working row\(s\) with loc4 not in \{1,2,3,4\}"):
        assemble_singles(p, f2)


def test_couples_invalid_working_loc4_raises():
    pc, fc = _couples()
    fc2 = fc.copy()
    # hh3 chosen male: hours=50 (working) but loc4=-2
    fc2.loc[(fc2.source_idhh == 3) & (fc2.alt == "chosen") & (fc2.dgn == 1), "loc4"] = -2
    with pytest.raises(ValueError, match=r"couples male: working row\(s\) with loc4_male not in"):
        assemble_couples(pc, fc2)
