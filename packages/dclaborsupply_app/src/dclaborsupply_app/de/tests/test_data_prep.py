"""Tests for the DE 2017 data-prep adapter.

Pure pandas/numpy. The loc4 test is the critical guard on the CERTIFIED
task-grouping convention (a reversed mapping silently inverts the DE occupation
model). A guarded end-to-end test runs on the real DE_2017_a2 file if present.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dclaborsupply_app.de.data_prep import (
    DE_CONFIG,
    DE_DROPPED_REGION_COLS,
    DEFAULT_DATA_PATH,
    apply_stepwise_filters,
    build_features,
    classify_households,
    collapse_loc_to_loc4,
    compute_is_worker,
    prepare_de_2017,
    reshape_couples_to_wide,
)

WPM = 52.0 / 12.0


def _person(idhh, k, *, dag, dgn, les, lhw, yivwg, idpartner=0, loc=2, deh=3,
            dec=0, ddi=0, yse=0.0, poa=0.0, byr=0.0, pdi=0.0, psu=0.0):
    yem = lhw * yivwg * WPM if les == 3 and lhw > 0 else 0.0
    return {
        "idhh": idhh, "idperson": idhh * 100 + k, "idpartner": idpartner,
        "idorighh": idhh, "dag": dag, "dgn": dgn, "deh": deh, "dehde": deh * 100,
        "dec": dec, "ddi": ddi, "lhw": lhw, "yivwg": yivwg, "yem": yem, "yse": yse,
        "yemse": yem + yse, "loc": loc, "les": les, "byr": byr, "pdi": pdi,
        "poa": poa, "psu": psu, "liwmy": 12,
        "drgn1": 0, "drgur": 0, "drgmd": 0, "drgru": 0, "dms": 2,
    }


def _fixture():
    rows = []
    # HH1 single clean employee -> SURVIVES
    rows += [_person(1, 1, dag=40, dgn=1, les=3, lhw=40, yivwg=20, loc=2)]
    # HH2 opposite-sex couple clean -> SURVIVES (2 deciders); mutual idpartner
    rows += [_person(2, 1, dag=42, dgn=1, les=3, lhw=40, yivwg=20, idpartner=202, loc=1),
             _person(2, 2, dag=39, dgn=0, les=3, lhw=20, yivwg=15, idpartner=201, loc=5)]
    # HH3 same-sex couple (mutual) -> excluded at classification
    rows += [_person(3, 1, dag=45, dgn=1, les=3, lhw=40, yivwg=20, idpartner=302),
             _person(3, 2, dag=43, dgn=1, les=3, lhw=40, yivwg=20, idpartner=301)]
    # HH4 two adults, no mutual link -> excluded
    rows += [_person(4, 1, dag=50, dgn=1, les=3, lhw=40, yivwg=20, idpartner=0),
             _person(4, 2, dag=48, dgn=0, les=3, lhw=20, yivwg=15, idpartner=0)]
    # HH5 three adults -> excluded
    rows += [_person(5, 1, dag=55, dgn=1, les=3, lhw=40, yivwg=20),
             _person(5, 2, dag=53, dgn=0, les=3, lhw=20, yivwg=15),
             _person(5, 3, dag=25, dgn=1, les=3, lhw=38, yivwg=18)]
    # HH6 single age 70 -> dropped (age)
    rows += [_person(6, 1, dag=70, dgn=1, les=3, lhw=40, yivwg=20)]
    # HH7 single in education dec=4 -> dropped (education)
    rows += [_person(7, 1, dag=22, dgn=0, les=3, lhw=40, yivwg=20, dec=4)]
    # HH8 single with old-age pension poa>0 -> dropped (retirement)
    rows += [_person(8, 1, dag=60, dgn=1, les=3, lhw=40, yivwg=20, poa=1000.0)]
    # HH9 single self-employed les=2 -> dropped (allowed_les)
    rows += [_person(9, 1, dag=45, dgn=1, les=2, lhw=40, yivwg=20)]
    # HH10 single parent + earning child(non-decider) -> dropped (other members)
    rows += [_person(10, 1, dag=45, dgn=0, les=3, lhw=40, yivwg=20),
             _person(10, 2, dag=16, dgn=1, les=3, lhw=20, yivwg=10, yse=0.0)]
    rows[-1]["yem"] = 500.0  # child has meaningful income -> triggers drop
    # HH11 single employee, yivwg out of bounds (>170) -> dropped (wage)
    rows += [_person(11, 1, dag=44, dgn=1, les=3, lhw=40, yivwg=500)]
    # HH12 single employee lhw=3 (<=5) -> inactive transition, SURVIVES (les->7)
    rows += [_person(12, 1, dag=50, dgn=0, les=3, lhw=3, yivwg=20)]
    # HH13 single employee lhw=80 -> capped to 70, SURVIVES; working_lh=1
    rows += [_person(13, 1, dag=38, dgn=1, les=3, lhw=80, yivwg=25)]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1. CERTIFIED loc4 mapping (the critical guard)                              #
# --------------------------------------------------------------------------- #
def test_loc4_certified_task_mapping():
    loc_ruro = pd.Series([-1, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    loc4, loc_armed = collapse_loc_to_loc4(loc_ruro)
    expected = [-1, -2, -2, 4, 4, 4, 3, 2, 1, 1, 1, 1]
    assert list(loc4) == expected
    # ISCO 0 = armed forces stays -2 with the armed flag set
    assert list(loc_armed) == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # explicit anti-inversion: nonroutine-cognitive must be {1,2,3} -> 4 (NOT 1)
    assert loc4.iloc[3] == 4 and loc4.iloc[5] == 4
    assert loc4.iloc[8] == 1  # ISCO 6 -> routine manual
    # ISCO 9 default goes to routine-manual (1), not nonroutine-manual (2)
    assert loc4.iloc[11] == 1
    alt, _ = collapse_loc_to_loc4(loc_ruro, elementary_as_nonroutine_manual=True)
    assert alt.iloc[11] == 2


def test_compute_is_worker_employees_only():
    df = pd.DataFrame({"les": [3, 3, 2, 5, 7], "lhw": [40, 0, 40, 40, 40]})
    w = compute_is_worker(df)
    assert list(w) == [True, False, False, False, False]  # self-employed excluded


# --------------------------------------------------------------------------- #
# 2. Household classification (DE idpartner only)                              #
# --------------------------------------------------------------------------- #
def test_classify_households():
    c = classify_households(_fixture())
    cls = c.groupby("idhh")["household_class"].first().to_dict()
    assert cls[1] == "single"
    assert cls[2] == "couple_mf"
    assert cls[3] == "excl_same_sex"
    assert cls[4] == "excl_2adult_no_link"
    assert cls[5] == "excl_3plus_adults"
    # ruro_decider: single->1 decider; couple->2; excluded->0
    dec = c.groupby("idhh")["ruro_decider"].sum().to_dict()
    assert dec[1] == 1 and dec[2] == 2
    assert dec[3] == 0 and dec[4] == 0 and dec[5] == 0


# --------------------------------------------------------------------------- #
# 3. FR-mirror stepwise eligibility filters                                    #
# --------------------------------------------------------------------------- #
def test_stepwise_filters_survivors():
    c = classify_households(_fixture())
    filt, stats = apply_stepwise_filters(c)
    survivors = set(filt["idhh"].unique())
    assert survivors == {1, 2, 12, 13}, survivors
    assert len(stats) >= 6  # funnel recorded

    # HH12: lhw<=5 employee -> inactive transition
    p12 = filt[filt["idhh"] == 12].iloc[0]
    assert p12["les"] == 7 and p12["lhw"] == 0 and p12["yem"] == 0.0
    # HH13: lhw>70 -> capped to 70
    assert filt[filt["idhh"] == 13].iloc[0]["lhw"] == 70


# --------------------------------------------------------------------------- #
# 4. Feature construction                                                       #
# --------------------------------------------------------------------------- #
def test_build_features_contract():
    c = classify_households(_fixture())
    filt, _ = apply_stepwise_filters(c)
    f = build_features(filt)

    # region cols dropped
    for col in DE_DROPPED_REGION_COLS:
        assert col not in f.columns

    # education mapping (deh==3 in fixture -> educM)
    assert (f["educM"] == 1).all() and (f["educL"] == 0).all() and (f["educH"] == 0).all()

    # wage split: wage_for_draws = yivwg always; wage_ruro 0 for non-workers
    assert np.allclose(f["wage_for_draws"], pd.to_numeric(f["yivwg"]))
    nonworker = f["is_worker"] == 0
    assert (f.loc[nonworker, "wage_ruro"] == 0.0).all()
    worker = f["is_worker"] == 1
    assert np.allclose(f.loc[worker, "wage_ruro"], pd.to_numeric(f.loc[worker, "yivwg"]))
    # not aliased: wage_for_draws nonzero where wage_ruro is zero (HH12 inactive)
    assert ((f["wage_for_draws"] > 0) & (f["wage_ruro"] == 0)).any()

    # earnings identity
    assert np.allclose(f["yemse"], pd.to_numeric(f["yem"]) + pd.to_numeric(f["yse"]))

    # working bands: HH13 lhw=70 -> working_lh=1, working_ft=0
    p13 = f[f["idhh"] == 13].iloc[0]
    assert p13["working_lh"] == 1 and p13["working_ft"] == 0
    # HH1 lhw=40 -> working_ft=1, working_lh=0
    p1 = f[f["idhh"] == 1].iloc[0]
    assert p1["working_ft"] == 1 and p1["working_lh"] == 0

    # female = (dgn==0)
    assert (f["female"] == (pd.to_numeric(f["dgn"]) == 0).astype(int)).all()

    # age_norm centred on the decider sample mean
    dec = f["ruro_sample"] == 1
    assert abs(float(f.loc[dec, "age_norm"].mean())) < 1e-9
    assert np.allclose(f["age_norm2"], f["age_norm"] ** 2)


# --------------------------------------------------------------------------- #
# 5. Couples reshape (dgn used only here)                                       #
# --------------------------------------------------------------------------- #
def test_reshape_couples_to_wide():
    c = classify_households(_fixture())
    filt, _ = apply_stepwise_filters(c)
    f = build_features(filt)
    couples = f[f["household_class"] == "couple_mf"]
    wide = reshape_couples_to_wide(couples)
    assert len(wide) == 1  # HH2
    row = wide.iloc[0]
    # male decider was dgn=1 (lhw=40), female dgn=0 (lhw=20)
    assert row["lhw_male"] == 40 and row["lhw_female"] == 20
    assert row["is_worker_male"] == 1 and row["is_worker_female"] == 1
    assert "idhh" in wide.columns


# --------------------------------------------------------------------------- #
# 6. End-to-end on the real DE_2017_a2 file (guarded)                          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not Path(DEFAULT_DATA_PATH).exists(),
                    reason="real DE_2017_a2 microdata not present")
def test_prepare_real_data_smoke():
    res = prepare_de_2017(write=False)
    singles, couples = res["singles"], res["couples"]
    assert len(singles) > 0 and len(couples) > 0
    # opposite-sex couples: exactly 2 deciders, one of each gender
    dec = couples[couples["ruro_decider"] == 1]
    g = dec.groupby("idhh")["dgn"].agg(lambda s: tuple(sorted(pd.to_numeric(s))))
    assert (g == (0, 1)).all()
    # region cols dropped; loc4 only in certified range; yemse identity holds
    for col in DE_DROPPED_REGION_COLS:
        assert col not in singles.columns
    assert set(pd.unique(singles["loc4"])).issubset({-2, -1, 1, 2, 3, 4})
    assert np.allclose(pd.to_numeric(singles["yemse"]),
                       pd.to_numeric(singles["yem"]) + pd.to_numeric(singles["yse"]))
    # deciders are employees/unemployed/inactive only (allowed_les)
    assert set(pd.unique(dec["les"])).issubset(set(DE_CONFIG["allowed_les"]))
