"""Tests for the DE canonical choice-state view + frozen earnings convention.

Self-contained (pandas/numpy + unchanged core generator; no scratch dependency).
Guards the fix for the working(=lhw>0) vs is_worker(=les==3&lhw>0) inconsistency:
non-worker deciders with stray hours must collapse to non-employment BEFORE draws.
"""
import numpy as np
import pandas as pd

import pandas.testing as pdt

from dclaborsupply.alternatives.continuous import generate_draws_long
from dclaborsupply_app.de.draws_prep import (
    canonicalize_choice_state,
    canonicalize_post_draws,
    assert_choice_state_consistent,
    COUPLES_GRID_NOTE,
    POST_MANAGED_FIELDS,
)

WPM = 52.0 / 12.0


def _state_fixture() -> pd.DataFrame:
    """Workers + non-workers-with-stray-hours (the 155-style inconsistency)."""
    rows = [
        # worker, full-time (untouched by canon)
        dict(idperson=1, is_worker=1, les=3, lhw=40, yivwg=20.0, wage=20.0, wage_ruro=20.0,
             working=1, working_pt1=0, working_pt2=0, working_ft=1, working_lh=0,
             loc4=2, yem=40 * 20.0 * WPM, yse=0.0, yemse=40 * 20.0 * WPM),
        # non-worker (les=5) BELOW 5h with stale positive yem  -> must zero
        dict(idperson=2, is_worker=0, les=5, lhw=3, yivwg=25.0, wage=0.0, wage_ruro=0.0,
             working=1, working_pt1=0, working_pt2=0, working_ft=0, working_lh=0,
             loc4=-1, yem=1500.0, yse=200.0, yemse=1700.0),
        # non-worker (les=7) WITHIN [5,70] -> the silent-pass hazard -> must zero
        dict(idperson=3, is_worker=0, les=7, lhw=30, yivwg=18.0, wage=0.0, wage_ruro=0.0,
             working=1, working_pt1=0, working_pt2=1, working_ft=0, working_lh=0,
             loc4=-1, yem=900.0, yse=0.0, yemse=900.0),
        # true non-employment (already consistent) -> idempotent
        dict(idperson=4, is_worker=0, les=5, lhw=0, yivwg=15.0, wage=0.0, wage_ruro=0.0,
             working=0, working_pt1=0, working_pt2=0, working_ft=0, working_lh=0,
             loc4=-1, yem=0.0, yse=0.0, yemse=0.0),
    ]
    return pd.DataFrame(rows)


def test_raw_frame_is_inconsistent_then_canon_fixes_it():
    df = _state_fixture()
    # raw frame has the documented inconsistency (rows 2 and 3)
    import pytest
    with pytest.raises(AssertionError):
        assert_choice_state_consistent(df)

    canon = canonicalize_choice_state(df)
    assert_choice_state_consistent(canon)  # no raise

    nonworkers = canon["is_worker"] == 0
    for col in ("lhw", "working", "working_pt1", "working_pt2", "working_ft",
                "working_lh", "wage", "wage_ruro", "yem"):
        assert (canon.loc[nonworkers, col] == 0).all(), col
    assert (canon.loc[nonworkers, "loc4"] == -1).all()

    # worker row untouched
    w = canon[canon["idperson"] == 1].iloc[0]
    assert w["lhw"] == 40 and w["working_ft"] == 1 and w["wage_ruro"] == 20.0


def test_frozen_earnings_convention_yem_zero_yemse_is_yse():
    df = _state_fixture()
    canon = canonicalize_choice_state(df)
    # row 2: non-worker, raw yem=1500 -> chosen state yem=0 (NOT restored), yemse=yse=200
    r2 = canon[canon["idperson"] == 2].iloc[0]
    assert r2["yem"] == 0.0
    assert r2["yemse"] == 200.0          # yem + yse = 0 + 200
    assert r2["yem_rawstate"] == 1500.0  # provenance preserved
    assert r2["yemse_rawstate"] == 1700.0
    # worker: yemse = yem + yse stays consistent
    r1 = canon[canon["idperson"] == 1].iloc[0]
    assert np.isclose(r1["yemse"], r1["yem"] + r1["yse"])


def test_raw_state_preserved():
    df = _state_fixture()
    canon = canonicalize_choice_state(df)
    r3 = canon[canon["idperson"] == 3].iloc[0]
    assert r3["lhw"] == 0 and r3["lhw_rawstate"] == 30      # raw hours kept
    assert r3["working_rawstate"] == 1                       # raw working kept


def test_canonicalized_nonworker_is_nonemployment_in_core_draws():
    """The formerly-inconsistent rows must be non-employment at draw 0 in the
    UNCHANGED core generator (hours==0 -> yem==0), not silently 'working'."""
    df = _state_fixture()
    # generator-required columns
    df = df.assign(idhh=df["idperson"], dgn=[1, 0, 1, 0], educL=0, educH=1)
    canon = canonicalize_choice_state(df)

    out = generate_draws_long(canon, n_draws=4, wage_spec="vw", occ_spec="fixed")

    d0 = out[out["draw"] == 0].set_index("idperson")
    # non-worker chosen rows: hours 0 and derived yem 0 (frozen convention)
    for pid in (2, 3, 4):
        assert d0.loc[pid, "hours"] == 0
        assert float(d0.loc[pid, "yem"]) == 0.0
    # worker chosen row: positive derived earnings
    assert float(d0.loc[1, "yem"]) > 0.0
    # exactly one chosen per person, and the generator did not raise on the
    # below-5h row (it is now non-employment, outside the support check)
    chosen = out.groupby("idperson")["is_chosen"].sum()
    assert (chosen == 1).all()
    # Wave-0.1 invariant: non-working draws carry zero H/W/Occ proposal components
    nonwork = pd.to_numeric(out["hours"], errors="coerce").fillna(0.0) <= 0
    assert (out.loc[nonwork, ["log_q_hours", "log_q_wage", "log_q_occ"]].abs().to_numpy() == 0).all()


def test_couples_grid_note_documents_non_fr_equivalence():
    assert "961" in COUPLES_GRID_NOTE and "901" in COUPLES_GRID_NOTE


# --------------------------------------------------------------------------- #
# POST-DRAW canonicalization (corrects the known core draw-0 wage limitation)  #
# --------------------------------------------------------------------------- #
def _generated_long():
    """Generator output (UNCHANGED core) over workers + non-workers, enough draws
    to yield both draw-0 and simulated non-employment rows."""
    rows = [
        dict(idperson=1, is_worker=1, les=3, lhw=40, yivwg=20.0, wage=20.0, wage_ruro=20.0,
             working=1, working_pt1=0, working_pt2=0, working_ft=1, working_lh=0,
             loc4=2, yem=40 * 20.0 * WPM, yse=0.0, yemse=40 * 20.0 * WPM),
        dict(idperson=2, is_worker=1, les=3, lhw=25, yivwg=18.0, wage=18.0, wage_ruro=18.0,
             working=1, working_pt1=0, working_pt2=1, working_ft=0, working_lh=0,
             loc4=3, yem=25 * 18.0 * WPM, yse=300.0, yemse=25 * 18.0 * WPM + 300.0),
        dict(idperson=3, is_worker=0, les=5, lhw=30, yivwg=22.0, wage=0.0, wage_ruro=0.0,
             working=1, working_pt1=0, working_pt2=0, working_ft=0, working_lh=0,
             loc4=-1, yem=900.0, yse=0.0, yemse=900.0),
        dict(idperson=4, is_worker=0, les=7, lhw=0, yivwg=15.0, wage=0.0, wage_ruro=0.0,
             working=0, working_pt1=0, working_pt2=0, working_ft=0, working_lh=0,
             loc4=-1, yem=0.0, yse=100.0, yemse=100.0),
    ]
    df = pd.DataFrame(rows).assign(
        idhh=lambda d: d["idperson"], dgn=lambda d: (d["is_worker"] * 0 + [1, 0, 1, 0]),
        educL=0, educH=1, wage_for_draws=lambda d: d["yivwg"],
    )
    canon_in = canonicalize_choice_state(df)
    return generate_draws_long(canon_in, n_draws=30, wage_spec="vw", occ_spec="fixed")


def test_reproduce_core_drawzero_wage_bug():
    out = _generated_long()
    d0 = out[out["draw"] == 0].set_index("idperson")
    # KNOWN CORE LIMITATION: non-worker chosen rows have hours==0 yet positive wage
    for pid in (3, 4):
        assert d0.loc[pid, "hours"] == 0
        assert float(d0.loc[pid, "wage"]) > 0.0
        assert float(d0.loc[pid, "wage_ruro"]) > 0.0


def test_post_draws_completes_nonemployment_state():
    out = _generated_long()
    post = canonicalize_post_draws(out)
    ne = pd.to_numeric(post["hours"], errors="coerce").fillna(0.0) <= 0.0
    assert ne.sum() > 0
    for c in ("working", "working_pt1", "working_pt2", "working_ft", "working_lh",
              "wage", "wage_ruro", "yem"):
        assert (post.loc[ne, c] == 0).all(), c
    assert (post.loc[ne, "loc4"] == -1).all()
    assert np.allclose(post.loc[ne, "yemse"], pd.to_numeric(post.loc[ne, "yse"]))
    # includes BOTH draw-0 non-employment and SIMULATED (draw>=1) non-employment rows
    assert (post.loc[ne, "draw"] == 0).any() and (post.loc[ne, "draw"] >= 1).any()
    # a worker's simulated non-employment row had working==1 pre-pass -> now 0
    sim_ne = (pd.to_numeric(out["hours"]).fillna(0) <= 0) & (out["draw"] >= 1)
    assert int((out.loc[sim_ne, "working"] == 1).sum()) > 0  # pre-pass inconsistency existed


def test_post_draws_hours_positive_byte_identical():
    out = _generated_long()
    post = canonicalize_post_draws(out)
    hp = pd.to_numeric(out["hours"], errors="coerce").fillna(0.0) > 0.0
    for c in POST_MANAGED_FIELDS:
        pdt.assert_series_equal(out.loc[hp, c], post.loc[hp, c], check_names=False)


def test_post_draws_offer_wage_unchanged():
    # wage_for_draws is the reliable offer-wage provenance. yivwg is unchanged
    # RELATIVE TO the generator output only — the core already overwrote yivwg with
    # the drawn wage on simulated rows (so yivwg is not a reliable offer carrier there).
    out = _generated_long()
    post = canonicalize_post_draws(out)
    for c in ("yivwg", "wage_for_draws"):
        pdt.assert_series_equal(out[c], post[c], check_names=False)
    # demonstrate the core's yivwg overwrite on a simulated non-employment row:
    sim_ne = out[(out["draw"] >= 1) & (pd.to_numeric(out["hours"]).fillna(0) <= 0)]
    if len(sim_ne):
        r = sim_ne.iloc[0]
        assert float(r["yivwg"]) == 0.0 and float(r["wage_for_draws"]) > 0.0


def test_post_draws_yemse_identity_all_rows():
    out = _generated_long()
    post = canonicalize_post_draws(out)
    assert np.allclose(post["yemse"], pd.to_numeric(post["yem"]) + pd.to_numeric(post["yse"]))


def test_post_draws_wave01_invariant_intact():
    out = _generated_long()
    post = canonicalize_post_draws(out)
    # post-pass leaves log_q_* untouched; non-working draws carry zero H/W/Occ
    nonwork = pd.to_numeric(post["hours"], errors="coerce").fillna(0.0) <= 0.0
    assert (post.loc[nonwork, ["log_q_hours", "log_q_wage", "log_q_occ"]].abs().to_numpy() == 0).all()
    total = post[["log_q_state", "log_q_hours", "log_q_wage", "log_q_occ"]].sum(axis=1)
    assert np.allclose(post["log_q_total"], total)
    # log_q columns are byte-identical to the generator output
    for c in ("log_q_state", "log_q_hours", "log_q_wage", "log_q_occ", "log_q_total"):
        pdt.assert_series_equal(out[c], post[c], check_names=False)


def test_post_draws_idempotent():
    out = _generated_long()
    once = canonicalize_post_draws(out)
    twice = canonicalize_post_draws(once)
    pdt.assert_frame_equal(once, twice)
