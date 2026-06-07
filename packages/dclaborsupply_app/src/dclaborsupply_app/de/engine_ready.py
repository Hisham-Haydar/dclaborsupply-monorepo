"""DE engine-ready ASSEMBLY: priced per-person rows -> engine-ready singles/couples.

Turns the pricing-runner's priced per-person output (raw nominal ``ils_dispy`` +
provenance) plus the pre-pricing per-alternative FEATURES (draws-derived: demographics,
choice/state columns, log_q/prior) into the engine-ready contract, applying the three
operations the authority doc governs and the pricing runner deliberately deferred:

  1. AGGREGATION ASYMMETRY (authority §4/§7.1): singles consumption = the DECIDER
     person's disposable income (ruro_decider==1); couples consumption = SUM over the
     tax unit (all members) per household-alternative key.
  2. CONSUMPTION FLOOR (authority §6/§7.3): consumption = clip(value, lower=1.0),
     applied HERE (not at pricing); a boolean floor flag + counts are recorded;
     alternatives are FLOORED, never excluded (row/key counts invariant).
  3. CLUSTER IDENTITY (authority §7.5): cluster_id = source_idorighh (the ORIGINAL
     household), validated household-consistent — never the synthetic pricing id.

``log_q_*`` and ``log_prior`` are carried through UNCHANGED (Wave-0.1 invariant). ``prior``
is NOT preserved: it is (re)canonicalized from ``log_prior`` as
``clip(exp(clip(log_prior,-700,700)),1e-16,None)`` — matching the certified harmonizer's
prior convention — so any incoming ``prior`` is overwritten by this canonical value.

DE income basis (this single-year smoke): income_source="ils_dispy", price_factor=1.0,
consumption is **nominal DE-2017 euros**. No ``ils_dispy_real`` column is created and FR
CPI is NOT reused; income_source/price_factor are injectable for a later real-income
policy. Estimation is OUT of this module; no df->PrecomputedData loading happens here.

Placement: implemented in de/ for now. RECOMMENDATION — the three operations
(aggregate_consumption / apply_consumption_floor / restore_cluster_id) and loc4 one-hot
construction are country-GENERAL and should be promoted to a shared euromod/engine_ready
module once a second country exists; the per-partner WIDENING column list and DE column
names stay country-specific. Kept in de/ to avoid premature generalization.

App layer only: pandas/numpy; no core/france/MNL imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DCM_MIN_POSITIVE = 1.0
TOTAL_LEISURE_HOURS = 80.0

# Hours bands (must equal the certified RURO definitions). Working/band flags are
# RECOMPUTED here from each alternative's hours: the core draw primitive leaves stale
# baseline flags on hours>0 simulated rows (per the draws-prep design, band recomputation
# is deferred to engine-ready), so the assembly is the single source of band truth.
_PT1, _PT2, _FT, _LH = (18.5, 20.5), (29.5, 30.5), (37.5, 40.5), (44.5, 70.0)


def _recompute_state_from_hours(df: pd.DataFrame, *, hours_col: str = "hours") -> pd.DataFrame:
    """Recompute working/working_* from hours and force non-employment loc4=-1.

    Deterministic function of the alternative's hours; overwrites any stale draws-stage
    band flags so working/bands/loc4 are mutually consistent before one-hot construction.
    """
    df = df.copy()
    h = pd.to_numeric(df[hours_col], errors="coerce").fillna(0.0)
    working = (h > 0.0).astype("int8")
    df["working"] = working
    df["working_pt1"] = ((h >= _PT1[0]) & (h <= _PT1[1])).astype("int8")
    df["working_pt2"] = ((h >= _PT2[0]) & (h <= _PT2[1])).astype("int8")
    df["working_ft"] = ((h >= _FT[0]) & (h <= _FT[1])).astype("int8")
    df["working_lh"] = ((working == 1) & (h >= _LH[0]) & (h <= _LH[1])).astype("int8")
    if "loc4" in df.columns:
        loc4 = pd.to_numeric(df["loc4"], errors="coerce").fillna(-2).astype(int)
        df["loc4"] = np.where(h <= 0.0, -1, loc4).astype(int)
    return df

# DE minimal spec references these per-decider feature columns (de_2017_minimal.yaml).
_SPEC_FEATURE_COLS = (
    "age_norm", "age_norm2", "n_children", "educL", "educM", "educH",
    "working", "working_pt1", "working_pt2", "working_ft", "working_lh", "loc4",
)
# Per-partner columns widened to _male/_female for couples.
_PARTNER_COLS = (
    "age_norm", "age_norm2", "educL", "educM", "educH",
    "working", "working_pt1", "working_pt2", "working_ft", "working_lh",
    "loc4", "loc4_1", "loc4_2", "loc4_3", "loc4_4", "hours", "lhw", "leisure", "log_l",
    "wage", "log_wage", "log_q_E", "log_q_H", "log_q_W", "log_q_Occ", "log_prior",
)
_LOGQ = ("log_q_E", "log_q_H", "log_q_W", "log_q_Occ")
# Features that must be present and finite on EVERY input row (fix 7).
_REQUIRED_FEATURES = (
    "age_norm", "age_norm2", "n_children", "educL", "educM", "educH",
    "hours", "wage", "log_q_E", "log_q_H", "log_q_W", "log_q_Occ", "log_prior", "prior",
)
# dataclass-required fields the DE minimal spec DROPS (region/year/market) -> 0 stubs.
_DE_ZERO_STUBS = ("drgn1", "gsur", "reg2", "reg3", "reg4", "reg5", "reg6", "reg7", "reg8",
                  "drgur", "drgmd", "drgru", "year_2015_indicator", "year_2017_indicator")


@dataclass
class EngineReadyResult:
    singles: pd.DataFrame
    couples: pd.DataFrame
    floor_report: Dict[str, Dict[str, float]]
    metadata: Dict[str, object]


# --------------------------------------------------------------------------- #
# General transforms (country-agnostic; promote to euromod/ when 2nd country)  #
# --------------------------------------------------------------------------- #
def aggregate_consumption(
    priced: pd.DataFrame,
    *,
    household_type: str,
    hh_key: str,
    alt_keys: Sequence[str],
    decider_flag: str = "ruro_decider",
    income_source: str = "ils_dispy",
    price_factor: float = 1.0,
) -> pd.DataFrame:
    """Per-alternative consumption with the aggregation ASYMMETRY.

    singles: the single decider's ``income_source`` (ruro_decider==1).
    couples: SUM of ``income_source`` over ALL tax-unit members.
    Returns one row per (hh_key, *alt_keys) with ``consumption_raw`` (= income*price_factor).
    """
    keys = [hh_key, *alt_keys]
    if income_source not in priced.columns:
        raise ValueError(f"priced output missing income_source column '{income_source}'.")
    if not (np.isfinite(price_factor) and price_factor > 0):   # gap 3
        raise ValueError(f"price_factor must be finite and strictly > 0; got {price_factor!r}.")
    # require income present + finite on EVERY priced person row before grouping (gap 1)
    _require_present_finite(priced, [income_source], tag=f"{household_type} priced income")
    g = priced.copy()
    g["_inc"] = pd.to_numeric(g[income_source], errors="coerce") * float(price_factor)
    if household_type == "singles":
        dec = g[pd.to_numeric(g[decider_flag], errors="coerce").fillna(0) == 1]
        cnt = dec.groupby(keys).size()
        if (cnt != 1).any():
            bad = cnt[cnt != 1].head(5).to_dict()
            raise ValueError(f"singles: expected exactly one decider per alternative; offending {bad}.")
        out = dec.groupby(keys, as_index=False)["_inc"].sum()
    elif household_type == "couples":
        out = g.groupby(keys, as_index=False)["_inc"].sum()   # sum over all members
    else:
        raise ValueError("household_type must be 'singles' or 'couples'.")
    return out.rename(columns={"_inc": "consumption_raw"})


def apply_consumption_floor(
    df: pd.DataFrame, *, raw_col: str = "consumption_raw",
    out_col: str = "consumption", floor: float = DCM_MIN_POSITIVE,
) -> pd.DataFrame:
    """consumption = clip(raw, lower=floor); add boolean ``consumption_floored``.
    Floored, NOT excluded — row count unchanged."""
    df = df.copy()
    raw = pd.to_numeric(df[raw_col], errors="coerce")
    df[out_col] = raw.clip(lower=floor)
    df[f"{out_col}_floored"] = (raw < floor)
    return df


def loc4_one_hots(loc4: pd.Series) -> Dict[str, pd.Series]:
    """loc4_1..4 one-hots (reference loc4==1). Non-workers (-1) / unknown (-2) -> all 0."""
    loc = pd.to_numeric(loc4, errors="coerce").fillna(-2).astype(int)
    return {f"loc4_{k}": (loc == k).astype("int8") for k in (1, 2, 3, 4)}


def restore_cluster_id(
    df: pd.DataFrame, *, hh_key: str, source_orighh_col: str = "source_idorighh",
    out_col: str = "cluster_id",
) -> pd.DataFrame:
    """cluster_id = original household id (source_idorighh), validated consistent per hh_key."""
    df = df.copy()
    chk = df.groupby(hh_key)[source_orighh_col].nunique()
    if (chk > 1).any():
        bad = chk[chk > 1].head(5).to_dict()
        raise ValueError(f"source_idorighh not household-consistent per {hh_key}: {bad}.")
    df[out_col] = pd.to_numeric(df[source_orighh_col], errors="coerce").astype("int64")
    return df


def _require_present_finite(df: pd.DataFrame, cols: Sequence[str], *, tag: str) -> None:
    """Reject missing columns or any missing/non-finite values (fix 7)."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{tag}: missing required columns {missing}.")
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        n_bad = int((~np.isfinite(v)).sum())
        if n_bad:
            raise ValueError(f"{tag}: column '{c}' has {n_bad} missing/non-finite value(s).")


def _log_wage(wage: pd.Series, working: pd.Series) -> np.ndarray:
    """log(wage) for workers (working==1 & wage>0); 0 otherwise."""
    w = pd.to_numeric(wage, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    wk = pd.to_numeric(working, errors="coerce").fillna(0).to_numpy()
    return np.where((wk == 1) & (w > 0), np.log(np.where(w > 0, w, 1.0)), 0.0)


def _normalize(df: pd.DataFrame, *, leisure_cols: Sequence[str]) -> pd.DataFrame:
    """c_scale=mean(consumption) all rows; per-leisure-column scale = min positive chosen
    leisure of THAT column (singles -> l_scale; couples -> l_male_scale / l_female_scale)."""
    df = df.copy()
    cons = pd.to_numeric(df["consumption"], errors="coerce")
    c_scale = float(cons.mean())
    df["c_scale"] = c_scale
    df["log_c"] = np.log(cons)
    df["c_norm"] = df["consumption"] / c_scale
    df["log_c_norm"] = np.log(df["c_norm"])
    chosen = df["is_chosen"] == 1
    for lc in leisure_cols:
        leis = pd.to_numeric(df[lc], errors="coerce")
        pos = leis[chosen & (leis > 0)]
        scale = float(pos.min()) if len(pos) else 1.0
        suffix = lc[len("leisure"):]                  # "" | "_male" | "_female"
        scale_name = "l_scale" if suffix == "" else f"l{suffix}_scale"
        df[scale_name] = scale
        df[f"l_norm{suffix}"] = leis / scale
        df[f"log_l_norm{suffix}"] = np.log(leis / scale)
    return df


def _validate_and_sort(df: pd.DataFrame, *, hh_key: str, alt_keys: Sequence[str],
                       finite_cols: Sequence[str], tag: str) -> pd.DataFrame:
    """Deterministic sort + structural validation (fixes 6,7)."""
    df = df.sort_values([hh_key, *alt_keys], kind="mergesort").reset_index(drop=True)
    # is_chosen must be strictly binary {0,1} (reject e.g. [0.5,0.5]) BEFORE the count check
    isch = pd.to_numeric(df["is_chosen"], errors="coerce")
    if not np.isin(isch.to_numpy(), [0, 1]).all():
        bad = sorted(set(isch.dropna().tolist()) - {0, 1, 0.0, 1.0})
        raise ValueError(f"{tag}: is_chosen must be binary {{0,1}}; found other values e.g. {bad[:5]}.")
    ch = df.groupby(hh_key)["is_chosen"].sum()
    if (ch != 1).any():
        raise ValueError(f"{tag}: not exactly one chosen per household: {ch[ch != 1].head().to_dict()}.")
    sz = df.groupby(hh_key, sort=False).size()
    if sz.nunique() != 1:
        raise ValueError(f"{tag}: non-constant alternatives per household: {sorted(set(sz))[:6]}.")
    grp = df[hh_key].to_numpy()
    blocks = int((pd.Series(grp) != pd.Series(grp).shift()).sum())
    if blocks != df[hh_key].nunique():
        raise ValueError(f"{tag}: household groups not contiguous after sort.")
    _require_present_finite(df, ["consumption", "prior", *finite_cols], tag=f"{tag} output")
    return df


def _canonical_prior(log_prior: np.ndarray) -> np.ndarray:
    return np.clip(np.exp(np.clip(np.asarray(log_prior, dtype="float64"), -700, 700)), 1e-16, None)


def _require_wage_positive_when_working(df: pd.DataFrame, *, wage_col: str,
                                        working_col: str, tag: str) -> None:
    """wage must be > 0 wherever working == 1 (reject; do not silently set log_wage=0)."""
    wk = pd.to_numeric(df[working_col], errors="coerce").fillna(0).astype(int)
    w = pd.to_numeric(df[wage_col], errors="coerce")
    bad = (wk == 1) & ~(w > 0)
    if bad.any():
        raise ValueError(f"{tag}: {int(bad.sum())} working row(s) with {wage_col} <= 0 or missing; "
                         f"wage must be > 0 when {working_col} == 1.")


# Strict tolerance for ALL prior identities: pure absolute, no relative slack
# (default np.allclose rtol=1e-5 would accept ~1e-6 identity errors).
_PRIOR_RTOL, _PRIOR_ATOL = 0.0, 1e-9


def _components_zero_on_nonworking(df: pd.DataFrame, *, working_col: str,
                                   logq_cols: Sequence[str], tag: str) -> None:
    """Require log_q_H/W/Occ == 0.0 on every non-working row (Wave-0.1 component-zero)."""
    nw = (pd.to_numeric(df[working_col], errors="coerce").fillna(0).astype(int) == 0).to_numpy()
    for c in logq_cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
        bad = int((nw & (v != 0.0)).sum())
        if bad:
            raise ValueError(f"{tag}: {bad} non-working row(s) with {c} != 0.0 "
                             "(Wave-0.1 component-zero violated).")


def _validate_wave01(df: pd.DataFrame, *, household_type: str) -> None:
    """Wave-0.1 proposal-density contract, AFTER state recomputation (gap 2).
    All identity comparisons use rtol=0.0, atol=1e-9 (no relative slack)."""
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")  # noqa: E731
    ac = lambda a, b: np.allclose(a, b, rtol=_PRIOR_RTOL, atol=_PRIOR_ATOL)          # noqa: E731
    lp = num("log_prior")
    prior = num("prior")
    if not (prior > 0).all():
        raise ValueError(f"{household_type}: prior must be strictly > 0.")
    if not ac(prior, _canonical_prior(lp)):
        raise ValueError(f"{household_type}: prior != clip(exp(clip(log_prior,-700,700)),1e-16,None).")
    if household_type == "singles":
        _components_zero_on_nonworking(df, working_col="working",
                                       logq_cols=["log_q_H", "log_q_W", "log_q_Occ"], tag="singles")
        ident = num("log_q_E") + num("working") * (num("log_q_H") + num("log_q_W") + num("log_q_Occ"))
        if not ac(lp, ident):
            raise ValueError("singles: log_prior != log_q_E + working*(log_q_H+log_q_W+log_q_Occ).")
    else:
        for s in ("male", "female"):
            _components_zero_on_nonworking(df, working_col=f"working_{s}",
                                           logq_cols=[f"log_q_H_{s}", f"log_q_W_{s}", f"log_q_Occ_{s}"],
                                           tag=f"couples {s}")
            ident = (num(f"log_q_E_{s}") + num(f"working_{s}")
                     * (num(f"log_q_H_{s}") + num(f"log_q_W_{s}") + num(f"log_q_Occ_{s}")))
            if not ac(num(f"log_prior_{s}"), ident):
                raise ValueError(f"couples: log_prior_{s} != log_q_E_{s} + working_{s}*(H+W+Occ).")
        if not ac(lp, num("log_prior_male") + num("log_prior_female")):
            raise ValueError("couples: joint log_prior != log_prior_male + log_prior_female.")


def _validate_occupation_state(df: pd.DataFrame, *, working_col: str, loc4_col: str,
                              onehot_cols: Sequence[str], tag: str) -> None:
    """DE occupation-state contract (gap 3), after state recomputation:
    working -> loc4 in {1,2,3,4} and one-hots sum to exactly 1;
    non-working -> loc4 == -1 and one-hots sum to exactly 0."""
    wk = pd.to_numeric(df[working_col], errors="coerce").fillna(0).astype(int) == 1
    loc4 = pd.to_numeric(df[loc4_col], errors="coerce")
    oh = df[list(onehot_cols)].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    nw = ~wk
    if int((wk & ~loc4.isin([1, 2, 3, 4])).sum()):
        raise ValueError(f"{tag}: working row(s) with {loc4_col} not in {{1,2,3,4}}.")
    if int((wk & (oh != 1)).sum()):
        raise ValueError(f"{tag}: working row(s) with one-hot sum != 1.")
    if int((nw & (loc4 != -1)).sum()):
        raise ValueError(f"{tag}: non-working row(s) with {loc4_col} != -1.")
    if int((nw & (oh != 0)).sum()):
        raise ValueError(f"{tag}: non-working row(s) with one-hot sum != 0.")


# --------------------------------------------------------------------------- #
# DE assembly (singles / couples)                                              #
# --------------------------------------------------------------------------- #
def _validate_one_to_one(features: pd.DataFrame, cons: pd.DataFrame, keys: List[str], tag: str) -> None:
    if features.duplicated(subset=keys).any():
        raise ValueError(f"{tag}: features have duplicate alternative keys {keys}.")
    fk = set(map(tuple, features[keys].itertuples(index=False, name=None)))
    ck = set(map(tuple, cons[keys].itertuples(index=False, name=None)))
    if fk != ck:
        raise ValueError(f"{tag}: alt-key set mismatch features vs priced "
                         f"(missing={sorted(ck - fk)[:5]}, extra={sorted(fk - ck)[:5]}).")


def assemble_singles(
    priced: pd.DataFrame, features: pd.DataFrame, *,
    hh_key: str = "source_idhh", alt_keys: Sequence[str] = ("alt",),
    orighh_col: str = "source_idorighh", income_source: str = "ils_dispy",
    price_factor: float = 1.0,
) -> pd.DataFrame:
    """Engine-ready SINGLES: one row per (hh, alt); consumption = decider disposable income."""
    alt_keys = list(alt_keys)
    keys = [hh_key, *alt_keys]
    feat = features.copy()
    _require_present_finite(feat, _REQUIRED_FEATURES, tag="singles features")  # fix 7
    cons = aggregate_consumption(priced, household_type="singles", hh_key=hh_key,
                                 alt_keys=alt_keys, income_source=income_source, price_factor=price_factor)
    _require_present_finite(cons, ["consumption_raw"], tag="singles aggregated income")
    _validate_one_to_one(feat, cons, keys, "singles")
    out = feat.merge(cons, on=keys, how="inner", validate="one_to_one")
    out = apply_consumption_floor(out)
    out = restore_cluster_id(out, hh_key=hh_key, source_orighh_col=orighh_col)
    out = _recompute_state_from_hours(out)   # working/bands/loc4 consistent with hours
    _require_wage_positive_when_working(out, wage_col="wage", working_col="working", tag="singles")
    hours = pd.to_numeric(out["hours"], errors="coerce").fillna(0.0)
    out["leisure"] = (TOTAL_LEISURE_HOURS - hours).clip(lower=DCM_MIN_POSITIVE)
    out["log_l"] = np.log(out["leisure"])
    out["log_wage"] = _log_wage(out["wage"], out["working"])   # fix 1: preserve wage + log_wage
    out["prior"] = _canonical_prior(pd.to_numeric(out["log_prior"], errors="coerce").to_numpy())
    for k, v in loc4_one_hots(out["loc4"]).items():
        out[k] = v
    out = _normalize(out, leisure_cols=["leisure"])
    out["female"] = (pd.to_numeric(out["dgn"], errors="coerce") == 0).astype("int8")
    out["in_couple"] = np.int8(0)
    out["household_type"] = "single"
    out["idhh"] = out[hh_key]               # fix 2: emit idhh/idorighh, keep provenance source_*
    out["idorighh"] = out[orighh_col]
    for z in _DE_ZERO_STUBS:
        out[z] = 0.0
    out = _validate_and_sort(out, hh_key=hh_key, alt_keys=alt_keys,
                             finite_cols=["leisure", "wage", "log_q_E", "log_prior"], tag="singles")
    _validate_wave01(out, household_type="singles")
    _validate_occupation_state(out, working_col="working", loc4_col="loc4",
                               onehot_cols=["loc4_1", "loc4_2", "loc4_3", "loc4_4"], tag="singles")
    return out


def assemble_couples(
    priced: pd.DataFrame, features: pd.DataFrame, *,
    hh_key: str = "source_idhh", alt_keys: Sequence[str] = ("alt",),
    orighh_col: str = "source_idorighh", dgn_col: str = "dgn",
    income_source: str = "ils_dispy", price_factor: float = 1.0,
) -> pd.DataFrame:
    """Engine-ready COUPLES: one row per joint alternative; consumption = tax-unit SUM.
    Per-partner features widened to _male/_female (dgn==1 male, dgn==0 female)."""
    alt_keys = list(alt_keys)
    keys = [hh_key, *alt_keys]
    cons = aggregate_consumption(priced, household_type="couples", hh_key=hh_key,
                                 alt_keys=alt_keys, income_source=income_source, price_factor=price_factor)

    _require_present_finite(features, _REQUIRED_FEATURES, tag="couples features")  # fix 7
    feat = _recompute_state_from_hours(features.copy())   # per-partner band/loc4 consistency
    _require_wage_positive_when_working(feat, wage_col="wage", working_col="working", tag="couples")
    g = pd.to_numeric(feat[dgn_col], errors="coerce")
    hrs = pd.to_numeric(feat["hours"], errors="coerce").fillna(0.0)
    feat["leisure"] = (TOTAL_LEISURE_HOURS - hrs).clip(lower=DCM_MIN_POSITIVE)
    feat["log_l"] = np.log(feat["leisure"])
    feat["log_wage"] = _log_wage(feat["wage"], feat["working"])   # fix 1
    for k, v in loc4_one_hots(feat["loc4"]).items():
        feat[k] = v

    # gap 1: exactly two deciders per alternative — exactly one dgn=1 and one dgn=0
    if not g.isin([0, 1]).all():
        bad = sorted(set(g.dropna().tolist()) - {0, 1, 0.0, 1.0})
        raise ValueError(f"couples: dgn must be in {{0,1}}; found other value(s) {bad[:5]}.")
    comp = feat.assign(_m=(g == 1).astype(int), _f=(g == 0).astype(int)).groupby(keys).agg(
        nm=("_m", "sum"), nf=("_f", "sum"), n=("_m", "size"))
    bad = comp[(comp.nm != 1) | (comp.nf != 1) | (comp.n != 2)]
    if len(bad):
        raise ValueError("couples: every alternative needs exactly one dgn=1 + one dgn=0 decider "
                         f"(2 rows); offending {bad.head().to_dict('index')}.")
    # gap 2: is_chosen must be binary BEFORE combining (no masking AND)
    for fr, who in ((feat[g == 1], "male"), (feat[g == 0], "female")):
        isc = pd.to_numeric(fr["is_chosen"], errors="coerce").to_numpy()
        if not np.isin(isc, [0, 1]).all():
            raise ValueError(f"couples {who}: is_chosen must be binary {{0,1}} before combining.")

    present = [c for c in _PARTNER_COLS if c in feat.columns]
    male, female = feat[g == 1].set_index(keys), feat[g == 0].set_index(keys)
    female = female.loc[male.index]   # aligned (gap 1 guarantees a 1:1 male/female key match)
    wide = pd.DataFrame(index=male.index)
    for col in present:
        wide[f"{col}_male"] = male[col]
        wide[f"{col}_female"] = female[col]
    part_cols = [f"{c}_{s}" for c in present for s in ("male", "female")]
    nbad = int(wide[part_cols].isna().to_numpy().sum())
    if nbad:
        raise ValueError(f"couples: {nbad} null partner field(s) after widening.")
    # gap 2: partner-shared fields must AGREE between male and female (do not combine/mask)
    shared = ["is_chosen", orighh_col, "n_children"] + (["data_year"] if "data_year" in male.columns else [])
    for col in shared:
        if not np.array_equal(male[col].to_numpy(), female[col].to_numpy()):
            raise ValueError(f"couples: partner-shared field '{col}' differs between male and female "
                             "within an alternative.")
    wide["is_chosen"] = pd.to_numeric(male["is_chosen"], errors="coerce").astype(int).values  # shared
    wide["n_children"] = pd.to_numeric(male["n_children"], errors="coerce").fillna(0).values
    wide[dgn_col] = -1
    wide[orighh_col] = male[orighh_col].values
    if "data_year" in male.columns:
        wide["data_year"] = male["data_year"].values
    wide = wide.reset_index()

    # joint proposal density: log_prior = male + female ; per-partner log_q carried unchanged
    wide["log_prior"] = (pd.to_numeric(wide["log_prior_male"], errors="coerce").fillna(0.0)
                         + pd.to_numeric(wide["log_prior_female"], errors="coerce").fillna(0.0))
    wide["prior"] = np.clip(np.exp(np.clip(wide["log_prior"].to_numpy(), -700, 700)), 1e-16, None)

    _validate_one_to_one(wide, cons, keys, "couples")
    out = wide.merge(cons, on=keys, how="inner", validate="one_to_one")
    out = apply_consumption_floor(out)
    out = restore_cluster_id(out, hh_key=hh_key, source_orighh_col=orighh_col)
    out = _normalize(out, leisure_cols=["leisure_male", "leisure_female"])
    out["household_type"] = "couple"
    out["idhh"] = out[hh_key]               # fix 2
    out["idorighh"] = out[orighh_col]
    for z in _DE_ZERO_STUBS:
        out[z] = 0.0
    # per-partner structural fields the couples PrecomputedData contract requires
    # (surfaced by the precompute smoke): completed here so the loader need not stub them.
    out["female_male"] = np.int8(0); out["female_female"] = np.int8(1)
    out["in_couple_male"] = np.int8(1); out["in_couple_female"] = np.int8(1)
    out["drgn1_male"] = 0.0; out["drgn1_female"] = 0.0
    out["gsur_male"] = 0.0; out["gsur_female"] = 0.0
    out = _validate_and_sort(out, hh_key=hh_key, alt_keys=alt_keys,
                             finite_cols=["leisure_male", "leisure_female", "wage_male",
                                          "wage_female", "log_q_E_male", "log_prior"], tag="couples")
    _validate_wave01(out, household_type="couples")
    for s in ("male", "female"):
        _validate_occupation_state(out, working_col=f"working_{s}", loc4_col=f"loc4_{s}",
                                   onehot_cols=[f"loc4_1_{s}", f"loc4_2_{s}", f"loc4_3_{s}", f"loc4_4_{s}"],
                                   tag=f"couples {s}")
    return out


def _floor_report(singles: pd.DataFrame, couples: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    def stat(df, chosen_col):
        n = len(df)
        fl = int(df["consumption_floored"].sum())
        ch = df[chosen_col] == 1 if chosen_col in df.columns else pd.Series(False, index=df.index)
        return {"n": n, "floored": fl, "floored_share": (fl / n if n else 0.0),
                "chosen_floored": int((df["consumption_floored"] & ch).sum())}
    return {"singles": stat(singles, "is_chosen"), "couples": stat(couples, "is_chosen")}


def assemble(
    priced_singles: pd.DataFrame, features_singles: pd.DataFrame,
    priced_couples: pd.DataFrame, features_couples: pd.DataFrame,
    *, alt_keys: Sequence[str] = ("alt",), income_source: str = "ils_dispy",
    price_factor: float = 1.0,
) -> EngineReadyResult:
    """Assemble both engine-ready tables + floor report + income-basis metadata."""
    s = assemble_singles(priced_singles, features_singles, alt_keys=alt_keys,
                         income_source=income_source, price_factor=price_factor)
    c = assemble_couples(priced_couples, features_couples, alt_keys=alt_keys,
                        income_source=income_source, price_factor=price_factor)
    meta = {
        "income_source": income_source,
        "price_factor": price_factor,
        "consumption_basis": "nominal DE-2017 euros (no CPI deflation; no ils_dispy_real)",
        "floor": {"constant": DCM_MIN_POSITIVE, "rule": "consumption=clip(value,lower=1.0)",
                  "stage": "engine-ready assembly", "excluded": False},
        "aggregation": {"singles": "decider ils_dispy (ruro_decider==1)",
                        "couples": "tax-unit sum over members"},
        "cluster_key": {"cluster_id_col": "cluster_id", "source_col": "source_idorighh"},
        "leisure": f"clip({TOTAL_LEISURE_HOURS}-hours, lower={DCM_MIN_POSITIVE})",
        "de_zero_stubs": list(_DE_ZERO_STUBS),
        "alt_keys": list(alt_keys),
        # non-blocking: DE contract working_pt1 upper bound = 20.5; the legacy FR loader
        # used 21.5. Recorded for traceability; the current DE slice has zero rows in the
        # (20.5, 21.5] band (see integration report), so it is immaterial here.
        "known_discrepancies": {
            "working_pt1_upper": {"de_contract": _PT1[1], "legacy_loader": 21.5,
                                  "blocking": False,
                                  "note": "rows in (20.5,21.5] would differ; 0 in current slice"},
        },
        # fix 5: certified-sidecar normalization / n_draws / row_counts structure
        "normalization": {
            "singles": {"c_scale": float(s["c_scale"].iloc[0]), "l_scale": float(s["l_scale"].iloc[0]),
                        "n_chosen": int((s["is_chosen"] == 1).sum())},
            "couples": {"c_scale": float(c["c_scale"].iloc[0]),
                        "l_male_scale": float(c["l_male_scale"].iloc[0]),
                        "l_female_scale": float(c["l_female_scale"].iloc[0]),
                        "n_chosen": int((c["is_chosen"] == 1).sum())},
        },
        "n_draws": {"singles": int(s.groupby("source_idhh").size().iloc[0]),
                    "couples": int(c.groupby("source_idhh").size().iloc[0])},
        "row_counts": {"singles": int(len(s)), "couples": int(len(c)),
                       "total": int(len(s) + len(c))},
    }
    return EngineReadyResult(singles=s, couples=c, floor_report=_floor_report(s, c), metadata=meta)
