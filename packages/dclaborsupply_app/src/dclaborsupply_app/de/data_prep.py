"""DE 2017 data-prep adapter: DE_2017_a2 -> processed singles / couples.

FROZEN DECISIONS (confirmed against MNL source, read-only; user-approved 2026-06-06)
==================================================================================

Eligibility rule = FULL FR MIRROR (user choice). The DE analytical sample applies
the complete FR ``stepwise_filter_households`` chain (enh_france_data_prep.py:739),
using DE fields, in this order:
  1. Age:        every decider dag in [18, 65]                  (FR DEFAULT_CONFIG age_range)
  2. Education:  every decider dec == 0 (not currently in education)
  3. Retire/dis: drop HH if sum over members of (byr+pdi+poa+psu) > 0
  4. Allowed LES: every decider les in {3, 5, 7} (drops self-employed les=2,
                  pensioner les=4, student les=6, etc. deciders)
  5. Opposite-sex couples only (enforced at classification)
  6. Other members: drop HH if any non-decider is working-age-healthy-not-student
                  (dag in [18,65] & ddi==0 & dec==0) OR has |yem|/|yse| > 50
  7. Hours cap + wage bounds for employee (les==3) deciders:
                  lhw>70 -> 70 ; 5<lhw<=10 -> 10 ; lhw<=5 -> inactive (les=7,
                  lhw=0, zero yem/yse/yemse, blank realised wage) ;
                  drop HH if employee-decider yivwg outside [2, 170].

Worker / occupation / wage conventions (confirmed against enh_RURO_prep.py):
  - is_worker = (les == 3) & (lhw > 0)            (enh_RURO_prep.py:441-460; DE has
                  no `lma` column so the rule reduces exactly to this)
  - loc_ruro:   nonworker -> -1 ; worker -> loc (missing/invalid -> -2)  (:570-587)
  - loc4 (CERTIFIED task-grouping, _collapse_loc_to_loc4 default :590-641):
        4 nonroutine-cognitive <- ISCO {1,2,3}
        3 routine-cognitive    <- ISCO {4}
        2 nonroutine-manual    <- ISCO {5}
        1 routine-manual       <- ISCO {6,7,8,9}      (9 defaults to 1)
       -1 nonworker ;  -2 worker with ISCO 0 / missing (loc_armed flag set for ISCO 0)
  - wage split (item 3, enh_RURO_prep.py:975): wage_for_draws = yivwg (all persons);
                  wage_ruro = wage = where(is_worker, yivwg, 0.0).  NOT aliased.
  - working bands (enh_RURO_prep.py:985-987 / estimation_utils.py:451,716):
        working      = lhw > 0
        working_pt1  = lhw in [18.5, 20.5]
        working_pt2  = lhw in [29.5, 30.5]
        working_ft   = lhw in [37.5, 40.5]
        working_lh   = working & lhw in [44.5, 70]
  - education (use deh, NOT dehde): educL = deh in {0,1,2}; educM = deh in {3,4};
                  educH = deh == 5
  - age_norm = dag - mean(dag over the DE analytical decider sample); age_norm2 = age_norm**2

Decider (item 2): ruro_decider = 1 for the single adult, and for BOTH mutually-linked
couple adults (idpartner symmetric in DE). hh_IsHead/hh_IsPartner are NOT emitted
(DE has no headship evidence); downstream RURO_prep accepts ruro_decider directly
(enh_RURO_prep.py:786-796). dgn is used ONLY at the couples reshape.

DE vs FR data facts: no yem00/yemxp overtime split (single monthly yem; keep
yemse = yem + yse); region/urbanisation (drgn1/drgur/drgmd/drgru) constant 0 in
this dataset -> dropped from the contract (and must be dropped from the DE spec).

Self-contained: pandas + numpy only; no France/MNL imports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Frozen configuration                                                         #
# --------------------------------------------------------------------------- #
DE_CONFIG: Dict[str, object] = {
    "adult_age": 18,
    "age_range": (18, 65),
    "allowed_les": (3, 5, 7),
    "wage_bounds": (2.0, 170.0),
    "other_member_income_threshold": 50.0,
    "hours_cap_high": 70,
    "hours_floor_low": 10,
    "hours_inactive_threshold": 5,
    "retire_cols": ("byr", "pdi", "poa", "psu"),
    "data_year": 2017,
    "system_year": 2016,  # DE_2017_a2 BestMatch system (DE_DataConfig.xml)
    "long_hours_band": (44.5, 70.0),
    "pt1_band": (18.5, 20.5),
    "pt2_band": (29.5, 30.5),
    "ft_band": (37.5, 40.5),
}

# Region/urbanisation fields confirmed constant 0 in DE_2017_a2 -> excluded.
DE_DROPPED_REGION_COLS = ("drgn1", "drgur", "drgmd", "drgru")

DEFAULT_DATA_PATH = Path(
    r"C:\Users\hisham\MNL\EUROMOD-STORAGE\Data\DE\DE_2017_a2.txt"
)
DEFAULT_OUT_DIR = Path(
    r"C:\Users\hisham\MNL\EUROMOD-STORAGE\scratch\staging\de_2017"
)


# --------------------------------------------------------------------------- #
# Worker / occupation primitives (faithful to enh_RURO_prep.py)               #
# --------------------------------------------------------------------------- #
def compute_is_worker(df: pd.DataFrame) -> pd.Series:
    """is_worker = (les == 3) & (lhw > 0).

    DE has no SILC ``lma`` column, so the FR hierarchical rule reduces exactly to
    the employee-and-positive-hours definition (enh_RURO_prep.py:441-460).
    """
    lhw = pd.to_numeric(df.get("lhw", pd.Series(0, index=df.index)), errors="coerce").fillna(0.0)
    les = pd.to_numeric(df.get("les", pd.Series(np.nan, index=df.index)), errors="coerce")
    return les.eq(3) & (lhw > 0.0)


def _build_loc_ruro(df: pd.DataFrame) -> pd.Series:
    """loc_ruro: nonworker -> -1 ; worker -> loc (missing/invalid -> -2)."""
    loc = pd.to_numeric(df["loc"], errors="coerce")
    is_worker = compute_is_worker(df)
    loc_ruro = loc.copy().fillna(-2)
    loc_ruro.loc[~is_worker] = -1
    return loc_ruro.astype("int16")


def collapse_loc_to_loc4(
    loc_ruro: pd.Series,
    *,
    elementary_as_nonroutine_manual: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Collapse ISCO-1d ``loc_ruro`` into the CERTIFIED 4 task groups.

    Returns (loc4, loc_armed). Mirrors enh_RURO_prep.py:_collapse_loc_to_loc4
    EXACTLY under the default ``elementary_as_nonroutine_manual=False``.
    """
    loc = pd.to_numeric(loc_ruro, errors="coerce").fillna(-2).astype(int)
    loc4 = pd.Series(-2, index=loc.index, dtype="int16")  # default: unknown worker
    loc4.loc[loc == -1] = -1                               # nonworker
    loc4.loc[loc.isin([1, 2, 3])] = 4                      # nonroutine cognitive
    loc4.loc[loc == 4] = 3                                 # routine cognitive
    loc4.loc[loc == 5] = 2                                 # nonroutine manual
    loc4.loc[loc.isin([6, 7, 8])] = 1                      # routine manual
    loc4.loc[loc == 9] = 2 if elementary_as_nonroutine_manual else 1
    # ISCO 0 (armed forces) stays -2 (explicit unknown task group); flag it.
    loc_armed = (loc == 0).astype("int8")
    return loc4.astype("int16"), loc_armed


# --------------------------------------------------------------------------- #
# Household classification (DE fields: idpartner + dag + dgn)                  #
# --------------------------------------------------------------------------- #
def classify_households(df: pd.DataFrame, *, adult_age: int = 18) -> pd.DataFrame:
    """Add ``household_class`` and ``ruro_decider`` using DE structure only.

    household_class in {single, couple_mf, excl_same_sex, excl_2adult_no_link,
    excl_3plus_adults, excl_no_adult}. ruro_decider = 1 for the single adult and
    for both mutually-linked adults of an opposite-sex couple. No FR role fields.
    """
    df = df.copy()
    dag = pd.to_numeric(df["dag"], errors="coerce").fillna(-1)
    idp = pd.to_numeric(df["idpartner"], errors="coerce").fillna(0).astype("int64")
    is_adult = dag >= adult_age

    id2partner = dict(zip(df["idperson"].astype("int64"), idp))
    idset = set(df["idperson"].astype("int64"))

    def _mutual(a: int, b: int) -> bool:
        return b != 0 and b in idset and id2partner.get(b, 0) == a

    cls: Dict[int, str] = {}
    for hh, g in df.groupby("idhh"):
        ad = g[g["dag"].astype(float) >= adult_age]
        n = len(ad)
        if n == 0:
            cls[hh] = "excl_no_adult"
        elif n == 1:
            cls[hh] = "single" if int(ad["idpartner"].iloc[0]) == 0 else "excl_2adult_no_link"
        elif n == 2:
            a, b = ad["idperson"].astype("int64").tolist()
            if _mutual(a, b) and _mutual(b, a):
                gens = sorted(pd.to_numeric(ad["dgn"], errors="coerce").tolist())
                cls[hh] = "couple_mf" if gens == [0, 1] else "excl_same_sex"
            else:
                cls[hh] = "excl_2adult_no_link"
        else:
            cls[hh] = "excl_3plus_adults"

    df["household_class"] = df["idhh"].map(cls)
    keep = df["household_class"].isin(["single", "couple_mf"])
    df["ruro_decider"] = (keep & is_adult).astype("int8")
    return df


# --------------------------------------------------------------------------- #
# FR-mirror stepwise eligibility filters                                       #
# --------------------------------------------------------------------------- #
def _drop_hh(df: pd.DataFrame, bad_idhh) -> pd.DataFrame:
    bad = pd.Index(pd.unique(bad_idhh))
    return df[~df["idhh"].isin(bad)].copy()


def _keep_hh_all_deciders(df: pd.DataFrame, cond: pd.Series) -> pd.DataFrame:
    """Keep a household only if EVERY decider satisfies ``cond``.

    Equivalent to FR's separate head+partner steps (net keep-set identical).
    """
    dec = df["ruro_decider"] == 1
    bad = df.loc[dec & ~cond, "idhh"]
    return _drop_hh(df, bad)


def apply_stepwise_filters(
    df: pd.DataFrame, *, config: Dict[str, object] = DE_CONFIG
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the full FR-mirror eligibility chain. Returns (filtered_df, stats)."""
    lo, hi = config["age_range"]  # type: ignore[misc]
    allowed = list(config["allowed_les"])  # type: ignore[arg-type]
    wlo, whi = config["wage_bounds"]  # type: ignore[misc]
    cap = int(config["hours_cap_high"])  # type: ignore[arg-type]
    floor = int(config["hours_floor_low"])  # type: ignore[arg-type]
    inact = int(config["hours_inactive_threshold"])  # type: ignore[arg-type]
    inc_thr = float(config["other_member_income_threshold"])  # type: ignore[arg-type]

    work = df.copy()
    stats = []

    def _stat(step: str) -> None:
        stats.append({"step": step, "households": int(work["idhh"].nunique()),
                      "persons": int(len(work))})

    # Baseline = singles + opposite-sex couples only
    work = work[work["household_class"].isin(["single", "couple_mf"])].copy()
    _stat("Baseline (single + couple_mf)")

    dag = pd.to_numeric(work["dag"], errors="coerce")
    # 1. Age: every decider in [lo, hi]
    work = _keep_hh_all_deciders(work, dag.between(lo, hi))
    _stat("Age (all deciders 18-65)")

    # 2. Education: every decider dec == 0 (not currently in education)
    if "dec" in work.columns:
        dec = pd.to_numeric(work["dec"], errors="coerce")
        work = _keep_hh_all_deciders(work, dec.eq(0))
    _stat("Education (deciders dec==0)")

    # 3. Retirement/disability benefits at household level
    retire_cols = [c for c in config["retire_cols"] if c in work.columns]  # type: ignore[union-attr]
    if retire_cols:
        retire = work[retire_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        work["_retire_sum"] = retire
        hh_retire = work.groupby("idhh")["_retire_sum"].sum()
        bad = hh_retire.index[hh_retire > 0]
        work = _drop_hh(work, bad).drop(columns="_retire_sum")
    _stat("Retirement/Disability (HH sum byr+pdi+poa+psu == 0)")

    # 4. Allowed LES for deciders
    les = pd.to_numeric(work["les"], errors="coerce")
    work = _keep_hh_all_deciders(work, les.isin(allowed))
    _stat("Allowed LES (deciders in {3,5,7})")

    # 5. Opposite-sex already enforced at classification (couple_mf).

    # 6. Other household members: drop HH if any non-decider has work capacity / income
    nondec = work["ruro_decider"] == 0
    dag = pd.to_numeric(work["dag"], errors="coerce")
    ddi = pd.to_numeric(work.get("ddi", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    dec = pd.to_numeric(work.get("dec", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    yem = pd.to_numeric(work.get("yem", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    yse = pd.to_numeric(work.get("yse", pd.Series(0.0, index=work.index)), errors="coerce").fillna(0.0)
    capable = dag.between(lo, hi) & ddi.eq(0) & dec.eq(0)
    earning = (yem > inc_thr) | (yse.abs() > inc_thr)
    bad = work.loc[nondec & (capable | earning), "idhh"]
    work = _drop_hh(work, bad)
    _stat("Other members (no capable/earning non-deciders)")

    # 7. Hours capping + inactive transition + wage-bounds filter (employee deciders)
    dec_mask = work["ruro_decider"] == 1
    les = pd.to_numeric(work["les"], errors="coerce")
    lhw = pd.to_numeric(work["lhw"], errors="coerce").fillna(0.0)
    emp = dec_mask & les.eq(3)

    work.loc[emp & (lhw > cap), "lhw"] = cap                                   # cap high
    lhw = pd.to_numeric(work["lhw"], errors="coerce").fillna(0.0)
    work.loc[emp & (lhw > inact) & (lhw <= floor), "lhw"] = floor             # floor low
    lhw = pd.to_numeric(work["lhw"], errors="coerce").fillna(0.0)

    very_low = emp & (lhw <= inact)
    become_inactive = very_low & les.isin(allowed)
    must_drop = very_low & ~les.isin(allowed)
    if become_inactive.any():
        work.loc[become_inactive, "lhw"] = 0
        work.loc[become_inactive, "les"] = 7
        for c in ("yem", "yse", "yemse"):
            if c in work.columns:
                work.loc[become_inactive, c] = 0.0
    work = _drop_hh(work, work.loc[must_drop, "idhh"])

    # wage bounds on employee-decider yivwg
    if "yivwg" in work.columns:
        dec_mask = work["ruro_decider"] == 1
        les = pd.to_numeric(work["les"], errors="coerce")
        yivwg = pd.to_numeric(work["yivwg"], errors="coerce")
        bad_wage = (dec_mask & les.eq(3) & yivwg.notna()
                    & ((yivwg < wlo) | (yivwg > whi)))
        work = _drop_hh(work, work.loc[bad_wage, "idhh"])
    _stat("Hours cap + wage bounds (employee deciders)")

    return work.reset_index(drop=True), pd.DataFrame(stats)


# --------------------------------------------------------------------------- #
# Feature construction (faithful to enh_RURO_prep.py / estimation_utils.py)    #
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, *, config: Dict[str, object] = DE_CONFIG) -> pd.DataFrame:
    """Add the DE labour-supply contract features to the filtered sample."""
    df = df.copy()
    lhw = pd.to_numeric(df["lhw"], errors="coerce").fillna(0.0)
    deh = pd.to_numeric(df["deh"], errors="coerce")
    dgn = pd.to_numeric(df["dgn"], errors="coerce")
    yem = pd.to_numeric(df.get("yem", 0.0), errors="coerce").fillna(0.0)
    yse = pd.to_numeric(df.get("yse", 0.0), errors="coerce").fillna(0.0)
    yivwg = pd.to_numeric(df["yivwg"], errors="coerce").fillna(0.0)

    # occupation
    if "loc_raw" not in df.columns:
        df["loc_raw"] = df["loc"]
    df["loc_ruro"] = _build_loc_ruro(df)
    loc4, loc_armed = collapse_loc_to_loc4(df["loc_ruro"])
    df["loc4"] = loc4
    df["loc_armed"] = loc_armed

    # worker flags
    is_worker = compute_is_worker(df)
    df["is_worker"] = is_worker.astype("int8")
    df["working"] = (lhw > 0.0).astype("int8")
    p1, p2 = config["pt1_band"]; q1, q2 = config["pt2_band"]  # type: ignore[misc]
    f1, f2 = config["ft_band"]; l1, l2 = config["long_hours_band"]  # type: ignore[misc]
    df["working_pt1"] = ((lhw >= p1) & (lhw <= p2)).astype("int8")
    df["working_pt2"] = ((lhw >= q1) & (lhw <= q2)).astype("int8")
    df["working_ft"] = ((lhw >= f1) & (lhw <= f2)).astype("int8")
    df["working_lh"] = ((df["working"] == 1) & (lhw >= l1) & (lhw <= l2)).astype("int8")

    # wages (item 3): never alias the two concepts
    df["wage_for_draws"] = yivwg
    wage_ruro = np.where(is_worker.to_numpy(), yivwg.to_numpy(), 0.0).astype(float)
    df["wage_ruro"] = wage_ruro
    df["wage"] = wage_ruro

    # earnings identity (DE: single yem; keep yemse = yem + yse)
    df["yemse"] = yem + yse

    # education (use deh)
    df["educL"] = deh.isin([0, 1, 2]).astype("int8")
    df["educM"] = deh.isin([3, 4]).astype("int8")
    df["educH"] = deh.eq(5).astype("int8")

    # demographics
    df["female"] = dgn.eq(0).astype("int8")
    df["in_couple"] = (df["household_class"] == "couple_mf").astype("int8")
    df["ruro_sample"] = ((df["ruro_decider"] == 1)
                         & (pd.to_numeric(df["dag"], errors="coerce") >= int(config["adult_age"]))).astype("int8")

    # age_norm centred on the DE analytical decider sample mean
    dag = pd.to_numeric(df["dag"], errors="coerce")
    sample_mean = float(dag[df["ruro_sample"] == 1].mean())
    df["age_mean_sample"] = sample_mean
    df["age_norm"] = dag - sample_mean
    df["age_norm2"] = df["age_norm"] ** 2

    # children per household
    nkids = (df["dag"].astype(float) < int(config["adult_age"])).groupby(df["idhh"]).transform("sum")
    df["n_children"] = nkids.astype("int16")

    # months worked (prefer liwmy)
    if "liwmy" in df.columns:
        df["months_worked"] = pd.to_numeric(df["liwmy"], errors="coerce").fillna(0.0)

    # year stamps
    df["data_year"] = np.int16(config["data_year"])      # type: ignore[arg-type]
    df["system_year"] = np.int16(config["system_year"])  # type: ignore[arg-type]
    df["input_year"] = np.int16(config["data_year"])     # type: ignore[arg-type]

    # drop constant-0 region/urbanisation fields (no variation in DE_2017_a2)
    df = df.drop(columns=[c for c in DE_DROPPED_REGION_COLS if c in df.columns])

    return df


# --------------------------------------------------------------------------- #
# Couples reshape (dgn used ONLY here)                                          #
# --------------------------------------------------------------------------- #
_RESHAPE_FIELDS = (
    "dag", "age_norm", "age_norm2", "deh", "educL", "educM", "educH",
    "lhw", "yivwg", "wage_for_draws", "wage_ruro", "wage", "yem", "yse", "yemse",
    "loc", "loc_ruro", "loc4", "is_worker", "working",
    "working_pt1", "working_pt2", "working_ft", "working_lh", "les",
)


def reshape_couples_to_wide(
    couples_long: pd.DataFrame, *, fields: Tuple[str, ...] = _RESHAPE_FIELDS
) -> pd.DataFrame:
    """One row per couple household with ``_male`` / ``_female`` decider fields.

    dgn (0=female, 1=male) is used ONLY here to assign roles. Deciders only.
    """
    dec = couples_long[couples_long["ruro_decider"] == 1].copy()
    dec["_role"] = np.where(pd.to_numeric(dec["dgn"], errors="coerce").eq(1), "male", "female")
    present = [f for f in fields if f in dec.columns]
    rows = []
    for hh, g in dec.groupby("idhh"):
        if set(g["_role"]) != {"male", "female"} or len(g) != 2:
            continue  # defensive: opposite-sex pairs only
        row: Dict[str, object] = {"idhh": hh, "n_children": int(g["n_children"].iloc[0])}
        for role in ("male", "female"):
            r = g[g["_role"] == role].iloc[0]
            for f in present:
                row[f"{f}_{role}"] = r[f]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def _read_microdata(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".txt", ".dat"}:
        return pd.read_csv(path, sep="\t")
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)  # type: ignore[arg-type]
    raise ValueError(f"Unsupported microdata format: {path}")


def _write_df(df: pd.DataFrame, base: Path) -> Path:
    try:
        out = base.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = base.with_suffix(".csv")
        df.to_csv(out, index=False, encoding="utf-8")
        return out


def prepare_de_2017(
    data_path: Path | str = DEFAULT_DATA_PATH,
    out_dir: Optional[Path | str] = DEFAULT_OUT_DIR,
    *,
    config: Dict[str, object] = DE_CONFIG,
    write: bool = True,
) -> Dict[str, object]:
    """Produce processed DE 2017 singles / couples (full FR-mirror sample).

    Returns a dict with ``singles``, ``couples`` (per-person long DataFrames),
    ``couples_wide``, ``stats`` (filter funnel), and ``paths`` (if written).
    Outputs are written under scratch/staging only.
    """
    raw = _read_microdata(Path(data_path))
    classified = classify_households(raw, adult_age=int(config["adult_age"]))  # type: ignore[arg-type]
    filtered, stats = apply_stepwise_filters(classified, config=config)
    feats = build_features(filtered, config=config)

    singles = feats[feats["household_class"] == "single"].reset_index(drop=True)
    couples = feats[feats["household_class"] == "couple_mf"].reset_index(drop=True)
    couples_wide = reshape_couples_to_wide(couples)

    result: Dict[str, object] = {
        "singles": singles, "couples": couples, "couples_wide": couples_wide,
        "stats": stats,
    }

    if write and out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "singles": str(_write_df(singles, out / "de_2017_singles")),
            "couples": str(_write_df(couples, out / "de_2017_couples")),
            "couples_wide": str(_write_df(couples_wide, out / "de_2017_couples_wide")),
            "stats": str((out / "de_2017_filter_stats.csv")),
        }
        stats.to_csv(out / "de_2017_filter_stats.csv", index=False)
        result["paths"] = paths

    return result
