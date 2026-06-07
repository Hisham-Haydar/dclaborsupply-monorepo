"""Package-native, country-general, spec-aware engine-ready loader.

Constructs the core ``PrecomputedDataSingles`` / ``PrecomputedDataCouples`` containers
(``likelihood/_numpy_primitives``) from harmonised *engine-ready* parquet files or
DataFrames, replacing the read-only dependency on
``MNL/scripts/enhanced/estimation_utils.py``.

Boundary discipline: imports ONLY stdlib, numpy, and pandas. NEVER imports
jax/gamspy/EUROMOD/Java, nor any MNL/app/France code. (``_numpy_primitives`` is numpy-only.)

Contract
--------
* Spec-aware: every variable referenced by the spec's utility / hours / wage / market /
  occupation shifters (resolved per group and gender) MUST resolve to a present, finite
  source column, else a clear error is raised. Wage/occupation loading is DERIVED FROM THE
  SPEC (a caller cannot disable a dimension the spec requires).
* No silent coercion: a column that is PRESENT but holds NaN / non-numeric / non-finite
  values ALWAYS raises. Zero/None defaults are used ONLY for columns that are both absent
  AND not referenced by the spec (unused structural fields).
* Hours bands are EXPLICIT via ``hours_band_policy``:
    - ``"assembled"`` — read and validate the assembled working/working_pt1/pt2/ft/lh
      columns (the engine-ready file is the source of truth).
    - ``"legacy_certified"`` — re-derive working/pt1/pt2/ft from ``hours`` using the
      HISTORICAL cutoffs below and read working_lh. These cutoffs (notably PT1 upper 21.5)
      are a legacy artifact, NOT a universal country-general definition; this policy exists
      to reproduce the certified FR/DE baselines, which were produced this way. The on-disk
      FR working_pt1 column disagrees with this re-derivation (~18.4k rows; ~859 nats at
      theta_hat), so the FR certified figure reproduces ONLY under ``legacy_certified``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from dclaborsupply.likelihood._numpy_primitives import (
    EPS,
    PrecomputedDataSingles,
    PrecomputedDataCouples,
)

# HISTORICAL focal-hours peak cutoffs (legacy_certified policy only — NOT universal).
_LEGACY_PT1 = (18.5, 21.5)
_LEGACY_PT2 = (29.5, 30.5)
_LEGACY_FT = (37.5, 40.5)

_HOURS_BANDS = ("working", "working_pt1", "working_pt2", "working_ft", "working_lh")
_LOC4_ONEHOTS = {"loc4_1", "loc4_2", "loc4_3", "loc4_4"}
_POLICIES = ("assembled", "legacy_certified")

DataSource = Union[pd.DataFrame, str, Path]


# --------------------------------------------------------------------------- #
# IO + metadata
# --------------------------------------------------------------------------- #
def _as_df(source: DataSource) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"engine-ready parquet not found: {path}")
    return pd.read_parquet(path)   # needs a pandas parquet engine; see [parquet] extra


def _load_metadata(metadata: Union[Dict[str, Any], str, Path, None], source: DataSource) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return metadata
    if metadata is not None:
        return json.loads(Path(metadata).read_text())
    if not isinstance(source, pd.DataFrame):
        p = Path(source)
        for marker in ("__singles", "__couples"):
            if marker in p.name:
                cand = p.with_name(p.name.split(marker)[0] + "__mnlmeta.json")
                if cand.exists():
                    return json.loads(cand.read_text())
    raise ValueError("metadata not provided and no sibling __mnlmeta.json could be inferred.")


def _scales(metadata: Dict[str, Any], group: str) -> Tuple[float, float]:
    if "normalization" not in metadata:
        raise ValueError(f"{group}: metadata has no 'normalization' block.")
    norm = metadata["normalization"]
    g = norm[group] if group in norm else norm
    c = float(g["c_scale"])
    l = float(g["l_scale"] if "l_scale" in g else g["l_male_scale"])
    checks = [("c_scale", c), ("l_scale", l)]
    if group == "couples":
        # validate BOTH per-gender leisure scales, not just the male one used downstream.
        checks = [("c_scale", c), ("l_male_scale", float(g["l_male_scale"])),
                  ("l_female_scale", float(g["l_female_scale"]))]
    for nm, val in checks:
        if not (np.isfinite(val) and val > 0):
            raise ValueError(f"{group}: metadata normalization {nm}={val!r} must be finite and > 0.")
    return c, l


# --------------------------------------------------------------------------- #
# column readers — strict (no silent coercion) vs explicit-optional
# --------------------------------------------------------------------------- #
def _present_finite(df: pd.DataFrame, name: str, tag: str) -> np.ndarray:
    """A PRESENT column converted to float; raises on any NaN/non-numeric/non-finite."""
    a = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")
    nbad = int((~np.isfinite(a)).sum())
    if nbad:
        raise ValueError(f"{tag}: column '{name}' has {nbad} NaN / non-numeric / non-finite value(s).")
    return a


def _strict(df: pd.DataFrame, name: str, tag: str) -> np.ndarray:
    if name not in df.columns:
        raise ValueError(f"{tag}: required column '{name}' is missing.")
    return _present_finite(df, name, tag)


def _optional(df: pd.DataFrame, name: str, tag: str, default: float = 0.0) -> np.ndarray:
    """Absent -> constant default (only valid for unused structural fields).
    Present -> validated (NaN/non-finite still raise)."""
    if name not in df.columns:
        return np.full(len(df), default, dtype="float64")
    return _present_finite(df, name, tag)


def _col(df: pd.DataFrame, name: str, required: Set[str], tag: str) -> np.ndarray:
    return _strict(df, name, tag) if name in required else _optional(df, name, tag)


def _opt_or_none(df: pd.DataFrame, name: str, required: Set[str], tag: str) -> Optional[np.ndarray]:
    """For Optional dataclass fields (wage/pexp): strict if required, validated if present, else None."""
    if name in required:
        return _strict(df, name, tag)
    if name in df.columns:
        return _present_finite(df, name, tag)
    return None


def _require_positive(a: np.ndarray, name: str, tag: str) -> np.ndarray:
    """Strictly positive (no silent EPS clipping). Finiteness already enforced upstream."""
    nbad = int((a <= 0).sum())
    if nbad:
        raise ValueError(f"{tag}: '{name}' must be finite and strictly positive; found {nbad} value(s) <= 0.")
    return a


def _wage_log(wage: Optional[np.ndarray], working: np.ndarray, name: str, tag: str) -> Optional[np.ndarray]:
    """Validate wage non-negative everywhere and strictly positive where working==1, then
    return log(max(wage, EPS)) — preserving the zero non-worker convention (masked by
    working in the engine). None passes through."""
    if wage is None:
        return None
    nneg = int((wage < 0).sum())
    if nneg:
        raise ValueError(f"{tag}: '{name}' must be non-negative; found {nneg} negative value(s).")
    bad = (working == 1) & (wage <= 0)
    if bad.any():
        raise ValueError(f"{tag}: '{name}' must be > 0 where the working flag is 1; "
                         f"found {int(bad.sum())} working row(s) with {name} <= 0.")
    return np.log(np.maximum(wage, EPS))


def _validate_binary(a: np.ndarray, name: str, tag: str) -> np.ndarray:
    if not np.isin(a, [0.0, 1.0]).all():
        raise ValueError(f"{tag}: column '{name}' must be binary {{0,1}}.")
    return a


# --------------------------------------------------------------------------- #
# spec-aware requirement resolution
# --------------------------------------------------------------------------- #
def _is_reg(v: str) -> bool:
    return v.startswith("reg") and v[3:].isdigit()


def _need(var: str, suffix: str, policy: str, req: Set[str], flags: Dict[str, Any]) -> None:
    """Add the source column(s) for one spec variable on one route (suffix
    ""=unsuffixed/household, "_male", "_female"). Mirrors engine variable resolution:
    bands<-hours/band col by policy; loc4_k<-loc4; pexp<-pexp_years; reg{k}<-region;
    gsur<-gsur flag (per route); else the (suffixed) source column."""
    if var in _HOURS_BANDS:
        if policy == "legacy_certified":
            req.add(f"hours{suffix}")
            if var == "working_lh":
                req.add(f"working_lh{suffix}")
        else:
            req.add(f"{var}{suffix}")
    elif var in _LOC4_ONEHOTS:
        req.add(f"loc4{suffix}")
    elif var in ("pexp_years", "pexp_years2"):
        req.add(f"pexp_years{suffix}")
    elif _is_reg(var):
        flags["region"] = True
    elif var == "gsur":
        flags["gsur"].add(suffix)
    else:
        req.add(f"{var}{suffix}")


def _requirements(spec: Any, *, couples: bool, policy: str, is_male: Optional[bool]
                  ) -> Tuple[Set[str], bool, Set[str]]:
    """Resolve required source columns by MIRRORING the engine's shifter routing
    (`applies_to` + gender_specific). Returns (cols, region_required, gsur_route_suffixes)."""
    if policy not in _POLICIES:
        raise ValueError(f"hours_band_policy must be one of {_POLICIES}; got {policy!r}.")
    req: Set[str] = ({"c_norm", "l_norm_male", "l_norm_female", "prior", "is_chosen", "idhh"}
                     if couples else {"c_norm", "l_norm", "prior", "is_chosen", "idhh"})
    flags: Dict[str, Any] = {"region": False, "gsur": set()}
    sides = ("_male", "_female") if couples else ("",)

    # leisure shifters. n_children: engine skips it for singles MALE only when the shifter
    # is gender_specific=true; if NOT gender-specific, male n_children IS read (loaded).
    for sh in spec.utility_leisure_shifters:
        var = sh["variable"]
        if var == "n_children":
            gs = bool(sh.get("gender_specific", False))
            if couples or not (gs and is_male):
                req.add("n_children")
            continue
        for s in sides:
            _need(var, s, policy, req, flags)
    # hours shifters (both partners for couples)
    for sh in spec.hours_shifters:
        for s in sides:
            _need(sh["variable"], s, policy, req, flags)
    # wage mean shifters + observed wage (only when wage is variable)
    if getattr(spec, "wage_spec", "fw") != "fw":
        for sh in spec.wage_mean_shifters:
            var = sh.get("variable")
            if var and var != "intercept":
                for s in sides:
                    _need(var, s, policy, req, flags)
        for s in sides:
            req.add(f"wage{s}")
    # market/occupation shifters with EXACT applies_to routing (mirrors engine_jax)
    for sh in (getattr(spec, "market_opportunity_shifters", None) or []):
        var = sh["variable"]
        applies = str(sh.get("applies_to", "both")).strip().lower()
        if couples:
            if applies == "household":
                _need(var, "", policy, req, flags)
            else:
                if applies in ("male", "cm", "both"):
                    _need(var, "_male", policy, req, flags)
                if applies in ("female", "cf", "both"):
                    _need(var, "_female", policy, req, flags)
        else:  # singles: skip cm/cf; honour male/sm & female/sf vs is_male
            if applies in {"cm", "cf"}:
                continue
            if applies in {"male", "sm"} and not is_male:
                continue
            if applies in {"female", "sf"} and is_male:
                continue
            _need(var, "", policy, req, flags)
    return req, flags["region"], flags["gsur"]


def _check_presence(df: pd.DataFrame, cols: Set[str], region: bool, gsur_suffixes: Set[str],
                    tag: str) -> None:
    missing = [c for c in sorted(cols) if c not in df.columns]
    if region:
        has_all_reg = all(f"reg_nuts1_{k}" in df.columns for k in range(2, 9))
        if not (has_all_reg or "drgn1" in df.columns):
            missing.append("ALL reg_nuts1_2..8 OR drgn1 (spec references region dummies)")
    for s in sorted(gsur_suffixes):
        if not (f"gsur{s}" in df.columns or f"u_rate{s}" in df.columns):
            missing.append(f"gsur{s} | u_rate{s} (spec references gsur)")
    if missing:
        raise ValueError(f"{tag}: missing spec-required variable column(s): {missing}.")


def _engine_attr_names(spec: Any, *, couples: bool, is_male: Optional[bool]) -> Set[str]:
    """The EXACT attribute names the engine resolves via getattr(data, name) for each
    ACTIVE shifter variable — mirroring engine_jax routing. Used for the post-construction
    binding audit: every name here must be a non-None attribute on the returned object."""
    names: Set[str] = set()

    def emit(var: str):
        names.update([f"{var}_male", f"{var}_female"]) if couples else names.add(var)

    for sh in spec.utility_leisure_shifters:
        var = sh["variable"]
        if var == "n_children":
            gs = bool(sh.get("gender_specific", False))
            if couples or not (gs and is_male):
                names.add("n_children")
            continue
        emit(var)
    for sh in spec.hours_shifters:
        emit(sh["variable"])
    if getattr(spec, "wage_spec", "fw") != "fw":
        for sh in spec.wage_mean_shifters:
            v = sh.get("variable")
            if v and v != "intercept":
                emit(v)
    for sh in (getattr(spec, "market_opportunity_shifters", None) or []):
        var = sh["variable"]
        applies = str(sh.get("applies_to", "both")).strip().lower()
        if couples:
            if applies == "household":
                names.add(var)
            else:
                if applies in ("male", "cm", "both"):
                    names.add(f"{var}_male")
                if applies in ("female", "cf", "both"):
                    names.add(f"{var}_female")
        else:
            if applies in {"cm", "cf"}:
                continue
            if applies in {"male", "sm"} and not is_male:
                continue
            if applies in {"female", "sf"} and is_male:
                continue
            names.add(var)
    return names


def _bind_and_audit(data: Any, df: pd.DataFrame, spec: Any, *, couples: bool,
                    is_male: Optional[bool], tag: str) -> None:
    """Ensure every active spec variable is bound on the returned object under the exact
    engine-resolved attribute name. Standard fields are already set; arbitrary
    parser-approved engine-ready variables (e.g. hours_bin_1, isco1_1) are attached here
    from their validated column. An active variable that resolves to neither a loaded
    field nor an available column is REJECTED — never accepted and silently dropped."""
    for name in sorted(_engine_attr_names(spec, couples=couples, is_male=is_male)):
        if getattr(data, name, None) is not None:
            continue
        if name in df.columns:
            setattr(data, name, _present_finite(df, name, tag))   # attach custom validated array
        else:
            raise ValueError(
                f"{tag}: active spec variable resolves to engine attribute '{name}', which is "
                "neither a loaded field nor an available engine-ready column — unsupported "
                "(it would be silently skipped by the engine). Add the column or remove it from the spec."
            )


# --------------------------------------------------------------------------- #
# grouping + structural validation
# --------------------------------------------------------------------------- #
def _group_bounds(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous choice-set bounds. Group key = idhh, combined with year_tag when pooled.
    Collision-free: detects group changes on the (idhh, year_tag) PAIR directly (no
    idhh*10+year_tag hash). Stable-sorts by the key first so chosen-first order is kept."""
    pooled = "year_tag" in df.columns and df["year_tag"].nunique() > 1
    sort_keys = ["idhh", "year_tag"] if pooled else ["idhh"]
    df.sort_values(sort_keys, kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)
    idhh = df["idhh"].to_numpy()
    if pooled:
        yt = df["year_tag"].to_numpy()
        change = (idhh[1:] != idhh[:-1]) | (yt[1:] != yt[:-1])
    else:
        change = idhh[1:] != idhh[:-1]
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    ends = np.concatenate([starts[1:], [len(df)]])
    if pooled:
        # genuinely-unique group_ids: collision-free factorization of (idhh, year_tag)
        # at the group starts (idhh alone repeats across years).
        pair = np.column_stack([idhh[starts], yt[starts]])
        _, group_ids = np.unique(pair, axis=0, return_inverse=True)
        group_ids = group_ids.astype("int64")
    else:
        group_ids = idhh[starts]   # one idhh per group -> already unique
    return starts, ends, group_ids


def _validate_groups(starts: np.ndarray, ends: np.ndarray, n_obs: int, tag: str) -> int:
    sizes = ends - starts
    if not np.all(ends > starts):
        raise ValueError(f"{tag}: empty/invalid group(s) detected.")
    if sizes.min() != sizes.max():
        raise ValueError(f"{tag}: non-constant alternatives per group "
                         f"({int(sizes.min())}..{int(sizes.max())}); engine needs rectangular sets.")
    n_alts = int(sizes[0])
    if n_obs % len(starts) != 0 or n_obs // len(starts) != n_alts:
        raise ValueError(f"{tag}: n_obs={n_obs} not divisible into {len(starts)} equal groups.")
    return n_alts


def _validate_chosen_first(df: pd.DataFrame, starts: np.ndarray, ends: np.ndarray, tag: str) -> np.ndarray:
    isc = pd.to_numeric(df["is_chosen"], errors="coerce").fillna(-1).to_numpy()
    if not np.isin(isc, [0, 1]).all():
        raise ValueError(f"{tag}: is_chosen must be binary {{0,1}}.")
    counts = np.array([isc[s:e].sum() for s, e in zip(starts, ends)])
    if not np.all(counts == 1):
        raise ValueError(f"{tag}: {int(np.sum(counts != 1))} group(s) without exactly one chosen.")
    if not np.all(isc[starts] == 1):
        raise ValueError(f"{tag}: {int(np.sum(isc[starts] != 1))} group(s) where chosen is not "
                         "first (column-0 contract for use_actual_choice=False).")
    return isc


def _cluster_ids(df: pd.DataFrame, starts: np.ndarray, ends: np.ndarray,
                 metadata: Dict[str, Any], tag: str) -> np.ndarray:
    ck = metadata.get("cluster_key", {}) or {}
    for cand in (ck.get("cluster_id_col", "cluster_id"), ck.get("source_col", "idorighh"),
                 "idorighh", "idhh"):
        if cand in df.columns:
            vals = pd.to_numeric(df[cand], errors="coerce").to_numpy()
            for s, e in zip(starts, ends):
                seg = vals[s:e]
                if not np.all(np.isfinite(seg)) or seg.min() != seg.max():
                    raise ValueError(f"{tag}: cluster id '{cand}' not constant within a choice group.")
            return vals[starts]
    raise ValueError(f"{tag}: no cluster identity column found (cluster_id / idorighh / idhh).")


def _validate_dgn(df: pd.DataFrame, is_male: bool, tag: str) -> None:
    if "dgn" not in df.columns:
        return
    d = pd.to_numeric(df["dgn"], errors="coerce").to_numpy()
    want = 1 if is_male else 0
    if not np.all(d == want):
        raise ValueError(f"{tag}: dgn must all equal {want} for is_male={is_male}; "
                         f"found {sorted(set(d[~np.isnan(d)].tolist()))[:5]}.")


# --------------------------------------------------------------------------- #
# derived families: hours bands, loc4 one-hots, region dummies
# --------------------------------------------------------------------------- #
def _bands(df: pd.DataFrame, suffix: str, policy: str, required: Set[str], tag: str) -> Dict[str, np.ndarray]:
    if policy == "assembled":
        out = {}
        for b in _HOURS_BANDS:
            col = f"{b}{suffix}"
            a = _col(df, col, required, tag)
            if col in df.columns:
                _validate_binary(a, col, tag)
            out[b] = a
        return out
    # legacy_certified: re-derive working/pt1/pt2/ft from hours; read working_lh from column
    h = _strict(df, f"hours{suffix}", tag)
    lh = _col(df, f"working_lh{suffix}", required, tag)
    if f"working_lh{suffix}" in df.columns:
        _validate_binary(lh, f"working_lh{suffix}", tag)
    return {
        "working": (h > 0).astype("float64"),
        "working_pt1": ((h >= _LEGACY_PT1[0]) & (h <= _LEGACY_PT1[1])).astype("float64"),
        "working_pt2": ((h >= _LEGACY_PT2[0]) & (h <= _LEGACY_PT2[1])).astype("float64"),
        "working_ft": ((h >= _LEGACY_FT[0]) & (h <= _LEGACY_FT[1])).astype("float64"),
        "working_lh": lh,
    }


def _loc4(df: pd.DataFrame, col: str, required: Set[str], tag: str
          ) -> Tuple[Optional[np.ndarray], ...]:
    v = _opt_or_none(df, col, required, tag)
    if v is None:
        return (None, None, None, None, None)
    return (v, *[(v == k).astype("float64") for k in (1, 2, 3, 4)])


def _region_dummies(df: pd.DataFrame, tag: str) -> Dict[str, np.ndarray]:
    """reg2..8 from EITHER all seven reg_nuts1_2..8 columns OR a drgn1 fallback. A PARTIAL
    direct dummy set (some but not all reg_nuts1_*, and no drgn1) raises clearly."""
    present = [k for k in range(2, 9) if f"reg_nuts1_{k}" in df.columns]
    if len(present) == 7:
        return {f"reg{k}": _present_finite(df, f"reg_nuts1_{k}", tag) for k in range(2, 9)}
    if "drgn1" in df.columns:
        d = _present_finite(df, "drgn1", tag)
        return {f"reg{k}": (d == k).astype("float64") for k in range(2, 9)}
    if present:
        raise ValueError(f"{tag}: partial region dummy set {[f'reg_nuts1_{k}' for k in present]} "
                         "present; require ALL reg_nuts1_2..8 OR a complete drgn1 fallback.")
    return {f"reg{k}": np.zeros(len(df)) for k in range(2, 9)}


def _drgn1(df: pd.DataFrame, tag: str) -> np.ndarray:
    if "drgn1" in df.columns:
        return _present_finite(df, "drgn1", tag)
    if "drgn" in df.columns:
        return (_present_finite(df, "drgn", tag) == 1).astype("float64")
    return np.zeros(len(df))


def _gsur(df: pd.DataFrame, suffix: str, tag: str) -> np.ndarray:
    for cand in (f"gsur{suffix}", f"u_rate{suffix}"):
        if cand in df.columns:
            return _present_finite(df, cand, tag)
    return np.zeros(len(df))


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_singles(source: DataSource, spec: Any, *, is_male: bool,
                 metadata: Union[Dict[str, Any], str, Path, None] = None,
                 hours_band_policy: str = "assembled") -> PrecomputedDataSingles:
    """Build PrecomputedDataSingles for one gender from an engine-ready singles frame.
    Wage/occupation loading are derived from ``spec`` (not caller-toggled)."""
    if spec is None:
        raise ValueError("spec is required (pass the EstimationSpec).")
    df = _as_df(source)
    meta = _load_metadata(metadata, source)
    tag = f"singles({'male' if is_male else 'female'})"
    required, region, gsur_suffixes = _requirements(spec, couples=False, policy=hours_band_policy, is_male=is_male)
    _check_presence(df, required, region, gsur_suffixes, tag=tag)

    starts, ends, group_ids = _group_bounds(df)
    n_obs = len(df)
    _validate_groups(starts, ends, n_obs, tag)
    isc = _validate_chosen_first(df, starts, ends, tag)
    _validate_dgn(df, is_male, tag)
    c_scale, l_scale = _scales(meta, "singles")

    # validate finite & strictly positive (raise on bad data — no silent coercion), then
    # apply the EPS log-stability floor that the certified engine contract uses (a no-op
    # for c_norm/l_norm which are >=1; it stabilizes sub-EPS-but-positive prior values).
    consumption = np.maximum(_require_positive(_strict(df, "c_norm", tag), "c_norm", tag), EPS)
    leisure = np.maximum(_require_positive(_strict(df, "l_norm", tag), "l_norm", tag), EPS)
    prior = np.maximum(_require_positive(_strict(df, "prior", tag), "prior", tag), EPS)
    # n_children: zero for male ONLY when the leisure shifter is gender_specific (engine rule).
    nkids_gs = any(sh.get("gender_specific") for sh in spec.utility_leisure_shifters
                   if sh.get("variable") == "n_children")
    bands = _bands(df, "", hours_band_policy, required, tag)
    loc4, loc4_1, loc4_2, loc4_3, loc4_4 = _loc4(df, "loc4", required, tag)
    pexp = _opt_or_none(df, "pexp_years", required, tag)
    wage = _opt_or_none(df, "wage", required, tag)
    reg = _region_dummies(df, tag)

    data = PrecomputedDataSingles(
        consumption=consumption, leisure=leisure, log_c=np.log(consumption), log_l=np.log(leisure),
        age_norm=_col(df, "age_norm", required, tag), age_norm2=_col(df, "age_norm2", required, tag),
        n_children=(np.zeros(n_obs) if (is_male and nkids_gs) else _col(df, "n_children", required, tag)),
        educL=_col(df, "educL", required, tag), educM=_col(df, "educM", required, tag),
        educH=_col(df, "educH", required, tag),
        working=bands["working"], working_pt1=bands["working_pt1"], working_pt2=bands["working_pt2"],
        working_ft=bands["working_ft"], working_lh=bands["working_lh"],
        gsur=_gsur(df, "", tag), female=(np.zeros(n_obs) if is_male else np.ones(n_obs)),
        in_couple=np.zeros(n_obs), drgn1=_drgn1(df, tag), **reg,
        drgur=_col(df, "drgur", required, tag), drgmd=_col(df, "drgmd", required, tag),
        drgru=_col(df, "drgru", required, tag),
        year_2015_indicator=_col(df, "year_2015_indicator", required, tag),
        year_2017_indicator=_col(df, "year_2017_indicator", required, tag),
        log_wage=_wage_log(wage, bands["working"], "wage", tag),
        pexp_years=pexp, pexp_years2=(pexp ** 2 if pexp is not None else None),
        loc4=loc4, loc4_1=loc4_1, loc4_2=loc4_2, loc4_3=loc4_3, loc4_4=loc4_4,
        prior=prior, c_scale=c_scale, l_scale=l_scale,
        group_ids=group_ids, group_starts=starts, group_ends=ends,
        n_groups=len(starts), n_obs=n_obs,
        actual_choice=(isc > 0.5).astype("float64"),
        cluster_ids=_cluster_ids(df, starts, ends, meta, tag), is_male=bool(is_male),
    )
    _bind_and_audit(data, df, spec, couples=False, is_male=is_male, tag=tag)
    return data


def load_couples(source: DataSource, spec: Any, *,
                 metadata: Union[Dict[str, Any], str, Path, None] = None,
                 hours_band_policy: str = "assembled") -> PrecomputedDataCouples:
    """Build PrecomputedDataCouples (one row per joint alternative, _male/_female widened)."""
    if spec is None:
        raise ValueError("spec is required (pass the EstimationSpec).")
    df = _as_df(source)
    meta = _load_metadata(metadata, source)
    tag = "couples"
    required, region, gsur_suffixes = _requirements(spec, couples=True, policy=hours_band_policy, is_male=None)
    _check_presence(df, required, region, gsur_suffixes, tag=tag)

    starts, ends, group_ids = _group_bounds(df)
    n_obs = len(df)
    _validate_groups(starts, ends, n_obs, tag)
    isc = _validate_chosen_first(df, starts, ends, tag)
    c_scale, l_scale = _scales(meta, "couples")

    # validate finite & strictly positive, then EPS log-stability floor (see load_singles).
    consumption = np.maximum(_require_positive(_strict(df, "c_norm", tag), "c_norm", tag), EPS)
    leisure_m = np.maximum(_require_positive(_strict(df, "l_norm_male", tag), "l_norm_male", tag), EPS)
    leisure_f = np.maximum(_require_positive(_strict(df, "l_norm_female", tag), "l_norm_female", tag), EPS)
    prior = np.maximum(_require_positive(_strict(df, "prior", tag), "prior", tag), EPS)
    bm = _bands(df, "_male", hours_band_policy, required, tag)
    bf = _bands(df, "_female", hours_band_policy, required, tag)
    loc4_m, l1m, l2m, l3m, l4m = _loc4(df, "loc4_male", required, tag)
    loc4_f, l1f, l2f, l3f, l4f = _loc4(df, "loc4_female", required, tag)
    pexp_m = _opt_or_none(df, "pexp_years_male", required, tag)
    pexp_f = _opt_or_none(df, "pexp_years_female", required, tag)
    wage_m = _opt_or_none(df, "wage_male", required, tag)
    wage_f = _opt_or_none(df, "wage_female", required, tag)
    reg = _region_dummies(df, tag)
    z, one = np.zeros(n_obs), np.ones(n_obs)

    data = PrecomputedDataCouples(
        consumption=consumption, log_c=np.log(consumption),
        leisure_male=leisure_m, log_l_male=np.log(leisure_m),
        leisure_female=leisure_f, log_l_female=np.log(leisure_f),
        age_norm_male=_col(df, "age_norm_male", required, tag), age_norm2_male=_col(df, "age_norm2_male", required, tag),
        educL_male=_col(df, "educL_male", required, tag), educM_male=_col(df, "educM_male", required, tag),
        educH_male=_col(df, "educH_male", required, tag),
        age_norm_female=_col(df, "age_norm_female", required, tag), age_norm2_female=_col(df, "age_norm2_female", required, tag),
        n_children=_col(df, "n_children", required, tag),
        educL_female=_col(df, "educL_female", required, tag), educM_female=_col(df, "educM_female", required, tag),
        educH_female=_col(df, "educH_female", required, tag),
        working_male=bm["working"], working_pt1_male=bm["working_pt1"], working_pt2_male=bm["working_pt2"],
        working_ft_male=bm["working_ft"], working_lh_male=bm["working_lh"], gsur_male=_gsur(df, "_male", tag),
        working_female=bf["working"], working_pt1_female=bf["working_pt1"], working_pt2_female=bf["working_pt2"],
        working_ft_female=bf["working_ft"], working_lh_female=bf["working_lh"], gsur_female=_gsur(df, "_female", tag),
        female_male=z, female_female=one, in_couple_male=one, in_couple_female=one,
        drgn1_male=_drgn1(df, tag), drgn1_female=_drgn1(df, tag), **reg,
        drgur=_col(df, "drgur", required, tag), drgmd=_col(df, "drgmd", required, tag),
        drgru=_col(df, "drgru", required, tag),
        year_2015_indicator=_col(df, "year_2015_indicator", required, tag),
        year_2017_indicator=_col(df, "year_2017_indicator", required, tag),
        log_wage_male=_wage_log(wage_m, bm["working"], "wage_male", tag),
        pexp_years_male=pexp_m, pexp_years2_male=(pexp_m ** 2 if pexp_m is not None else None),
        loc4_male=loc4_m, loc4_1_male=l1m, loc4_2_male=l2m, loc4_3_male=l3m, loc4_4_male=l4m,
        log_wage_female=_wage_log(wage_f, bf["working"], "wage_female", tag),
        pexp_years_female=pexp_f, pexp_years2_female=(pexp_f ** 2 if pexp_f is not None else None),
        loc4_female=loc4_f, loc4_1_female=l1f, loc4_2_female=l2f, loc4_3_female=l3f, loc4_4_female=l4f,
        prior=prior, c_scale=c_scale, l_scale=l_scale,
        group_ids=group_ids, group_starts=starts, group_ends=ends,
        n_groups=len(starts), n_obs=n_obs,
        actual_choice=(isc > 0.5).astype("float64"),
        cluster_ids=_cluster_ids(df, starts, ends, meta, tag),
    )
    _bind_and_audit(data, df, spec, couples=True, is_male=None, tag=tag)
    return data


def load_engine_ready_stem(
    stem: Union[str, Path], spec: Any, *, year_tags: Optional[Sequence[int]] = None,
    hours_band_policy: str = "assembled",
) -> Tuple[PrecomputedDataSingles, PrecomputedDataSingles, PrecomputedDataCouples]:
    """Load (singles_male, singles_female, couples) from a
    ``<stem>__{singles,couples}.parquet`` + ``<stem>__mnlmeta.json`` bundle. Singles are
    split by ``dgn`` (1=male, 0=female). ``year_tags`` optionally restricts pooled data."""
    stem = Path(stem)
    meta = json.loads(Path(f"{stem}__mnlmeta.json").read_text())
    s = pd.read_parquet(f"{stem}__singles.parquet")
    c = pd.read_parquet(f"{stem}__couples.parquet")
    if year_tags is not None:
        yt = list(year_tags)
        if "year_tag" not in s.columns or "year_tag" not in c.columns:
            raise ValueError("year_tags requested but 'year_tag' column is absent.")
        s = s[s["year_tag"].isin(yt)].copy()
        c = c[c["year_tag"].isin(yt)].copy()
        if len(s) == 0 or len(c) == 0:
            raise ValueError(f"year_tags={yt} produced empty singles ({len(s)}) or couples ({len(c)}).")
    if "dgn" not in s.columns:
        raise ValueError("singles frame lacks 'dgn'; cannot split by gender.")
    sm = load_singles(s[s["dgn"] == 1].reset_index(drop=True), spec, is_male=True,
                      metadata=meta, hours_band_policy=hours_band_policy)
    sf = load_singles(s[s["dgn"] == 0].reset_index(drop=True), spec, is_male=False,
                      metadata=meta, hours_band_policy=hours_band_policy)
    cou = load_couples(c, spec, metadata=meta, hours_band_policy=hours_band_policy)
    return sm, sf, cou
