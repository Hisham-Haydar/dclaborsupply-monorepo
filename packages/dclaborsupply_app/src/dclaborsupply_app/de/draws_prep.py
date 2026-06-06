"""DE app-layer draws-prep: canonical choice-state view + frozen earnings convention.

WHY
---
The unchanged core ``dclaborsupply.alternatives.continuous.generate_draws_long``
defines the observed working state as ``hours > 0``. The DE adapter (faithful to the
FR ``enh_RURO_prep``) sets ``working = (lhw > 0)`` but ``is_worker = (les==3)&(lhw>0)``.
So non-employee (``les in {5,7}``) deciders with stray positive hours carry an
INCONSISTENT state: ``working=1, is_worker=0, lhw>0, wage_ruro=0, loc4=-1``.

In the processed DE 2017 files this affects 155 deciders (91 singles + 64 couples):
15 below the 5h support floor (the core would RAISE) and 140 within ``[5,70]`` (the
core would SILENTLY treat them as working). Both are wrong — these are unemployed /
inactive people. (``loc4`` and ``wage_ruro`` are already consistent at -1 / 0 because
the adapter derives them from ``is_worker``; the live break is ``lhw``/``working``/``yem``.)

DECISION (2026-06-06): ZERO, not clip; applied to ALL non-worker deciders.
Clipping to 5h would wrongly convert unemployed/inactive people into employed
alternatives. So force the observed/chosen state of every ``is_worker==0`` row to
non-employment before draws. ``wage_for_draws`` is the reliable preserved OFFER-wage
field (RURO still offers hypothetical jobs to non-workers; π₀ depends only on gender).
NOTE: ``yivwg`` is NOT a reliable offer-wage carrier — the unchanged core overwrites
``yivwg`` on simulated rows with the drawn wage (e.g. ``yivwg=0`` on a non-employment
draw while ``wage_for_draws=20``). Raw observed values are preserved in ``*_rawstate``
columns for provenance and for raw-baseline EUROMOD pricing.

FROZEN draw-zero earnings convention
------------------------------------
The chosen alternative (draw 0) is priced with DERIVED earnings, identical to the
rule used for simulated alternatives, so that derived earnings place chosen and
simulated alternatives on a consistent pricing convention:

    yem_chosen = wage * hours * (52/12)      # the core recomputes this at draw 0
    => non-employment (hours == 0) ⇒ yem = 0  (NOT the observed raw/part-year yem)
    yemse = yem + yse                         # DE earnings identity, restored app-side

This intentionally does NOT restore observed raw ``yem`` for the chosen state; the
raw value is kept only in ``yem_rawstate`` for provenance. The convention matches the
unchanged core's draw-0 recompute, so no core edit is needed.

This module is APP-LAYER only (pandas/numpy; no core/france/MNL imports).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# Columns forced to the non-employment value for is_worker==0 rows.
# (Only those present in the frame are touched.)
_ZERO_COLS: tuple[str, ...] = (
    "lhw", "hours",
    "working", "working_pt1", "working_pt2", "working_ft", "working_lh",
    "wage", "wage_ruro",
    "yem",
)
# loc4 is set to the non-employment sentinel (-1); yemse is rebuilt as yem + yse.
_NONEMP_LOC4 = -1

# Couples joint-grid convention:
#   The later pricing smoke uses the FR-COMPARABLE design: N x N SIMULATED joint
#   alternatives PLUS one joint chosen alternative (N=30 -> 30x30 + 1 = 901), matching
#   the certified FR B-pool. A naive full Cartesian product of per-partner alternatives
#   INCLUDING both chosen rows would instead be (N+1) x (N+1) = 961 — that is NOT
#   FR-equivalent and is not the design used. Document the chosen design per run.
COUPLES_GRID_NOTE = (
    "Pricing smoke uses FR-comparable couples design: NxN simulated joint + 1 joint "
    "chosen (N=30 -> 901, matches FR B-pool). The naive (N+1)x(N+1)=961 full product "
    "incl. both chosen rows is NOT FR-equivalent and is not used."
)


def canonicalize_choice_state(
    df: pd.DataFrame,
    *,
    preserve_raw: bool = True,
    raw_suffix: str = "_rawstate",
) -> pd.DataFrame:
    """Return a copy of ``df`` with the canonical RURO choice-state view applied.

    For every row with ``is_worker == 0`` the observed/chosen state is forced to
    non-employment: hours/working/realized-wage zeroed, ``loc4 = -1``, and the
    earnings convention applied (``yem = 0``, ``yemse = yse``). The offer wage
    ``wage_for_draws`` / ``yivwg`` is left untouched. Raw pre-canonical values are
    stored in ``<col>{raw_suffix}`` columns when ``preserve_raw`` (default True).

    Intended to be called on the deciders frame fed to ``generate_draws_long``
    (non-deciders are not drawn and keep their raw baseline for pricing).
    """
    if "is_worker" not in df.columns:
        raise KeyError("canonicalize_choice_state requires an 'is_worker' column.")
    out = df.copy()
    isw = pd.to_numeric(out["is_worker"], errors="coerce").fillna(0).astype(int)
    mask = (isw == 0).to_numpy()

    touched: List[str] = [c for c in _ZERO_COLS if c in out.columns]
    if "loc4" in out.columns:
        touched.append("loc4")
    if "yemse" in out.columns:
        touched.append("yemse")

    if preserve_raw:
        for c in touched:
            raw_col = f"{c}{raw_suffix}"
            if raw_col not in out.columns:
                out[raw_col] = out[c].copy()

    # Zero the chosen/observed working state for non-workers.
    for c in _ZERO_COLS:
        if c in out.columns:
            out.loc[mask, c] = 0
    if "loc4" in out.columns:
        out.loc[mask, "loc4"] = _NONEMP_LOC4

    # Frozen earnings convention: non-employment chosen state -> yem=0; yemse=yem+yse.
    if "yemse" in out.columns:
        out["yemse"] = _yem_plus_yse(out)

    return out


def _yem_plus_yse(df: pd.DataFrame) -> pd.Series:
    """yem + yse as a float Series (missing column treated as 0)."""
    yem = (pd.to_numeric(df["yem"], errors="coerce").fillna(0.0)
           if "yem" in df.columns else pd.Series(0.0, index=df.index))
    yse = (pd.to_numeric(df["yse"], errors="coerce").fillna(0.0)
           if "yse" in df.columns else pd.Series(0.0, index=df.index))
    return (yem + yse).astype(float)


# ---------------------------------------------------------------------------
# POST-DRAW canonicalization (corrects a KNOWN unchanged-core limitation)
# ---------------------------------------------------------------------------
# KNOWN CORE-PRIMITIVE LIMITATION (generate_draws_long, NOT patched here):
#   The core leaves generated non-employment rows in an INCOMPLETE state, on BOTH
#   row kinds:
#     - draw 0: it restores realized wage/wage_ruro from yivwg_base even when draw-0
#       hours are 0 (lines ~599-601), producing hours==0 with a POSITIVE realized wage;
#     - simulated (draw>=1) non-employment: the row inherits the decider's baseline
#       working/working_* flags and a stale yemse.
#   Both conflict with the non-employment convention (loc=-1, yem=0, no realized wage).
#   In this DE agnosticism smoke we do NOT edit core; we correct the resulting DE draw
#   VIEW app-side: every hours==0 row is forced to a complete non-employment state.
#   The post-pass leaves yivwg unchanged RELATIVE TO the generator output (it does not
#   restore the original offer wage on simulated rows, where the core has already
#   overwritten yivwg with the drawn wage); wage_for_draws is the reliable offer-wage.

# Fields zeroed on every non-employment (hours<=0) row. Excludes yemse (rebuilt to
# yem+yse on ALL rows) and loc4 (set to the -1 sentinel). On hours>0 rows these are
# left byte-identical.
_POST_NONEMP_ZERO: tuple[str, ...] = (
    "working", "working_pt1", "working_pt2", "working_ft", "working_lh",
    "wage", "wage_ruro", "yem",
)
# Managed fields asserted byte-identical on hours>0 rows before/after the post-pass.
POST_MANAGED_FIELDS: tuple[str, ...] = _POST_NONEMP_ZERO + ("loc4",)


def canonicalize_post_draws(long_df: pd.DataFrame) -> pd.DataFrame:
    """Correct the post-draw non-employment state in a generated long draws frame.

    For every row with ``hours <= 0`` enforce the complete non-employment state:
    ``working`` and all ``working_*`` flags = 0, ``wage`` = ``wage_ruro`` = 0,
    ``yem`` = 0, ``loc4`` = -1. ``yemse`` is rebuilt to ``yem + yse`` on ALL rows
    (so non-employment ⇒ ``yemse = yse``; working ⇒ the DE earnings identity).

    ``wage_for_draws`` (the reliable offer-wage provenance) is left UNCHANGED. ``yivwg``
    is also left unchanged RELATIVE TO the generator output, but note the core already
    overwrote ``yivwg`` with the drawn wage on simulated rows, so ``yivwg`` is NOT a
    reliable offer-wage carrier there. Every managed field on ``hours > 0`` rows and all
    ``log_q_*`` proposal components are byte-identical. Idempotent.
    """
    if "hours" not in long_df.columns:
        raise KeyError("canonicalize_post_draws requires an 'hours' column.")
    out = long_df.copy()
    hours = pd.to_numeric(out["hours"], errors="coerce").fillna(0.0)
    nonemp = (hours <= 0.0).to_numpy()

    for c in _POST_NONEMP_ZERO:
        if c in out.columns:
            out.loc[nonemp, c] = 0
    if "loc4" in out.columns:
        out.loc[nonemp, "loc4"] = -1

    # DE earnings identity on ALL rows (non-emp yem already 0 -> yemse = yse).
    if "yemse" in out.columns or "yem" in out.columns:
        out["yemse"] = _yem_plus_yse(out)

    return out


def assert_choice_state_consistent(df: pd.DataFrame) -> None:
    """Raise AssertionError if any row has the working/is_worker inconsistency.

    A consistent choice-state frame has NO row with ``is_worker==0`` yet
    ``lhw>0`` or ``working==1`` or ``wage_ruro!=0`` or ``loc4`` other than -1.
    """
    isw = pd.to_numeric(df["is_worker"], errors="coerce").fillna(0).astype(int)
    nonworker = isw == 0
    problems = {}
    if "lhw" in df.columns:
        problems["lhw>0"] = int((nonworker & (pd.to_numeric(df["lhw"], errors="coerce").fillna(0.0) > 0)).sum())
    if "working" in df.columns:
        problems["working==1"] = int((nonworker & (pd.to_numeric(df["working"], errors="coerce").fillna(0) == 1)).sum())
    if "wage_ruro" in df.columns:
        problems["wage_ruro!=0"] = int((nonworker & (pd.to_numeric(df["wage_ruro"], errors="coerce").fillna(0.0) != 0)).sum())
    if "loc4" in df.columns:
        problems["loc4!=-1"] = int((nonworker & (pd.to_numeric(df["loc4"], errors="coerce").fillna(-1) != -1)).sum())
    bad = {k: v for k, v in problems.items() if v > 0}
    assert not bad, f"choice-state inconsistency for is_worker==0 rows: {bad}"
