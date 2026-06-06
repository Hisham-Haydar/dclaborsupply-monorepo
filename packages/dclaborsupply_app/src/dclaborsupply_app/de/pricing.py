"""DE earnings-mutation policy for the EUROMOD pricing runner (validated rule).

Implements the EarningsMutationPolicy interface for Germany only. The validated DE
pricing rule (DE pricing smoke; project memory) mutates a decider's labour inputs to a
labour-supply alternative's (hours, wage):

    lhw   = hours
    yem   = wage * hours * (52/12)      # single monthly employment income (no FR yem00/yemxp split)
    yemse = yem + yse                    # DE earnings identity (employment + self-employment)
    bun / bsa and all other inputs: UNCHANGED

``yivwg`` (the hourly/offer wage) is intentionally NOT mutated here: it is not a EUROMOD
tax base, so it does not affect ``ils_dispy``. The policy is country-injected; FR can
later implement the SAME interface with its yem00/yemxp 35h split (not implemented here).

App layer only — pandas/numpy not even required; no core/france/MNL imports.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

WEEKS_PER_MONTH = 52.0 / 12.0


def de_earnings_policy(
    member: Mapping[str, Any],
    *,
    hours: float,
    wage: float,
    weeks_per_month: float = WEEKS_PER_MONTH,
) -> Dict[str, float]:
    """Return DE decider-input overrides for one priced alternative.

    Reads ``yse`` from the decider's baseline row to maintain ``yemse = yem + yse``.
    Returns only the mutated columns (``lhw``, ``yem``, ``yemse``); the runner applies
    them and leaves every other column (incl. ``bun``/``bsa``/``yivwg``) untouched.
    """
    try:
        yse = float(member.get("yse", 0.0) or 0.0)
    except (TypeError, ValueError):
        yse = 0.0
    yem = float(wage) * float(hours) * float(weeks_per_month)
    return {"lhw": float(hours), "yem": yem, "yemse": yem + yse}
