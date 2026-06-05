"""Portable param-binding gate (migration matrix Wave 2.5).

A GENERIC, callable gate distilled from the perturbation core of
MNL/scripts/bpool/phase_a_param_binding.py. The B-pool runner (58-param
assumptions, hardcoded parquet/spec/meta paths, mixed-year slices, CLI/report) was
NOT lifted. This is the engine-agnostic primitive: structural checks plus an
optional perturbation test over an injected negative-log-likelihood objective.

The certified provenance gate (jax_recovery_gate.py) is NOT referenced. Zero
old-repo imports; no jax at import (the objective is supplied by the caller).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _WarnCapture(logging.Handler):
    """Collect WARNING records (e.g. an engine's 'skipping <param>' messages)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.msgs: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.msgs.append(record.getMessage())


def base_perturbation_theta(spec) -> np.ndarray:
    """Non-trivial base theta for the perturbation test.

    Start from the spec's initial vector and bump exact-0.0 entries to 0.3 so a
    nudge is meaningful (many spec defaults are 0.0). Box-Cox thetas (negative
    initials) are left as-is. Lifted from phase_a._base_theta.
    """
    th = spec.get_initial_vector().astype(float).copy()
    for i in range(len(th)):
        if th[i] == 0.0:
            th[i] = 0.3
    return th


def check_param_binding(
    spec,
    neg_ll_fn: Optional[Callable[[np.ndarray], float]] = None,
    *,
    theta: Optional[np.ndarray] = None,
    applicable: Optional[Callable[[str], bool]] = None,
    tol: float = 1e-7,
    step: float = 0.2,
    capture_logger: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify every spec parameter binds to theta with no silent drops.

    Structural checks (always run): no duplicate names; ``fixed_params`` disjoint
    from the free vector; ``get_param_index`` round-trips; the (base or supplied)
    theta length matches the free-parameter count.

    Perturbation test (only when ``neg_ll_fn`` is given): from a non-trivial base
    theta, nudge each APPLICABLE free param and require the objective to move
    (|Δ negLL| > tol). A param that does not move the objective is a SILENT DROP.
    ``neg_ll_fn(theta) -> float`` is engine-agnostic (the caller closes it over the
    data and chosen backend). Optionally captures WARNING records on
    ``capture_logger`` during the base evaluation (e.g. an engine's
    'skipping <param>' messages).

    Returns a dict with the structural results, bound / not_bound / na lists, any
    captured warnings, and an overall ``passed`` flag.
    """
    names = list(spec.all_param_names)
    fixed = dict(getattr(spec, "fixed_params", {}) or {})

    duplicates = sorted({n for n in names if names.count(n) > 1})
    fixed_free_collision = sorted(set(fixed) & set(names))
    index_round_trip = all(spec.get_param_index(n) == i for i, n in enumerate(names))

    base = base_perturbation_theta(spec) if theta is None else np.asarray(theta, dtype=float)
    theta_len_ok = len(base) == len(names)

    bound: List[str] = []
    not_bound: List[str] = []
    na: List[str] = []
    warnings: List[str] = []
    ll0: Optional[float] = None

    if neg_ll_fn is not None:
        cap = _WarnCapture()
        target = logging.getLogger(capture_logger) if capture_logger else None
        if target is not None:
            target.addHandler(cap)
            prev_level = target.level
            target.setLevel(logging.WARNING)
        try:
            ll0 = float(neg_ll_fn(base))
        finally:
            if target is not None:
                target.removeHandler(cap)
                target.setLevel(prev_level)
        warnings = [m for m in cap.msgs
                    if "skipping" in m.lower() or "not found" in m.lower()]

        bt = spec.get_bounds_tuple()
        for i, name in enumerate(names):
            if applicable is not None and not applicable(name):
                na.append(name)
                continue
            thp = base.copy()
            s = step
            lo, hi = -np.inf, np.inf
            if bt and i < len(bt) and bt[i] is not None:
                lo = bt[i][0] if bt[i][0] is not None else -np.inf
                hi = bt[i][1] if bt[i][1] is not None else np.inf
            if thp[i] + s > hi:
                s = -s
            elif thp[i] + s < lo:
                s = -s
            thp[i] = thp[i] + s
            llp = float(neg_ll_fn(thp))
            moved = np.isfinite(llp) and abs(llp - ll0) > tol
            (bound if moved else not_bound).append(name)

    passed = (
        not duplicates
        and not fixed_free_collision
        and index_round_trip
        and theta_len_ok
        and not not_bound
        and not warnings
    )
    return {
        "n_free": len(names),
        "n_fixed": len(fixed),
        "fixed_params": fixed,
        "duplicates": duplicates,
        "fixed_free_collision": fixed_free_collision,
        "index_round_trip": index_round_trip,
        "theta_len_ok": theta_len_ok,
        "ll0": ll0,
        "bound": sorted(bound),
        "not_bound": sorted(not_bound),
        "na": sorted(na),
        "warnings": warnings,
        "passed": passed,
    }
