"""Portable synthetic-recovery gate (migration matrix Wave 2.5).

Generic recovery primitives distilled from MNL/scripts/bpool/joint_recovery_test.py:
spec-driven theta* generation, a vectorised Gumbel-max synthetic-choice draw, a
slim PD-Hessian verdict, synthetic-choice installation via the lifted NumPy engine,
and an objective-injected recovery runner that uses the LIFTED solver
(solvers.optimize_lbfgsb) + exact-JAX Hessian (se.compute_hessian_se).

NOT lifted (B-pool / provenance): the CONOPT runners, parquet/data loaders,
contamination / smoke / two-start / group-specific suites, CLI/report code, and
58-param assumptions. The certified provenance gate jax_recovery_gate.py is NOT
referenced. This PORTABLE recovery validates that the package recovers a known
synthetic theta* end-to-end; it does NOT reproduce the certified provenance gate.

Heavy lifted modules (engine_jax / engine_numpy / solvers / se) are imported
LAZILY inside the functions, so ``import dclaborsupply.gates`` pulls no
jax/gamspy/numba.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def generate_theta_star(spec, rng, scale_perturb: float = 0.25,
                        shifter_frac: float = 0.12) -> np.ndarray:
    """Build a plausible, non-trivial, in-bounds theta* from the spec.

    No parameter names are hardcoded: per param, perturb a non-zero initial by
    ``scale_perturb`` (alternating sign), or give a zero initial an
    alternating-sign signal scaled to the finite bound half-width. Lifted verbatim
    from the joint_recovery_test generate_theta_star helper.
    """
    names = spec.all_param_names
    th = np.array([float(spec.initial_values.get(n, 0.0)) for n in names], dtype=float)
    bt = spec.get_bounds_tuple()
    eps = 1e-6
    sign = 1.0
    for i, n in enumerate(names):
        v0 = th[i]
        lo, hi = -np.inf, np.inf
        if bt and i < len(bt) and bt[i] is not None:
            lo = bt[i][0] if bt[i][0] is not None else -np.inf
            hi = bt[i][1] if bt[i][1] is not None else np.inf
        if abs(v0) > eps:
            val = v0 * (1.0 + scale_perturb * sign)
        else:
            half = 0.5 * (hi - lo) if (np.isfinite(hi) and np.isfinite(lo)) else 1.0
            val = sign * shifter_frac * half
        pad = 1e-6
        if np.isfinite(lo):
            val = max(val, lo + pad)
        if np.isfinite(hi):
            val = min(val, hi - pad)
        th[i] = val
        sign = -sign
    return th


def draw_synthetic_choice(V: np.ndarray, group_starts: np.ndarray,
                          group_ends: np.ndarray, rng) -> np.ndarray:
    """One Gumbel-max draw per group from softmax(V), fully vectorised.

    Returns actual_choice (1.0 at the drawn alt per group, else 0.0). Lifted
    verbatim from the joint_recovery_test draw_synthetic_choice helper.
    """
    n = V.shape[0]
    g = np.repeat(np.arange(len(group_starts)), group_ends - group_starts)
    gumbel = -np.log(-np.log(rng.uniform(size=n)))
    key = V + gumbel
    order = np.lexsort((-key, g))
    g_sorted = g[order]
    first_in_group = np.ones(n, dtype=bool)
    first_in_group[1:] = g_sorted[1:] != g_sorted[:-1]
    chosen_global = order[first_in_group]
    out = np.zeros(n)
    out[chosen_global] = 1.0
    return out


def hessian_pd_verdict(H: np.ndarray, param_names: Sequence[str],
                       loading_thresh: float = 0.20) -> Dict[str, Any]:
    """Slim PD verdict for a Hessian: ``pd_ok``, ``min_eig``, eigenvalues, and the
    loadings of any non-positive direction.

    Distilled from joint_recovery_test._hessian_verdict; the B-pool
    market-collinearity / SE-covariance diagnostics were intentionally dropped.
    """
    H = 0.5 * (H + H.T)
    eig_w, eig_v = np.linalg.eigh(H)
    pd_ok = bool(np.all(eig_w > 0))
    min_eig = float(eig_w.min())
    bad_dirs: List[Tuple[float, List[Tuple[str, float]]]] = []
    for k in range(len(eig_w)):
        if eig_w[k] <= 0:
            v = eig_v[:, k]
            loaders = sorted(
                ((param_names[i], float(abs(v[i]))) for i in range(len(v))
                 if abs(v[i]) > loading_thresh),
                key=lambda kv: -kv[1])
            bad_dirs.append((float(eig_w[k]), loaders))
    return {"pd_ok": pd_ok, "min_eig": min_eig, "eig": eig_w, "bad_dirs": bad_dirs}


def synthesize_actual_choices(spec, data_tuple, theta_star, rng):
    """Install synthetic ``actual_choice`` arrays drawn from the model at theta*.

    For each non-None group, compute the per-alternative value V at theta* via the
    lifted NumPy engine (return_components) and draw one choice per group
    (Gumbel-max). Returns shallow copies with ``actual_choice`` set; the caller's
    data is not mutated. Mirrors joint_recovery_test.run_synthetic_dgp (lifted
    engine). Lazy imports.

    The NumPy engine has no native ``fixed_params`` pinning, so any pinned params
    are folded into theta (and removed from fixed_params) via the Wave-1.4
    ``_fold_fixed_params`` helper before the V evaluation — both backends then
    evaluate uniformly.
    """
    import copy
    from dclaborsupply.likelihood.engine_numpy import (
        compute_likelihood_singles, compute_likelihood_couples,
    )
    from dclaborsupply.likelihood.index import _fold_fixed_params

    data_sm, data_sf, data_cou = data_tuple
    theta_star = np.asarray(theta_star, dtype=float)
    spec_eval, theta_eval = _fold_fixed_params(spec, theta_star)
    out = []
    plan = (
        (data_sm, compute_likelihood_singles),
        (data_sf, compute_likelihood_singles),
        (data_cou, compute_likelihood_couples),
    )
    for data, fn in plan:
        if data is None:
            out.append(None)
            continue
        comp = fn(theta_eval, data, spec_eval, return_components=True)
        ac = draw_synthetic_choice(comp["V"], data.group_starts, data.group_ends, rng)
        d = copy.copy(data)
        d.actual_choice = ac
        out.append(d)
    return tuple(out)


def build_synthetic_recovery_objective(spec, data_sm, data_sf, data_cou,
                                       gender_split=None):
    """Joint negLL with ``use_actual_choice=True`` (the synthetic-recovery objective).

    Sums the lifted per-group JAX builders so the observed term reads the drawn
    ``actual_choice`` (not column 0) — REQUIRED for synthetic recovery. Returns the
    jitted callable. jax imported lazily.
    """
    from dclaborsupply.likelihood.engine_jax import (
        build_jax_singles_ll, build_jax_couples_ll, _load_jax,
    )

    jax, _jnp = _load_jax()
    parts = []
    if data_sm is not None:
        f, _ = build_jax_singles_ll(data_sm, spec, is_male=True,
                                    use_actual_choice=True, gender_split=gender_split)
        parts.append(f)
    if data_sf is not None:
        f, _ = build_jax_singles_ll(data_sf, spec, is_male=False,
                                    use_actual_choice=True, gender_split=gender_split)
        parts.append(f)
    if data_cou is not None:
        f, _ = build_jax_couples_ll(data_cou, spec,
                                    use_actual_choice=True, gender_split=gender_split)
        parts.append(f)

    def joint(theta):
        return sum(f(theta) for f in parts)

    return jax.jit(joint)


def recover(
    neg_ll_fn,
    theta_star,
    bounds,
    *,
    seed: int = 0,
    band: float = 0.5,
    perturb: float = 0.1,
    maxiter: int = 2000,
    param_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Portable recovery: perturb a start off theta*, re-estimate via the lifted
    SciPy L-BFGS-B solver, then check the PD Hessian (exact JAX) and per-param band.

    ``neg_ll_fn`` maps a jnp parameter vector to the scalar negLL (e.g. the
    synthetic-recovery objective from build_synthetic_recovery_objective, or any
    jax objective). Returns theta* vs recovered theta, per-param deviations, max
    deviation + band status, ``min_eig`` and ``pd_ok``, and an overall ``passed``.
    """
    from dclaborsupply.solvers.jax_optimize import optimize_lbfgsb
    from dclaborsupply.se.numerical import compute_hessian_se

    theta_star = np.asarray(theta_star, dtype=float)
    rng = np.random.default_rng(seed)
    theta0 = theta_star + perturb * rng.standard_normal(theta_star.shape)
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            theta0[i] = max(theta0[i], lo)
        if hi is not None:
            theta0[i] = min(theta0[i], hi)

    res = optimize_lbfgsb(neg_ll_fn, theta0, bounds, gtol=1e-6, maxiter=maxiter)
    theta_hat = res["theta_hat"]

    dev = np.abs(theta_hat - theta_star)
    max_dev = float(dev.max())
    within_band = bool(max_dev <= band)

    H = np.asarray(compute_hessian_se(neg_ll_fn, theta_hat)["hessian"])
    names = (list(param_names) if param_names is not None
             else [f"p{i}" for i in range(len(theta_star))])
    verdict = hessian_pd_verdict(H, names)
    worst = names[int(dev.argmax())] if len(names) == len(dev) else None

    return {
        "theta_star": theta_star,
        "theta_hat": theta_hat,
        "deviations": dev,
        "max_dev": max_dev,
        "worst_param": worst,
        "band": band,
        "within_band": within_band,
        "min_eig": verdict["min_eig"],
        "pd_ok": verdict["pd_ok"],
        "optimizer_success": res["success"],
        "passed": bool(within_band and verdict["pd_ok"]),
    }
