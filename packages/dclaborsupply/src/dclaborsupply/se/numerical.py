"""Hessian-based standard errors (migration matrix Wave 2.2).

Two lanes, co-located here because both turn a Hessian of the negative LL into
SE/VCV/t/p via one shared finalizer (`_finalize_se_from_hessian`):

- ``compute_standard_errors`` — central-difference NUMERICAL Hessian SE helper,
  lifted from MNL/scripts/enhanced/compute_standard_errors.py (only the reusable
  function; the CLI / argparse / data-loading / sys.path hacks / old-repo imports
  and the scipy dependency were NOT lifted). Portable numerical fallback; generic
  over any ``grad_func``; no jax.

- ``compute_hessian_se`` — NEW thin exact-JAX Hessian SE wrapper over a negLL
  function (``jax.hessian``). jax is imported lazily (light import preserved).

PROVENANCE: the certified ``se_hessian`` column in theta_hat_realdata_901_v1.csv
was produced by **exact ``jax.hessian``** in step4_realdata_baseline.py — NOT by
the central-difference helper. So ``compute_standard_errors`` carries no
byte-identical certified provenance (it is gated on a synthetic quadratic with a
known exact Hessian); ``compute_hessian_se`` is the certified-reproduction path
(gated against the se_hessian column).

p-values use the stdlib normal CDF (``math.erfc``); scipy is NOT a dependency.
Central-difference scheme preserved exactly: ``eps=1e-5``,
``H[:, i] = (g(theta+h_i) - g(theta-h_i)) / (2*eps)``, symmetrize, ``pinv``
``rcond=1e-10``. Zero old-repo imports.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _two_sided_normal_pvalues(t_values: np.ndarray) -> np.ndarray:
    """Two-sided p-values 2*(1 - Phi(|t|)) == erfc(|t|/sqrt(2)), stdlib only (no scipy).

    NaN t -> NaN p; |t|=inf -> 0; t=0 -> 1 (matches the scipy.stats.norm path).
    """
    z = np.abs(np.asarray(t_values, dtype=float)) / math.sqrt(2.0)
    _erfc = np.vectorize(math.erfc, otypes=[float])
    return _erfc(z)


def _finalize_se_from_hessian(
    H: np.ndarray,
    theta: np.ndarray,
    use_pinv: bool = True,
    rcond: float = 1e-10,
) -> dict:
    """Turn a symmetrized Hessian of the negative LL into SE/VCV/t/p.

    varcov = pinv(H, rcond) (use_pinv) else inv(H); se = sqrt(|diag|);
    negative-variance diagonals -> se = NaN; t = theta/se; p via stdlib normal.
    """
    n_params = len(theta)
    try:
        cond = float(np.linalg.cond(H))
        if cond > 1e10:
            logger.warning(f"Hessian is ill-conditioned (cond={cond:.2e})")
    except np.linalg.LinAlgError:
        logger.warning("Could not compute Hessian condition number")

    try:
        if use_pinv:
            varcov = np.linalg.pinv(H, rcond=rcond)
        else:
            varcov = np.linalg.inv(H)
    except np.linalg.LinAlgError as exc:
        logger.error(f"Hessian inversion failed: {exc}")
        nan = np.full(n_params, np.nan)
        return {"se": nan, "varcov": None, "t_values": nan, "p_values": nan, "hessian": H}

    diag = np.diag(varcov)
    se = np.sqrt(np.abs(diag))  # abs absorbs tiny negative round-off
    neg_var = diag < 0
    if np.any(neg_var):
        logger.warning(f"{int(np.sum(neg_var))} parameter(s) have negative variance "
                       "(identification issue); SE set to NaN")
        se = se.copy()
        se[neg_var] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = np.asarray(theta, dtype=float) / se
    p_values = _two_sided_normal_pvalues(t_values)

    return {"se": se, "varcov": varcov, "t_values": t_values,
            "p_values": p_values, "hessian": H}


def compute_standard_errors(
    theta: np.ndarray,
    grad_func: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-5,
    use_pinv: bool = True,
    rcond: float = 1e-10,
) -> dict:
    """Standard errors via a central-difference numerical Hessian (portable fallback).

    Central differences on ``grad_func`` (gradient of the NEGATIVE log-likelihood)
    approximate the Hessian, which is then inverted for the VCV. Generic: works
    with any gradient callable. Returns a dict with ``se``, ``varcov``,
    ``t_values``, ``p_values``, ``hessian``.
    """
    theta = np.asarray(theta, dtype=float)
    n_params = len(theta)
    logger.info(f"Computing numerical Hessian ({n_params}x{n_params})...")

    H = np.zeros((n_params, n_params))
    for i in range(n_params):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps
        g_plus = np.asarray(grad_func(theta_plus), dtype=float)
        g_minus = np.asarray(grad_func(theta_minus), dtype=float)
        # Second derivative: (g(x+h) - g(x-h)) / (2h)
        H[:, i] = (g_plus - g_minus) / (2 * eps)
        if (i + 1) % 10 == 0:
            logger.info(f"  Hessian column {i+1}/{n_params} computed")

    H = 0.5 * (H + H.T)  # symmetrize
    return _finalize_se_from_hessian(H, theta, use_pinv=use_pinv, rcond=rcond)


def _load_jax():
    """Lazy jax import (float64); keeps `import ...se.numerical` jax-free."""
    try:
        import jax
    except ImportError as exc:  # pragma: no cover - exercised only without jax
        raise ImportError(
            "exact-JAX Hessian SE requires jax. Install with "
            "`pip install dclaborsupply[jax]`."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    return jax


def compute_hessian_se(
    neg_ll_fn: Callable,
    theta: np.ndarray,
    use_pinv: bool = True,
    rcond: float = 1e-10,
) -> dict:
    """Standard errors via the EXACT Hessian (``jax.hessian(neg_ll_fn)``).

    The certified-reproduction path: the certified ``se_hessian`` was produced by
    exact ``jax.hessian``. ``neg_ll_fn`` maps a jnp parameter vector to the scalar
    negative log-likelihood (e.g. ``engine_jax.build_joint_neg_ll(...)``). jax is
    imported lazily. Returns the same dict shape as ``compute_standard_errors``.
    """
    jax = _load_jax()
    import jax.numpy as jnp

    theta = np.asarray(theta, dtype=float)
    th = jnp.asarray(theta, dtype=jnp.float64)
    H = np.asarray(jax.hessian(neg_ll_fn)(th))
    H = 0.5 * (H + H.T)  # symmetrize
    return _finalize_se_from_hessian(H, theta, use_pinv=use_pinv, rcond=rcond)
