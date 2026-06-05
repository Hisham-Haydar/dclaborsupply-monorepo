"""SciPy L-BFGS-B optimizer over the JAX joint negLL (migration matrix Wave 2.3).

Lifted from MNL/scripts/bpool/jax_optimize.py — reusable optimizer routines only
(the CLI / argparse / sys.path hacks / probe / output-CSV code were NOT lifted).
jax, scipy, and (optional) optimistix are imported LAZILY inside the functions, so
``import dclaborsupply.solvers`` works with all three absent.

The SciPy L-BFGS-B scheme is preserved exactly:
    method="L-BFGS-B", bounds from spec, gtol, ftol=1e-15, maxls=60, maxiter;
    objective = the JAX joint negLL, jac = jax.grad(negLL) (both jit-compiled).

Dependencies (see pyproject):
- scipy is an OPTIONAL extra: ``pip install dclaborsupply[solver]``.
- jax via ``pip install dclaborsupply[jax]``. The L-BFGS-B path needs BOTH.
- optax is NOT required (the source does not use it).
- optimistix is a further-optional pure-JAX polish (not installed by any extra).
  Do NOT use it on bound-active solutions (it is unconstrained); the certified
  47-param point has bound-active params, so the certified path is SciPy only.

Zero old-repo imports; uses the lifted ``engine_jax.build_joint_neg_ll`` as the
objective at the call site.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BoundsList = List[Tuple[Optional[float], Optional[float]]]


def build_bounds_list(spec) -> BoundsList:
    """Bounds aligned to ``spec.all_param_names``; ``(None, None)`` when unbounded.

    SciPy-L-BFGS-B-ready. Lifted verbatim from jax_optimize._bounds_list.
    """
    out: BoundsList = []
    for n in spec.all_param_names:
        if n in spec.bounds:
            lo, hi = spec.bounds[n]
            out.append((None if lo is None else float(lo),
                        None if hi is None else float(hi)))
        else:
            out.append((None, None))
    return out


def _load_jax():
    """Lazy jax import (float64); keeps ``import ...solvers`` jax-free."""
    try:
        import jax
    except ImportError as exc:  # pragma: no cover - only without jax
        raise ImportError(
            "JAX optimizer requires jax. Install with `pip install dclaborsupply[jax]`."
        ) from exc
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def _load_scipy_minimize():
    """Lazy scipy import; keeps ``import ...solvers`` scipy-free."""
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - only without scipy
        raise ImportError(
            "SciPy L-BFGS-B optimizer requires scipy. "
            "Install with `pip install dclaborsupply[solver]`."
        ) from exc
    return minimize


def optimize_lbfgsb(
    neg_ll_fn: Callable,
    theta0: Sequence[float],
    bounds: BoundsList,
    *,
    gtol: float = 1e-6,
    maxiter: int = 2000,
    callback: Optional[Callable] = None,
) -> dict:
    """Box-constrained SciPy L-BFGS-B over a JAX joint negLL.

    ``neg_ll_fn`` maps a jnp parameter vector to the scalar negative log-likelihood
    (e.g. ``engine_jax.build_joint_neg_ll(spec, data_sm, data_sf, data_cou)``). The
    objective is jit-compiled and its ``jax.grad`` supplies the analytic jacobian.

    Scheme preserved exactly: ``method="L-BFGS-B"``, ``bounds``, ``gtol``,
    ``ftol=1e-15``, ``maxls=60``, ``maxiter`` (``disp=False``).

    Returns a dict: ``theta_hat``, ``neg_ll``, ``max_abs_grad``, ``success``,
    ``status``, ``message``, ``n_iter``, ``result`` (raw scipy OptimizeResult).
    """
    jax, jnp = _load_jax()
    minimize = _load_scipy_minimize()

    t0 = np.asarray(theta0, dtype=np.float64)
    jval = jax.jit(neg_ll_fn)
    jgrad = jax.jit(jax.grad(neg_ll_fn))

    def fun(x):
        return float(jval(jnp.asarray(x, dtype=jnp.float64)))

    def grad(x):
        return np.asarray(jgrad(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)

    res = minimize(
        fun, t0, jac=grad, method="L-BFGS-B", bounds=list(bounds),
        callback=callback,
        options={"maxiter": int(maxiter), "gtol": float(gtol),
                 "ftol": 1e-15, "maxls": 60, "disp": False},
    )
    theta_hat = np.asarray(res.x, dtype=np.float64)
    return {
        "theta_hat": theta_hat,
        "neg_ll": float(res.fun),
        "max_abs_grad": float(np.max(np.abs(grad(theta_hat)))),
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "n_iter": int(res.nit),
        "result": res,
    }


def polish_optimistix(
    neg_ll_fn: Callable,
    theta0: Sequence[float],
    *,
    gtol: float = 1e-6,
    maxiter: int = 2000,
) -> np.ndarray:
    """OPTIONAL pure-JAX BFGS polish (optimistix); returns the polished theta.

    Lazy-imports optimistix (not installed by any extra). UNCONSTRAINED — do NOT
    use on bound-active solutions (e.g. the certified 47-param point). ``verbose``
    is left off: optimistix 0.1.0 ``BFGS(verbose=...)`` crashes at solve time.
    """
    _jax, jnp = _load_jax()
    try:
        import optimistix as optx
    except ImportError as exc:  # pragma: no cover - only without optimistix
        raise ImportError(
            "optimistix polish requires optimistix (pip install optimistix)."
        ) from exc

    def ox_fn(y, _args):
        return neg_ll_fn(y)

    solver = optx.BFGS(rtol=float(gtol), atol=float(gtol))
    sol = optx.minimise(
        ox_fn, solver, jnp.asarray(np.asarray(theta0, dtype=np.float64)),
        max_steps=int(maxiter), throw=False,
    )
    return np.asarray(sol.value, dtype=np.float64)
