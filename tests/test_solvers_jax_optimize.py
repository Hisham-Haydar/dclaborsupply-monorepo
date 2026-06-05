"""Wave 2.3 tests for the SciPy L-BFGS-B / JAX optimizer (self-contained).

Light-import is checked in a fresh subprocess (jax/scipy must NOT load on import).
The optimizer itself needs jax+scipy and is gated on a synthetic quadratic with a
known box-constrained optimum (importorskip). The certified no-drift gate runs as a
validation script, not a committed test.
"""
import subprocess
import sys

import numpy as np
import pytest

from dclaborsupply.solvers import build_bounds_list, optimize_lbfgsb


class _Spec:
    """Minimal duck-typed spec for build_bounds_list (only the two attrs it reads)."""
    all_param_names = ["a", "b", "c"]
    bounds = {"a": (0.0, 1.0), "c": (-2.0, None)}  # b unbounded; c half-open


def test_build_bounds_list():
    out = build_bounds_list(_Spec())
    assert out == [(0.0, 1.0), (None, None), (-2.0, None)]
    # floats coerced, None preserved
    assert isinstance(out[0][0], float) and out[1] == (None, None)


def test_solvers_import_is_light():
    code = (
        "import sys, dclaborsupply.solvers\n"
        "assert 'jax' not in sys.modules, 'jax imported!'\n"
        "assert 'scipy' not in sys.modules, 'scipy imported!'\n"
        "assert 'optimistix' not in sys.modules, 'optimistix imported!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_optimize_lbfgsb_quadratic_with_bounds():
    jax = pytest.importorskip("jax")
    pytest.importorskip("scipy")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    x_star = jnp.array([3.0, -2.0])

    def neg_ll(th):
        return jnp.sum((th - x_star) ** 2)

    # x2 lower bound 0 > -2  =>  optimum is the projection [3, 0] (bound-active).
    bounds = [(0.0, 5.0), (0.0, 5.0)]
    res = optimize_lbfgsb(neg_ll, [1.0, 1.0], bounds, gtol=1e-9)

    assert res["success"]
    assert np.allclose(res["theta_hat"], [3.0, 0.0], atol=1e-5)
    assert res["neg_ll"] == pytest.approx(4.0, abs=1e-6)  # (0)^2 + (0-(-2))^2
    # x1 interior, x2 at its lower bound
    assert abs(res["theta_hat"][0] - 3.0) < 1e-5
    assert abs(res["theta_hat"][1] - 0.0) < 1e-5
