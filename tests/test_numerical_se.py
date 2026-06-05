"""Wave 2.2 tests for Hessian SEs (self-contained; central-diff is jax-free).

Central-difference lane is gated on a synthetic quadratic with a KNOWN exact
Hessian (not on the certified column). The exact-JAX wrapper is checked on a tiny
jax quadratic (importorskip) and gated against the certified se_hessian in a
separate validation script.
"""
import math

import numpy as np
import pytest

from dclaborsupply.se import compute_standard_errors, compute_hessian_se
from dclaborsupply.se.numerical import _two_sided_normal_pvalues

# Symmetric PD "true Hessian"; gradient of 0.5 x'Ax is A x (linear) -> central
# differences recover A EXACTLY (up to float round-off).
A = np.array([[2.0, 0.5], [0.5, 1.0]])
THETA = np.array([1.0, 1.0])


def _grad(t):
    return A @ t


def test_central_diff_recovers_exact_hessian():
    out = compute_standard_errors(THETA, _grad)
    assert np.allclose(out["hessian"], A, atol=1e-7)


def test_hessian_symmetry():
    H = compute_standard_errors(THETA, _grad)["hessian"]
    assert np.allclose(H, H.T, atol=0, rtol=0)


def test_se_is_sqrt_diag_pinv():
    out = compute_standard_errors(THETA, _grad)
    expected = np.sqrt(np.diag(np.linalg.pinv(out["hessian"], rcond=1e-10)))
    assert np.allclose(out["se"], expected)
    # and equals the analytic sqrt(diag(inv(A)))
    assert np.allclose(out["se"], np.sqrt(np.diag(np.linalg.inv(A))))


def test_pinv_eq_inv_for_pd():
    se_pinv = compute_standard_errors(THETA, _grad, use_pinv=True)["se"]
    se_inv = compute_standard_errors(THETA, _grad, use_pinv=False)["se"]
    assert np.allclose(se_pinv, se_inv)


def test_negative_variance_gives_nan_se():
    # Indefinite "Hessian": grad = diag(1,-1) x -> H = diag(1,-1); var[1] < 0.
    D = np.array([[1.0, 0.0], [0.0, -1.0]])
    out = compute_standard_errors(THETA, lambda t: D @ t)
    assert np.isfinite(out["se"][0]) and out["se"][0] == pytest.approx(1.0)
    assert np.isnan(out["se"][1])


def test_pvalues_stdlib_normal():
    # erfc(|t|/sqrt2): t=0 -> 1; |t|=inf -> 0; t=1.959963985 -> ~0.05
    p = _two_sided_normal_pvalues(np.array([0.0, np.inf, 1.959963984540054]))
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(0.0, abs=1e-12)
    assert p[2] == pytest.approx(0.05, abs=1e-6)
    # NaN t -> NaN p
    assert np.isnan(_two_sided_normal_pvalues(np.array([np.nan]))[0])


def test_tvalues_and_pvalues_shapes():
    out = compute_standard_errors(THETA, _grad)
    assert out["t_values"].shape == THETA.shape
    assert out["p_values"].shape == THETA.shape
    assert np.all((out["p_values"] >= 0) & (out["p_values"] <= 1))


def test_exact_jax_hessian_wrapper_quadratic():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    A_j = jnp.asarray(A)

    def neg_ll(th):
        return 0.5 * jnp.dot(th, A_j @ th)

    out = compute_hessian_se(neg_ll, THETA)
    assert np.allclose(out["hessian"], A, atol=1e-9)
    assert np.allclose(out["se"], np.sqrt(np.diag(np.linalg.inv(A))), atol=1e-9)
