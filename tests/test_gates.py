"""Wave 2.5 tests for the portable gates (self-contained).

Param-binding is tested with a duck-typed spec + trivial objectives (no engine).
Recovery primitives are pure; the full recover() loop is checked on a tiny jax
quadratic (importorskip). The certified param-binding + portable synthetic recovery
run as a validation script with MNL read-only helpers, not as committed tests.
"""
import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from dclaborsupply.gates import (
    base_perturbation_theta,
    check_param_binding,
    draw_synthetic_choice,
    generate_theta_star,
    hessian_pd_verdict,
)


class _Spec:
    """Minimal duck-typed spec exposing only what the gates read."""

    def __init__(self, names, initial, bounds=None, fixed=None):
        self.all_param_names = list(names)
        self.initial_values = dict(initial)
        self._bounds = bounds or {}
        self.fixed_params = dict(fixed or {})

    def get_param_index(self, n):
        return self.all_param_names.index(n)

    def get_initial_vector(self):
        return np.array([float(self.initial_values.get(n, 0.0))
                         for n in self.all_param_names], dtype=float)

    def get_bounds_tuple(self):
        return [self._bounds.get(n, (None, None)) for n in self.all_param_names]


def _spec():
    return _Spec(["a", "b", "c"], {"a": 1.0, "b": 0.0, "c": 2.0},
                 bounds={"a": (0.0, 5.0), "b": (-5.0, 5.0), "c": (0.0, 5.0)},
                 fixed={"d": -0.8})


# --- param binding ------------------------------------------------------------
def test_base_perturbation_theta_bumps_zeros():
    th = base_perturbation_theta(_spec())
    assert th.tolist() == [1.0, 0.3, 2.0]  # 0.0 -> 0.3


def test_param_binding_structural_passes():
    r = check_param_binding(_spec())
    assert r["n_free"] == 3 and r["n_fixed"] == 1
    assert not r["duplicates"] and not r["fixed_free_collision"]
    assert r["index_round_trip"] and r["theta_len_ok"]
    assert r["passed"]  # structural-only (no neg_ll_fn) still passes


def test_param_binding_detects_fixed_free_collision():
    spec = _Spec(["a", "b"], {"a": 1.0, "b": 1.0}, fixed={"a": 0.5})  # 'a' both free and pinned
    r = check_param_binding(spec)
    assert r["fixed_free_collision"] == ["a"] and not r["passed"]


def test_param_binding_all_bound_with_full_objective():
    spec = _spec()
    w = np.array([1.0, 1.0, 1.0])
    r = check_param_binding(spec, lambda th: float(np.asarray(th) @ w))
    assert set(r["bound"]) == {"a", "b", "c"} and not r["not_bound"] and r["passed"]


def test_param_binding_detects_silent_drop():
    spec = _spec()
    # objective ignores 'c' -> nudging 'c' does not move the LL -> silent drop
    r = check_param_binding(spec, lambda th: float(th[0] + th[1]))
    assert r["not_bound"] == ["c"] and not r["passed"]


# --- recovery primitives ------------------------------------------------------
def test_generate_theta_star_in_bounds_and_nontrivial():
    spec = _spec()
    rng = np.random.default_rng(0)
    ts = generate_theta_star(spec, rng)
    bt = spec.get_bounds_tuple()
    for v, (lo, hi) in zip(ts, bt):
        assert lo <= v <= hi
    assert np.any(ts != spec.get_initial_vector())  # non-trivial


def test_draw_synthetic_choice_one_hot_per_group():
    # 2 groups x 3 alts; argmax-dominated V so the draw is (near) deterministic
    V = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    gs = np.array([0, 3])
    ge = np.array([3, 6])
    rng = np.random.default_rng(0)
    ac = draw_synthetic_choice(V, gs, ge, rng)
    assert ac.sum() == 2.0                      # one chosen per group
    assert ac[0:3].sum() == 1.0 and ac[3:6].sum() == 1.0


def test_hessian_pd_verdict():
    pd = hessian_pd_verdict(np.array([[2.0, 0.5], [0.5, 1.0]]), ["a", "b"])
    assert pd["pd_ok"] and pd["min_eig"] > 0 and not pd["bad_dirs"]
    indef = hessian_pd_verdict(np.array([[1.0, 0.0], [0.0, -1.0]]), ["a", "b"])
    assert not indef["pd_ok"] and indef["min_eig"] < 0 and indef["bad_dirs"]


def test_recover_quadratic():
    # jax+scipy needed; run in a SUBPROCESS so jax never loads in the main pytest
    # process (that would break the in-process light-import test in test_imports.py).
    if importlib.util.find_spec("jax") is None or importlib.util.find_spec("scipy") is None:
        pytest.skip("jax/scipy not installed")
    code = (
        "import numpy as np, jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "import jax.numpy as jnp\n"
        "from dclaborsupply.gates import recover\n"
        "star = np.array([3.0, -2.0]); sj = jnp.asarray(star)\n"
        "neg_ll = lambda th: jnp.sum((th - sj) ** 2)\n"
        "out = recover(neg_ll, star, [(-10.0, 10.0), (-10.0, 10.0)], seed=1,\n"
        "              band=1e-3, perturb=0.5, param_names=['x', 'y'])\n"
        "assert out['optimizer_success'], out\n"
        "assert out['within_band'] and out['max_dev'] < 1e-3, out['max_dev']\n"
        "assert out['pd_ok'] and out['min_eig'] > 0, out['min_eig']\n"
        "assert out['passed']\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- light import (fresh process) ---------------------------------------------
def test_gates_import_is_light():
    code = (
        "import sys, dclaborsupply.gates\n"
        "for m in ('jax', 'gamspy', 'scipy', 'numba'):\n"
        "    assert m not in sys.modules, m + ' imported at gates import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
