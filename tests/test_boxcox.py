"""Wave 3.1 tests: confirm the lifted Box-Cox primitives match the engine path.

box_cox_transform must match the JAX engine's jbox_cox; box_cox_derivative_theta
must match central finite differences (R1-fixed Taylor branch from Wave 0.2/1.3);
box_cox_derivative_x must equal x**(theta-1). The transform-vs-jbox_cox check runs
in a subprocess so jax never loads in the main pytest process.
"""
import importlib.util
import subprocess
import sys

import numpy as np

from dclaborsupply.utility.boxcox import (
    box_cox_derivative_theta,
    box_cox_derivative_x,
    box_cox_transform,
)

_X = np.array([0.2, 0.5, 1.0, 1.5, 3.0, 8.0])


def test_box_cox_derivative_x_matches_power():
    # (c) dBC/dx = x**(theta-1), exactly.
    for theta in (0.0, 0.3, -0.8, 1.5):
        d = np.asarray(box_cox_derivative_x(_X, theta))
        assert np.allclose(d, _X ** (theta - 1.0), rtol=0, atol=1e-12)


def test_box_cox_derivative_theta_matches_central_fd():
    # (b) dBC/dtheta vs central FD of the transform. Exact branch (|theta|>=0.05)
    # matches tightly; Taylor branch (|theta|<0.05) matches to FD precision (FD
    # near theta=0 carries catastrophic-cancellation noise, so a looser bound).
    h = 1e-5
    for theta in (0.0, 0.01, 0.03, -0.02, 0.049, 0.1, 0.3, -0.8, 1.5):
        d = np.asarray(box_cox_derivative_theta(_X, theta))
        fd = (np.asarray(box_cox_transform(_X, theta + h))
              - np.asarray(box_cox_transform(_X, theta - h))) / (2 * h)
        tol = 1e-3 if abs(theta) < 0.05 else 1e-6
        assert np.max(np.abs(d - fd)) < tol, (theta, np.max(np.abs(d - fd)))


def test_box_cox_transform_matches_jbox_cox():
    # (a) transform vs engine_jax.jbox_cox, theta near 0 and away. Subprocess so
    # jax does not pollute the main process's light-import test.
    if importlib.util.find_spec("jax") is None:
        import pytest
        pytest.skip("jax not installed")
    code = (
        "import numpy as np, jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "import jax.numpy as jnp\n"
        "from dclaborsupply.utility.boxcox import box_cox_transform\n"
        "from dclaborsupply.likelihood.engine_jax import jbox_cox, _load_jax\n"
        "_load_jax()\n"
        "x = np.array([0.2, 0.5, 1.0, 1.5, 3.0, 8.0])\n"
        "worst = 0.0\n"
        "for th in (1e-10, 1e-9, 0.03, 0.3, -0.8, 1.5):\n"
        "    t = np.asarray(box_cox_transform(x, th))\n"
        "    j = np.asarray(jbox_cox(jnp.asarray(x), jnp.asarray(float(th))))\n"
        "    worst = max(worst, float(np.max(np.abs(t - j))))\n"
        "assert worst < 1e-12, worst\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_utility_import_is_light():
    # (e) importing the utility package must not eagerly load jax/gamspy/java
    # (nor numba — the acceleration loads only on direct boxcox import).
    code = (
        "import sys, dclaborsupply.utility\n"
        "for m in ('jax', 'gamspy', 'jpype', 'java', 'numba'):\n"
        "    assert m not in sys.modules, m + ' imported at utility import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
