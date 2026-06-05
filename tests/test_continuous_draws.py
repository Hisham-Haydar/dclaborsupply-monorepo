"""Wave 3.2 tests for continuous opportunity-draw generation (self-contained).

Covers: draw=0 + simulated draws reproducible from a fixed seed, decider draw-grid
completeness, the Wave-0.1 mask (non-working H/W/Occ exactly 0.0), and the masked
log_q_total convention. No FR/EUROMOD/old-repo deps.
"""
import subprocess
import sys

import numpy as np
import pandas as pd

from dclaborsupply.alternatives import build_continuous_alternatives, generate_draws_long

_N = 5


def _synth():
    # heads/partners are deciders; p5 is a non-decider; p3 is a non-working decider.
    return pd.DataFrame({
        "idperson": [1, 2, 3, 4, 5, 6],
        "idhh": [1, 1, 2, 3, 3, 4],
        "lhw": [40.0, 20.0, 0.0, 35.0, 0.0, 50.0],
        "yivwg": [20.0, 15.0, 0.0, 18.0, 0.0, 25.0],
        "hh_IsHead": [1, 0, 1, 1, 0, 1],
        "hh_IsPartner": [0, 1, 0, 0, 0, 0],
        "dgn": [1, 0, 1, 0, 1, 1],
        "lma": [1, 1, 1, 1, 1, 1],
        "loc4": [2, 3, 1, 4, 1, 2],
        "educL": [0, 1, 0, 0, 1, 0],
        "educH": [1, 0, 1, 1, 0, 1],
    })


def test_draws_present_grid_complete_and_reproducible():
    out1 = generate_draws_long(_synth(), n_draws=_N, rng_seed=17, wage_spec="vw")
    out2 = generate_draws_long(_synth(), n_draws=_N, rng_seed=17, wage_spec="vw")

    # (a) draw=0 plus simulated draws present
    assert sorted(int(d) for d in out1["draw"].unique()) == list(range(_N + 1))

    # (e) seed reproducibility — identical numeric output
    cols = ["draw", "hours", "wage", "log_q_state", "log_q_hours",
            "log_q_wage", "log_q_occ", "log_q_total"]
    assert out1[cols].reset_index(drop=True).equals(out2[cols].reset_index(drop=True))

    # (d) decider draw-grid completeness {0..N}
    dec = out1[out1["is_decider"] == 1]
    expected = set(range(_N + 1))
    assert all(set(g["draw"]) == expected for _, g in dec.groupby("idperson_true"))

    # build_continuous_alternatives is a thin alias
    assert build_continuous_alternatives(_synth(), n_draws=_N, rng_seed=17,
                                         wage_spec="vw")[cols].equals(out1[cols])


def _check_mask_and_total(out):
    h = out["log_q_hours"].to_numpy()
    w = out["log_q_wage"].to_numpy()
    o = out["log_q_occ"].to_numpy()
    s = out["log_q_state"].to_numpy()
    tot = out["log_q_total"].to_numpy()
    nonwork = out["hours"].to_numpy() <= 0.0
    z = np.zeros(int(nonwork.sum()))

    # (b) non-working rows: H/W/Occ exactly 0.0
    assert np.array_equal(h[nonwork], z)
    assert np.array_equal(w[nonwork], z)
    assert np.array_equal(o[nonwork], z)

    # (c) Wave-0.1 invariant. Non-working rows: total == state, bit-exact.
    assert np.array_equal(tot[nonwork], s[nonwork])
    # Masked convention over all rows: equal up to IEEE float association
    # (state + (h+w+o) regrouped vs the engine's state+h+w+o); ~1 ULP.
    working = (out["hours"].to_numpy() > 0).astype(float)
    masked = s + working * (h + w + o)
    assert np.allclose(masked, tot, rtol=0.0, atol=1e-12)


def test_mask_and_total_fixed_occ():
    _check_mask_and_total(generate_draws_long(_synth(), n_draws=_N, rng_seed=17,
                                              wage_spec="vw", occ_spec="fixed"))


def test_mask_and_total_empirical_occ():
    # empirical occ contributes log_q_occ on working rows; non-working still exactly 0.0
    _check_mask_and_total(generate_draws_long(_synth(), n_draws=_N, rng_seed=17,
                                              wage_spec="vw", occ_spec="empirical"))


def test_fixed_wage_spec_zeroes_log_q_wage():
    # wage_spec="fw": wage is degenerate at observed -> log_q_wage == 0.0 everywhere
    out = generate_draws_long(_synth(), n_draws=_N, rng_seed=17, wage_spec="fw")
    assert np.array_equal(out["log_q_wage"].to_numpy(),
                          np.zeros(len(out)))


def test_continuous_import_is_light():
    code = (
        "import sys, dclaborsupply.alternatives\n"
        "for m in ('jax', 'gamspy', 'jpype', 'java', 'numba', 'scipy'):\n"
        "    assert m not in sys.modules, m + ' imported at alternatives import!'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
