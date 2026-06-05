"""Choice-index dispatcher for RUM and RURO (migration matrix Wave 1.4).

`compute_index` is a thin DISPATCHER over the already-lifted engines
(`engine_jax`, `engine_numpy`). It does NOT re-implement the index — both engines
already build

    v = u + log_h + log_w + log_occ + log_market - log_prior

and return the joint negative log-likelihood. (Occupation is currently folded
into `log_market` in the lifted engines, not a separate `log_occ` array.)

RUM vs RURO is the only genuinely new logic here, and it is expressed as a
*view*, never a re-implementation of v:

  - ruro=True  -> call the lifted engine path unchanged (reproduces the certified
                 negLL).
  - ruro=False -> the SAME engine with the opportunity-density and
                 importance-correction terms identically zero, leaving standard
                 MNL  v = u  over a fixed choice set. The four terms are nulled in
                 `build_rum_view` WITHOUT mutating the caller's spec/data:
                   * hours opportunity   : hours_shifters = []            -> log_h    == 0
                   * wage opportunity    : wage_spec = "fw"               -> log_w    == 0
                   * market + occupation : market_opportunity_shifters=[] -> log_market == 0
                   * IS correction       : data.prior = 1                 -> log_prior == 0

`fixed_params` pinning stays JAX-only for v0.1 (design decision). `compute_index`
boundary-folds pinned values into the parameter vector at their pinned values for
BOTH backends before dispatch (the NumPy engine has no native pinning), so the two
backends evaluate uniformly. The caller's spec/data are never mutated.

Returns a negative-log-likelihood scalar by default (the public name is
historical; not changed in this wave). JAX/NumPy engines are imported lazily
inside the functions, so `import dclaborsupply.likelihood.index` pulls in no jax.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

_GROUP_KEYS = ("singles_male", "singles_female", "couples")


def _as_group_tuple(data: Any) -> Tuple[Any, Any, Any]:
    """Normalize joint `data` to (data_sm, data_sf, data_cou); validate explicitly.

    Accepts either a 3-sequence (data_sm, data_sf, data_cou) or a mapping with
    keys 'singles_male', 'singles_female', 'couples'. Entries may be None (a group
    can be absent); the engines skip None groups.
    """
    if isinstance(data, Mapping):
        missing = [k for k in _GROUP_KEYS if k not in data]
        if missing:
            raise ValueError(
                f"data mapping is missing required keys {missing}; "
                f"expected {list(_GROUP_KEYS)}."
            )
        return data["singles_male"], data["singles_female"], data["couples"]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) != 3:
            raise ValueError(
                f"data sequence must have exactly 3 elements "
                f"(data_sm, data_sf, data_cou); got {len(data)}."
            )
        return data[0], data[1], data[2]
    raise TypeError(
        "data must be a 3-tuple (data_sm, data_sf, data_cou) or a mapping with "
        f"keys {list(_GROUP_KEYS)}; got {type(data).__name__}."
    )


def _fold_fixed_params(spec: Any, theta: Any) -> Tuple[Any, np.ndarray]:
    """Boundary-fold the JAX-only generic `fixed_params` pins into (spec, theta).

    Each pinned param is appended to ``all_param_names`` at its pinned value and
    removed from ``fixed_params``, so both backends evaluate uniformly (the NumPy
    engine never implemented native pinning). The caller's spec is NOT mutated
    (a dataclasses.replace copy is returned). No-op when fixed_params is empty.
    """
    theta = np.asarray(theta, dtype=np.float64)
    fixed = dict(getattr(spec, "fixed_params", {}) or {})
    if not fixed:
        return spec, theta
    new_names = list(spec.all_param_names) + list(fixed.keys())
    theta_eval = np.concatenate(
        [theta, np.array([fixed[k] for k in fixed], dtype=np.float64)]
    )
    spec_eval = dataclasses.replace(spec, all_param_names=new_names, fixed_params={})
    return spec_eval, theta_eval


def build_rum_view(spec: Any, data_tuple: Tuple[Any, Any, Any]) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """Return (spec_rum, data_rum) that null the opportunity + correction terms.

    Produces a standard-MNL evaluation view (v = u over a fixed choice set) by
    making the lifted engine compute every opportunity term and the IS correction
    as identically zero — WITHOUT re-implementing v and WITHOUT mutating the
    caller's spec/data. See module docstring for the per-term nulling.
    """
    spec_rum = dataclasses.replace(
        spec,
        wage_spec="fw",                    # log_w == 0 (engine's fw branch)
        hours_shifters=[],                 # log_h == 0
        market_opportunity_shifters=[],    # log_market (incl. occupation) == 0
    )
    data_rum = []
    for d in data_tuple:
        if d is None:
            data_rum.append(None)
            continue
        dv = copy.copy(d)                  # shallow copy; only `prior` is replaced
        dv.prior = np.ones_like(np.asarray(d.prior, dtype=np.float64))  # log_prior == 0
        data_rum.append(dv)
    return spec_rum, tuple(data_rum)


def compute_index(
    spec: Any,
    data: Any,
    theta: Any,
    *,
    ruro: bool,
    backend: str = "jax",
) -> float:
    """Dispatch to the lifted engines; return the joint negative log-likelihood.

    Parameters
    ----------
    spec : EstimationSpec
    data : (data_sm, data_sf, data_cou) tuple or mapping with the group keys.
    theta : free-parameter vector (length == len(spec.all_param_names)).
    ruro : True  -> full RURO index via the lifted engine (certified path).
           False -> standard MNL (v = u); opportunity + correction nulled via
                    `build_rum_view`, evaluated through the NumPy engine.
    backend : "jax" or "numpy"; validated for all calls. Drives the RURO
              (ruro=True) path. For ruro=False it is IGNORED — the NumPy engine
              is required because the RUM nulling is verified via its per-term
              return_components.
    """
    if backend not in ("jax", "numpy"):
        raise ValueError(f"backend must be 'jax' or 'numpy'; got {backend!r}.")
    data_sm, data_sf, data_cou = _as_group_tuple(data)
    spec_eval, theta_eval = _fold_fixed_params(spec, theta)

    if ruro:
        if backend == "jax":
            # Local imports keep `import ...index` jax-free (light import).
            from dclaborsupply.likelihood.engine_jax import build_joint_neg_ll
            import jax.numpy as jnp
            joint = build_joint_neg_ll(spec_eval, data_sm, data_sf, data_cou)
            return float(joint(jnp.asarray(theta_eval, dtype=jnp.float64)))
        # backend == "numpy"
        from dclaborsupply.likelihood.engine_numpy import compute_likelihood_joint
        return float(
            compute_likelihood_joint(theta_eval, data_sm, data_sf, data_cou, spec_eval)
        )

    # RUM: `backend` is validated above but IGNORED here -- the NumPy engine is
    # required because ruro=False relies on its return_components for the per-term
    # zero verification. Null opportunity + correction via the RUM view, evaluate.
    from dclaborsupply.likelihood.engine_numpy import compute_likelihood_joint
    spec_rum, (d_sm, d_sf, d_cou) = build_rum_view(spec_eval, (data_sm, data_sf, data_cou))
    return float(compute_likelihood_joint(theta_eval, d_sm, d_sf, d_cou, spec_rum))
