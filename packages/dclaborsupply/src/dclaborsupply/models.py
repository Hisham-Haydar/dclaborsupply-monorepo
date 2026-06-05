"""Thin front-end models over the lifted core (migration matrix Wave 3.3).

``RUMModel`` / ``RUROModel`` are thin front-ends: they accept an ``EstimationSpec``
and data in the core engine contract — a ``(data_sm, data_sf, data_cou)`` tuple or
a mapping with ``singles_male`` / ``singles_female`` / ``couples`` — build the
objective by summing the lifted ``engine_jax`` per-group builders, optimize through
``solvers.jax_optimize.optimize_lbfgsb``, take Hessian SEs through
``se.numerical.compute_hessian_se``, run synthetic recovery through
``gates.recovery``, and return a ``Result``. NO engine/optimizer/score/SE math is
re-implemented here.

Heavy deps (jax/scipy/numba, via the lifted modules) are imported LAZILY inside
methods, so ``import dclaborsupply`` stays light (no jax/scipy/gamspy/numba). No
old-repo/MNL imports; no dataframe->PrecomputedData conversion (deferred).

fixed_params: pinned parameters (e.g. ``theta_l_m``) are boundary-folded through the
lifted objective path — ``engine_jax`` pins them natively in the objective, and the
NumPy V-evaluation used for synthetic recovery folds them via
``index._fold_fixed_params`` — consistent with Waves 1.4 and 2.2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from dclaborsupply.spec.parser import EstimationSpec

_BLOCK_NAMES = ("preference", "hours", "wage", "market", "occupation")


# ---------------------------------------------------------------------------
# Parameter-block classification (conservative, name-based partition)
# ---------------------------------------------------------------------------
def classify_param_block(name: str) -> str:
    """Assign a parameter name to exactly one of the five blocks.

    First-match, exhaustive (``preference`` is the catch-all), so any set of names
    is partitioned with every name assigned exactly once.
    """
    if name.startswith("beta_occ"):
        return "occupation"
    if name.startswith("beta_h_"):
        return "hours"
    if name.startswith("beta_w") or name == "sigma":
        return "wage"
    if name.startswith("beta_E"):
        return "market"
    return "preference"


def _partition_blocks(param_names, theta) -> dict:
    blocks: dict = {b: {} for b in _BLOCK_NAMES}
    for n, v in zip(param_names, theta):
        blocks[classify_param_block(n)][n] = float(v)
    return blocks


# ---------------------------------------------------------------------------
# Data-contract normalization + objective assembly (thin glue over engine_jax)
# ---------------------------------------------------------------------------
def _normalize_data(data):
    """Return a ``(singles_male, singles_female, couples)`` tuple (entries may be None)."""
    if isinstance(data, dict):
        return (data.get("singles_male"), data.get("singles_female"), data.get("couples"))
    if isinstance(data, (tuple, list)):
        if len(data) != 3:
            raise ValueError(
                "data tuple must be (singles_male, singles_female, couples); "
                f"got length {len(data)}."
            )
        return tuple(data)
    raise TypeError(
        "data must be a (singles_male, singles_female, couples) tuple or a mapping "
        "with those keys (precomputed engine data; raw dataframes are not accepted)."
    )


def _build_objective(spec, data_sm, data_sf, data_cou, *,
                     use_actual_choice: bool = False, gender_split=None) -> Callable:
    """Sum the lifted per-group JAX builders for the non-None groups.

    Thin glue over ``engine_jax`` (mirrors ``build_joint_neg_ll`` /
    ``gates.recovery.build_synthetic_recovery_objective``, generalized to partial
    group sets); no likelihood math here. Returns the jitted negLL callable. jax is
    imported lazily.
    """
    from dclaborsupply.likelihood.engine_jax import (
        build_jax_singles_ll, build_jax_couples_ll, _load_jax,
    )

    jax, _jnp = _load_jax()
    parts = []
    if data_sm is not None:
        parts.append(build_jax_singles_ll(data_sm, spec, is_male=True,
                                          use_actual_choice=use_actual_choice,
                                          gender_split=gender_split)[0])
    if data_sf is not None:
        parts.append(build_jax_singles_ll(data_sf, spec, is_male=False,
                                          use_actual_choice=use_actual_choice,
                                          gender_split=gender_split)[0])
    if data_cou is not None:
        parts.append(build_jax_couples_ll(data_cou, spec,
                                          use_actual_choice=use_actual_choice,
                                          gender_split=gender_split)[0])
    if not parts:
        raise ValueError("no data groups provided (all of sm/sf/cou are None).")

    def joint(theta):
        return sum(f(theta) for f in parts)

    return jax.jit(joint)


def _clamp_into_bounds(theta: np.ndarray, bounds) -> np.ndarray:
    out = np.asarray(theta, dtype=float).copy()
    for i, (lo, hi) in enumerate(bounds):
        if lo is not None:
            out[i] = max(out[i], lo)
        if hi is not None:
            out[i] = min(out[i], hi)
    return out


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class Result:
    """Estimation result container (JSON-safe fields + runtime-only caches)."""

    params: dict = field(default_factory=dict)
    theta: list = field(default_factory=list)
    param_names: list = field(default_factory=list)
    blocks: dict = field(default_factory=dict)
    convergence: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    se_hessian: Optional[list] = None
    se_cluster: Optional[list] = None

    def __post_init__(self) -> None:
        # Runtime-only caches; never serialized.
        self._objective: Any = None
        self._spec: Any = None
        self._theta_arr: Any = None
        self._hessian: Any = None
        self._scores_all: Any = None
        self._cluster_ids_all: Any = None

    # -- standard errors -----------------------------------------------------
    def _ensure_hessian_se(self) -> None:
        from dclaborsupply.se.numerical import compute_hessian_se
        out = compute_hessian_se(self._objective, np.asarray(self._theta_arr, dtype=float))
        self.se_hessian = np.asarray(out["se"], dtype=float).tolist()
        self._hessian = np.asarray(out["hessian"], dtype=float)
        eig = np.linalg.eigvalsh(self._hessian)
        self.diagnostics["hessian_min_eig"] = float(eig.min())
        self.diagnostics["hessian_pd"] = bool(eig.min() > 0)

    def attach_cluster_inputs(self, scores_all, cluster_ids_all, hessian=None) -> None:
        """Provide the per-cluster score vectors (+ Hessian) needed for cluster SE.

        The chunked score extraction is upstream and NOT implemented in this
        front-end; pass its outputs here to enable ``se("cluster")``.
        """
        self._scores_all = np.asarray(scores_all, dtype=float)
        self._cluster_ids_all = np.asarray(cluster_ids_all)
        if hessian is not None:
            self._hessian = np.asarray(hessian, dtype=float)

    def se(self, kind: str = "hessian") -> dict:
        if kind == "hessian":
            if self.se_hessian is None:
                if self._objective is None or self._theta_arr is None:
                    raise ValueError(
                        "Hessian SE unavailable: no cached se_hessian and no cached "
                        "objective/theta (e.g. after from_json). Re-fit to compute it."
                    )
                self._ensure_hessian_se()
            return dict(zip(self.param_names, self.se_hessian))

        if kind == "cluster":
            if self.se_cluster is None:
                if (self._scores_all is None or self._cluster_ids_all is None
                        or self._hessian is None):
                    raise ValueError(
                        "Cluster-robust SE requires scores_all + cluster_ids_all + "
                        "hessian (per-cluster score vectors from the chunked score "
                        "extraction). None are cached; supply them via "
                        "attach_cluster_inputs(...). This front-end does not implement "
                        "score math."
                    )
                from dclaborsupply.se.cluster_robust import compute_cluster_robust_se
                se_rob, _vc, _B = compute_cluster_robust_se(
                    self._hessian, self._scores_all, self._cluster_ids_all)
                self.se_cluster = np.asarray(se_rob, dtype=float).tolist()
            return dict(zip(self.param_names, self.se_cluster))

        raise ValueError(f"unknown se kind {kind!r}; use 'hessian' or 'cluster'.")

    # -- reporting / prediction ---------------------------------------------
    def summary(self) -> dict:
        return {
            "model": self.metadata.get("model"),
            "n_free": self.metadata.get("n_free"),
            "n_fixed": self.metadata.get("n_fixed"),
            "neg_ll": self.convergence.get("neg_ll"),
            "converged": self.convergence.get("success"),
            "status": self.convergence.get("status"),
            "hessian_min_eig": self.diagnostics.get("hessian_min_eig"),
            "blocks": {b: len(self.blocks.get(b, {})) for b in _BLOCK_NAMES},
        }

    def predict(self, data: Any) -> dict:
        """Evaluate the fitted objective on precomputed data (returns its negLL).

        Raw-dataframe prediction is out of scope in v0.1 (no df->PrecomputedData
        conversion) and raises NotImplementedError.
        """
        if type(data).__name__ == "DataFrame" or hasattr(data, "to_parquet"):
            raise NotImplementedError(
                "raw dataframe prediction is not supported in v0.1; pass precomputed "
                "engine data (a (sm, sf, cou) tuple/mapping)."
            )
        if self._spec is None or self._theta_arr is None:
            raise ValueError("predict requires a fitted Result (cached spec + theta).")
        sm, sf, cou = _normalize_data(data)
        obj = _build_objective(self._spec, sm, sf, cou, use_actual_choice=False)
        import jax.numpy as jnp
        neg_ll = float(obj(jnp.asarray(self._theta_arr, dtype=jnp.float64)))
        return {"neg_ll": neg_ll}

    # -- serialization (JSON-safe fields only) ------------------------------
    _JSON_FIELDS = ("params", "theta", "param_names", "blocks", "convergence",
                    "metadata", "diagnostics", "se_hessian", "se_cluster")

    def to_json(self) -> str:
        return json.dumps({k: getattr(self, k) for k in self._JSON_FIELDS})

    @classmethod
    def from_json(cls, s: str) -> "Result":
        d = json.loads(s)
        return cls(**{k: d.get(k) for k in cls._JSON_FIELDS})


# ---------------------------------------------------------------------------
# Front-end models
# ---------------------------------------------------------------------------
def _resolve_spec(spec) -> EstimationSpec:
    if spec is None:
        raise ValueError(
            "no spec set; construct the model with from_spec(EstimationSpec) or pass "
            "spec=... (this front-end does not synthesize a spec)."
        )
    return spec


def _make_result(spec, res: dict, objective, *, model: str, backend: str,
                 compute_se: bool) -> Result:
    theta = np.asarray(res["theta_hat"], dtype=float)
    names = list(spec.all_param_names)
    result = Result(
        params={n: float(v) for n, v in zip(names, theta)},
        theta=theta.tolist(),
        param_names=names,
        blocks=_partition_blocks(names, theta),
        convergence={
            "success": bool(res["success"]),
            "status": int(res["status"]),
            "message": str(res["message"]),
            "n_iter": int(res["n_iter"]),
            "neg_ll": float(res["neg_ll"]),
            "max_abs_grad": float(res["max_abs_grad"]),
        },
        metadata={
            "model": model,
            "backend": backend,
            "spec_name": getattr(spec, "name", None),
            "n_free": len(names),
            "n_fixed": len(getattr(spec, "fixed_params", {}) or {}),
            "fixed_params": {k: float(v) for k, v in (getattr(spec, "fixed_params", {}) or {}).items()},
        },
        diagnostics={},
    )
    result._objective = objective
    result._spec = spec
    result._theta_arr = theta
    if compute_se:
        result._ensure_hessian_se()
    return result


@dataclass(slots=True)
class RUMModel:
    """RUM front-end (fixed choice sets; observed alternative at column 0)."""

    utility: str = "box_cox"
    choice_col: str = "chosen"
    unit_col: str = "idorighh"
    spec: Optional[EstimationSpec] = None

    @classmethod
    def from_spec(cls, spec: EstimationSpec) -> "RUMModel":
        return cls(spec=spec)

    def fit(self, data: Any, *, backend: str = "jax", warm_start: Any | None = None,
            compute_se: bool = True, gender_split=None, maxiter: int = 2000) -> Result:
        spec = _resolve_spec(self.spec)
        sm, sf, cou = _normalize_data(data)
        objective = _build_objective(spec, sm, sf, cou, use_actual_choice=False,
                                     gender_split=gender_split)
        from dclaborsupply.solvers.jax_optimize import build_bounds_list, optimize_lbfgsb
        bounds = build_bounds_list(spec)
        theta0 = (np.asarray(warm_start, dtype=float) if warm_start is not None
                  else np.asarray(spec.get_initial_vector(), dtype=float))
        theta0 = _clamp_into_bounds(theta0, bounds)
        res = optimize_lbfgsb(objective, theta0, bounds, maxiter=maxiter)
        return _make_result(spec, res, objective, model="RUM", backend=backend,
                            compute_se=compute_se)


@dataclass(slots=True)
class RUROModel:
    """RURO front-end (latent opportunities; importance-sampling-corrected index)."""

    utility: str = "box_cox"
    opportunity: Any | None = None
    correction: str = "importance_sampling"
    choice_col: str = "chosen"
    unit_col: str = "idorighh"
    spec: Optional[EstimationSpec] = None

    @classmethod
    def from_spec(cls, spec: EstimationSpec) -> "RUROModel":
        return cls(spec=spec)

    def fit(self, data: Any, *, backend: str = "jax", warm_start: Any | None = None,
            compute_se: bool = True, gender_split=None, maxiter: int = 2000) -> Result:
        spec = _resolve_spec(self.spec)
        sm, sf, cou = _normalize_data(data)
        objective = _build_objective(spec, sm, sf, cou, use_actual_choice=False,
                                     gender_split=gender_split)
        from dclaborsupply.solvers.jax_optimize import build_bounds_list, optimize_lbfgsb
        bounds = build_bounds_list(spec)
        theta0 = (np.asarray(warm_start, dtype=float) if warm_start is not None
                  else np.asarray(spec.get_initial_vector(), dtype=float))
        theta0 = _clamp_into_bounds(theta0, bounds)
        res = optimize_lbfgsb(objective, theta0, bounds, maxiter=maxiter)
        return _make_result(spec, res, objective, model="RURO", backend=backend,
                            compute_se=compute_se)

    def recover_synthetic(self, data: Any, *, seed: int = 0, band: float = 0.5,
                          perturb: float = 0.05, theta_star=None, gender_split=None) -> Result:
        """Portable synthetic recovery via the Wave-2.5 ``gates.recovery`` path.

        Draws synthetic choices from a known theta* (``use_actual_choice=True`` joint),
        re-estimates via the lifted solver, and reports the PD-Hessian verdict + band.
        Delegates entirely to ``gates.recovery`` — no recovery math here.
        """
        spec = _resolve_spec(self.spec)
        sm, sf, cou = _normalize_data(data)
        from dclaborsupply.gates.recovery import (
            generate_theta_star, synthesize_actual_choices,
            build_synthetic_recovery_objective, recover,
        )
        from dclaborsupply.solvers.jax_optimize import build_bounds_list

        rng = np.random.default_rng(seed)
        ts = (np.asarray(theta_star, dtype=float) if theta_star is not None
              else generate_theta_star(spec, rng))
        sm2, sf2, cou2 = synthesize_actual_choices(spec, (sm, sf, cou), ts, rng)
        objective = build_synthetic_recovery_objective(spec, sm2, sf2, cou2,
                                                       gender_split=gender_split)
        bounds = build_bounds_list(spec)
        names = list(spec.all_param_names)
        out = recover(objective, ts, bounds, seed=seed + 1, band=band, perturb=perturb,
                      param_names=names)
        theta = np.asarray(out["theta_hat"], dtype=float)
        result = Result(
            params={n: float(v) for n, v in zip(names, theta)},
            theta=theta.tolist(),
            param_names=names,
            blocks=_partition_blocks(names, theta),
            convergence={"success": bool(out["optimizer_success"])},
            metadata={
                "model": "RURO-recovery",
                "spec_name": getattr(spec, "name", None),
                "n_free": len(names),
                "n_fixed": len(getattr(spec, "fixed_params", {}) or {}),
                "seed": seed, "band": band, "perturb": perturb,
            },
            diagnostics={
                "hessian_min_eig": float(out["min_eig"]),
                "hessian_pd": bool(out["pd_ok"]),
                "max_dev": float(out["max_dev"]),
                "within_band": bool(out["within_band"]),
                "worst_param": out["worst_param"],
                "theta_star": np.asarray(ts, dtype=float).tolist(),
            },
        )
        result._objective = objective
        result._spec = spec
        result._theta_arr = theta
        result._hessian = None
        return result
