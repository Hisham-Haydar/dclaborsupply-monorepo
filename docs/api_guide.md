# API guide (verified)

Every call below was verified against the source. Signatures are quoted as they
appear in the package. Private/internal modules (for example
`dclaborsupply.likelihood._numpy_primitives`) are **not** part of the public API
and do not appear in any example here.

## Public surface

Top-level (`import dclaborsupply`) — `spec/parser.py` and `models.py`:

- `EstimationSpec.from_yaml(path) -> EstimationSpec`
- `EstimationSpec.get_initial_vector(self) -> np.ndarray`
- `RUMModel.from_spec(spec) -> RUMModel`
- `RUROModel.from_spec(spec) -> RUROModel`
- `RUMModel.fit` / `RUROModel.fit`:
  `(data, *, backend="jax", warm_start=None, compute_se=True,`
  `gender_split=None, maxiter=2000) -> Result`
- `RUROModel.recover_synthetic(data, *, seed=0, band=0.5,`
  `perturb=0.05, theta_star=None, gender_split=None) -> Result`
- `Result.summary(self) -> dict`
- `Result.se(self, kind="hessian") -> dict`
- `Result.to_json(self) -> str`, `Result.from_json(s) -> Result`

Public submodule APIs:

- `dclaborsupply.data.load_engine_ready_stem(stem, spec, *,`
  `year_tags=None, hours_band_policy="assembled") -> (sm, sf, cou)`
  — `data/loader.py`
- `dclaborsupply.data.load_singles(source, spec, *, is_male,`
  `metadata=None, hours_band_policy="assembled")` — `data/loader.py`
- `dclaborsupply.data.load_couples(source, spec, *, metadata=None,`
  `hours_band_policy="assembled")` — `data/loader.py`
- `dclaborsupply.likelihood.index.compute_index(spec, data, theta,`
  `*, ruro, backend="jax") -> float` — `likelihood/index.py`
- `dclaborsupply.diagnostics.build_diagnostics_bundle(*, profile,`
  `results_data, parsed_params, fit_stats, bound_diagnostics, ...)`
  `-> DiagnosticsBundle` — `diagnostics/bundle.py`

## Parsing a specification

```python
from dclaborsupply import EstimationSpec

spec = EstimationSpec.from_yaml("path/to/spec.yaml")
names = spec.all_param_names          # free-parameter names, in order
theta0 = spec.get_initial_vector()    # initial values aligned to names
```

## Engine-ready fitting workflow (verified API template)

This is a **verified API template**, not a repository-runnable script: it
requires a **caller-provided engine-ready stem** (see
[engine_ready_contract.md](engine_ready_contract.md)). The repository ships no
real engine-ready data, so substitute your own bundle path. For a path you can
run as-is, use the synthetic CLI quickstart in the [README](../README.md).

```python
from dclaborsupply import EstimationSpec, RUROModel
from dclaborsupply.data import load_engine_ready_stem

spec = EstimationSpec.from_yaml("path/to/spec.yaml")

# Bundle: <stem>__singles.parquet, <stem>__couples.parquet,
#         <stem>__mnlmeta.json  (caller-provided)
sm, sf, cou = load_engine_ready_stem(
    "path/to/engine_ready_stem", spec, hours_band_policy="assembled"
)

result = RUROModel.from_spec(spec).fit(
    (sm, sf, cou), backend="jax", compute_se=True
)

print(result.summary())          # model, n_free, neg_ll, converged, ...
standard_errors = result.se("hessian")   # {param_name: SE}
payload = result.to_json()       # JSON-safe string for persistence
```

`fit` only supports `backend="jax"` (the objective is JAX-built and optimized
with L-BFGS-B). `compute_se=True` computes Hessian standard errors at the fitted
point.

## Likelihood evaluation only (NumPy)

`compute_index` evaluates the joint negative log-likelihood at a parameter
vector. The NumPy backend is for **evaluation only** — there is no NumPy
optimizer; fitting is JAX-only.

```python
from dclaborsupply.likelihood.index import compute_index

theta = spec.get_initial_vector()
neg_ll = compute_index(spec, (sm, sf, cou), theta, ruro=True, backend="numpy")
```

Use `ruro=True` for the random-opportunity model and `ruro=False` for RUM.

## Diagnostics (honest)

There are two distinct things, at two levels:

- **`Result.diagnostics`** is a small dictionary attached to a fitted result.
  When `compute_se=True` it carries `hessian_min_eig` and `hessian_pd`. It is a
  convenience summary, not a full report.
- **`build_diagnostics_bundle`** is a **lower-level public builder**. It does
  *not* take a `Result`; it assembles a model-generic `DiagnosticsBundle` from
  already-computed payloads (`results_data`, `parsed_params`, `fit_stats`,
  `bound_diagnostics`, and optional `hessian_diagnostics`, `solver_diag`, …).
  Missing optional payloads degrade to sections marked unavailable with a
  reason. **There is no one-call `Result` → `DiagnosticsBundle` helper** — the
  caller supplies the payloads.

```python
result.diagnostics
# e.g. {"hessian_min_eig": -0.0019, "hessian_pd": False}

from dclaborsupply.diagnostics import build_diagnostics_bundle
# build_diagnostics_bundle(profile=..., results_data=..., parsed_params=...,
#                          fit_stats=..., bound_diagnostics=..., ...)
# Assemble the payloads yourself; there is no Result-to-bundle shortcut.
```

## Notes

- The only runnable **CLI** example is the synthetic smoke (README
  quickstart); the repository also ships two runnable notebooks
  (`notebooks/01_rum_quickstart.ipynb`, `notebooks/02_ruro_workflow.ipynb`).
  The engine-ready workflow above is a template needing caller data.
- `fw` and standard `vw` wage specs are JAX-validated; `loc_empirical` and
  `vw_occupation` must not be used for JAX estimation
  (see [known_limitations.md](known_limitations.md)).
