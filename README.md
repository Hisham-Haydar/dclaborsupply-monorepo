# dclaborsupply

A spec-driven discrete-choice **labour-supply estimation** package. It expresses
a utility/opportunity model from a YAML specification, builds the joint
likelihood, fits it, and reports standard errors and diagnostics.

This repository is a two-package monorepo. This README orients you to the
**core** package (`packages/dclaborsupply`); the application layer is summarized
below and documented separately.

## What the package currently does

- Parses an estimation specification from YAML
  (`EstimationSpec.from_yaml`) — Box-Cox utility over consumption and leisure,
  optional hours, wage, market, and occupation opportunity blocks.
- Builds the joint negative log-likelihood for two model families:
  **RUM** (fixed choice sets) and **RURO** (random utility / random
  opportunity).
- Fits the model with a **JAX** objective and an **L-BFGS-B** optimizer
  (`RUMModel` / `RUROModel`), and computes **Hessian standard errors**.
- Provides a **NumPy reference engine** for likelihood *evaluation* and a
  cross-backend index function (`compute_index`).
- Loads harmonised **engine-ready** parquet/DataFrame bundles into the core
  data containers (`dclaborsupply.data`), spec-aware and country-general.
- Assembles a model-generic **diagnostics bundle**
  (`dclaborsupply.diagnostics.build_diagnostics_bundle`).
- Ships a small **CLI** (`dcls`) with a runnable **synthetic** estimation mode.

The core imports cleanly without Java, JAX, or GAMSPy; those are optional
extras (see Installation). JAX is required only to *fit*.

## Core / app boundary

- **`packages/dclaborsupply` (core).** Microsimulator-free estimation: spec,
  likelihood, solvers, standard errors, gates, the engine-ready loader, and
  diagnostics. No EUROMOD/Java dependency at import.
- **`packages/dclaborsupply_app` (app).** EUROMOD integration, country-specific
  data preparation (e.g. the DE adapter), welfare, and reporting. Depends on
  the core; builds the engine-ready bundles the core consumes.

The contract between a country adapter and the core is the **engine-ready
bundle** — see [docs/engine_ready_contract.md](docs/engine_ready_contract.md).

## Capabilities and caveats (honest)

What is actually proven, and the scope of each claim:

- **Certified FR reproduction.** The lifted core reproduces the certified
  France baseline negative log-likelihood **238504.6360973987** exactly.
- **One non-FR end-to-end configuration.** A DE 2017 minimal configuration runs
  end-to-end on the unchanged core (load → fit → SE → diagnostics). This proves
  **one** non-FR configuration, **not** universal country support.
- **JAX wage specifications.** `fw` (fixed wage) and standard `vw` (log-normal
  variable wage) are JAX-validated. **`loc_empirical` and `vw_occupation` have
  no dedicated JAX implementation and must not be used for JAX estimation**
  until separately implemented and gated — see
  [docs/known_limitations.md](docs/known_limitations.md).
- **Fixed-wage dimension dropping.** Recovery is proven for one identified
  20-free hours-only fixed-wage configuration only — see
  [the validation note](docs/validation/dimension_drop_hours_only_fw.md).
- **CLI real-data estimation is not implemented.** The CLI `estimate` command
  supports only `cli.mode: synthetic`. Real-data fitting uses the Python API
  with caller-provided engine-ready data.

## Installation

The base core depends only on `numpy`, `pandas`, and `pyyaml`:

```bash
pip install -e packages/dclaborsupply
```

Optional extras:

```bash
# Fitting (JAX objective + jax.grad + SciPy L-BFGS-B):
pip install -e "packages/dclaborsupply[jax,solver]"

# Reading parquet engine-ready stems (load_engine_ready_stem / parquet paths):
pip install -e "packages/dclaborsupply[parquet]"

# Optional GAMSPy solver path:
pip install -e "packages/dclaborsupply[gamspy]"
```

DataFrame loading APIs work with base `numpy`+`pandas`; only parquet-*path*
reading needs `[parquet]`. Fitting needs `[jax,solver]`.

## Quickstart (synthetic CLI / front-end smoke)

Requires the `[jax,solver]` extra. Uses
[docs/examples/synthetic_cli_smoke.yaml](docs/examples/synthetic_cli_smoke.yaml),
a runnable synthetic **CLI / front-end smoke**. It exercises the current
RUM-labeled front-end path (spec parser → JAX objective → L-BFGS-B → Hessian
SE) end-to-end. It is **not** a canonical correction-null RUM demonstration:
`RUMModel.fit()` fits the same full JAX objective as `RUROModel.fit()` and does
not null opportunity terms or set `prior == 1`, and this fixture has a
non-uniform `prior` (see [docs/core_concepts.md](docs/core_concepts.md) and
[docs/known_limitations.md](docs/known_limitations.md)).

```bash
dcls validate-config docs/examples/synthetic_cli_smoke.yaml

dcls estimate --config docs/examples/synthetic_cli_smoke.yaml --backend jax \
    --out result.json

dcls summarize --result result.json
```

Expected (abridged): `estimate` prints a finite `neg_ll` and writes
`result.json`; `summarize` prints `"model": "RUM"`, `"n_free": 8`,
`"converged": true`. On this small synthetic fixture one parameter can have a
near-zero/negative Hessian variance (SE reported as NaN) — that is an expected
artifact of the toy fixture, not a fitting failure.

This path is synthetic only. For real-data estimation use the Python API with
your own engine-ready bundle (see the API guide).

## Documentation

- [docs/core_concepts.md](docs/core_concepts.md) — RUM/RURO, the index, the
  core/app split, and what is proven vs. limited.
- [docs/api_guide.md](docs/api_guide.md) — verified public API with examples.
- [docs/engine_ready_contract.md](docs/engine_ready_contract.md) — the
  country-adapter / core data contract.
- [docs/known_limitations.md](docs/known_limitations.md) — JAX wage-path limits
  and the deferred draw canonicalization.
- [docs/validation/dimension_drop_hours_only_fw.md](docs/validation/dimension_drop_hours_only_fw.md)
  — the fixed-wage dimension-drop recovery proof.

## Status

Core v0.1: likelihood, fitting, SE, gates, engine-ready loader, diagnostics, and
a synthetic CLI are implemented and tested. App-layer lifts (EUROMOD, France
prep, welfare, reports) and core optimization/de-redundancy passes are tracked
in [03_migration_matrix.md](03_migration_matrix.md).
