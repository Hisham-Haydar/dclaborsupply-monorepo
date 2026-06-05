# 02 — `dclaborsupply` Package Architecture Memo

**Status:** FROZEN v1 (signed off 2026-06-05). Changes require a v2.
**Document class:** Architecture blueprint — defines the API, module tree, config schema, and v0.1 scope for the `dclaborsupply` monorepo. Contains no code.
**Grounded against:** `01_repo_inventory.md`, certified spec `estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml`, `estimation_spec_parser.py`, `RURO_welfare_scaffold_design_contract_v2.md`, `JMP_welfare_spec_v5.md`.

---

## Resolved decisions (this freeze)

1. **Repo location:** a **separate new repo** (the monorepo). The existing JMP/research repo *depends on* it; package code is never developed inside `MNL/`. This keeps the "don't blindly rewrite" firewall and the published-package boundary clean.
2. **GAMSPy:** lives in core `solvers/` but as an **optional extra** (`pip install dclaborsupply[gamspy]`). Base install stays light and Java-free. Must be **well-documented**: README + a dedicated docs page on when GAMSPy is needed, install prerequisites, and the fact that the default backend is JAX.
3. **Package name:** `dclaborsupply` (import alias `dcls`). Final.

---

## A. Mission

`dclaborsupply` is a discrete-choice labour-supply estimation library with two user trajectories — **RUM** (preferences over a fixed choice set) and **RURO** (preferences *and* a latent opportunity density) — usable identically from notebooks, scripts, and CLI. The core is microsimulator-agnostic and testable without Java. EUROMOD pricing, French data prep, and opportunity-sensitive welfare/decomposition live in a separate application package `dclaborsupply_app` that depends on the core.

**The unifying design fact** (confirmed from the certified YAML): the model is *already* fully config-driven, and the choice index decomposes into named blocks:

```
v_ij = u(c,ℓ; prefs)  +  log_h  +  log_w  +  log_occ  +  log_market  −  log_prior
       └─ preference ─┘  └──────── opportunity density g ────────┘   └─ IS correction ─┘
```

RUM is the special case `log g − log_prior ≡ 0`. **The package is one estimator over this index; the RUM/RURO fork is whether the opportunity/correction terms are present.** Every API choice below follows from this.

---

## B. Core conceptual objects

| Object | Role | Maps to existing |
|---|---|---|
| `Household` / unit | Estimation + welfare unit; couples never split | `idorighh` cluster, 4-group `_sm/_sf/_m/_f` |
| `AlternativeSet` | The job packages a unit chooses among | continuous draws (v1) / job-grid (stub) |
| `UtilitySpec` | Preference block: Box-Cox over (c, ℓ), shifters | `utility:` YAML block |
| `OpportunityDensity` | The 4 opportunity blocks + IS correction (RURO only) | `hours_/wage_/market_/occupation_opportunity:` |
| `EstimationSpec` | Frozen parsed spec; param names, bounds, fixed_params | existing dataclass — **reused, not rebuilt** |
| `ChoiceModel` | The likelihood machine; RUM or RURO by presence of `OpportunityDensity` | `compute_log_likelihood`, JAX builders |
| `EstimationResult` | θ̂, SEs (Hessian + cluster-robust), diagnostics, convergence | `estimation_results.json` shape |
| `WelfareProtocol` | **Interface only in core**; impl in app | welfare contract §3 |

**Block partition is first-class.** The result object exposes `result.blocks = {preference, hours, wage, market, occupation}` so the decomposition can equalize a block without refactoring (welfare contract §7). This is the single most important interface guarantee for the JMP.

---

## C. RUM workflow

```python
import dclaborsupply as dcls

model = dcls.RUMModel(
    utility="box_cox",          # box_cox | log | linear
    choice_col="chosen",
    unit_col="idorighh",
)
result = model.fit(long_df)     # fixed alternatives already in long_df
result.summary()
result.predict(long_df)
```

No opportunity density, no `log_prior`. Likelihood = Σ log P(chosen | alternatives).

---

## D. RURO workflow

```python
model = dcls.RUROModel(
    utility="box_cox",
    opportunity=dcls.OpportunityDensity(
        employment="logit",                 # participation
        hours="discrete_mixture",
        wage="log_normal",                   # ← or "occupation_specific_log_normal" (4-density)
        occupation="categorical",            # ISCO/loc4 — opportunity layer ONLY
        market=["gsur", "region", "year"],
    ),
    correction="importance_sampling",        # the −log_prior term
    unit_col="idorighh",
)
result = model.fit(long_df)
g_hat = result.opportunity_density(long_df)  # predicted feasible-set object
```

The `wage="occupation_specific_log_normal"` switch *is* the four-density question — already a config value in the parser, exposed here as one API knob. **This makes the spec-matrix experiments config changes, not code changes** — the real test of the agnosticism claim.

Both `RUMModel` and `RUROModel` are thin front-ends constructing the same `EstimationSpec` and calling the same `ChoiceModel.fit`. They are convenience constructors, not separate engines.

---

## E. Minimal public API

```python
dcls.RUMModel(...) / dcls.RUROModel(...)        # constructors
dcls.EstimationSpec.from_yaml(path)             # the existing parser, surfaced
dcls.load_alternatives(...)                     # long-format choice-set loader/validator
model.fit(df, *, backend="jax"|"numpy", warm_start=None)
result.summary() / .params / .blocks / .se(kind="cluster"|"hessian") / .predict(df)
result.opportunity_density(df)                  # RURO only
result.to_json(path) / dcls.Result.from_json(path)
```

Backends: `numpy` (reference/portable) and `jax` (production; the certified path). GAMSPy lives in core `solvers/` but is an optional extra (decision 2).

---

## F. Internal module tree (separate monorepo)

```
dclaborsupply-monorepo/
├── packages/
│   ├── dclaborsupply/                 # CORE — no EUROMOD, no FR constants
│   │   ├── pyproject.toml
│   │   └── src/dclaborsupply/
│   │       ├── __init__.py            # RUMModel, RUROModel, EstimationSpec, Result
│   │       ├── models.py              # front-end constructors
│   │       ├── spec/parser.py         # ← estimation_spec_parser.py (lifted)
│   │       ├── likelihood/
│   │       │   ├── index.py           # v = u + log g − log_prior  (ONE site; R3 fix)
│   │       │   ├── engine_numpy.py    # ← estimation_engine.py (R1 fixed first)
│   │       │   └── engine_jax.py      # ← jax_ll_probe.py + joint_hessian
│   │       ├── utility/boxcox.py
│   │       ├── opportunity/           # hours / wage / market / occupation densities
│   │       ├── alternatives/continuous.py   # (+ job_grid.py STUB)
│   │       ├── se/{cluster_robust,numerical}.py
│   │       ├── solvers/{jax_optimize,gamspy_vectorized}.py
│   │       ├── diagnostics/bundle.py
│   │       ├── gates/{recovery,param_binding}.py   # portable recovery tests
│   │       ├── welfare/protocol.py    # INTERFACE ONLY (Protocol/ABC)
│   │       ├── config/{schema,loader}.py
│   │       └── cli.py                 # dcls estimate/summarize/validate-config
│   └── dclaborsupply_app/             # APPLICATION — depends on core
│       └── src/dclaborsupply_app/
│           ├── euromod/{connector,runner}.py   # the MicrosimConnector impl
│           ├── france/{data_prep,gsur,cpi}.py
│           ├── welfare/{core,vdir,measures}.py # implements core's WelfareProtocol
│           └── reports/post_estimation.py
├── notebooks/{01_rum_quickstart,02_ruro_workflow}.ipynb
└── tests/  (per-package)
```

The JMP/paper-reproduction repo (provenance gates, `step4_realdata_baseline.py`, certified specs) **stays where it is** and *depends on* this monorepo. Provenance is never moved.

---

## G. Config schema

The core config schema **is the certified YAML schema** — `specification / utility / hours_opportunity / wage_opportunity / market_opportunity / occupation_opportunity / fixed_params / optimization / reporting`, plus the generic `fixed_params` and `gender_split` mechanisms. The package does not invent a schema; it adopts and documents the one already in production. `welfare.blocks`, `unit.pool_opportunity_share`, and `decomposition_readiness.*` (welfare contract §7) are app-layer config extensions.

---

## H. Notebook usage

```python
import dclaborsupply as dcls
spec = dcls.EstimationSpec.from_yaml("configs/ruro_baseline.yaml")
result = dcls.RUROModel.from_spec(spec).fit(df)
result.summary()
```

## I. CLI usage

```bash
dcls validate-config configs/ruro_baseline.yaml
dcls estimate --config configs/ruro_baseline.yaml --backend jax --out result.json
dcls summarize --result result.json
```

---

## J. Stays OUT of core

EUROMOD/Java, FR CPI/NUTS1/INSEE/GSUR, 35h rule, welfare *implementation*, Shapley decomposition, all provenance gates and certified artifacts. Core exposes welfare only as a `Protocol`.

---

## K. Migrate first (v0.1 spine order)

1. **`log_prior` consolidation** (R3) — one canonical `index.py` site, before any boundary.
2. **Box-Cox NumPy fix** (R1) — proven by analytical-vs-FD gate; no re-estimation (certified used JAX/CONOPT).
3. `spec/parser.py` (lifts cleanly, no EUROMOD imports).
4. `likelihood/` (numpy + jax) behind the single index function.
5. `se/`, `diagnostics/`, `gates/` (portable recovery test).
6. RUM + RURO front-ends + CLI; synthetic-DGP tests; two notebooks.

## L. Do NOT migrate yet

Job-choice RURO (stub the `job_grid` constructor), nested/mixed/probit (declare in API, raise `NotImplementedError`), welfare impl, France data prep, EUROMOD runner — all app-layer, post-spine.

---

## M. Resolved open decisions

1. Repo location → **separate new repo** (the monorepo); JMP repo depends on it.
2. GAMSPy → **optional extra** in core `solvers/`, base install Java-free, well-documented.
3. Name → **`dclaborsupply`** (alias `dcls`), final.

---

## N. MVP v0.1 definition

Installs editable; `import dclaborsupply` succeeds with **no Java on PATH**; `EstimationSpec.from_yaml(certified_spec)` parses to **47 free params**; RUM fits a synthetic fixed-choice dataset; RURO fits a synthetic latent-jobs dataset and **recovers known θ\*** through the portable recovery gate; `dcls --help` works; two notebooks run top-to-bottom. No EUROMOD, no welfare numbers, no France data required.

---

*End of frozen architecture memo. Implementation is gated on the v0.1 skeleton (Step 3) plus the spine-migration steps in §K.*
