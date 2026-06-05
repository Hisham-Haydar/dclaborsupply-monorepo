# 03 — `dclaborsupply` Migration Matrix

**Status:** DRAFT v1 for review → freeze after sign-off.
**Document class:** Ordered, gated migration plan. Converts `01_repo_inventory.md` classifications into per-file actions with pre-registered validation gates. No code.
**Grounded against:** `01_repo_inventory.md`, `02_package_architecture_memo.md` §K/§L.

---

## Governing rules (apply to every row)

1. **One file (or one consolidation) → one gate → read result → decide next.** No batch lifts.
2. **Dependencies move before dependents.** A file is not migrated until everything it imports is already in core (or is a base dep: numpy/pandas/pyyaml/scipy).
3. **Pre-registered gate before action.** The "Gate" column states the pass criterion *before* the work; passing is what licenses the next row.
4. **Provenance never moves.** `stays_in_jmp_repo` / `never` rows are out of scope here by construction.
5. **Lift = copy into target module + adapt imports only.** No behavior change during a lift. Behavior fixes (R1) are their own rows with their own gates.
6. **Synthetic recovery is the standard of evidence** for the likelihood path, above in-sample fit.

---

## Wave 0 — In-place fixes in `MNL/` BEFORE anything leaves (decision A)

These happen in the old repo, validated against the certified pipeline, so the thing later lifted is already correct. Reversible refactors only.

| # | Action | Files (in `MNL/`) | Risk | Gate (pre-registered) |
|---|---|---|---|---|
| 0.1 | **R3 — verify `log_prior` provenance.** ✅ DONE (verify-only, no edit). FINDING: R3-as-duplication premise was inaccurate. The `log_prior` formula `log_q_E + working*(log_q_H+log_q_W+log_q_Occ)` is **single-sited** in `enh_RURO_prep_mnl_basic.py` (`_component_log_q_singles/_couples`). `enh_RURO_draws.py` only produces the component columns + a differently-defined unconditional `log_q_total` (distinct quantity). The real risk is a milder producer/consumer convention split: draws pre-zeroes H/W/Occ for non-workers (implicit mask) ⇔ prep masks explicitly with `working*`. **This invariant must be preserved by core `index.py` (1.4).** | `scripts/enhanced/enh_RURO_prep_mnl_basic.py` (formula site), `scripts/enhanced/enh_RURO_draws.py` (component producer) | high | ✅ PASSED. Read-only recompute on certified parquets (singles 505,707 / couples 6,701,638 rows): prep formula vs stored `log_prior` = 3.55e-15 / 7.11e-15 (IEEE round-trip floor; exact-0.0 unreachable by from-scratch recompute since parquet stores post-`exp` only). draws convention ≡ prep convention bit-for-bit (non-working H/W/Occ ≡ 0.0 on full data). **Deferred to Wave 4:** the genuine in-`prep` duplication (two near-identical inline blocks) is `app_package` churn, de-duplicated when prep moves — not a core-spine blocker. |
| 0.2 | **R1 — fix Box-Cox NumPy gradient.** ✅ DONE. Taylor branch (|θ|<0.05) of `box_cox_derivative_theta` had wrong coeffs (linear 1/6→1/3, quad 1/24→1/8) + added cubic θ³L³/15 term → correct to O(θ⁴). Both backend copies (numba+numpy) fixed in one edit. | `scripts/enhanced/estimation_utils.py` | high | ✅ PASSED. Max element-wise rel-error vs central-FD = **9.58e-10** across 55-pt θ-grid spanning 0 (threshold 1e-4, ~5 orders margin). NumPy↔JAX agree 8.3e-8 at all |θ|≥1e-4 points; |θ|≤1e-8 JAX blow-up is JAX's own cancellation (vindicates Taylor branch). Certified θ̂ provably untouched (helper used only by NumPy engine; certification ran JAX/CONOPT). |

**Wave-0 stop condition:** both gates pass; commit in `MNL/` with the gate artifacts. **Status: COMPLETE — 0.1 DONE (verify-only, R3 premise corrected); 0.2 DONE (Box-Cox fixed, FD gate 9.58e-10). Wave 1 may begin.**

---

## Wave 1 — Core spine: spec + likelihood (the load-bearing lift)

Order within the wave is dependency-forced.

| # | Source (`MNL/`) | → Target (core) | Action | Dep prereq | Gate |
|---|---|---|---|---|---|
| 1.1 | `scripts/enhanced/estimation_spec_parser.py` | `spec/parser.py` | Lift full parser, replacing the skeleton stub. No EUROMOD imports to strip (already clean). | none | `EstimationSpec.from_yaml(certified_spec)` parses to **47 free params**, `fixed_params={theta_l_m:-0.8}`, `beta_ll` absent (pinned). Param-binding gate (1.4) will re-confirm. |
| 1.2 | `scripts/bpool/jax_ll_probe.py` (+ `jax_joint_hessian.py`) | `likelihood/engine_jax.py` | Lift JAX builders `build_jax_singles_ll`, `build_jax_couples_ll`, joint assembler. Lazy jax import preserved. | 1.1 | On certified engine-ready data at certified θ̂: negLL reproduces the certified **238362.79** (or the session's 238502.866 staged figure for the staged stem) to tolerance ≤ 1e-3 nats. |
| 1.3 | `scripts/enhanced/estimation_engine.py` (R1-fixed) | `likelihood/engine_numpy.py` | Lift NumPy reference engine. | 0.2, 1.1 | NumPy vs JAX negLL agree ≤ 1e-4 nats at certified θ̂ (cross-check that the R1 fix closed the gap the JAX probe found). |
| 1.4 | (consolidated `log_prior` from 0.1) + index assembly | `likelihood/index.py` | Implement `compute_index(spec, data, theta, *, ruro)`: `v = u + log_h + log_w + log_occ + log_market − log_prior`; `ruro=False` zeroes opportunity + correction. Single canonical `log_prior` site. | 0.1, 1.2, 1.3 | (a) `ruro=True` path reproduces 1.2 negLL. (b) `ruro=False` on a fixed-choice synthetic dataset gives a finite standard-MNL negLL with the opportunity/correction terms identically zero. (c) Param-binding: every spec param binds to θ with no silent drops. |

**Wave-1 stop condition:** certified negLL reproduced through the lifted core (1.2), NumPy↔JAX agree (1.3), index function carries both RUM and RURO paths (1.4). This is the MVP likelihood.

---

## Wave 2 — Core: SE, diagnostics, gates, solvers

| # | Source | → Target | Action | Dep | Gate |
|---|---|---|---|---|---|
| 2.1 | `scripts/enhanced/cluster_robust_se.py` | `se/cluster_robust.py` | Lift; keep chunked sandwich (naive jacrev = 11TB OOM) and T1–T5 verification gates. | 1.2 | Reproduces certified clustered SEs (47/47) at θ̂; T1–T5 pass; 9,657 idorighh clusters. |
| 2.2 | `scripts/enhanced/compute_standard_errors.py` | `se/numerical.py` | Lift central-difference Hessian SEs. | 1.3 | Reproduces certified Hessian SEs at θ̂. |
| 2.3 | `scripts/bpool/jax_optimize.py` | `solvers/jax_optimize.py` | Lift JAX optimizer (used in recovery cert). | 1.2 | Converges from certified warm-start to PD Hessian min_eig > 0. |
| 2.4 | `scripts/enhanced/gamspy_estimation_vectorized.py` (+ `expression_constraints.py`) | `solvers/gamspy_vectorized.py` | Lift behind optional `[gamspy]` extra; lazy import. | 1.1 | Imports without gamspy installed (raises documented ImportError only when called). GAMSPy-present parity vs scipy/JAX LL deferred to app validation. |
| 2.5 | `scripts/bpool/phase_a_param_binding.py`, `phase_b_recovery_test.py`, `joint_recovery_test.py` | `gates/{param_binding,recovery}.py` | Lift PORTABLE recovery + param-binding gates (NOT the provenance `jax_recovery_gate.py`, which stays in JMP). | 1.4 | Param-binding gate passes on certified spec; portable recovery recovers a known synthetic θ* (PD Hessian, all params within band). |
| 2.6 | `scripts/enhanced/diagnostics_bundle.py` | `diagnostics/bundle.py` | Lift model-generic metrics (no EUROMOD imports). | 1.4 | Runs on a synthetic fit; produces finite metrics; no FR constants referenced. |

**Wave-2 stop condition:** SEs reproduced, portable recovery gate green on synthetic θ*, solvers import-clean.

---

## Wave 3 — Core: front-ends, draws, utility, CLI, tests, notebooks

| # | Source | → Target | Action | Dep | Gate |
|---|---|---|---|---|---|
| 3.1 | utility math from `estimation_utils.py` (R1-fixed) | `utility/boxcox.py` | Lift Box-Cox utility evaluation. | 0.2, 1.3 | Matches engine's internal Box-Cox at sample points. |
| 3.2 | `scripts/enhanced/enh_RURO_draws.py` (draw-gen only; EUROMOD-free) | `alternatives/continuous.py` | Lift continuous opportunity-draw generation. Uses consolidated `log_prior`. | 0.1, 1.4 | Regenerates draws with `log_prior` matching the consolidated function (Wave-0 0.1 identity holds). |
| 3.3 | (new, thin) | `models.py` | Replace skeleton stubs: `RUMModel`/`RUROModel` construct `EstimationSpec` + call `compute_index`/optimizer. Thin front-ends, no engine logic. | 1.4, 2.3 | RUM fits synthetic fixed-choice data; RURO fits synthetic latent-jobs data and recovers θ* via 2.5 gate. |
| 3.4 | existing skeleton `cli.py` + emitter logic from `step4_emit_results_json.py` (export shape only) | `cli.py` | Wire `dcls estimate --config --backend --out` to real fit; `summarize` reads result JSON. | 3.3 | `dcls estimate` on synthetic config produces a result JSON; `dcls summarize` reads it. |
| 3.5 | (new) | `tests/`, `notebooks/` | Synthetic-DGP tests (RUM + RURO recovery); fill the two notebooks to run top-to-bottom. | 3.3, 3.4 | `pytest` green incl. recovery; both notebooks execute end-to-end. |

**Wave-3 stop condition = MVP v0.1 (memo §N):** import Java-free; certified spec → 47 params; RUM + RURO synthetic recovery; CLI works; notebooks run.

---

## Wave 4 — App package (post-spine, separate authorisations each)

Lifted only after core v0.1 is frozen. Each is `app_package` per inventory. Listed for completeness; gates drafted when reached.

| Group | Source → Target | Note |
|---|---|---|
| EUROMOD | `enh_RURO_euromod.py`, `run_bpool_euromod*.py`, `src/mnl/integration/euromod.py` → `euromod/{runner,connector}.py` | Implements core `MicrosimConnector`/`WelfareProtocol` interfaces. FR 35h rule, CPI φ stay here. |
| France prep | `enh_france_data_prep.py`, `enh_prepare_FR_gsur_v2.py`, `enh_RURO_prep.py`, `multi_year/m1_*` → `france/{data_prep,gsur,cpi}.py` | All FR SILC/INSEE/NUTS1 constants live here. |
| Pipeline glue | `enh_RURO_prep_mnl_basic.py` (R3-consolidated), `enh_RURO_estimate_FR.py` → app `pipeline/` | `prep_mnl_basic` is EUROMOD-merging → app, but its `log_prior` call uses the consolidated core function. |
| Welfare | `scripts/welfare/welfare_core.py`, `welfare_vdir.py` → `welfare/{core,vdir,measures}.py` | Implements core `WelfareProtocol`. Honors decomposition-readiness flags (contract §7). |
| Reports | `RURO_post_estimation_styled.py` → `reports/post_estimation.py` | Fully dynamic on params already; FR output paths parameterised. |

---

## Never-move register (firewall — reproduced from inventory)

`jax_recovery_gate.py`, `step4_realdata_baseline.py`, `step4_lr_pooling_test.py`, all `validate_chosen_*.py`, certified + gsplit spec YAMLs, `theta_hat_realdata_901_v1.csv`, `scripts/archive/**`. These stay in the JMP repo, which *depends on* the published package. Migrating any of these is out of scope permanently.

---

## Wave dependency summary

```
Wave 0 (in-place R3+R1)  →  Wave 1 (spec+likelihood)  →  Wave 2 (SE/gates/solvers)
                                                          →  Wave 3 (front-ends/CLI/MVP)
                                                                     →  Wave 4 (app, per-group authorisations)
```

*End of migration matrix. Wave 0.1 (R3 `log_prior` consolidation, in-place in `MNL/`) is the first executable step.*
