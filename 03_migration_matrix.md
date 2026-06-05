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
| 1.2 | `scripts/bpool/jax_ll_probe.py` (+ `jax_joint_hessian.py`) | `likelihood/engine_jax.py` | Lift JAX builders `build_jax_singles_ll`, `build_jax_couples_ll`, joint assembler. Lazy jax import preserved. ✅ DONE. Lifted `jbox_cox`, `jgroup_logsumexp`, `_center_proposal`, both builders, `build_joint_neg_ll`. Dropped CLI/argparse/sys.path + all old-repo imports (`estimation_utils/engine/spec_parser/joint_recovery_test`). Module top-level imports = only `__future__` + numpy; jax bound lazily via `_load_jax()` inside builders. | 1.1 | ✅ PASSED (exact, certified production stem). **TARGET CORRECTED:** original target 238362.79 was the **49-param gsplit** negLL (the spec that FAILED synthetic recovery — `RURO_realdata_2016_2017_joint_901_gsplit_v1.md`), NOT the certified 47-param baseline. Certified 47-param figure = **238504.636097** (`RURO_realdata_2016_2017_joint_901_v1.md`); ~142-nat gap = the 2 extra gsplit free params. On certified production stem `fr_p3a_bpool_engine_ready` at certified θ̂: lifted negLL = 238504.6360973987, **|lift − certified| = 0.0e+00**; lifted ≡ source bit-for-bit (|diff| = 0.0 on sm/sf/couples/JOINT). Lazy jax confirmed (absent from sys.modules after import); zero MNL/old-repo imports in `engine_jax`. Paths: `fr_p3a_bpool_engine_ready__{singles,couples,mnlmeta}` + `theta_hat_realdata_901_v1.csv`. |
| 1.3 | `scripts/enhanced/estimation_engine.py` (R1-fixed) | `likelihood/engine_numpy.py` | Lift NumPy reference engine. ✅ DONE. Byte-faithful `cp` of the engine + import adaptation; BC math → `utility/boxcox.py`, containers/LSE/EPS/HAS_NUMBA → `likelihood/_numpy_primitives.py` (both lifted from `estimation_utils.py`); spec → `dclaborsupply.spec.parser`. **NOT a fully byte-identical lift — two deliberate core-boundary deviations (acceptable for the certified 47-param baseline):** (1) `expression_constraints` penalty hooks were **excluded from the likelihood** and assigned to the solver/optimization layer (Wave 2.4) — the core reference likelihood is the PURE negLL (matches JAX + certified figure; penalty kept as 0.0 for return-shape parity); (2) the AC2013 helper import (`estimation_utils_AC2013.compute_log_age_terms`) was **not lifted** and now **raises NotImplementedError** on that non-certified `is_ac2013()` branch (out of scope, memo §L). Also dropped dead numba decorator block. Zero MNL/old-repo imports in the 3 new modules. | 0.2, 1.1 | ✅ PASSED. NumPy joint negLL = **238504.6360973987**, JAX = **238504.6360973987** → **\|NumPy − JAX\| = 0.0** at full precision (≤ 1e-4 gate) on stem `fr_p3a_bpool_engine_ready` at certified θ̂. R1 fix confirmed lifted (BC θ-derivative vs central-FD max rel-err 8.66e-07, O(θ⁴)). Light import holds (no jax/gamspy on `import dclaborsupply`). MNL sources untouched. NB: NumPy engine doesn't implement the JAX-only generic `fixed_params` pin; validation folds pinned `theta_l_m=-0.8` into the vector at its pinned value (identical likelihood). |
| 1.4 | (RUM/RURO over the lifted engines) | `likelihood/index.py` | Implement `compute_index(spec, data, theta, *, ruro, backend="jax")`. ✅ DONE. **Thin negLL dispatcher** (the name is historical — returns the joint negative log-likelihood scalar, not the raw V vector), **not a third index re-implementation**. `ruro=True` dispatches **unchanged** to the lifted engines (`engine_jax.build_joint_neg_ll` / `engine_numpy.compute_likelihood_joint`). `ruro=False` (RUM) uses the **NumPy** engine via a **non-mutating RUM view** that nulls the opportunity terms + IS correction: `hours_shifters=[]` → log_h=0; `wage_spec="fw"` → log_w=0; `market_opportunity_shifters=[]` → log_market/occupation=0 (occupation is folded into log_market in the lifted engines, no separate `log_occ`); `prior=1` → log_prior correction=0; leaving v=u over a fixed choice set. `fixed_params` are **boundary-folded** into the θ-vector at their pinned values before dispatch (JAX-only pin mechanism made uniform across backends); caller's spec/data are never mutated (`dataclasses.replace`/shallow copy). `backend` validated for all calls; ignored for `ruro=False` (NumPy required for component checks). Data contract `(sm,sf,cou)` tuple or mapping, validated. Zero MNL/old-repo imports; jax imported only inside the function. | 0.1, 1.2, 1.3 | ✅ PASSED (all three). **(a)** `ruro=True` via `compute_index` reproduces the certified RURO negLL on `fr_p3a_bpool_engine_ready` at certified θ̂: jax=**238504.6360973987** (abs-diff vs 238504.636097 = 3.99e-07 ≤1e-4), numpy=238504.6360973987 (abs(NumPy−JAX)=0.0). **(b)** `ruro=False` on a synthetic fixed-choice dataset: finite negLL (3.990391, ll<0) with per-term assertions `log_h==0`, `log_w==0`, `log_market==0`, `V==u` (⇒ log_prior term 0), `prior==1` — all pass. **(c)** Param-binding on the certified spec: 47 free + 1 pinned (`theta_l_m`)=48 covered, no duplicates, pinned-disjoint, index round-trip, θ covers free; runtime JAX eval resolved every `P()` lookup ⇒ no silent drops. Light import holds (`import dclaborsupply`/`.likelihood.index` pull no jax/gamspy). Tests: **13 passed**. |

**Wave-1 stop condition:** certified negLL reproduced through the lifted core (1.2), NumPy↔JAX agree (1.3), index function carries both RUM and RURO paths (1.4). This is the MVP likelihood.

---

## Wave 2 — Core: SE, diagnostics, gates, solvers

| # | Source | → Target | Action | Dep | Gate |
|---|---|---|---|---|---|
| 2.1 | `scripts/enhanced/cluster_robust_se.py` | `se/cluster_robust.py` | Lift the clustered sandwich + T1–T5 checks. ✅ DONE. **Verbatim copy** (the module is pure NumPy — `logging`/`typing`/`numpy` only, zero old-repo imports; nothing to adapt). Lifted `assemble_meat_matrix`, `compute_cluster_robust_se`, `run_t1..t5`; `se/__init__.py` exports them. CLI/workflow (`run_cluster_robust_se.py`) NOT lifted. **Chunking clarification:** `cluster_robust_se.py` itself has NO chunking — it takes pre-computed `scores_all`/`cluster_ids_all`/`hessian` as inputs. The memory-safe (chunked `jax.jacrev`) per-choice-set score extraction lives in the workflow (`step4_realdata_baseline._chunked_scores`/`_slice_data_groups`, chunk 400); the certified `se_clustered` was produced by step4's inline sandwich (same math). No dense full-data jacrev introduced. In core, scores come from the lifted engines. | 1.2 | ✅ PASSED. On certified parquets `fr_p3a_bpool_engine_ready__{singles,couples}` at θ̂ (`theta_hat_realdata_901_v1.csv`), the lifted functions + lifted `engine_jax` (chunked jacrev) reproduce certified `se_clustered` for **all 47/47** params: max abs dev **3.19e-13**, max rel dev **3.80e-12** (tol 1e-6 abs / 1e-4 rel). T1 sign PASS (max_abs_diff 7.4e-12); T2 meat symmetry PASS (0.0); T3 cluster count PASS (**9,657** idorighh); T4 SE positivity PASS (0/47 non-positive); T5 completes, n_below=2 (robust<hessian; informational, not a failure). Light import holds (no jax on `import dclaborsupply.se`); zero MNL/old-repo imports; MNL sources + never-move step4 untouched. Tests: 23 passed. |
| 2.2 | `scripts/enhanced/compute_standard_errors.py` (+ new exact-JAX wrapper) | `se/numerical.py` | ✅ DONE. **Two lanes, both in `se/numerical.py`** (shared `_finalize_se_from_hessian`). **Lane A — `compute_standard_errors`**: central-difference numerical Hessian SE helper, LIFTED from compute_standard_errors.py (only the reusable function; CLI/argparse/data-loading/sys.path + the scipy dep NOT lifted). Scheme preserved exactly (eps=1e-5, central diff on grad, symmetrize, pinv rcond=1e-10). p-values via stdlib `math.erfc` (no scipy). Portable fallback (no jax). **Lane B — `compute_hessian_se`** (NEW): thin exact-JAX wrapper over `engine_jax` (`jax.hessian`), lazy jax import. **Provenance:** the certified `se_hessian` was produced by exact `jax.hessian` (step4), NOT by the central-diff helper — so Lane A carries no byte-identical certified provenance and is gated on a synthetic quadratic, not the certified column. `se/__init__.py` exports both. Zero old-repo imports. | 1.3 | ✅ PASSED. **Lane B (exact-JAX) reproduces certified `se_hessian` for all 47/47** params on `fr_p3a_bpool_engine_ready` at θ̂ (`theta_hat_realdata_901_v1.csv`): max abs 8.8e-14, max rel **2.1e-12** (47/47 within rel≤1e-6). **Lane A (central-diff)** gated on a synthetic quadratic with known exact Hessian (committed test); on real data, fed the exact jax gradient, it agrees with the exact path to max rel **4.2e-9** (a numerical approximation, not byte-identical provenance). **Certified-matching path = exact JAX.** Light import holds (no jax/scipy on `import dclaborsupply.se`). MNL sources + never-move step4 untouched. Tests: 31 passed (8 new). |
| 2.3 | `scripts/bpool/jax_optimize.py` | `solvers/jax_optimize.py` | ✅ DONE. Lift reusable optimizer routines only (CLI/argparse/sys.path/probe/CSV NOT lifted). `build_bounds_list` (verbatim from `_bounds_list`); `optimize_lbfgsb` (SciPy L-BFGS-B over the lifted `engine_jax.build_joint_neg_ll`, jac=`jax.grad`); `polish_optimistix` (optional pure-JAX BFGS, lazy, **not** used for bound-active solutions). Scheme preserved exactly: `method="L-BFGS-B"`, spec bounds, gtol, `ftol=1e-15`, `maxls=60`, maxiter. jax/scipy/optimistix all **lazy** (inside functions). **Dependency decision:** scipy added as a NEW optional extra `solver=["scipy"]` (the L-BFGS-B path needs `[jax]` too → `pip install dclaborsupply[jax,solver]`); optax NOT required; optimistix further-optional (no extra). `solvers/__init__.py` exports the three. Zero old-repo imports. | 1.2 | ✅ PASSED. Lifted SciPy L-BFGS-B no-drift from certified θ̂ on `fr_p3a_bpool_engine_ready` (`theta_hat_realdata_901_v1.csv`): success=True (status 0, 97 iters, CONVERGENCE F<=FACTR*EPSMCH). init negLL 238504.6360973987; final 238504.63609722524; abs(final−238504.6360973987)=**1.7e-7**; max param movement **1.2e-4**. **Exact-JAX Hessian min_eig = 0.459 > 0 → PD (PASS)** (matches certified report). Bound-active params = beta_l_age2_sf hi, beta_l0_m lo, beta_l_age2_f hi (as expected) — so NOT an interior unconstrained Check-5: max raw \|grad\| = 44.2 (bound-active), but max KKT-projected \|grad\| = 0.026. optimistix NOT used in the gate. Light import holds (no jax/scipy/optimistix on `import dclaborsupply.solvers`). MNL source + never-move step4 untouched. Tests: 34 passed (3 new). |
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

## Provenance note (RESOLVED)

**Wave 1.2 target reconciliation — RESOLVED:** the figure `238362.79` originally cited as the Wave 1.2 gate target is the **49-param gsplit** negLL (`238362.788142`, spec `joint_pooled_v1_bll0_tlmpin_gsplit`, `RURO_realdata_2016_2017_joint_901_gsplit_v1.md`) — a *different, non-certified* spec, NOT the certified 47-param baseline. The certified **47-param** baseline negLL is **238504.636097** (`RURO_realdata_2016_2017_joint_901_v1.md`); the ~142-nat gap is exactly the 2 extra gsplit free parameters. The lifted core engine reproduces the certified 47-param figure **exactly** (|lift − certified| = 0.0e+00) on the on-disk production stem `fr_p3a_bpool_engine_ready` at `theta_hat_realdata_901_v1.csv` — there is **no data-provenance blocker**. The earlier "d1w1 `__mnlmeta.json` missing" concern was a red herring induced by the wrong target number; `fr_p3a_bpool_engine_ready` IS the loadable certified production build.

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
