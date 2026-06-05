# MNL Repository — Architectural Inventory (v3)

**Survey date:** 2026-06-05  
**Method:** Read-only; paths verified via `find`, line counts via `wc -l` on `C:\Users\hisham\Repo\MNL`.  
**Note on repo compliance:** The original prompt said output-only / do not modify the repo. This file was subsequently requested as a repo-resident artefact; it is tracked as `?? 01_repo_inventory.md` (untracked) and can be deleted without affecting any production file.  
**Note on line counts:** Measured in the canonical local repo. A contested figure (user cited 2054 / 1588 for `estimation_engine.py` / `estimation_utils.py`; `wc -l` gives 2446 / 1799) likely reflects a stale network-replica copy.

---

## Classification schema

**Class** (exactly one per row):  
`reusable_core_candidate` · `application_layer_candidate` · `EUROMOD_specific` ·  
`welfare_specific` · `diagnostics_reporting` · `configuration` · `tests_or_gates` ·  
`output_or_provenance` · `scratch_or_temporary` · `unclear_needs_review`

**Target** (one per row):  
`core_package` · `app_package` · `stays_in_jmp_repo` · `never`

**Priority** (migration urgency): `high` · `medium` · `low` · `never`

**Risk** (migration risk if file is moved/renamed/edited without care): `high` · `medium` · `low` · `never` (`never` extends the original `{high/medium/low}` enum — it marks files that are not migration candidates; migrating something that must never move is distinct from `low`)

---

## A. Executive Summary

This repository implements a **Random Utility Random Opportunity (RURO)** discrete-choice labour supply model for France (SILC/EUROMOD, 2015–2017), estimated by maximum likelihood with importance-sampling correction. Two active estimation branches exist: continuous RURO (`scripts/enhanced/`) and job-choice RURO (`scripts/Job_model/`). An archived translog/Box-Cox RUM approach (`scripts/archive/rum_approach/`) predates RURO and must not be migrated.

**Certified baseline:** 47-param spec `scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml` — negLL 238 362.79, Hessian min\_eig +1.706. Script: `scripts/bpool/step4_realdata_baseline.py`. NEVER move or edit.

**49-param gsplit** (`scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_tlmpin_gsplit.yaml`): FAILED synthetic recovery gate (tight-SE bias; beta\_h\_pt2\_m err/SE = 19). NOT the paper baseline.

**Five critical risks:**

| # | Risk | Evidence |
|---|------|----------|
| R1 | Box-Cox Taylor bug in `scripts/enhanced/estimation_utils.py:box_cox_derivative_theta` (~0.5 off near θ = 0) | Confirmed by JAX/FD cross-check in `scripts/bpool/jax_ll_probe.py`; documented in project memory |
| R2 | `src/mnl/models/mnl.py` exposes statsmodels MNLogit — wrong model entirely | 39-line file wraps `MNLogit`; no RURO content |
| R3 | `log_prior` formula split across `enh_RURO_draws.py` and `enh_RURO_prep_mnl_basic.py` | Byte-identical required; silent divergence risk on refactor |
| R4 | `ensure_local_workdir()` in `scripts/enhanced/path_helpers.py` is a hard runtime dependency for GAMSPy on network drives | Must survive any import restructuring |
| R5 | `gsplit` spec shares directory with certified spec, not labeled as non-certified | Both in `scripts/bpool/specs/`; risk of confusion |

---

## B. File Classification Table

Columns: **Current path** (from repo root) · **Lines** · **Purpose** · **Class** · **Target** · **Priority** · **Risk** · **Notes**

### scripts/enhanced/ — production estimation engine

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/enhanced/estimation_engine.py` | 2,446 | NumPy/Numba likelihood + analytical gradients; `compute_likelihood_singles/couples()`, `compute_gradient_singles/couples()`, Box-Cox utility, hours/wage opportunity derivatives | `reusable_core_candidate` | `core_package` | high | high | ⚠ R1: Box-Cox Taylor bug in `estimation_utils`; `_USE_NUMBA` flag sensitive to import restructuring; fix bug before publishing |
| `scripts/enhanced/estimation_utils.py` | 1,799 | `PrecomputedData` containers, Box-Cox math, log-sum-exp | `reusable_core_candidate` | `core_package` | high | high | ⚠ R1: `box_cox_derivative_theta` Taylor bug confirmed by JAX probe (`jax_ll_probe.py`); ~0.5 off near θ = 0; fix before publish |
| `scripts/enhanced/estimation_utils_AC2013.py` | 770 | Utility math variants following Aaberge–Colombino 2013 | `reusable_core_candidate` | `core_package` | low | low | Variant of `estimation_utils` used by AC2013 spec |
| `scripts/enhanced/estimation_spec_parser.py` | 1,905 | YAML → `EstimationSpec`; 4-group architecture (`_sm _sf _m _f`); fixed\_params, gender-split, warmstart | `reusable_core_candidate` | `core_package` | high | medium | No EUROMOD imports; safe to migrate; gender-split block must stay spec-driven |
| `scripts/enhanced/gamspy_estimation_vectorized.py` | 1,871 | Vectorised GAMSPy CONOPT/IPOPT; `estimate_joint_vectorized_gamspy()`; 3–5× faster expression build | `reusable_core_candidate` | `core_package` | high | medium | Depends on `path_helpers`; R4 UNC workaround required; 94% wall time is model generation (R7) |
| `scripts/enhanced/gamspy_estimation.py` | 2,563 | Non-vectorised GAMSPy variant (predecessor to vectorised) | `unclear_needs_review` | `stays_in_jmp_repo` | low | low | Exists alongside vectorised version; likely superseded but existence not confirmed as unused |
| `scripts/enhanced/expression_constraints.py` | 748 | GAMSPy expression constraints builder | `reusable_core_candidate` | `core_package` | low | low | Used by vectorised solver |
| `scripts/enhanced/enh_RURO_draws.py` | 1,631 | Vectorised continuous opportunity-set generation; `generate_draws_long()` | `reusable_core_candidate` | `core_package` | high | high | ⚠ R3: `log_prior` formula here must be byte-identical to `enh_RURO_prep_mnl_basic.py` |
| `scripts/enhanced/enh_RURO_prep_mnl_basic.py` | 2,433 | Merge draws + EUROMOD output, reshape couples wide, compute `log_prior` | `EUROMOD_specific` | `app_package` | high | high | ⚠ R3: `log_prior` must match `enh_RURO_draws.py`; largest file in data pipeline |
| `scripts/enhanced/enh_RURO_euromod.py` | 1,151 | Single EUROMOD run on all draws; 35h French overtime split; decider-only logic | `EUROMOD_specific` | `app_package` | medium | medium | Java/pythonnet dependency; FR 35h rule hard-coded |
| `scripts/enhanced/enh_france_data_prep.py` | 2,621 | FR SILC/EUROSTAT filtering; `clean_harmonize_fr()`, `stepwise_filter_households()` | `application_layer_candidate` | `app_package` | medium | low | FR-specific column names and filter thresholds throughout |
| `scripts/enhanced/enh_RURO_estimate_FR.py` | 1,821 | 8-step estimation orchestrator; `main()`, `compute_standard_errors()`, `save_results_json()` | `application_layer_candidate` | `app_package` | medium | medium | FR-specific config; mixes orchestration and FR business logic |
| `scripts/enhanced/RURO_post_estimation_styled.py` | 10,232 | Styled HTML/Markdown post-estimation report; `ParsedParameters`, `compute_marginal_utility_*()`; fully dynamic parameter handling | `diagnostics_reporting` | `app_package` | medium | low | Largest file in repo; fully dynamic on parameters; FR-specific output paths |
| `scripts/enhanced/diagnostics_bundle.py` | 2,505 | 40+ metrics; 4 sections (A/B/C/D); CONOPT log parsing; `build_diagnostics_bundle()` | `diagnostics_reporting` | `core_package` | medium | low | No EUROMOD imports; model-generic |
| `scripts/enhanced/cluster_robust_se.py` | 239 | `compute_cluster_robust_se()`, `assemble_meat_matrix()`; T1–T5 verification; 9,657 idorighh clusters | `reusable_core_candidate` | `core_package` | high | medium | Sandwich must be chunked (naive jacrev = 11 TB OOM); T1–T5 verification gates must stay |
| `scripts/enhanced/compute_standard_errors.py` | 379 | Numerical Hessian SEs via central differences | `reusable_core_candidate` | `core_package` | medium | low | Pure NumPy central-difference Hessian; no EUROMOD imports; safe to migrate |
| `scripts/enhanced/occupation_choice_utils.py` | 504 | Occupation preferences, wage/hours density, availability weights | `reusable_core_candidate` | `core_package` | medium | medium | ⚠ R10: may contain FR SILC column assumptions; verify before publishing |
| `scripts/enhanced/mcfadden_sampler.py` | 539 | McFadden (1978) choice-set expansion; 400 alternatives | `reusable_core_candidate` | `core_package` | medium | low | Pure Python/NumPy; no FR-specific logic; safe to migrate |
| `scripts/enhanced/path_helpers.py` | 265 | EUROMOD-STORAGE resolution, UNC workaround for GAMSPy, `ensure_local_workdir()` | `application_layer_candidate` | `app_package` | high | high | ⚠ R4: hard runtime dependency; must survive any restructuring |
| `scripts/enhanced/enh_prepare_FR_gsur_v2.py` | 858 | GSUR lookup; FR Eurostat + INSEE benchmark; 10 validation checks | `application_layer_candidate` | `app_package` | medium | low | FR/INSEE data-specific |
| `scripts/enhanced/enh_prepare_FR_gsur.py` | 717 | Earlier version of GSUR builder (v1) | `unclear_needs_review` | `stays_in_jmp_repo` | low | low | Likely superseded by v2; unclear if still referenced |
| `scripts/enhanced/enh_RURO_prep.py` | 1,339 | RURO variable construction (NUTS1, educ dummies, experience) | `application_layer_candidate` | `app_package` | medium | low | FR SILC variable names throughout |
| `scripts/enhanced/enh_RURO_mnl_rebuild_GSURv2_stageA.py` | 1,115 | GSUR v2 stage-A MNL rebuild | `application_layer_candidate` | `app_package` | low | low | FR-specific rebuild script |
| `scripts/enhanced/enh_RURO_post_estimation.py` | 1,654 | Earlier post-estimation reporter | `unclear_needs_review` | `stays_in_jmp_repo` | low | low | Likely superseded by `RURO_post_estimation_styled.py`; unclear if still used |
| `scripts/enhanced/parallel_estimation.py` | 648 | Parallel estimation orchestration | `application_layer_candidate` | `app_package` | low | low | Thin orchestration layer |
| `scripts/enhanced/run_cluster_robust_se.py` | 1,226 | Cluster-robust SE runner | `application_layer_candidate` | `app_package` | low | low | Thin runner over `cluster_robust_se.py` |
| `scripts/enhanced/diagnostic_consumption_variation.py` | 209 | Consumption variation diagnostic | `diagnostics_reporting` | `app_package` | low | low | FR-specific consumption columns expected; verify column names before migrating |
| `scripts/enhanced/sanity_checks.py` | 667 | Sanity checks on estimation data | `diagnostics_reporting` | `app_package` | low | low | FR SILC variable names expected; verify before migrating to `app_package` |
| `scripts/enhanced/validate_specs.py` | 174 | Spec validation utility | `reusable_core_candidate` | `core_package` | low | low | Validates spec YAML files pre-estimation; no EUROMOD imports |
| `scripts/enhanced/fix_spec_initial_values.py` | 283 | One-off spec init-value fixer | `scratch_or_temporary` | `never` | never | never | Not production |
| `scripts/enhanced/quick_verify.py` | 196 | Quick verification script | `scratch_or_temporary` | `never` | never | never | One-off verification script; not part of production pipeline |
| `scripts/enhanced/reduce_draws_files.py` | 496 | Draw file size reducer | `scratch_or_temporary` | `never` | never | never | One-off draw-file size reduction; not part of production pipeline |
| `scripts/enhanced/reduce_mnl_columns.py` | 644 | MNL column reducer | `scratch_or_temporary` | `never` | never | never | One-off MNL column reduction; not part of production pipeline |
| `scripts/enhanced/enh_RURO_explore_predrop.py` | 906 | Pre-drop exploration script | `scratch_or_temporary` | `never` | never | never | Exploratory script predating production pipeline |
| `scripts/enhanced/specifications/` (24 YAML files) | — | Continuous RURO model specs; general-purpose specifications | `configuration` | `core_package` | medium | low | Does NOT contain certified bll0/tlmpin specs — those are in `scripts/bpool/specs/` |

### scripts/bpool/ — identification analysis suite

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/bpool/jax_recovery_gate.py` | 560 | Synthetic recovery certification (Checks 1–6) | `tests_or_gates` | `stays_in_jmp_repo` | never | high | ⚠ PROVENANCE GATE — NEVER MOVE; basis of all identification claims |
| `scripts/bpool/step4_realdata_baseline.py` | 678 | Certified paper baseline run: negLL 238 362.79, min\_eig +1.706 | `output_or_provenance` | `stays_in_jmp_repo` | never | high | ⚠ PROVENANCE ARTIFACT — NEVER MOVE OR EDIT |
| `scripts/bpool/step4_lr_pooling_test.py` | 298 | LR test: beta\_E + beta\_h\_pt2 pooling rejection | `tests_or_gates` | `stays_in_jmp_repo` | never | high | ⚠ FREEZE; establishes pooling rejection; not safely re-runnable without full data |
| `scripts/bpool/step4_emit_results_json.py` | 336 | Export certified results to per-group JSON | `output_or_provenance` | `stays_in_jmp_repo` | low | low | Run once after certified baseline |
| `scripts/bpool/jax_ll_probe.py` | 614 | JAX likelihood builders for singles/couples | `reusable_core_candidate` | `core_package` | high | medium | Cross-checks NumPy engine; confirmed R1 Box-Cox bug |
| `scripts/bpool/jax_joint_hessian.py` | 210 | Exact JAX Hessian for joint likelihood | `reusable_core_candidate` | `core_package` | high | low | Minutes vs hours for CONOPT |
| `scripts/bpool/jax_optimize.py` | 244 | JAX-based optimization routines | `reusable_core_candidate` | `core_package` | high | low | Used in recovery certification |
| `scripts/bpool/jax_profile_couples_leisure.py` | 285 | JAX profile of couples-leisure likelihood direction | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | Identification analysis artefact |
| `scripts/bpool/joint_recovery_test.py` | 1,800 | Joint synthetic recovery test suite | `tests_or_gates` | `core_package` | medium | low | Comprehensive portable recovery tests |
| `scripts/bpool/recovery_test.py` | 627 | Synthetic recovery test (earlier version) | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Likely superseded by `joint_recovery_test.py` |
| `scripts/bpool/phase_a_param_binding.py` | 232 | Gate: all spec params bind without silent drops | `tests_or_gates` | `core_package` | high | low | All spec params bind to theta vector without silent drops; run before any estimation |
| `scripts/bpool/phase_b_recovery_test.py` | 279 | Parameter recovery test (58-param bpool design) | `tests_or_gates` | `core_package` | high | low | 58-param bpool synthetic recovery; precursor to `jax_recovery_gate.py` |
| `scripts/bpool/build_joint_theta_star.py` | 133 | Build joint θ* vector for recovery tests | `tests_or_gates` | `core_package` | medium | low | Builds joint θ* vector needed by phase_b and jax_recovery_gate |
| `scripts/bpool/build_bpool_precompute.py` | 653 | Per-year EUROMOD precompute with 7 gate checks | `EUROMOD_specific` | `app_package` | medium | medium | EUROMOD/Java dependency |
| `scripts/bpool/run_bpool_euromod.py` | 673 | Chunked EUROMOD batch pricing; FR CPI deflators phi\_2015/2016/2017 | `EUROMOD_specific` | `app_package` | medium | medium | FR CPI constants hard-coded |
| `scripts/bpool/run_bpool_euromod_chunk.py` | 238 | Chunk-level EUROMOD pricing (sub-script of above) | `EUROMOD_specific` | `app_package` | low | low | Called by `run_bpool_euromod.py` |
| `scripts/bpool/run_bpool_draws.py` | 294 | 100 singles + 30×30 couples bpool draws | `reusable_core_candidate` | `core_package` | medium | medium | ⚠ R3: `log_prior` consistency required |
| `scripts/bpool/assemble_bpool_priced.py` | 170 | Assemble chunk parquets with 4 canary checks | `EUROMOD_specific` | `app_package` | medium | low | Assembles EUROMOD-priced chunk parquets with canary validation |
| `scripts/bpool/build_bpool_estimation_ready.py` | 617 | Build estimation-ready bpool dataset | `EUROMOD_specific` | `app_package` | medium | low | Final prep of estimation-ready dataset with FR EUROMOD tax/benefit vars |
| `scripts/bpool/build_bpool_singles.py` | 413 | Build singles bpool draws | `reusable_core_candidate` | `core_package` | medium | low | Generates bpool draws for singles (_sm/_sf) without EUROMOD calls |
| `scripts/bpool/build_bpool_couples.py` | 520 | Build couples bpool draws | `reusable_core_candidate` | `core_package` | medium | low | Generates bpool draws for couples (_m/_f); EUROMOD-free draw logic |
| `scripts/bpool/harmonise_bpool_engine_ready.py` | 289 | Harmonise engine-ready dataset across years | `EUROMOD_specific` | `app_package` | low | low | Harmonises column names and units across 2015–2017 engine-ready files |
| `scripts/bpool/slice_engine_ready.py` | 102 | Slice engine-ready dataset | `EUROMOD_specific` | `app_package` | low | low | Slices to relevant columns for estimation engine input |
| `scripts/bpool/check_bpool_engine_ready.py` | 288 | Engine-ready validation gate | `tests_or_gates` | `app_package` | low | low | Gate: validates engine-ready dataset shape, dtypes, and sentinel values |
| `scripts/bpool/_bpool_paths.py` | 59 | Path constants delegating to `path_helpers` | `application_layer_candidate` | `app_package` | low | low | Thin bpool-local path adapter; delegates to shared path_helpers |
| `scripts/bpool/hours_mixture_d1.py` | 237 | Hours mixture distribution D1 | `reusable_core_candidate` | `core_package` | low | low | Fits and samples hours-mixture D1 distribution; no FR-specific constants |
| `scripts/bpool/occ_draw_empirical.py` | 122 | Empirical occupation draw utilities | `reusable_core_candidate` | `core_package` | low | low | Empirical occupation-draw sampler; data-driven, no hard-coded FR codes |
| `scripts/bpool/diag_gsplit_nonid_structure.py` | 203 | Non-ID structure diagnostic for gsplit | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | Documents gsplit failure; retain as provenance |
| `scripts/bpool/diag_nchildren_per_parent.py` | 147 | nchildren-per-parent diagnostic | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Diagnostic: nchildren-per-parent tabulation for bpool draw validation |
| `scripts/bpool/bench_conopt_modelgen.py` | 589 | CONOPT model-generation benchmark | `diagnostics_reporting` | `stays_in_jmp_repo` | never | never | Documents R7 bottleneck; threads/memory tuning all measured dead ends |
| `scripts/bpool/phase0_repricing_variation.py` | 133 | Phase 0 repricing variation check | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | Checks repricing variation across EUROMOD draws in phase 0 |
| `scripts/bpool/check_urbanisation_spec.py` | 150 | Urbanisation spec check | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Verifies NUTS1 urbanisation codes match expected FR SILC specification |
| `scripts/bpool/dump_theta_star.py` | 65 | Dump θ* to disk | `output_or_provenance` | `stays_in_jmp_repo` | low | never | One-off script to persist certified θ* vector as provenance artifact |
| `scripts/bpool/rebuild_meta.py` | 51 | Rebuild metadata index | `unclear_needs_review` | `stays_in_jmp_repo` | low | low | Purpose unclear from name alone |
| `scripts/bpool/proto_gamspy_intermediate_var.py` | 173 | Prototype GAMSPy intermediate-variable approach | `scratch_or_temporary` | `never` | never | never | Abandoned prototype for intermediate-variable GAMSPy formulation |
| `scripts/bpool/_tmp_benchmark_multistart.py` | 157 | Multistart benchmark scratch | `scratch_or_temporary` | `never` | never | never | Scratch: multistart timing benchmark; no longer relevant |
| `scripts/bpool/_tmp_benchmark_scipy_newton.py` | 152 | Scipy Newton benchmark scratch | `scratch_or_temporary` | `never` | never | never | Scratch: scipy Newton vs BFGS timing comparison; superseded |
| `scripts/bpool/validate_chosen_anchors.py` | 159 | Provenance validation: anchor draws | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks anchor-draw consistency |
| `scripts/bpool/validate_chosen_flips.py` | 175 | Provenance validation: sign flips | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks sign-flip consistency in chosen draws |
| `scripts/bpool/validate_chosen_vs_canonical.py` | 80 | Provenance validation: vs canonical dataset | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks chosen vs canonical dataset |
| `scripts/bpool/validate_chosen_vs_tminus1.py` | 217 | Provenance validation: vs t-1 | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks chosen vs t-1 consistency |
| `scripts/bpool/validate_chosen_yem_couples.py` | 85 | Provenance validation: YEM couples | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks YEM couples in chosen draws |
| `scripts/bpool/validate_female_repricing.py` | 118 | Provenance validation: female repricing | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Freeze alongside `jax_recovery_gate.py`; checks female-repricing consistency |
| `scripts/bpool/verify_lh_coverage.py` | 231 | Verify labour-hours coverage | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Verifies labour-hours coverage across draws before estimation |
| `scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml` | — | CERTIFIED 47-param spec (beta\_ll=0, theta\_l\_m=−0.8 pinned) | `output_or_provenance` | `stays_in_jmp_repo` | never | high | ⚠ Certified baseline spec — never modify |
| `scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_tlmpin_gsplit.yaml` | — | 49-param gsplit — FAILED synthetic recovery gate | `output_or_provenance` | `stays_in_jmp_repo` | never | high | ⚠ NOT certified; label prominently before any use |
| `scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0.yaml` | — | 47-param spec without theta\_l\_m pin (pre-certified version) | `output_or_provenance` | `stays_in_jmp_repo` | never | medium | Pre-certified 47-param spec before theta\_l\_m=-0.8 was pinned; superseded |
| `scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_gsplit_draw.yaml` | — | gsplit draw spec | `output_or_provenance` | `stays_in_jmp_repo` | never | medium | Draw-side spec for non-certified gsplit estimation |
| `scripts/bpool/specs/estimation_spec_joint_pooled_v1.yaml` | — | Base pooled joint spec (no pin) | `output_or_provenance` | `stays_in_jmp_repo` | never | low | Earliest pooled joint spec; no beta\_ll or theta\_l\_m constraints |
| `scripts/bpool/specs/estimation_spec_bpool_p3a_v1.yaml` | — | bpool P3a spec | `output_or_provenance` | `stays_in_jmp_repo` | never | low | Phase 3a spec preceding joint pooling; retained for provenance |
| `scripts/bpool/specs/theta_hat_realdata_901_gsplit_v1.csv` | — | gsplit θ̂ values from real-data run | `output_or_provenance` | `stays_in_jmp_repo` | never | low | Non-certified gsplit θ̂; retained as provenance for failed gate analysis |

### scripts/Job_model/ — job-choice RURO branch

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/Job_model/enh_job_universe.py` | 1,493 | Discrete job grid; `build_job_universe_from_ruro_ready()` | `reusable_core_candidate` | `core_package` | medium | low | Alternative draw model; same likelihood kernel as continuous RURO |
| `scripts/Job_model/enh_job_draws.py` | 1,111 | Person-level job draws; `generate_job_draws_long()` | `reusable_core_candidate` | `core_package` | medium | low | Same log\_prior convention as continuous draws; verify consistency with `enh_RURO_draws.py` |
| `scripts/Job_model/run_job_ruro_pipeline.py` | 499 | Orchestrator subprocess chain | `application_layer_candidate` | `app_package` | medium | low | FR-specific paths |
| `scripts/Job_model/sanity_checks_job.py` | 542 | Job-model sanity checks | `diagnostics_reporting` | `app_package` | low | low | FR SILC variable names expected; verify before migrating |
| `scripts/Job_model/plot_loc_by_dehde.py` | 170 | Location-by-DEHDE visualisation | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Job-model diagnostic: plots location distribution by DEHDE classification |

### scripts/welfare/ — post-estimation welfare simulation

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/welfare/welfare_core.py` | 603 | Core welfare computation utilities | `welfare_specific` | `app_package` | medium | medium | EUROMOD-dependent |
| `scripts/welfare/welfare_vdir.py` | 558 | V\_i^dir direct-utility welfare | `welfare_specific` | `app_package` | medium | medium | Direct-utility welfare using EUROMOD-priced budget sets |
| `scripts/welfare/welfare_resim_probe.py` | 186 | Resimulation probe for welfare | `welfare_specific` | `app_package` | low | low | Probes welfare sensitivity by resimulating draws |
| `scripts/welfare/welfare_correction_prep.py` | 197 | Correction prep for welfare pipeline | `welfare_specific` | `app_package` | low | low | Prepares selection-correction factors for welfare pipeline |
| `scripts/welfare/welfare_assessment_unit_diag.py` | 313 | Assessment-unit diagnostic | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Diagnoses assessment-unit assignment across household types |
| `scripts/welfare/welfare_chosen_contamination.py` | 112 | Chosen-outcome contamination audit | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Audits contamination from chosen-outcome draws in welfare pipeline |
| `scripts/welfare/welfare_couples_contamination_audit.py` | 224 | Couples contamination audit | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Audits chosen-outcome contamination for the couples group |
| `scripts/welfare/welfare_cross_track_residual_diag.py` | 111 | Cross-track residual diagnostic | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Diagnoses cross-track residuals in welfare pipeline |
| `scripts/welfare/run_stage1_w3.py` | 229 | Stage 1: welfare wave-3 | `welfare_specific` | `app_package` | medium | low | Runs welfare stage 1 using wave-3 draws |
| `scripts/welfare/run_stage2_assessment_unit_diag.py` | 205 | Stage 2: assessment-unit diagnostic | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Stage 2 diagnostic for assessment-unit classification |
| `scripts/welfare/run_stage2_chosen_measure.py` | 351 | Stage 2: chosen-measure computation | `welfare_specific` | `app_package` | medium | low | Stage 2: computes chosen welfare measure from EUROMOD-priced draws |
| `scripts/welfare/run_stage2_chosen_task1.py` | 280 | Stage 2: chosen task 1 | `welfare_specific` | `app_package` | medium | low | Stage 2 task 1: chosen-outcome welfare computation |
| `scripts/welfare/run_stage2_chunk_writeback_validation.py` | 367 | Stage 2: chunk writeback validation | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 2: validates chunk-writeback integrity across parquet shards |
| `scripts/welfare/run_stage2_correction_prep.py` | 421 | Stage 2: correction prep | `welfare_specific` | `app_package` | low | low | Stage 2: runs correction-prep for welfare draws |
| `scripts/welfare/run_stage2_couples_audit.py` | 115 | Stage 2: couples audit | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Stage 2 audit of couples welfare draws |
| `scripts/welfare/run_stage2_couples_reprice.py` | 268 | Stage 2: couples repricing | `welfare_specific` | `app_package` | medium | low | Stage 2: reprices couples draws with EUROMOD |
| `scripts/welfare/run_stage2_cross_track_diag.py` | 260 | Stage 2: cross-track diagnostic | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Stage 2 cross-track residual diagnostic |
| `scripts/welfare/run_stage2_full_rebuild_staging.py` | 310 | Stage 2: full rebuild staging | `welfare_specific` | `app_package` | low | low | Stage 2: full draw-set rebuild (staging step) |
| `scripts/welfare/run_stage2_full_rebuild_validation.py` | 196 | Stage 2: full rebuild validation | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 2: validates full-rebuild output integrity |
| `scripts/welfare/run_stage2_parity.py` | 67 | Stage 2: parity check | `tests_or_gates` | `stays_in_jmp_repo` | low | never | Stage 2: parity check across groups |
| `scripts/welfare/run_stage2_resim.py` | 140 | Stage 2: resimulation | `welfare_specific` | `app_package` | medium | low | Stage 2: resimulation of draws for welfare computation |
| `scripts/welfare/run_stage2_singles_vdir_gate.py` | 196 | Stage 2: singles V\_i^dir gate | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 2 gate: singles V\_i^dir vs chosen contamination |
| `scripts/welfare/run_stage2_twoH_validation.py` | 357 | Stage 2: two-household validation | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 2: validates two-household welfare accounting |
| `scripts/welfare/run_stage2_vdir.py` | 242 | Stage 2: V\_i^dir computation | `welfare_specific` | `app_package` | medium | low | Stage 2: computes V\_i^dir welfare metric from draws |
| `scripts/welfare/run_stage3a_pinned_baseline_validation.py` | 594 | Stage 3a: pinned baseline validation | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 3a: validates welfare with pinned certified baseline |
| `scripts/welfare/run_stage3b1_engine_ready_parity.py` | 600 | Stage 3b1: engine-ready parity | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 3b1: parity check between engine-ready and welfare draws |
| `scripts/welfare/run_stage3b2_controlled_reestimation.py` | 338 | Stage 3b2: controlled re-estimation | `welfare_specific` | `app_package` | low | low | Stage 3b2: controlled re-estimation for welfare robustness |
| `scripts/welfare/run_stage3b3_synthetic_recovery.py` | 384 | Stage 3b3: synthetic recovery | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 3b3: synthetic recovery gate for welfare re-estimation |
| `scripts/welfare/run_stage4a_baseline_policy.py` | 234 | Stage 4a: baseline policy simulation | `welfare_specific` | `app_package` | low | low | Stage 4a: simulates baseline policy with certified estimates |
| `scripts/welfare/run_stage4b_population_parity_gate.py` | 287 | Stage 4b: population parity gate | `tests_or_gates` | `stays_in_jmp_repo` | low | low | commit 0e66325 |
| `scripts/welfare/run_stage4c_singles_vdir_smoke.py` | 873 | Stage 4c: singles V\_i^dir smoke test | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Stage 4c: full smoke test for singles V\_i^dir welfare metric |
| `scripts/welfare/run_stage4c2_vdir_bias_calibration.py` | 405 | Stage 4c2: V\_i^dir bias calibration | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | commit 5c8eb88 |

### scripts/multi\_year/ — CPI harmonization and year stacking

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/multi_year/m1_harmonise_cpi.py` | 363 | FR CPI harmonisation 2015–2017 | `application_layer_candidate` | `app_package` | medium | low | FR CPI constants; not country-generic |
| `scripts/multi_year/m1_stack_years.py` | 454 | Stack per-year estimation-ready datasets | `application_layer_candidate` | `app_package` | medium | low | Stacks annual estimation-ready datasets into pooled panel |
| `scripts/multi_year/m1_validate.py` | 821 | Validation of stacked dataset | `tests_or_gates` | `app_package` | medium | low | Validates stacked panel for consistency across years |
| `scripts/multi_year/m1_add_cluster_key.py` | 281 | Add cluster key to stacked dataset | `application_layer_candidate` | `app_package` | low | low | Adds cluster key for cluster-robust SE computation |
| `scripts/multi_year/m1_config.py` | 268 | Multi-year config constants | `configuration` | `app_package` | medium | low | FR-specific year/path constants for multi-year pipeline |
| `scripts/multi_year/m1_identity_validation.py` | 504 | Identity validation across years | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Validates household identity linkage consistency across 2015–2017 |
| `scripts/multi_year/m1_isf_check_2018.py` | 595 | ISF check for 2018 year | `scratch_or_temporary` | `never` | never | never | One-off year-specific check |

### scripts/maintenance/

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/maintenance/prepare_pooled_estimation_ready.py` | 578 | Pooled estimation dataset prep | `application_layer_candidate` | `app_package` | medium | low | Prepares FR pooled dataset; EUROMOD-specific variable construction |
| `scripts/maintenance/run_pooled_P3a_estimation.py` | 325 | Phase 3a pooled estimation runner | `application_layer_candidate` | `app_package` | medium | low | Phase 3a runner; precedes bpool certified pipeline |
| `scripts/maintenance/run_pooled_P3a_presolver_checks.py` | 316 | Pre-solver checks | `tests_or_gates` | `stays_in_jmp_repo` | low | low | Pre-solver sanity checks before phase 3a CONOPT run |
| `scripts/maintenance/run_pooled_P3a_S5_S8_hessian_recompute.py` | 227 | Hessian recompute stages 5–8 | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Recomputes Hessian at stages 5–8 for phase 3a PD verification |
| `scripts/maintenance/run_pooled_P3a_S6_preference_comparison.py` | 118 | Preference comparison stage 6 | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Stage 6: preference parameter comparison before/after phase 3a re-estimation |
| `scripts/maintenance/run_pooled_P3a_S6_theta_c_singles_profile.py` | 133 | theta\_c singles profile | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Profile LL in theta\_c direction for singles; informed Box-Cox identification |
| `scripts/maintenance/validate_occ_dummies.py` | 49 | Occupation dummy validation | `tests_or_gates` | `stays_in_jmp_repo` | low | never | Validates occupation dummy encoding; no collinear/missing categories |
| `scripts/maintenance/validate_v1.py` | 20 | V1 spec validation | `tests_or_gates` | `stays_in_jmp_repo` | low | never | Minimal validation for v1 spec; superseded by bpool gate suite |
| `scripts/maintenance/validate_v7.py` | 109 | V7 spec validation | `tests_or_gates` | `stays_in_jmp_repo` | low | never | Validation for v7 spec variant; retained for historical comparison |
| `scripts/maintenance/rename_stijn_to_ruro.py` | 444 | One-off rename script | `scratch_or_temporary` | `never` | never | never | One-off mass-rename of "stijn" → "ruro" identifiers; already applied |

### scripts/diagnostics/

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/diagnostics/RURO_post_estimation_M1_diagnostics.py` | 773 | M1 post-estimation diagnostics | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | Full post-estimation diagnostics for M1 single-group model |
| `scripts/diagnostics/RURO_post_estimation_M1_naive_diagnostics.py` | 723 | M1 naive diagnostics | `diagnostics_reporting` | `stays_in_jmp_repo` | low | low | Naive (no importance weights) diagnostic counterpart to M1 |
| `scripts/diagnostics/check_nchildren_simple.py` | 61 | nchildren check (simple) | `diagnostics_reporting` | `stays_in_jmp_repo` | never | never | Quick tabulation of nchildren in FR SILC data |
| `scripts/diagnostics/check_nchildren_variation.py` | 101 | nchildren variation check | `diagnostics_reporting` | `stays_in_jmp_repo` | never | never | Checks within-household nchildren variation across years |
| `scripts/diagnostics/check_nchildren_variation_v2.py` | 51 | nchildren variation v2 | `diagnostics_reporting` | `stays_in_jmp_repo` | never | never | Revised nchildren variation check after data prep fix |
| `scripts/diagnostics/check_preference_diagnostics.py` | 158 | Preference diagnostics | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Tabulates preference heterogeneity across estimated groups |
| `scripts/diagnostics/check_type_ids.py` | 107 | Type ID validation | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Verifies household-type IDs consistent across draws and prep |
| `scripts/diagnostics/compare_scipy_gamspy.py` | 174 | Scipy vs GAMSPy solver comparison | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | One-off comparison that confirmed CONOPT/scipy LL parity |
| `scripts/diagnostics/run_stage5a2_cluster_se_artifact.py` | 183 | Stage 5a2 cluster SE artifact check | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Checks cluster-SE sandwich for OOM/chunking artifact |
| `scripts/diagnostics/run_stage5a_postestimation_descriptives.py` | 951 | Post-estimation descriptives | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Full post-estimation descriptive tables and fit statistics |
| `scripts/diagnostics/test_gamspy_vs_scipy.py` | 310 | GAMSPy vs scipy numerical test | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Numerical regression test confirming CONOPT/scipy LL parity |

### scripts/pilot/ — pre-production experiments

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/pilot/pilot_wage_draw.py` | — | Pilot wage draw utilities | `scratch_or_temporary` | `never` | never | never | Pilot predecessor to production draws |
| `scripts/pilot/build_pilot_couples_product.py` | — | Build pilot couples product | `scratch_or_temporary` | `never` | never | never | Pilot-era couples product builder; superseded by bpool pipeline |
| `scripts/pilot/build_precompute_ready.py` | — | Build pilot precompute | `scratch_or_temporary` | `never` | never | never | Pilot-era precompute prep; superseded by bpool engine-ready |
| `scripts/pilot/fit_pilot_mincer.py` | — | Fit pilot Mincer equation | `scratch_or_temporary` | `never` | never | never | Pilot Mincer wage equation; coefficients saved to pilot_mincer_coefficients_v1.json |
| `scripts/pilot/export_pilot_euromod_inputs.py` | — | Export pilot EUROMOD inputs v1 | `scratch_or_temporary` | `never` | never | never | Pilot-era EUROMOD input exporter; replaced by bpool repricing |
| `scripts/pilot/export_pilot_euromod_inputs_v2.py` | — | Export pilot EUROMOD inputs v2 | `scratch_or_temporary` | `never` | never | never | V2 of pilot EUROMOD exporter; still superseded by bpool pipeline |
| `scripts/pilot/merge_pilot_em_outputs.py` | — | Merge pilot EM outputs | `scratch_or_temporary` | `never` | never | never | Merges pilot EUROMOD output files; replaced by assemble_bpool_priced |
| `scripts/pilot/run_pilot_em_blocks.py` | — | Run pilot EM blocks | `scratch_or_temporary` | `never` | never | never | Pilot block-level EUROMOD runner; replaced by chunked bpool repricing |
| `scripts/pilot/_bisect_ll.py` | — | LL bisect diagnostic | `scratch_or_temporary` | `never` | never | never | Scratch: bisection search for LL anomaly; not referenced anywhere |
| `scripts/pilot/_precompute_gate.py` | — | Precompute gate probe | `scratch_or_temporary` | `never` | never | never | Scratch: precompute correctness probe predating bpool gate |
| `scripts/pilot/_rebuild_c_norm.py` | — | c\_norm rebuild | `scratch_or_temporary` | `never` | never | never | Scratch: one-off c\_norm recalculation; not in active pipeline |
| `scripts/pilot/_resolve_hnpos.py` | — | hn/pos resolution probe | `scratch_or_temporary` | `never` | never | never | Scratch: hn/pos alignment investigation; resolved in bpool build |
| `scripts/pilot/_run_beta_l0_m_diagnostic.py` | — | beta\_l0\_m diagnostic runner | `scratch_or_temporary` | `never` | never | never | Scratch: beta\_l0\_m floor diagnostic; findings absorbed into jax\_recovery\_gate |
| `scripts/pilot/_run_diagnostic_estimation.py` | — | Diagnostic estimation runner | `scratch_or_temporary` | `never` | never | never | Scratch: early diagnostic estimation; replaced by step4_realdata_baseline |
| `scripts/pilot/_run_diagnostic_estimation_rerun.py` | — | Diagnostic estimation rerun | `scratch_or_temporary` | `never` | never | never | Scratch: rerun of above; superseded |
| `scripts/pilot/_run_jax_optimizer_benchmark.py` | — | JAX optimizer benchmark | `scratch_or_temporary` | `never` | never | never | Scratch: JAX optimizer timing; findings absorbed into jax\_recovery\_gate |
| `scripts/pilot/_run_jax_validation_estimation.py` | — | JAX validation estimation | `scratch_or_temporary` | `never` | never | never | Scratch: early JAX validation run; superseded by jax\_recovery\_gate |
| `scripts/pilot/_run_ll_equivalence_prototype.py` | — | LL equivalence prototype | `scratch_or_temporary` | `never` | never | never | Scratch: LL equivalence between enhanced and bpool engines; confirmed |
| `scripts/pilot/_run_loc4_precompute_augmentation.py` | — | loc4 precompute augmentation | `scratch_or_temporary` | `never` | never | never | Scratch: augments loc4 features in precompute; not in current pipeline |
| `scripts/pilot/_run_optimizer_protocol_diagnostic.py` | — | Optimizer protocol diagnostic | `scratch_or_temporary` | `never` | never | never | Scratch: optimizer protocol comparison (scipy/CONOPT/JAX); superseded |
| `scripts/pilot/_run_precompute.py` | — | Precompute runner | `scratch_or_temporary` | `never` | never | never | Scratch: pilot precompute execution; replaced by bpool build pipeline |
| `scripts/pilot/_run_scaled_jax_validation.py` | — | Scaled JAX validation | `scratch_or_temporary` | `never` | never | never | Scratch: scaled JAX validation run; findings absorbed into certified gate |
| `scripts/pilot/_validate_draw_patch.py` | — | Draw patch validation | `scratch_or_temporary` | `never` | never | never | Scratch: validates draw patch after log\_prior bug fix; now retired |
| `scripts/pilot/_tmp_*/` (4 dirs, JSON outputs) | — | Temporary output dirs from pilot runs | `output_or_provenance` | `stays_in_jmp_repo` | low | never | Retain as identification experiment record |
| `scripts/pilot/specs/estimation_spec_nc_pilot_couples_2016.yaml` | — | Pilot couples spec | `scratch_or_temporary` | `never` | never | never | Pilot-era couples spec for 2016; superseded by bpool joint specs |
| `scripts/pilot/config/pilot_mincer_coefficients_v1.json` | — | Pilot Mincer coefficients | `output_or_provenance` | `stays_in_jmp_repo` | low | never | Mincer wage-equation coefficients used in pilot draw generation |

### scripts/ root-level loose files

These are early-generation scripts that predate `scripts/enhanced/`. They are NOT superseded in the sense of being deleted; the enhanced versions diverged significantly.

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `scripts/RURO_draws.py` | 913 | Pre-enhanced RURO draws | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | 913 vs 1631 lines in enhanced version; not a simple copy |
| `scripts/RURO_estimate_FR.py` | 6,201 | Pre-enhanced FR estimator | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | 6201 vs 1821 lines in enhanced version — significantly different |
| `scripts/RURO_euromod.py` | 810 | Pre-enhanced EUROMOD runner | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | Legacy EUROMOD runner predating enhanced/ split; review before any use |
| `scripts/RURO_post_estimation.py` | 2,836 | Pre-enhanced post-estimation | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | Monolithic legacy post-estimation; superseded by enhanced/ styled reports |
| `scripts/RURO_prep.py` | 729 | Pre-enhanced prep | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | Legacy data prep preceding enhanced/ split; not in active pipeline |
| `scripts/RURO_prep_mnl_basic.py` | 815 | Pre-enhanced MNL prep | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | 815 vs 2433 lines in enhanced version — heavily diverged |
| `scripts/path_helpers.py` | 266 | Root-level copy of path\_helpers | `unclear_needs_review` | `never` | never | medium | Possible duplicate of `scripts/enhanced/path_helpers.py` (265 lines); verify canonical copy before deleting |
| `scripts/france_data_prep.py` | 1,718 | Earlier FR data prep | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | Earlier FR SILC prep; relationship to maintenance/ prep unclear — review |
| `scripts/prepare_FR_gsur.py` | 453 | Earlier GSUR builder | `unclear_needs_review` | `stays_in_jmp_repo` | never | low | Earlier FR GSUR dataset builder; superseded status unclear — review |
| `scripts/generate_html_report.py` | 325 | HTML report generator | `diagnostics_reporting` | `stays_in_jmp_repo` | low | never | Generates HTML diagnostic report; may duplicate enhanced/ styled output |
| `scripts/run_post_estimation_standalone.py` | 186 | Standalone post-estimation runner | `application_layer_candidate` | `app_package` | low | low | Thin runner invoking enhanced/ post-estimation; potential app entry point |
| `scripts/extract_excel_text.py` | 24 | Excel text extractor | `scratch_or_temporary` | `never` | never | never | 24-line scratch utility; no production use |
| `scripts/init_params_singles_template.csv` | — | Singles init params template | `configuration` | `core_package` | low | low | Default θ₀ for singles; spec-agnostic starting point for optimisation |
| `scripts/seed_boxcox_init.csv` | — | Box-Cox seed init values | `configuration` | `core_package` | low | low | Box-Cox θ₀ seeds; shared across specs for reproducible warm start |
| `scripts/runners/` | — | Additional runner scripts (contents not individually listed) | `unclear_needs_review` | `stays_in_jmp_repo` | low | low | Contents not individually enumerated |
| `scripts/run_fr_2016_joint_only.ps1` | — | FR 2016 joint-only runner | `application_layer_candidate` | `app_package` | low | never | PowerShell entry point for FR 2016 joint estimation only |
| `scripts/run_fr_2016_pipeline.ps1` | — | FR 2016 full pipeline runner | `application_layer_candidate` | `app_package` | low | never | PowerShell entry point for full FR 2016 pipeline execution |
| `scripts/run_post_estimation.ps1` | — | Post-estimation PowerShell runner | `application_layer_candidate` | `app_package` | low | never | PowerShell wrapper for enhanced/ post-estimation scripts |
| `scripts/sync_backup.ps1` | — | Backup sync script | `scratch_or_temporary` | `never` | never | never | Backup sync utility; superseded by EUROMOD-STORAGE canonical store |
| `scripts/tdo.ps1` | — | TODO helper | `scratch_or_temporary` | `never` | never | never | Personal TODO helper script; no project function |
| `scripts/run_pipeline_explicit.ipynb` | — | Notebook pipeline runner | `scratch_or_temporary` | `never` | never | never | Notebook for exploratory pipeline runs; not production |

### scripts/archive/ — superseded (never migrate)

| Current path | Purpose | Class | Target | Priority | Risk |
|---|---|---|---|---|---|
| `scripts/archive/rum_approach/RUM/` (21 × .py) | Legacy translog/Box-Cox RUM: DCM1/DCM2, biogeme, train\_mnl, scenarios\_de | `output_or_provenance` | `never` | never | never |
| `scripts/archive/old_ruro_pre_enhanced/` (7 × .py) | Pre-enhanced RURO: RURO\_boxcox\_\*, full\_RURO, run\_fr\_2021 | `output_or_provenance` | `never` | never | never |
| `scripts/archive/experimental/` (5 × .py) | Interactive/memory-only pipeline experiments | `scratch_or_temporary` | `never` | never | never |
| `scripts/archive/fixes/` (2 × .py) | One-off SE and post-estimation rerun fixes | `scratch_or_temporary` | `never` | never | never |
| `scripts/archive/old_data_prep/data_prep2.py` | Old data prep | `output_or_provenance` | `never` | never | never |
| `scripts/archive/backups_2025_12/` (3 files) | Manual backups from December 2025 | `output_or_provenance` | `never` | never | never |
| `scripts/archive/run_gamspy.ps1` | Archive PowerShell runner | `scratch_or_temporary` | `never` | never | never |

### src/mnl/ — package scaffold (stub; does NOT expose RURO engine)

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `src/mnl/__init__.py` | 11 | Version string only | `unclear_needs_review` | `core_package` | high | high | ⚠ Exposes nothing of the RURO engine; replace before any release |
| `src/mnl/models/mnl.py` | 39 | statsmodels MNLogit wrapper | `unclear_needs_review` | `never` | never | high | ⚠ R2: completely wrong model; delete before any public release |
| `src/mnl/integration/euromod.py` | 152 | EUROMOD connector with lazy load | `EUROMOD_specific` | `app_package` | low | low | Pattern sound; no real implementation yet |
| `src/mnl/pipelines/estimation.py` | 47 | Minimal stub pipeline | `unclear_needs_review` | `app_package` | low | low | Shell only |
| `src/mnl/config.py` | 45 | Package-level config stubs | `configuration` | `core_package` | low | low | Stub config in wrong src/mnl layout; replace with packages/dclaborsupply/ |
| `src/mnl/data/loaders.py` | 39 | Data loader stubs | `unclear_needs_review` | `core_package` | low | low | Stub data loaders in wrong layout; review for content before migrating |
| `src/mnl/evaluation/metrics.py` | 22 | Evaluation metrics stubs | `unclear_needs_review` | `core_package` | low | low | Stub evaluation metrics; 22 lines, review for content before migrating |
| `src/mnl.egg-info/` | — | Build artefacts from `pip install -e .` | `output_or_provenance` | `never` | never | never | Generated by editable install; should be in .gitignore |

### tests/

| Current path | Lines | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|---|
| `tests/test_recovery_cov_verdict.py` | 139 | Hessian verdict regression tests | `tests_or_gates` | `core_package` | high | low | Migrate to `packages/dclaborsupply/tests/` |
| `tests/test_imports.py` | 5 | Package import sanity | `tests_or_gates` | `core_package` | high | low | Only current test; migrate and expand as Gate 1 boundary test |

### stijn/ — R reference implementation

| Current path | Purpose | Class | Target | Priority | Risk |
|---|---|---|---|---|---|
| `stijn/Ruro_estimation_H.Rmd` | R RURO estimation (Hisham variant) | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `stijn/Ruro_estimation_new.Rmd` | R RURO estimation (new variant) | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `stijn/Ruro_functions_EMRWS.R` | R RURO functions (EMRWS) | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `stijn/Ruro_simulation_H.Rmd` | R RURO simulation | `output_or_provenance` | `stays_in_jmp_repo` | never | never |

### config/ and configs/ — pipeline configuration

| Current path | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|
| `config/multi_year/fr_p2_stage_m1.yaml` | FR phase 2 multi-year stage config | `configuration` | `app_package` | medium | low | Stage config for FR phase 2 multi-year pipeline |
| `config/multi_year/fr_p3a_gsurv2_stage_m1.yaml` | FR phase 3a GSUR v2 stage config | `configuration` | `app_package` | medium | low | Stage config for FR phase 3a with GSUR v2 specification |
| `config/multi_year/fr_p3a_stage_m1.yaml` | FR phase 3a stage config | `configuration` | `app_package` | medium | low | Stage config for FR phase 3a (standard GSUR) |
| `config/multi_year/fr_p3b_stage_m1.yaml` | FR phase 3b stage config | `configuration` | `app_package` | medium | low | Stage config for FR phase 3b (identification diagnostics) |
| `config/multi_year/fr_p4_stage_m1.yaml` | FR phase 4 stage config | `configuration` | `app_package` | medium | low | Stage config for FR phase 4 (real-data baseline estimation) |
| `configs/default.yaml` | Default MNL pipeline config (generic column/path names) | `configuration` | `core_package` | low | low | Generic; not FR-specific |

### notes/

| Current path | Purpose | Class | Target | Priority | Risk |
|---|---|---|---|---|---|
| `notes/EUROMO_sys_france_2015.md` | EUROMOD system notes for FR 2015 | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `notes/R_REFERENCE_vs_PYTHON_SPECIFICATION.md` | R vs Python spec comparison | `output_or_provenance` | `stays_in_jmp_repo` | never | never |

### Root-level project files

| Current path | Purpose | Class | Target | Priority | Risk | Notes |
|---|---|---|---|---|---|---|
| `pyproject.toml` | Package build config (current `src/mnl` package) | `configuration` | `core_package` | high | medium | Needs significant revision for monorepo |
| `requirements.txt` | Dependency spec | `configuration` | `core_package` | high | medium | Conflates core and app deps; needs split |
| `README.md` | Project readme | `output_or_provenance` | `stays_in_jmp_repo` | low | never | Project overview; update when packages/ structure is established |
| `RURO_MNL_project_files_structure.md` | Files structure documentation | `output_or_provenance` | `stays_in_jmp_repo` | low | never | Earlier manual file map; superseded by this inventory |
| `debug.log` | Debug log | `output_or_provenance` | `never` | never | never | Session debug output; add to .gitignore |
| `gate_gsplit_901_run.log` | gsplit 901 gate run log | `output_or_provenance` | `stays_in_jmp_repo` | never | never | Log from failed 49-param gsplit gate run; provenance for FAILED classification |
| `gate_output.txt` | Gate output artefact | `output_or_provenance` | `stays_in_jmp_repo` | never | never | Gate stdout capture; retained as provenance |
| `.markdownlint.json` | Markdown lint config | `configuration` | `stays_in_jmp_repo` | never | never | MD lint rules for this repo (e.g. heading-level skip suppression) |
| `.mplconfig` | Matplotlib config | `configuration` | `stays_in_jmp_repo` | low | never | Matplotlib rcParams overrides for styled diagnostic figures |

### outputs/, Results/, Data/, Pdfs/, Prompts/, literature/, notebooks/

| Current path | Purpose | Class | Target | Priority | Risk |
|---|---|---|---|---|---|
| `outputs/opportunity_diagnostics_certified_v1.parquet` | Certified opportunity diagnostics parquet | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `outputs/figures/stage5a_*.png` (6 files) | Stage 5a diagnostic figures | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `outputs/welfare/stage1_w3/` | Welfare stage 1 outputs | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `outputs/KEEP_RESULTS.md` | Results preservation note | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `Results/JMP_*.csv/.tex/.json/.npy/.log` (30+ files) | JMP paper estimation results, tables, SE matrices, Hessians | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `Data/documentation/`, `Data/external/` | Data documentation and external inputs | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `Pdfs/` (8 PDFs) | Reference papers + model documentation | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `Prompts/` (12+ .md/.txt) | LLM prompts used during development | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `Prompts/welfare/` (21 .txt) | Welfare-stage LLM prompts | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `literature/` | Literature files | `output_or_provenance` | `stays_in_jmp_repo` | never | never |
| `notebooks/estimation_notebook.ipynb` | Estimation notebook (exploration) | `scratch_or_temporary` | `stays_in_jmp_repo` | never | never |
| `Microsoft/` | Unknown directory (contents not examined) | `unclear_needs_review` | `stays_in_jmp_repo` | low | low |

---

## C. Dependency Map

### Main estimation pipeline (continuous RURO, FR)

```text
FR SILC/EUROSTAT raw data
        │
        ▼
scripts/enhanced/enh_france_data_prep.py    clean_harmonize_fr(), stepwise_filter_households()
        │   enh_prepare_FR_gsur_v2.py       GSUR lookup (INSEE/Eurostat)
        ▼
scripts/enhanced/enh_RURO_prep.py           NUTS1, educ dummies, experience
        │
        ▼
scripts/enhanced/enh_RURO_draws.py   ─── ⚠ log_prior formula A (R3) ────────────────┐
        │                                                                             │
        ▼                                                                             │
scripts/enhanced/enh_RURO_euromod.py        EUROMOD Java/pythonnet; 35h FR overtime  │
        │                                                                             │
        ▼                                                                             │
scripts/enhanced/enh_RURO_prep_mnl_basic.py ─── ⚠ log_prior formula B (must = A) ───┘
        │   2,433 lines; couples reshape here
        ▼
scripts/enhanced/estimation_spec_parser.py  YAML → EstimationSpec; 4-group _sm _sf _m _f
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
scripts/enhanced/gamspy_estimation_vectorized.py  scripts/enhanced/estimation_engine.py
(GAMSPy CONOPT/IPOPT)                            (NumPy/Numba scipy)
        │   ⚠ R4: path_helpers required            │   ⚠ R1: Box-Cox bug in estimation_utils
        └──────────────┬───────────────────────────┘
                       ▼
         scripts/enhanced/enh_RURO_estimate_FR.py   8-step orchestrator
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
  compute_standard_errors.py  cluster_robust_se.py  (chunked; T1–T5 verification)
                       │
                       ▼
             save_results_json()
                       │
          ┌────────────┴─────────────┐
          ▼                          ▼
  diagnostics_bundle.py   RURO_post_estimation_styled.py  (10,232 lines)
                                     │
                                     ▼
                            scripts/welfare/
```

### bpool / JAX certification pipeline

```text
scripts/bpool/run_bpool_draws.py
    ├── scripts/bpool/build_bpool_precompute.py   per-year EUROMOD + 7 gate checks
    │        └── scripts/bpool/run_bpool_euromod.py    chunked FR CPI pricing
    │                └── scripts/bpool/run_bpool_euromod_chunk.py
    ├── scripts/bpool/assemble_bpool_priced.py    chunk parquets + 4 canary checks
    └── scripts/bpool/validate_chosen_*.py (×7)  provenance gates
              │
              ▼
    jax_ll_probe.py  jax_joint_hessian.py  jax_optimize.py
              │  (cross-checks NumPy; R1 Box-Cox bug confirmed here)
              ▼
    scripts/bpool/jax_recovery_gate.py   Checks 1–6  ⚠ NEVER MOVE (provenance)
              │
    scripts/bpool/phase_a_param_binding.py
    scripts/bpool/phase_b_recovery_test.py
              │
              ▼
    scripts/bpool/step4_realdata_baseline.py   certified run  ⚠ NEVER MOVE OR EDIT
              │
    scripts/bpool/step4_emit_results_json.py   per-group JSON
              │
    scripts/enhanced/RURO_post_estimation_styled.py  (per-group --joint-baseline-json mode)
```

### Certified spec location (critical correction from earlier drafts)

```text
scripts/bpool/specs/
    estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml        ← CERTIFIED baseline spec
    estimation_spec_joint_pooled_v1_bll0_tlmpin_gsplit.yaml ← NON-CERTIFIED (failed gate)

scripts/enhanced/specifications/  (24 YAML files)
    estimation_spec_ruro_occ_P3a_pooled.yaml, estimation_spec_v2.yaml, ...
    Does NOT contain the certified bll0/tlmpin family.
```

### Import graph — critical chains

```text
scripts/enhanced/estimation_engine.py
    └── scripts/enhanced/estimation_utils.py  ⚠ R1: box_cox_derivative_theta Taylor bug
            └── (numpy, numba, scipy)

scripts/enhanced/gamspy_estimation_vectorized.py
    ├── scripts/enhanced/estimation_spec_parser.py
    ├── scripts/enhanced/path_helpers.py       ⚠ R4: ensure_local_workdir() UNC dependency
    └── (gamspy, gams)

scripts/enhanced/enh_RURO_estimate_FR.py
    ├── estimation_engine.py
    ├── estimation_spec_parser.py
    ├── gamspy_estimation_vectorized.py
    ├── cluster_robust_se.py
    ├── diagnostics_bundle.py
    └── path_helpers.py

scripts/bpool/ JAX files
    ├── jax, optax, optimistix
    └── estimation_spec_parser.py  (shared spec parsing)
```

---

## D. Package Boundary Analysis

### EUROMOD boundary criterion

A module belongs to **`core_package`** only if it can be imported and exercised without:

- EUROMOD/Java on PATH
- FR CPI constants (`phi_2015`, `phi_2016`, `phi_2017`)
- NUTS1 codes or INSEE benchmark data
- EUROMOD-STORAGE path resolution (`ensure_local_workdir()`)
- The 35h French overtime split rule

### Proposed `dclaborsupply` (core package)

```text
dclaborsupply/
  likelihood/
    engine.py          ← estimation_engine.py         ⚠ fix R1 (Box-Cox) before publish
    utils.py           ← estimation_utils.py           ⚠ fix R1 (Box-Cox) before publish
    utils_ac2013.py    ← estimation_utils_AC2013.py
  spec/
    parser.py          ← estimation_spec_parser.py
    constraints.py     ← expression_constraints.py
    schemas/           ← scripts/enhanced/specifications/ (generic YAMLs)
  solvers/
    gamspy_vectorized.py  ← gamspy_estimation_vectorized.py
    jax_optimize.py       ← scripts/bpool/jax_optimize.py
  se/
    cluster_robust.py  ← cluster_robust_se.py
    numerical.py       ← compute_standard_errors.py
  draws/
    continuous.py      ← enh_RURO_draws.py  (draw generation; no EUROMOD)
    job_universe.py    ← Job_model/enh_job_universe.py
    job_draws.py       ← Job_model/enh_job_draws.py
    hours_mixture.py   ← scripts/bpool/hours_mixture_d1.py
  jax/
    ll_probe.py        ← scripts/bpool/jax_ll_probe.py
    joint_hessian.py   ← scripts/bpool/jax_joint_hessian.py
  sampler/
    mcfadden.py        ← mcfadden_sampler.py
  occupation/
    utils.py           ← occupation_choice_utils.py   ⚠ R10: verify no FR SILC names
  diagnostics/
    bundle.py          ← diagnostics_bundle.py
  gates/
    phase_a.py         ← scripts/bpool/phase_a_param_binding.py
    phase_b.py         ← scripts/bpool/phase_b_recovery_test.py
    joint_recovery.py  ← scripts/bpool/joint_recovery_test.py
```

### Proposed `dclaborsupply_app` (FR + EUROMOD application layer)

```text
dclaborsupply_app/
  france/
    data_prep.py       ← enh_france_data_prep.py
    gsur_v2.py         ← enh_prepare_FR_gsur_v2.py
    prep_ruro.py       ← enh_RURO_prep.py
    cpi.py             ← FR CPI constants (phi_2015/2016/2017)
  euromod/
    connector.py       ← src/mnl/integration/euromod.py (improve)
    runner.py          ← enh_RURO_euromod.py
    bpool_runner.py    ← scripts/bpool/run_bpool_euromod.py
    bpool_precompute.py ← scripts/bpool/build_bpool_precompute.py
  pipeline/
    prep_mnl.py        ← enh_RURO_prep_mnl_basic.py  ⚠ R3: consolidate log_prior first
    estimate_fr.py     ← enh_RURO_estimate_FR.py
    job_pipeline.py    ← Job_model/run_job_ruro_pipeline.py
  welfare/
    core.py            ← scripts/welfare/welfare_core.py
    vdir.py            ← scripts/welfare/welfare_vdir.py
    resim_probe.py     ← scripts/welfare/welfare_resim_probe.py
    correction_prep.py ← scripts/welfare/welfare_correction_prep.py
    stages/            ← scripts/welfare/run_stage*.py
  multi_year/
    harmonise_cpi.py   ← multi_year/m1_harmonise_cpi.py
    stack_years.py     ← multi_year/m1_stack_years.py
    config.py          ← multi_year/m1_config.py
  paths.py             ← scripts/enhanced/path_helpers.py + scripts/bpool/_bpool_paths.py
  reports/
    post_estimation.py ← RURO_post_estimation_styled.py
```

### `stays_in_jmp_repo/` — paper-reproduction only, never published

```text
jmp/
  baseline/
    step4_realdata_baseline.py    ⚠ FREEZE — provenance artifact
    step4_lr_pooling_test.py      ⚠ FREEZE
    step4_emit_results_json.py
  gates/
    jax_recovery_gate.py          ⚠ FREEZE — certification gate
    validate_chosen_*.py (×7)     ⚠ FREEZE
    run_stage3a_pinned_baseline_validation.py
    run_stage4b_population_parity_gate.py
  specs/
    estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml   ⚠ CERTIFIED — never modify
    estimation_spec_joint_pooled_v1_bll0_tlmpin_gsplit.yaml  ⚠ NON-CERTIFIED — label
    (remaining bpool specs)
  stijn/   Results/   outputs/   Prompts/   notes/   docs/   Pdfs/   literature/
  scripts/archive/
```

---

## E. Proposed Monorepo Tree

```text
MNL/
├── packages/
│   ├── dclaborsupply/
│   │   ├── pyproject.toml
│   │   └── src/dclaborsupply/      (see §D for detailed layout)
│   └── dclaborsupply_app/
│       ├── pyproject.toml          (depends on dclaborsupply)
│       └── src/dclaborsupply_app/  (see §D for detailed layout)
│
├── jmp/                            paper-reproduction (no package)
│   ├── baseline/   gates/   specs/ stijn/
│   ├── Results/   outputs/   Prompts/   notes/   docs/   Pdfs/   literature/
│   └── scripts/archive/
│
├── scripts/                        CURRENT LOCATION — do not move until test_boundary passes
│   ├── enhanced/   bpool/   Job_model/   welfare/
│   ├── multi_year/   maintenance/   diagnostics/   pilot/   archive/
│   └── (root loose files — keep as-is; not migrated)
│
├── src/mnl/                        CURRENT STUB — delete after packages/ is live
├── tests/                          migrate to packages/dclaborsupply/tests/
├── config/   configs/              → split to app_package / core_package
├── pyproject.toml                  update to workspace (uv/hatch monorepo)
└── requirements.txt                split: core / app / dev
```

---

## F. RUM vs RURO — API Implications

### Model inventory

| Model | Location | Status |
|---|---|---|
| Continuous RURO | `scripts/enhanced/` | Production; certified |
| Job-choice RURO | `scripts/Job_model/` | Active; same likelihood kernel |
| Legacy RUM (translog/Box-Cox MNL, biogeme) | `scripts/archive/rum_approach/RUM/` (21 files) | Archived; NEVER migrate |
| R RURO reference | `stijn/` (4 files) | Read-only reference |
| statsmodels MNLogit stub | `src/mnl/models/mnl.py` (39 lines) | ⚠ R2: wrong model; delete |

### Critical distinction: importance-sampling correction

**RUM:** `log_likelihood = Σ log P_MNL(chosen | alternatives)`
**RURO:** `log_likelihood = Σ [log P_MNL(chosen | draws) − log_prior(draws)]`

The `log_prior` subtraction is the importance-sampling correction. Any refactor that silently drops or modifies it produces RUM results with no error. The formula is currently split across two files (R3); consolidate before any package boundary is drawn.

### Proposed public API surface

```python
def compute_log_likelihood(
    spec: EstimationSpec,
    precomputed: PrecomputedData,
    theta: np.ndarray,
    *,
    backend: Literal["numpy", "jax"] = "numpy",
) -> float:
    """RURO importance-sampling log-likelihood.
    If spec.proposal_density is None, reduces to RUM (for future extensibility only)."""
    ...
```

---

## G. Risk Register

| # | Risk | Severity | Probability | Evidence | Mitigation |
|---|---|---|---|---|---|
| R1 | Box-Cox Taylor bug in `scripts/enhanced/estimation_utils.py:box_cox_derivative_theta` (~0.5 off near θ = 0) | high | certain | Confirmed via JAX/FD cross-check in `scripts/bpool/jax_ll_probe.py`; documented in project memory `project_box_cox_theta_grad_bug.md` | Fix before publish; Gate 3 in §H will surface it |
| R2 | `src/mnl/models/mnl.py` exposes statsmodels MNLogit — wrong model | high | certain | File inspection: 39 lines wrapping `statsmodels.MNLogit` | Delete before any public release |
| R3 | `log_prior` formula split across `enh_RURO_draws.py` and `enh_RURO_prep_mnl_basic.py` | high | medium (refactor trigger) | Byte-identical formula required; two separate files each computing it | Consolidate into one canonical function before splitting packages |
| R4 | `ensure_local_workdir()` in `scripts/enhanced/path_helpers.py` is a hard runtime dependency for GAMSPy | high | high (network env) | Documented failure mode on UNC paths | Must remain importable before any GAMSPy call; enforce via test |
| R5 | gsplit spec not labeled non-certified; shares `scripts/bpool/specs/` with certified spec | high | medium | gsplit FAILED synthetic recovery gate (err/SE = 19); both YAMLs in same directory | Add `STATUS: NON-CERTIFIED` to YAML header |
| R6 | Numba JIT import sensitivity | medium | medium (refactor trigger) | `_USE_NUMBA` flag; JIT decorators sensitive to module rename | Test Numba path after any `__init__.py` or module rename |
| R7 | CONOPT model-generation bottleneck (94% of wall time) | medium | certain | Benchmarked in `scripts/bpool/bench_conopt_modelgen.py`; threads/memory/listing tuning all measured as dead ends | Not a migration blocker; document in package README |
| R8 | EUROMOD transitive import leak into core | medium | medium | Several scripts import `enh_RURO_euromod` at module level | CI gate: `python -c "import dclaborsupply"` must succeed without Java on PATH |
| R9 | Provenance gate scripts moved accidentally | critical | low | `jax_recovery_gate.py` and `step4_realdata_baseline.py` underpin all identification and paper claims | Mark read-only; add CI check comparing file hashes to certified commits |
| R10 | FR-specific constants in nominally generic functions | medium | medium | `scripts/enhanced/occupation_choice_utils.py` not fully read; FR SILC column assumptions plausible | Gate 5 in §H will surface these |

---

## H. Next Step — Boundary-Proving Integration Test

Write **`tests/test_boundary.py`** (new file only; no edits to any existing file). Proves the core can be imported and exercised without EUROMOD, without FR constants, and without EUROMOD-STORAGE. Surfaces R1, R8, and R10 before any file is moved.

### Gate 1 — Import without EUROMOD/Java on PATH

```python
import sys
from scripts.enhanced.estimation_engine import compute_likelihood_singles
from scripts.enhanced.estimation_utils import build_precomputed_data_singles
from scripts.enhanced.estimation_spec_parser import parse_estimation_spec
assert "euromod" not in sys.modules
```

### Gate 2 — Synthetic-data likelihood is finite and negative

```python
# Construct minimal synthetic PrecomputedData; call compute_likelihood_singles()
# Assert: np.isfinite(ll) and ll < 0
```

### Gate 3 — Analytical gradient matches finite-difference to 1e-4

```python
# |grad_analytical - grad_FD| / (|grad_FD| + 1e-8) < 1e-4 element-wise
# If R1 (Box-Cox bug) is present, this gate will fail for theta_c near 0 — expected
```

### Gate 4 — Certified spec parses without error

```python
spec = parse_estimation_spec(
    "scripts/bpool/specs/estimation_spec_joint_pooled_v1_bll0_tlmpin.yaml"
)
assert spec is not None
assert "beta_ll" in spec.fixed_params   # pinned to 0
```

### Gate 5 — No EUROMOD-STORAGE path resolution in core path

```python
from unittest.mock import patch
with patch("scripts.enhanced.path_helpers.ensure_local_workdir",
           side_effect=AssertionError("core called EUROMOD path")):
    # Run compute_likelihood_singles() on synthetic data
    # Assert: no AssertionError raised — core is clean
```

### What this test does NOT do

- Does not move, rename, or edit any existing production file
- Does not create packages or modify `pyproject.toml`
- Does not require GAMSPy, EUROMOD, or EUROMOD-STORAGE
- Does not touch `scripts/bpool/jax_recovery_gate.py` or `scripts/bpool/step4_realdata_baseline.py`

---

*End of inventory. Paths verified via `find`; line counts via `wc -l` on `C:\Users\hisham\Repo\MNL` (canonical local repo). No production files modified.*
