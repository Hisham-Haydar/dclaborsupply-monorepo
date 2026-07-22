# FR P2a Region-Live — Promotion Readiness — v1

Read-only validation audit. Date: 2026-07-22. Auditor: automated cross-repo audit per
`Job_Market_paper/docs/prompts/JMP_cross_repo_state_audit_prompt_v1.md`.

Cross-repo companion: `Job_Market_paper/docs/JMP_cross_repo_manager_handoff_v1.md`.

**Rules honoured:** file/artifact evidence overrides prose; a notebook cell is not a production
pipeline; an improved likelihood is not sufficient for promotion; a PD real-data Hessian is not
sufficient without recovery or a narrower diagnostic interpretation; proposal occupation conditioning
is distinguished from structural occupation-wage modelling; no welfare readiness is declared; the
active baseline is not changed; no computational pipeline was run.

---

## 1. Audit verdict

**READY AFTER PRODUCTION REBUILD.**

The region-live improvement is a legitimate, understood **data-wiring correction** (region/urbanisation/
gsur columns were zero-stubbed at engine-ready assembly and have been revived), lowering the singles
negLL from 19071.6562 to 19053.4655 with the same spec, bounds, start and JAX engine. But region-live
currently exists only as **one executed notebook cell** plus a **propagated artifact set produced by a
script that is absent from disk**. It has **no** committed gradient, Hessian, eigenvalues, rank,
condition number, region-live cluster-robust SE, region-live post-estimation report, or verified
cold-reload anchor; its downstream welfare numbers were nonetheless already recomputed on it. The very
identification question the repair should settle — whether reviving the region/urbanisation block lifts
the flat direction seen region-dead — is **unverified**. Promotion is therefore blocked until the fit
is rebuilt through production code with a full diagnostic bundle. This is not a REJECT: the wiring fix
is sound and the likelihood gain is real; it is not DIAGNOSTIC-ONLY either, because the fix is
production-legitimate and should be carried forward — it simply must be rebuilt and diagnosed first.

## 2. Source notebook evidence

- `dclaborsupply/notebooks/fr_data_walkthrough.ipynb` is the **only** home of region-live:
  `19053.4655` and the tokens `region-live`/`region-dead` appear only here.
- **P2a-10** (cell id `7c42e9bd`, executed): performs the revival and re-fit —
  `region-live fit: negLL 19053.4655 (region-dead was 19071.6562) iters=353`, L-BFGS-B, converged.
- **P2a-9** (executed): region-dead Hessian flat-direction diagnostic (see §10).
- **`regionlive00`–`regionlive05`** (re-freeze, region-live SE, region-live post-estimation report,
  region-live Hessian, cold-reload anchor `assert abs(_nll−19053.4655)<1e-2`, welfare re-run) are
  authored but have **`execution_count: null` — never executed**.
- `fr_singles_pipeline_v1.ipynb` reproduces **region-dead only** (stated anchor and asserts at
  target 19071.6562; no P2a-9/P2a-10, no region-live cells). It has not been updated with the repair.

## 3. Production-pipeline evidence

- **None in this monorepo.** No `outputs/`, no `*.parquet`, no theta/estimation/SE artifacts are
  committed here; the notebook writes to relative paths and reads from the external MNL tree and
  `EUROMOD-STORAGE`. No `packages/*` or `exports/*` file contains the region-revival wiring or a
  region-live fit path — the reusable loader *supports* region/gsur, but the act of wiring FR-2016
  columns is inline notebook code.
- In the **MNL** repo, the region-live values were propagated into committed artifacts
  (`outputs/p2a_singles2016/estimation_results_p2a_singles2016.json`, `p2a_fit_provenance.json`,
  `theta_p2a_singles_2016_v2.csv`, `fr_singles_engine_ready_p2a_bpool_v2.parquet` + mnlmeta) by
  `command_line = propagate_regionlive.py`. **That script is not present on disk in either repo and has
  no git record** — the propagation is not reproducible.

## 4. Data and wiring changes

The repair (P2a-10) maps five columns from the source `singles_dec` into the engine-ready parquet and
re-writes it as `_v2`:

```
for c in ['drgn1', 'drgur', 'drgmd', 'drgru', 'gsur']:
    er_c[c] = er_c['idhh'].map(pd.to_numeric(src[c])).astype('float64')
er_c.to_parquet('fr_singles_engine_ready_p2a_bpool_v2.parquet')
```

Population by layer:

| Layer | Region/urb/gsur populated? | Evidence |
|---|---|---|
| Source data (`singles_dec`) | YES | `drgn1∈[1,8]` counts {1:245,…,8:163}; `gsur∈[0.05,0.23]`, mean 0.098; urban one-hots sum to 1 |
| Engine-ready parquet (default assembly) | NO (zero-stubbed) | repair note: "region/urbanization/gsur zero-stubbed at assembly" |
| Engine-ready `_v2` (after P2a-10) | YES | loader sees `reg2` mean 0.181, `gsur` mean 0.098 |
| Core loader arrays | Fully SUPPORTED | `packages/dclaborsupply/.../data/loader.py` `_reg`/`_drgn1`/`_gsur` |
| Likelihood index | SUPPORTED (data-driven) | `beta_E_gsur`, `beta_E_drgn2..8`, `beta_E_drgur/drgmd` move once data revived |

The change is **data wiring only — not a specification change** (same spec, same bounds `b4`, same
start `t0`, same `build_jax_singles_ll`). `beta_E` moved −4.31 → −2.90 because it had been absorbing
the regional variation that is now carried by the revived region block.

## 5. Specification compatibility

- Structural **`wage_spec = vw`** (standard log-normal variable wage) — inside the JAX-validated set.
- **`loc_empirical` and `vw_occupation` are NOT used structurally.** They are parser-recognised but
  have no dedicated JAX implementation and must not be used for JAX estimation
  (`docs/known_limitations.md`, README). The P2a path uses `vw`.
- **Occupation conditioning is in the proposal, not the structural wage branch:** occupation enters
  (a) the proposal density `log_q` as "W1 occ-conditional wages, empirical occ" and (b) estimated
  occupation *dummies* (`beta_occ_*`, a free block). The structural wage distribution is `vw`, not
  `vw_occupation`. This is exactly the proposal-vs-structural distinction the rules require.
- The path is **compatible with the validated JAX engine** (`likelihood/engine_jax.py`), which
  reproduces the certified FR baseline negLL 238504.6360973987.

## 6. JAX support status

**Supported.** Region-live uses `build_jax_singles_ll` / `compute_index` on the `vw` spec via the
validated JAX engine (float64). No unsupported structural feature (`loc_empirical`/`vw_occupation`) is
invoked. Region/urb/gsur enter as ordinary linear regressors in the opportunity/`beta_E` block, which
the engine already handles.

## 7. Likelihood comparison

| Fit | negLL | Δ vs region-dead |
|---|---|---|
| Region-dead (zero-stubbed region block) | 19071.6562 | — |
| Region-live (region block revived) | 19053.4655 | **−18.19** improvement |

Real and consistent across the notebook output, `p2a_fit_provenance.json`, and the MNL committed
estimation JSON summary (`joint_ll −19053.46553`). Per the rules, this improvement is **not**
sufficient for promotion on its own.

## 8. Convergence status

Region-live converged in the notebook cell: L-BFGS-B, `iters=353`, success flag set. **But** the only
**committed** solver-diagnostics file (`MNL/outputs/p2a_singles2016/p2a_singles2016_solver_diagnostics.json`)
is the **region-dead** vintage (objective −19071.656, solver "notebook P2a (warm→fit)",
`gradient_norm_results_json: null`, `n_iterations: null`). There is no committed region-live optimizer
record beyond the propagated summary. Convergence is asserted by a notebook cell, not by a persisted
production optimizer status.

## 9. Gradient status

**MISSING for region-live.** The gradient is implicit in the notebook fit (`jac=True`) but is not
persisted; the committed solver diagnostics carry `gradient_norm = null`. No max|grad| is recorded for
19053.4655. (For contrast, the certified pooled baseline records max|grad| = 44.20.)

## 10. Hessian / rank status

**MISSING for region-live; adverse evidence region-dead.**

- Region-dead (P2a-9, executed): the region block is a **flat direction** —
  `5 smallest eigenvalues: [-0. -0. -0. -0. -0.]`, flat-direction loadings on
  `beta_E_drgn3 −0.616, beta_E_drgur −0.420, beta_E_drgmd −0.405, beta_E_drgn8 0.363, beta_E_drgn4 −0.328`.
  This is expected: with zero-stubbed data the 10 regional params rest on no variation.
- Region-live (P2a-9c and the min-eig/condition print inside P2a-7c): **never executed**. The committed
  `p2a_singles2016_inference_diagnostics.json` reports `hessian.available = false`
  ("Hessian diagnostics not present in results JSON").
- Consequence: there is **no eigenvalue / rank / condition-number / Hessian evidence that region and
  urbanisation are jointly identified in the region-live model.** The notebook only *expects*
  "min_eig > 0 now that the regional params rest on real variation" — unverified. Whether reviving the
  data lifts the region-dead flat direction is precisely the open identification question.

## 11. Bound status

Two parameters at bound in region-live: **`beta_l_age2_sm`, `beta_l_age2_sf`** (leisure age² for
single males/females), recorded in the estimation-JSON metadata and `p2a_fit_provenance.json`. These
are leisure-curvature params, **not** region/urbanisation params, so they do not directly threaten the
region block. Pinned params (10): `beta_l0_m`, `beta_l_age_m`, `beta_l_age2_m`, `beta_l0_f`,
`beta_l_age_f`, `beta_l_age2_f`, `beta_l_nkids_f`, `theta_l_f`, `beta_E_y2015`, `beta_E_y2017`.
A full bound diagnostic (with the region-live Hessian) has not been produced.

## 12. Cluster-robust inference status

**MISSING for region-live.** The committed `p2a_se_clustered.csv` is the **region-dead** vintage
(2026-07-12, cluster=idorighh); the region-live SE cell (`regionlive01`) was never executed, and the
inference-diagnostics report shows `robust_se.available = false`. The region-live `theta_v2.csv`
carries two extra numeric columns, but these are bounds, not a cluster-robust sandwich. No valid
region-live standard errors exist.

## 13. Post-estimation status

**MISSING for region-live.** All committed post-estimation artifacts in
`MNL/outputs/p2a_singles2016/` (parameter table, elasticities, MU/MUC/MUL contours, fit and
distribution PNGs, diagnostics bundle) are the **region-dead** vintage (2026-07-12). The region-live
re-freeze regenerated only three files (results JSON, `_v2` parquet, mnlmeta; 2026-07-13). The
region-live post-estimation cell (`regionlive02`) was never executed. `P2A_MASTER_RECORD.md` still
documents negLL 19071.6562.

## 14. Reproducibility status

**Not reproducible as committed.**
- The region-live fit is a single notebook cell; the propagation into MNL artifacts was done by
  `propagate_regionlive.py`, which is **absent from disk with no git record**.
- The cold-reload regression anchor at 19053.4655 (`regionlive04`, `assert abs(_nll−19053.4655)<1e-2`)
  was **never executed** — the region-live number has never been re-loaded and re-verified from disk.
- The P3-1 verify cell captured the pre-propagation state as `REGION-DEAD (stale)` / `region live?
  False | gsur live? False`, confirming the region-live state was assembled after the notebook save,
  outside a reproducible pipeline.

## 15. Missing artifacts

For region-live, the following do not exist in committed form (only region-dead equivalents or
unexecuted stubs):
1. gradient / max|grad|
2. Hessian, eigenvalues, rank, condition number (joint region × urbanisation identification)
3. cluster-robust SE (cluster=idorighh)
4. post-estimation report (parameter table, elasticities, fit/MU/contour plots)
5. persisted production optimizer status
6. verified cold-reload regression anchor at 19053.4655
7. the producing script `propagate_regionlive.py`
8. a production (non-notebook) rebuild path in `dclaborsupply` packages/exports
9. corrected provenance (master record + `p2a_fit_provenance.json` theta pointer still region-dead / `_v1`)

## 16. Promotion decision

**Do not promote now. READY AFTER PRODUCTION REBUILD.** Retain the region-live wiring fix and the
19053.4655 result as the target to reproduce; keep it out of inference, writing and certified welfare
until the rebuild passes. The certified 47-param pooled baseline (negLL 238504.636097) remains the
formal active baseline and is unaffected — region-live is a separate FR-2016 singles track, not a
replacement for it.

## 17. Required next action

Rebuild the FR-2016 singles P2a region-live fit **through the reusable `dclaborsupply` package/exports
path** (not a notebook), starting from `fr_singles_engine_ready_p2a_bpool_v2.parquet`, and emit a
committed, reproducible bundle:
- engine-ready `_v2` provenance + theta CSV + estimation JSON,
- persisted optimizer status + gradient (max|grad|),
- **PD Hessian with eigenvalues, rank and condition number** demonstrating that region × urbanisation
  are jointly identified (i.e. the region-dead flat direction is lifted),
- cluster-robust SE (cluster=idorighh),
- full post-estimation report,
- a committed producing/rebuild script (replacing the missing `propagate_regionlive.py`),
- a verified cold-reload regression anchor at negLL 19053.4655,
- corrected provenance (`P2A_MASTER_RECORD.md` and the `theta_csv` pointer).

Only after this bundle exists and the identification check passes should region-live be re-submitted
for a strict-estimation verdict, and only then should the P2a welfare numbers built on it be
re-examined. Do not change the certified baseline and do not declare welfare readiness in the interim.
