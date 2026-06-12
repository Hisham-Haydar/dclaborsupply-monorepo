# Welfare migration — inventory & architecture plan (v1)

**Document class:** Inspection + planning only. No migration code, no engine/builder/spec/data
changes, no welfare production, no commit. Companion to `01_repo_inventory.md`,
`02_package_architecture_memo.md`, `03_migration_matrix.md`. Wave-4 (app) lives in
`dclaborsupply_app`.

**Provenance discipline (held throughout).** Two distinct baselines must never be conflated:
(1) the **certified estimation baseline** — spec `joint_pooled_v1_bll0_tlmpin` (47 params,
`beta_ll=0`, `theta_l_m=-0.8` pinned), `theta_hat_realdata_901_v1.csv`, stem
`fr_p3a_bpool_engine_ready`; and (2) the **staged reproducible welfare-pricing reference** —
stem `fr_p3a_bpool_engine_ready_staged_threeB1`, chunks `staging_twoN/`, priced
`staging_threeB1_priced/`. The staged reference is a *pricing oracle candidate only*; the MNL
config marks `production_swap_authorised: false`, `promote_staged_to_canonical: false`. It is
**not** automatically the canonical production baseline.

---

## 1. Executive verdict

The welfare **machinery is more complete than "design-only"** but its **reportable scope is
narrow and its strongest exact oracles are not yet frozen**. Concretely:

- **Country-general welfare math is implemented and oracle-backed for porting:** inclusive-value
  integration (`V_i^IS`), ESS/max-weight diagnostics, `W3` equivalent-income inversion, Gini
  arithmetic, and a Gate-0 estimator/welfare parity check all run today and reproduce the
  estimator's own negLL to machine zero (singles_male **28489.042816294535**, singles_female
  **35411.86351549324**, couples **174603.72976561091**, max|Δ| = **0.0** at production tol 1e-6).
  These are **PORT** units.
- **Singles `V_i^dir`** (redraw → population-faithful pricing → analytic integration) is
  **implemented and has run as a bounded diagnostic smoke + node-count calibration**, with a
  verified non-employment canonicalization and a clean utility-only-vs-full-V distinction. It is
  **PORT**, but its exact replay artifacts (frozen node + priced tables) have been **deleted**
  (both scratch dirs are empty), so today only per-household *scalars* survive as anchors.
- **Couples `V_i^IS`** ran in the Stage-1 production gate (couples Gate-0..4 all `ok`, 7438
  households, 901 alts), but **per-household couples scalars are not stored** and **couples
  `V_i^dir` (redraw pricing/integration) was never built** — `redraw_nodes_couples` exists but is
  never invoked. Couples `V_i^dir` is **BUILD**.
- **W1/W2/W4/W5/W6** reference objects and the **Shapley–Shorrocks** decomposition are
  **design-only / absent** (config slots `null`; zero `shapley|shorrocks` references in code).
  These are **BUILD**.
- **One core hook is required.** `welfare_core.py` reconstructs the **per-alternative `V` grid**
  itself (it does not use a scalar negLL). The migrated core exposes only a **scalar**
  `compute_index`; the per-alternative `V`/`u` components exist **only in an internal, unexported**
  NumPy path (`engine_numpy.compute_likelihood_singles/couples(return_components=True)`). Welfare
  integration in the app therefore needs a **public, backend-neutral per-alternative
  V/utility-component extractor** in the core (singles + couples, utility and full corrected index,
  consumption overrides) — surfacing the existing per-alternative components with **no new index
  math**, gated on NumPy/JAX parity and certified-FR reproduction.

**Authorized next (smallest evidence-backed step):** **W4.0-DESIGN only** — *specify* (not
implement) the backend-neutral per-alternative component API the welfare layer needs. In parallel,
**W4.1A** may PORT only pure/general welfare *arithmetic* validated by **synthetic** unit tests.
Reproduction and freezing of the live single-household `V_i^IS` anchor (a **candidate**, UID
200001593700 = `11.496632024594227`) remains **blocked** until the W4.0 API is implemented and
gated — the anchor cannot be reproduced through package-native machinery until that hook exists.
Nothing else is authorized.

---

## 2. Source inventory and two-axis classification

Axis A = COUNTRY-GENERAL mechanism vs FRANCE/INSTITUTIONAL rule. Axis B = implementation status
verified from code/artifacts: RUN (implemented-and-run), IMPL (implemented, not production-run),
BLOCKED, DESIGN. PORT = implemented + MNL oracle; BUILD = net-new or no usable oracle.

| Unit | Source (file:line) | Axis A | Axis B | PORT/BUILD |
|---|---|---|---|---|
| Inclusive value / `_group_lse_and_V` | `welfare_core.py:119-124` | general | RUN | PORT |
| `compute_group_welfare` orchestration | `welfare_core.py:463-493` | general | RUN | PORT |
| `_build_V_extractor` (singles V grid) | `welfare_core.py:137-280` | general | RUN | PORT (needs core V hook) |
| `_build_V_extractor_couples` (incl. `beta_ll`) | `welfare_core.py:283-443` | general | RUN (Stage-1) | PORT (needs core V hook) |
| Singles `V_i^IS` / ESS / max-weight | `welfare_core.py:485-490` | general | RUN | PORT |
| Couples `V_i^IS` / ESS / max-weight | `welfare_core.py:283-443` + `production_results.json` | general | RUN (Stage-1; per-HH not stored) | PORT (anchor must be minted) |
| `w3_inversion` (equivalent income) | `welfare_core.py:523-575` | general | RUN (singles+couples Stage-1) | PORT |
| `gate0_parity` (engine parity) | `welfare_core.py:499-517` | general | RUN | PORT |
| `gini` / `_gini_mad_stream` | `welfare_core.py:581-603` | general | RUN (internal-validation only) | PORT |
| `redraw_nodes_singles` | `welfare_vdir.py:98-136` | general (FR Mincer/year_tag injected) | RUN | PORT |
| `redraw_nodes_couples` | `welfare_vdir.py:232-250` | general | IMPL (never invoked, no oracle) | BUILD/ADAPT |
| Population-faithful node pricing | `run_stage4c_singles_vdir_smoke.py:94-195` | general structure, FR runner | RUN | PORT/ADAPT |
| Non-employment canonicalization (redraw) | `run_stage4c_singles_vdir_smoke.py:161-165` | FR earnings identity | RUN | PORT (FR rule) |
| Singles `V_i^dir` utility-only (contract) | `run_stage4c...:705-744` (`_vdir_for_hh`) | general | RUN (bounded smoke) | PORT (replay artifacts deleted) |
| Full-V diagnostic + like-for-like compare | `run_stage4c...:738,747-815` | general | RUN | PORT |
| Stage-4C2 1/S bias calibration | `run_stage4c2_vdir_bias_calibration.py` | general calibration | RUN | PORT |
| Couples `V_i^dir` pricing/integration | absent (no joint-record pricer) | general+FR | DESIGN | BUILD |
| `W3` reference object | `welfare_stage1_w3.yaml:38`; `run_stage1_w3.py` | general | RUN (singles+couples Stage-1) | PORT |
| `W1`, `W2` reference objects | absent | general | absent | BUILD |
| `W4`/`W5`/`W6` + `Abar`/`J`/`o` sets | `welfare_stage1_w3.yaml:44-46` (`null`) | will be FR/EUROMOD-institutional | DESIGN | BUILD |
| Shapley–Shorrocks (opp/ability/pref) | none (`shapley`/`shorrocks` = 0 hits); readiness interfaces only `run_stage1_w3.py:184-191` | general | DESIGN | BUILD |
| FR pricing/aggregation/deflation rules | `welfare_stage1_w3.yaml:246-299` (`stage4`); `run_stage4c...:53,187,628,724` | FR/institutional | RUN | PORT/ADAPT (as injected rules) |

**Reconciliation note (couples V_i^IS).** `welfare_core.py` docstrings/comments repeatedly say
"COUPLES_DEFERRED", but the implemented `_build_V_extractor_couples` is a complete jit extractor
**and** `production_results.json` shows couples computed end-to-end (Gate-0 couples max_abs
**0.0**, negLL **174603.72976561091**; Gate-2 couples `omega` computed; Gate-3
`couples_joint_unit: true`). Verified status: **couples `V_i^IS` and `W3` inversion ran at
Stage-1**; only **couples `V_i^dir`** is genuinely deferred. (Flagged again in §10.)

---

## 3. Dependency and coupling map

### 3.1 Current MNL welfare dependencies

- **Estimator-index coupling (load-bearing).** `welfare_core.py:59-72` injects `scripts/enhanced`
  and `scripts/bpool` onto `sys.path` and imports `estimation_spec_parser` (`sp`),
  `joint_recovery_test` (`jrt`, the data/theta loaders), and `jax_ll_probe` (`jllp`, the estimator
  builder — `build_jax_singles_ll/couples_ll`, `jbox_cox`, `_center_proposal`). The V extractors
  re-derive `V = u + log_h + log_w + log_market - log_prior` row-by-row from `jllp` internals
  (`welfare_core.py:274,437`).
- **JAX/NumPy.** JAX x64 is the compute path (jit V extractors, logsumexp); NumPy for the bisection
  solve and Gini.
- **EUROMOD.** Not imported in `welfare_core.py`; arrives via the data objects + resolved config.
  In `welfare_vdir.py`/Stage-4C it is reused from the **build module** `run_bpool_euromod_chunk`
  (`_SYSTEM_PAIRING`, `_CPI`, `_RAW_SCHEMA`, `EuromodRunner`, `_EM_ROOT`, `_stamp_draw_ids`) via
  `welfare_vdir._build_constants` (`welfare_vdir.py:142-155`).
- **MNL loader.** Data via `jrt.build_data_objects` on stems `fr_p3a_bpool_engine_ready`
  (estimation) / `fr_p3a_bpool_engine_ready_staged_threeB1` (welfare pricing).
- **Staged-stem / storage layout.** Stage-4 reads `{staged_stem}__{mode}.parquet`,
  `{staged_stem}__mnlmeta.json`, `{precomp_stem}__{year}__{mode}__long.parquet`, and writes
  scratch dirs (now empty).
- **Country config.** FR system pairing (2015→FR_2014, 2016→FR_2015, 2017→FR_2016), French CPI φ
  (2015:1.0031, 2016:1.0, 2017:0.9886), per-year EUROMOD schema (122/124/128 cols), 35h
  `yem00/yemxp` split, all in `stage4` / build module.

### 3.2 Package-native equivalents already available

- **Loader:** `dclaborsupply.data.load_engine_ready_stem(stem, spec, *, year_tags=None,
  hours_band_policy="assembled")` produces the exact `(sm, sf, cou)` `PrecomputedData*` containers
  the engines/components consume — direct replacement for `jrt.build_data_objects`.
- **Likelihood engines (proven):** `likelihood/engine_jax.py` (fit) and `engine_numpy.py`
  (evaluation, with `return_components`). Certified FR negLL **238504.6360973987** reproduced.
- **App EUROMOD/pricing:** `dclaborsupply_app.euromod` exports `EuromodConnector`,
  `PricingConnector` (Protocol), `EuromodPricingRunner` (`.build_inputs`, `.price` →
  `PricingResult` with `taxunit_totals`), `EarningsMutationPolicy` (Protocol), `PricingColumns`.
- **Country adapters:** `dclaborsupply_app.de` — `canonicalize_post_draws` /
  `canonicalize_choice_state`, `de_earnings_policy`, `engine_ready.assemble*` (with
  country-general `aggregate_consumption`, `apply_consumption_floor`, `restore_cluster_id`,
  `loc4_one_hots`). `france/` is placeholders only.
- **App welfare:** `dclaborsupply_app.welfare` exists but is pure `NotImplementedError`
  placeholders (`welfare/core.py`, `welfare/vdir.py`, `welfare/measures.py`) — the migration
  *fills* these.

### 3.3 Estimator-index coupling — the precise finding

- `welfare_core.py` needs the **per-alternative `V` grid** (and, for the contract `V_i^dir`, the
  **utility-only `u(c,l)`** component), because both `V_i` integration and the `W3` consumption-
  shift inversion (`lse_at`, re-evaluating the grid at `c+w`) are impossible from a scalar.
- `dclaborsupply.likelihood.compute_index(spec, data, theta, *, ruro, backend="jax") -> float`
  returns a **scalar** negLL on every path (`index.py:119-165`); `ruro=False` only supplies a
  correction-null scalar. It cannot supply the welfare integration object.
- The only existing per-alternative extractor is **internal and unexported**:
  `engine_numpy.compute_likelihood_singles/couples(theta, data, spec, return_components=True)`
  returns a dict with keys `V, u, log_h, log_w, log_market, lse, V_obs, ll, neg_ll` (couples adds
  `log_h_male/female`, `log_w_male/female`). `V`/`u` are **flat per-row** arrays sliced by
  `data.group_starts/ends`. Not in any `__init__` (no `__all__`).
- **Finding (do not solve here):** migration requires a **public core hook for per-alternative
  V/utility components**. Do **not** copy or re-implement index math in `dclaborsupply_app`. See
  §8 for the minimal proposed signature.

---

## 4. Correctness-critical constraints (findings only)

### 4a. Population-batch faithfulness

`V_i^dir` prices counterfactual nodes through EUROMOD, and means-tested benefits depend on the
**population batch**. The MNL Stage-4 path already does this correctly: it prices the **whole
production-chunk band** with the population intact (`run_stage4c...:96-104,107,567-571`;
`n_population_rows_priced=241895`, `n_population_hh=1676` for the 6-HH singles smoke), overwriting
only the target household's decider draw slots. The Four-B method correction is explicit: *"a draw
sub-slice, even with all households present, is a DIFFERENT EUROMOD batch"* — the full chunk band
must be repriced. **Constraint to honor:** the migrated path MUST reuse the complete
production-chunk population batch, never price isolated households or node sub-bands. Map to
`EuromodPricingRunner.price`/`build_inputs`; preserve household relationships, decider flags
(`ruro_decider`), and all means-tested-benefit inputs (the Four-B/Two-L `ils_ben` signature is the
failure canary — Two-K showed 16–22% `ils_ben` mismatch on sub-faithful repricing).

### 4b. Redraw-to-pricing state audit — NO core draw-zero bug reproduced

Full path audited (`redraw node → template overwrite → population-batch pricing input → priced
output → integration object`). The Stage-4C overwrite **explicitly canonicalizes non-employment**
(`run_stage4c...:161-165`): for any node with `wk==0 or h<=0 or w<=0` it sets `working=0.0` and
zeroes `lhw,yivwg,yem,yem00,yemxp,yem_hour,hours,wage`; working nodes get the FR identity
`yem=yem00+yemxp` with the 35h split (`:143-160`). `redraw_nodes_singles` pre-fills non-working
slots `hours=0,wage=0,loc4=-1` (`welfare_vdir.py:111-113`). Checked explicitly: **no** positive
wage at `hours==0`, **no** stale `working`/hours-band/`yem*` fields. **Finding:** the welfare
redraw path is already canonicalized; the core opportunity-generator draw-zero issue (documented
in `docs/known_limitations.md`) is **not** reproduced here. The validation that proves it is the
`:161-165` zeroing branch + the population-parity gate machine-zero result. (The app-side
`de.canonicalize_post_draws` is the package-native analogue to carry forward.) No app-layer
post-draw canonicalization is *demonstrated necessary* for this path, but it MUST be re-verified
after migration (the FR yem identity is FR-specific and must be ported as an injected rule).

### 4c. France pricing and assembly rules (injected FR/app rules, not universal math)

Locate and preserve as **France/app** rules: FR singles consumption = **decider disposable
income** (`run_stage4c...:189-190,723` reads `ils_dispy_real` of the `ruro_decider==1` node); FR
couples consumption = **tax-unit sum** (app: `PricingResult.taxunit_totals`;
`de.engine_ready.aggregate_consumption`); consumption **floor clip(.,1.0)** at engine-ready
assembly (`DCM_MIN_POSITIVE=1.0`, `run_stage4c...:53,724`; app `apply_consumption_floor`);
counterfactual wage **nominal before EUROMOD** (`:628-629,673`); **real via φ_y after** pricing,
**no double deflation** (`:187`, `out["ils_dispy_real"]=ils_dispy*φ`); **cluster identity** from
`source_idorighh` (NOT present in the welfare scripts — it is restored upstream/at assembly; app
analogue `de.engine_ready.restore_cluster_id`). Keep all of these out of Tier-1 general math.

### 4d. `V_i^dir` object distinction (must not collapse)

Preserve two distinct objects: (a) the **contract welfare object** — `V_i^dir = log mean_s
exp(u(c_is, ℓ_is))`, **own-preference utility only** (because drawing from `ĝ` makes `ĝ/π`
uniform, so opportunity terms and `-log_prior` are the sampling density, not the integrand;
`run_stage4c...:736-737`); and (b) the **full-V diagnostic** — `log mean_s exp(u + log_h + log_w +
log_market - log_prior)`, which *does* have a like-for-like `V_i^IS` comparison via
`delta_common = V_dir_full − [V_IS − log(n_draws)]` (`:738,781`). The **gate/licensing object is
the utility-only one**; the full-V object is the validation instrument. The utility-only object
has **no valid `V_i^IS` counterpart** (a large raw gap is expected and meaningless). Do not merge
them.

---

## 5. Oracle-anchor proposal (smallest deterministic granularity)

### 5.1 Primary CANDIDATE anchor — single-household `V_i^IS`

A per-household `V_i^IS` scalar already exists in the Stage-4C artifacts and is
node-count-independent (derived from the 101 existing draws). It is a **candidate anchor only**: it
becomes a frozen never-move anchor once its **exact input rows**, **θ/spec hashes**, and **full
provenance** are captured (the artifacts store the scalar and seed but **no θ/spec/version hash** —
see §10.3) AND it is reproducible through package-native machinery (which requires the W4.0 hook,
§8). Until then it is a target, not a locked anchor. Proposed candidate:

- **Spec/θ:** certified `joint_pooled_v1_bll0_tlmpin` (47 params), `theta_hat_realdata_901_v1.csv`.
- **Data:** existing engine-ready alternatives/draws, staged stem
  `fr_p3a_bpool_engine_ready_staged_threeB1`, `c_scale=2034.988978049439`, `l_scale=10.0`,
  `n_existing_draws=101`, year 2016, mode singles.
- **Household:** UID **200001593700** (high-ESS, group singles).
- **Expected scalar:** `V_i^IS = 11.496632024594227` (identical in S=20 and S=60 files).
  Companions: `ESS = 40.452586177800676`, `max_w = 0.06736301317123719`.
- **Source artifact:** `outputs/welfare/stage1_w3/stage4c_singles_vdir_smoke.json`
  (`task4_vdir_vs_vis`).
- **Tolerance:** ≤ 1e-9 (a deterministic logsum on fixed draws; treat like Gate-0) — **applies
  once reproducible through package-native machinery + W4.0**.

Also available as redundant candidate anchors: the other five singles UIDs (200001687502,
200001793700, 200001813600, 200001917500, 200001981300). **Couples and any non-listed UID have NO
stored per-HH scalar** — those anchors must be minted by the fast lane.

A complementary **aggregate Gate-0 parity anchor** (already exact) is the estimator/welfare negLL
identity: singles_male **28489.042816294535**, singles_female **35411.86351549324**, couples
**174603.72976561091**, max|Δ| **0.0** (tol 1e-6). This is the strongest existing migration oracle.

### 5.2 Secondary anchor — `V_i^dir` (exact replay NOT currently freezable)

The preferred anchor is a **frozen exact-replay**: household UID → frozen node table → frozen
population-batch pricing input identity → frozen priced-node output → utility-only `V_i^dir`
scalar → full-V diagnostic scalar → metadata. **However, the frozen node/priced artifacts have
been deleted** (`scratch_four_c_vdir` and `scratch_four_c2_bias` are both **empty**;
`n_population_rows_priced=241895` was transient). Today only the **scalars** survive, e.g. UID
200001593700: `V_i^dir(util, S=20)=2.236098528111568`, `S=60=2.333468890508914`,
`full-V(S=20)=4.899766610436251`, `delta_common(S=20)=-1.981744897316716`,
`bias-corrected V_dir@Smax=2.299346321529319`, `cv2@Smax=0.08995778953810254`.

Therefore:

- A **scalar regression anchor** (the six per-HH `V_i^dir` utility-only + full-V + ESS + max_w +
  delta_common values, seed **20260604**, S∈{20,60,100}) can be frozen from existing JSON **as a
  fragile reproducibility check** only.
- A **fixed-seed RNG replay** would require pinning ALL of: NumPy version + bit generator, seed
  20260604 (and the per-HH `seed + uid%1_000_000` scheme in 4C2), draw ordering
  (employment→occupation→hours→Halton-wage), the Halton wage configuration + harvested sub-seed,
  any JAX version/config, row/household ordering, node count S, and the country-pricing inputs
  (staged stem, system pairing, CPI φ, EUROMOD schema). This is **fragile** and is explicitly NOT
  the strongest oracle.
- **Recommendation:** the fast lane must **re-mint and FREEZE** the node + priced-node tables (not
  just scalars) for ≥1 high-ESS household so a durable exact-replay `V_i^dir` oracle exists. Until
  then, treat `V_i^dir` migration as gated by the scalar anchors + a re-mint requirement.

### 5.3 Honest oracle-availability statement

- **No** reportable full-decomposition oracle and **no** broader W-family (W1/W2/W4/W5/W6) oracle
  currently exists.
- `W3` smoke and production **have run**; `W3` inversion runs (zero-recovery `phi0 = 0.0`,
  monotone, bracketed, converged for all households).
- The `W3` `Omega`/`Gini` output is a **degenerate INTERNAL-VALIDATION artifact** (`Omega ≈
  -2.91e-10` for every household; `Gini = 0` by construction against the own set), explicitly
  **not** a reportable welfare number.
- **No oracle may be invented.** Couples `V_i^dir`, full singles `V_i^dir` production, and any
  W-family/decomposition number have no oracle and are BUILD-with-new-gate.

### 5.4 Fast-lane capture list (to create durable anchors)

For the upcoming singles production run, capture and freeze, per selected household ID(s):
existing-draw `V_i^IS`; the redrawn node table; the full population-batch pricing input identity
(stem, band, decider slots, HH set); priced-node output (`ils_dispy`, `ils_dispy_real`); utility-
only `V_i^dir`; full-V diagnostic; ESS + max-weight; `delta_common`; seeds + RNG/bit-generator +
Halton config; **θ-vector hash + spec hash** (currently absent from artifacts); pricing reference
(staged stem, system pairing, CPI φ, EUROMOD schema version); and batch identity
(`n_population_rows_priced`, `n_population_hh`).

---

## 6. Architecture proposal (three tiers in `dclaborsupply_app`)

### Tier 1 — pure/general welfare math → `welfare/general/`

Path: `packages/dclaborsupply_app/src/dclaborsupply_app/welfare/general/`. Pure functions, no
EUROMOD/JAX-at-import, no country constants:

- inclusive-value / log-sum helpers (port of `_group_lse_and_V`);
- equivalent-income solve given utility + budget objects (port of `w3_inversion`);
- inequality indices (port of `gini`/`_gini_mad_stream`);
- Shapley–Shorrocks arithmetic (BUILD).

These are country-general and *could* live in core, but they consume the per-alternative V grid
and welfare-specific objects, so the app `welfare/general/` is the right home **unless** the core
chooses to expose them alongside the V hook. **Do not move or implement here** — recorded as a
placement decision for W4.1A.

### Tier 2 — app pricing & country adapters → `welfare/pricing/`, `france/welfare/`, `de/welfare/`

Paths: `welfare/pricing/`, `france/welfare/`, `de/welfare/`. EUROMOD node→disposable-income
mapping (reuse `euromod.EuromodPricingRunner.price`); full population-batch construction (4a);
France pricing/aggregation/deflation (4c) as injected rules; DE rules (reuse
`de.de_earnings_policy`, `de.engine_ready`). Couples joint-record pricing (BUILD) lands here.

### Tier 3 — welfare integration glue → `welfare/integration/`

Path: `welfare/integration/`. `V_i^IS` construction; `V_i^dir` orchestration (redraw → Tier-2
pricing → Tier-1 integration); singles/couples integration; oracle/gate runners. **Required
interface to the core:** the public per-alternative V/utility-component extractor (§8). **Do not
duplicate core utility/index math in this tier** — it calls the core hook.

---

## 7. Gated Wave-4 migration matrix (welfare row)

**Numbering.** These `W4.0`–`W4.7` sub-stages **refine only the welfare row of the existing Wave 4**
in `03_migration_matrix.md` (the `Welfare` group: `welfare/{core,vdir,measures}.py`). They do
**not** replace or renumber the rest of the Wave-4 plan (EUROMOD, France prep, pipeline, reports),
which stands as written. The `W4.x` labels are local to this welfare row.

Matches `03_migration_matrix.md` discipline: one unit → one pre-registered gate → result licenses
the next. PORT requires an MNL oracle; BUILD does not (and must state "no oracle"). A stage is
**not** PORT merely because a design doc describes it. Welfare-gate map uses the contract gates:
**G0** estimator/welfare parity; **G1** integration stability + ESS/max-weight + `V_i^dir`
diagnostics; **G2** inversion sanity; **G3** household-unit integrity; **G4** reference coverage.

### W4.0-DESIGN (prerequisite, CORE) — specify backend-neutral per-alternative component API — **BUILD (core, design)**

- **Source → Target:** the existing internal per-alternative components
  (`engine_numpy.compute_likelihood_singles/couples(return_components=True)`) → a *specified*
  public, backend-neutral core API (signature in §8). Design only.
- **Dep:** frozen core; certified FR negLL anchor; welfare Gate-0 negLLs.
- **Authorized scope:** *write the API design* — must cover **singles and couples**, expose both
  **utility-only** `u(c,l)` and **full corrected index** `V` per alternative, accept a
  **consumption override** (for the `W3` `c+w` inversion), preserve **light imports**, and define
  **NumPy/JAX parity** + **certified-FR regression** gates. **No implementation.**
- **Gate (design acceptance):** the design names the exact signature, return shape (per-group V/u),
  backend dispatch, and the two regression gates below. **Map:** G0 (design). **Oracle:** n/a (design).
- **Stop:** design cannot be expressed without new index math, or cannot serve both groups +
  override.

### W4.0-IMPL (CORE, NOT YET AUTHORIZED) — implement the API — **BUILD (core)**

- **Source → Target:** as designed in W4.0-DESIGN → public function re-exported from
  `dclaborsupply.likelihood`; thin surface over the existing components, **no new index math**.
- **Dep:** W4.0-DESIGN accepted.
- **Authorized scope:** **NOT AUTHORIZED in this memo** (separate authorization).
- **Gate (pre-registered):** (i) FR certified negLL **238504.6360973987** unchanged (≤1e-4);
  (ii) **NumPy/JAX parity** of the returned per-alternative `V`/`u` (≤1e-9);
  (iii) per-group logsum reproduces the welfare Gate-0 negLLs (28489.0428 / 35411.8635 /
  174603.7298) to ≤1e-6. **Map:** G0. **Oracle:** YES (Gate-0 negLLs + FR anchor).
- **Stop:** any FR-negLL drift, NumPy≠JAX beyond tol, or logsum≠estimator negLL beyond tol.

### W4.1A — PORT pure/general welfare *arithmetic* (synthetic unit tests) — **PORT**

- **Source → Target:** `welfare_core.py:119-124,523-575,581-603` → `welfare/general/`.
- **Dep:** **W4.0-DESIGN** (API shape known); does **not** depend on W4.0-IMPL.
- **Scope:** PORT only the pure/general *arithmetic* — inclusive value, equivalent-income
  inversion, Gini — as pure functions validated by **synthetic** unit tests. **No live data, no
  package-native `V_i^IS` reproduction** (that needs W4.0-IMPL and is out of scope here).
- **Gate (synthetic only):** unit tests reproduce `w3_inversion` zero-recovery (`phi0=0.0`),
  monotonicity, and bracket convergence on a **fixed synthetic** monotone fixture; Gini matches
  `welfare_core.gini` on synthetic arrays to ≤1e-12. **Map:** G2 (arithmetic). **Oracle:**
  synthetic self-consistency only (the live `V_i^IS` candidate anchor is **deferred to W4.3**,
  after W4.0-IMPL).
- **Stop:** any synthetic-test mismatch.

### 4.2 — PORT/ADAPT app pricing with population-batch faithfulness — **PORT/ADAPT**

- **Source → Target:** `run_stage4c...:94-195` + FR rules (4c) → `welfare/pricing/` +
  `france/welfare/` (reuse `euromod.EuromodPricingRunner`).
- **Dep:** W4.1A; W4.0-IMPL; `dclaborsupply_app.euromod`; `load_engine_ready_stem`.
- **Scope:** population-faithful pricing of existing nodes; FR consumption/deflation/floor as
  injected rules; non-employment canonicalization (4b) re-verified.
- **Gate:** reprice the staged reference and reproduce stored priced columns to **machine zero**
  (replicate Four-B: all headline + simulated components `max abs = 0.0`); `ils_ben` parity holds
  at full-chunk scope. **Map:** G4 (pricing/coverage). **Oracle:** YES (Four-B parity).
- **Stop:** any non-zero parity residual, or `ils_ben` mismatch (the Two-K canary).

### 4.3 — PORT single-household `V_i^IS` + Gate-0 parity — **PORT**

- **Source → Target:** `compute_group_welfare`/`gate0_parity` → `welfare/integration/`.
- **Dep:** W4.0-IMPL, W4.1A.
- **Scope:** construct `V_i^IS` from the public core component hook; Gate-0 parity vs estimator
  negLL; **promote the candidate single-household anchor to a frozen anchor** here (this is the
  first point it can be reproduced through package-native machinery).
- **Gate:** Gate-0 negLLs reproduced (28489.0428 / 35411.8635 / 174603.7298, max|Δ|≤1e-6);
  candidate per-HH `V_i^IS` anchor (200001593700 = 11.496632024594227, with frozen input rows +
  θ/spec hashes) reproduced ≤1e-9; ESS/max-weight summaries match production (singles_male median
  ESS 20.26; couples median 63.16). **Map:** G0 + G1(ii). **Oracle:** YES (once frozen).
- **Stop:** parity or anchor failure.

### 4.4 — PORT singles `V_i^dir` calibration/oracle reproduction — **PORT** (with re-mint)

- **Source → Target:** `welfare_vdir.redraw_nodes_singles` + `run_stage4c/4c2` →
  `welfare/integration/` + Tier-2 pricing.
- **Dep:** 4.2, 4.3.
- **Scope:** redraw → population-faithful pricing → utility-only `V_i^dir` + full-V diagnostic;
  keep objects distinct (4d); re-mint + FREEZE node/priced tables for the anchor HH.
- **Gate:** reproduce the six-HH scalar anchors (seed 20260604, S∈{20,60,100}); high-ESS full-V
  `delta_common` S→∞ intercept ≤ 0.5 nat (Four-C2: abs_max 0.395); bias-corrected utility-only
  `V_i^dir` stable. **Map:** G1(ii). **Oracle:** scalars YES; exact-replay only after re-mint.
- **Stop:** intercept > 0.5 nat (persistent offset), or no frozen replay artifact produced.

### 4.5 — PORT/BUILD couples `V_i` integration — **PORT (V_i^IS) / BUILD (V_i^dir)**

- **Source → Target:** `_build_V_extractor_couples` (V_i^IS, PORT); couples redraw pricing/
  integration (BUILD) → `welfare/integration/` + `welfare/pricing/`.
- **Dep:** 4.3, 4.4.
- **Scope:** couples `V_i^IS` (port; mint per-HH couples anchor from a run); couples `V_i^dir`
  joint-record pricing (build the two-partner overwrite path).
- **Gate:** couples Gate-0 negLL **174603.72976561091** (≤1e-6); couples joint-unit integrity
  (one `Omega` per couple, `couples_joint_unit=true`, no per-capita split); couples `V_i^dir`
  population-faithful parity. **Map:** G0 + G3 (+ G1 for V_i^dir). **Oracle:** V_i^IS YES (negLL);
  V_i^dir NONE (BUILD — must mint).
- **Stop:** any per-capita leakage; couples parity residual.

### 4.6 — PORT/BUILD `W3` + broader W-family reference objects — **PORT (W3) / BUILD (W1/2/4/5/6)**

- **Source → Target:** `welfare_core.w3_inversion`/`run_stage1_w3.py` (W3, PORT); W1/W2/W4/W5/W6 +
  `Abar`/`J`/`o` reference sets (BUILD) → `welfare/integration/` + Tier-2 reference pricing.
- **Dep:** 4.3, 4.5.
- **Scope:** port `W3` (own-set, no EUROMOD reference coverage needed); BUILD the reference-set
  construction + EUROMOD reference-package coverage gate for the others.
- **Gate:** `W3` inversion sanity (G2: zero-recovery, monotone, bracketed); for any W4/W5/W6,
  reference-coverage gate (G4): every required `c_ij` for `Abar`/`J`/`o` exists, finite, positive,
  evaluated at 2016-real basis + EUROMOD system year, **no wholesale rerun, no silent
  interpolation**. **Map:** G2 + G4. **Oracle:** `W3` YES (internal-validation only); others NONE.
- **Stop:** any missing/ non-finite reference `c_ij`; non-convergent inversion.

### 4.7 — BUILD Shapley–Shorrocks decomposition — **BUILD**

- **Source → Target:** none (readiness interfaces only `run_stage1_w3.py:184-191`) → new module.
- **Dep:** 4.6.
- **Scope:** build the access/ability/preference 3-channel Shapley average (6 orderings) over a
  Full-Responsibility anchor (W2/W3) + duals (W1/W5).
- **Gate:** **exhaustiveness** — components sum **exactly** to `I(Omega^k)` (order-independence);
  block membership = config (`preference`/`ability`/`access`). **Map:** forward decomposition gate.
  **Oracle:** NONE — pure BUILD with a self-consistency (exhaustiveness) gate.
- **Stop:** components do not sum to total inequality.

---

## 8. Core-change forecast

Migration **cannot remain entirely in `dclaborsupply_app`**: it needs **one** public core hook.

- **Package-native loading:** ✅ available (`load_engine_ready_stem`).
- **Scalar Gate-0 parity:** ✅ via `compute_index` (scalar negLL).
- **Per-alternative V & utility-component extraction:** ❌ **required and missing publicly.**
  `compute_index(..., ruro=False/True)` returns only a scalar; the per-alternative `V`/`u` live in
  the **internal, unexported** `engine_numpy.compute_likelihood_singles/couples(return_components=
  True)`.
- **Fixed-choice/no-opportunity counterfactual:** the correction-null RUM view exists for scalar
  *evaluation* (`build_rum_view`), but welfare needs the per-alternative grid, not the scalar.
- **Singles & couples support:** both component paths exist internally (couples adds gendered
  hours/wage); the hook must cover both.
- **Parameter/block exposure:** the decomposition (4.7) needs block membership
  (preference/ability/access) — config-level, no core change.

**Required core hook (record as finding; do not implement):**

- **Behavior:** given `(spec, data, theta)`, return the per-alternative `V` grid (per group,
  sliceable by `group_starts/ends`) **and** the per-alternative utility-only component `u(c,l)`;
  optionally accept a consumption override (for the `W3` `c+w` shift) so the inversion need not
  re-enter app math.
- **Why existing APIs are insufficient:** `compute_index` is scalar-only;
  `Result`/`models.py` expose no per-alternative object; the only extractor is unexported engine
  internals (fragile to depend on; violates layering).
- **Minimal proposed location/signature (backend-NEUTRAL):** e.g.
  `dclaborsupply.likelihood.index.compute_components(spec, data, theta, *, ruro,
  backend="numpy"|"jax", consumption=None) -> dict` returning per-group, per-alternative `V`
  (full corrected index) **and** `u` (utility-only) — for **both singles and couples** — with an
  optional **consumption override** for the `W3` `c+w` inversion. It surfaces the existing
  per-alternative components (NumPy via `return_components`; JAX via the engine's pre-reduction
  `V`) and is re-exported from `dclaborsupply.likelihood`. **No new index math**; do **not**
  prescribe a NumPy-only wrapper — the API is backend-neutral and both backends are gated for
  parity. Preserve light imports (JAX lazy).
- **Regression gates for the change:** (i) FR certified negLL **238504.6360973987** unchanged
  (≤1e-4); (ii) **NumPy/JAX parity** of the returned per-alternative `V`/`u` (≤1e-9); (iii) welfare
  Gate-0 negLLs (28489.0428 / 35411.8635 / 174603.7298) reproduced from the hook's per-group
  logsum to ≤1e-6.

---

## 9. Fast-lane status readout

Direct status for the conference fast lane (source · status · runs as-is today · already run ·
artifact · next-run requirement).

| Item | Source | Status | Runs today | Already run | Artifact | Next-run requirement |
|---|---|---|---|---|---|---|
| Singles `V_i^IS` | `welfare_core.py` + `run_stage1_w3.py` | IMPLEMENTED | yes | YES (Stage-1 prod + Stage-4C per-HH) | `production_results.json`, `stage4c_singles_vdir_smoke.json` | freeze exact input rows + θ/spec hashes + provenance; package-native reproduction follows W4.0-IMPL |
| Couples `V_i^IS` | `_build_V_extractor_couples` + `run_stage1_w3.py` | IMPLEMENTED | yes | YES (Stage-1 prod; aggregates only) | `production_results.json` (couples negLL 174603.7298) | run to capture per-HH couples scalars |
| Singles `V_i^dir` | `welfare_vdir.py` + `run_stage4c*.py` | IMPLEMENTED (bounded smoke) | yes (S≈100 design-ready) | YES (smoke + calibration; **NOT** production) | `stage4c_singles_vdir_smoke[_n60].json`, `stage4c2_vdir_bias_calibration.json` | full production run at S≈100; **re-mint + freeze node/priced tables**; capture θ/spec hashes |
| Couples redraw-node construction | `welfare_vdir.py:232-250` | IMPLEMENTED, never invoked | yes (locations only) | NO | none | invoke + validate per-partner node sets |
| Couples `V_i^dir` | absent (no joint-record pricer) | DESIGN/BLOCKED | no | NO | none | BUILD the two-partner joint-record pricing path |
| Reference-set construction (`Abar`/`J`/`o`) | `welfare_stage1_w3.yaml:44-46` (`null`) | DESIGN | no | NO | none | BUILD reference sets + coverage gate |
| `W3` equivalent-income inversion | `welfare_core.w3_inversion` | IMPLEMENTED | yes | YES (singles+couples Stage-1) | `production_results.json` (Gate-2) | none (internal-validation only) |
| `W1`/`W2`/`W4`/`W5`/`W6` | absent / named-deferred | DESIGN (W1/W2 absent) | no | NO | none | BUILD |
| Inequality / Gini | `welfare_core.gini` | IMPLEMENTED | yes | YES (internal-validation; Gini=0 degenerate) | `production_results.json` (`gini_INTERNAL_VALIDATION_ONLY=0.0`) | NOT reportable until decomposition imposes an equalised channel |
| Shapley–Shorrocks | none (readiness interfaces only) | DESIGN/ABSENT | no | NO | `decomposition_readiness` block only | BUILD (exhaustiveness gate) |

**Did the `W3` path execute end-to-end?** YES — `run_stage1_w3.py` ran both `smoke` (25 HH/group)
and `production` (all HH) passes; `production_results.json` and `smoke_results.json` show Gate-0
through Gate-4 all `ok`/`pass:true` for all three groups, including couples at 901 alts.
**Distinction:** the W3 *machinery* (V_i^IS, inversion, ESS, Gate-0 parity) ran and validated; the
W3 *Omega/Gini* output is the **degenerate internal-validation artifact** (`Omega ≈ -2.91e-10`,
`Gini = 0` by construction), explicitly **not** a reportable welfare result. `gate1_draw_growth`
and `gate1_vdir_crosscheck` are **BLOCKED** (2×/4× datasets absent; V_i^dir redraw not wired into
Stage-1).

---

## 10. Explicit uncertainties and unresolved questions

1. **Couples `V_i^IS` "deferred" vs run.** Code comments say COUPLES_DEFERRED, but the implemented
   couples extractor + `production_results.json` show couples Gate-0..4 ran. Resolve by confirming
   the production run used the non-None couples extractor (artifact strongly implies yes). Per-HH
   couples scalars are not stored regardless.
2. **`V_i^dir` exact-replay oracle is not freezable today** — scratch dirs are empty; only scalars
   survive. A durable exact oracle needs a re-mint (§5.2).
3. **No θ/spec/version hashes** are stored in any welfare artifact (only seed 20260604). A
   self-describing anchor needs these captured by the fast lane.
4. **Singles `V_i^dir` smoke did not certify** at smoke scale (high-ESS `delta_common` 1.1–2.0
   nats > 0.5); the bias-calibration argues this is finite-S/integration noise (S→∞ intercept
   abs_max 0.395 ≤ 0.5) but `slopes_negative=false` — "finite-S error sufficient for design", not
   a proven pure-Jensen mechanism. Production must confirm at S≈100.
5. **Tier-1 placement** (app `welfare/general/` vs core): the inversion/inequality are
   country-general but consume the V grid; decide at W4.1A whether any belong beside the V hook in
   core.
6. **`source_idorighh` cluster restore** is not in the welfare scripts — confirm where it is
   applied (assembly/loader) so couples joint-unit integrity (G3) is preserved post-migration.
7. **Staged vs canonical provenance for welfare numbers:** which baseline a *reportable* welfare
   number must use is a supervisor decision (`production_swap_authorised: false` today).

---

## 11. Authorized next

**W4.0-DESIGN only:** specify the public per-alternative component API required
by welfare integration. The design must cover singles and couples, expose utility
and full corrected index values, support consumption overrides for equivalent-
income inversion, preserve light imports, and define NumPy/JAX parity plus
certified-FR regression gates. No implementation is authorized yet.

In parallel, **W4.1A** may port only pure/general welfare arithmetic that can be
validated with synthetic unit tests. Reproduction and freezing of the live
single-household V_i^IS anchor remains blocked until the W4.0 API is implemented
and gated.

---

## 12. Not authorized

- W4.0 implementation
- Live V_i^IS reproduction through package-native machinery
- EUROMOD pricing migration
- V_i^dir migration or production runs
- Couples V_i^dir construction
- Measure-family or Shapley-Shorrocks implementation
- No implementation commit; this planning memo may be committed after review

---

## 13. REQUIRED NEXT INPUT for the W4.0-DESIGN / W4.1A prompt

To author the next prompt, the next input must supply:

1. **Authorization scope:** confirm W4.0-DESIGN (specify the backend-neutral per-alternative
   component API) + W4.1A (PORT pure/general arithmetic, synthetic tests only) as the authorized
   pair, and that W4.0-IMPL and live `V_i^IS` reproduction are **separately** authorized later
   (under FR-negLL + NumPy/JAX-parity + Gate-0 gates).
2. **Tier-1 placement decision:** app `welfare/general/` (default) vs core for the inversion/Gini.
3. **Candidate anchor set:** confirm UID 200001593700 (`V_i^IS=11.496632024594227`) as the primary
   **candidate** anchor (frozen only at W4.3 with input rows + θ/spec hashes + provenance), which of
   the other five singles UIDs to lock as redundant candidates, and the tolerance (proposed 1e-9).
4. **Fast-lane coordination:** whether the conference singles production run will (a) run at S≈100
   and (b) **re-mint and freeze** node/priced tables + θ/spec hashes, so Wave-4.4 gets a durable
   exact-replay `V_i^dir` oracle (vs scalar-only).
5. **Provenance ruling:** confirm the staged welfare-pricing reference
   (`fr_p3a_bpool_engine_ready_staged_threeB1`) is the pricing oracle for migration gates, and that
   no promotion to canonical production is in scope.
6. **Test-runner conventions:** where W4.1A unit tests live (app `tests/`), and whether they may
   import the certified spec/θ for the `V_i^IS` anchor or must use a frozen fixture.

---

*End of welfare migration inventory & plan v1. Inspection-only; one document written; no code,
data, runs, or commits. PORT/BUILD tags and oracle claims are backed by the cited
`file:line` / artifact evidence above; absence of an oracle is stated explicitly where it applies.*
