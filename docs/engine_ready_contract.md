# Engine-ready contract (country-adapter / core boundary)

The core consumes data only as an **engine-ready bundle**. Any country adapter
(the app's DE adapter, or your own) must produce a bundle that satisfies this
contract; the core then loads it with `dclaborsupply.data` and never needs to
know the data's origin. This page documents the actual contract enforced by
`data/loader.py`.

## Bundle naming

`load_engine_ready_stem(stem, spec, hours_band_policy=...)` expects three files:

```text
<stem>__singles.parquet
<stem>__couples.parquet
<stem>__mnlmeta.json
```

`load_singles` / `load_couples` accept a DataFrame or parquet path directly
(with `metadata=` for the normalization scales).

## Structural requirements (enforced; violations raise)

- **Chosen-first, column-0 convention.** Within each choice group the chosen
  alternative is row 0. `is_chosen` must be binary `{0,1}`.
- **Exactly one chosen per group.** Groups without exactly one chosen are
  rejected.
- **Constant alternatives per group.** Every group must have the same number of
  alternatives.
- **Singles split by `dgn`.** `load_engine_ready_stem` splits the singles frame
  by `dgn` (1 = male, 0 = female); `load_singles` validates that `dgn` matches
  the requested `is_male`.
- **Positive, finite normalized quantities.** `c_norm`, and `l_norm` (singles)
  or `l_norm_male` / `l_norm_female` (couples), and `prior` must be finite and
  strictly positive — otherwise the loader raises (no silent coercion). A
  documented EPS log-stability floor is then applied to positive sub-EPS values
  of `c_norm`, leisure, and `prior` (a no-op for the typical values ≥ 1).
- **Usable household/group and cluster identity.** Choice groups are formed from
  the household identity (`idhh`, paired with `year_tag` for pooled multi-year
  bundles), and the cluster id must be constant within each choice group.

## Normalization metadata

`<stem>__mnlmeta.json` supplies the normalization scales read by the loader:
`c_scale` and a leisure scale for singles, and `c_scale` plus `l_male_scale`
and `l_female_scale` for couples. These must be finite and strictly positive.

## Spec-aware required columns

Beyond the structural columns above, **required columns are spec-aware**: every
variable activated by the specification (leisure shifters such as `age_norm` /
`n_children`, hours bands, wage, occupation, region, …) must resolve to a finite
engine attribute or column. A spec-active variable that resolves to neither a
loaded field nor an available column is rejected — never silently skipped. Any
present column consumed by the loader or bound to an active spec variable must
be finite.

- **Hours-band policy.** `hours_band_policy="assembled"` reads and validates the
  assembled `working` / `working_pt1` / `working_pt2` / `working_ft` /
  `working_lh` columns (the engine-ready file is the source of truth).
  `"legacy_certified"` re-derives the bands from `hours` using the historical
  cutoffs and reads `working_lh` — this exists only to reproduce certified
  baselines and is not a universal rule.
- **Wage and occupation columns** are required **only when the spec activates
  them**: `wage` is required whenever `spec.wage_spec != "fw"`; an occupation
  block needs `loc4` / `loc4_*`. Region dummies are derived when present —
  either all seven `reg_nuts1_2..8` columns or a complete `drgn1` fallback; a
  partial set raises.

## What the core does NOT universally require

To keep the boundary honest, the core does **not** assume:

- a particular **EUROMOD** implementation (the core is microsimulator-free; any
  adapter that produces a conforming bundle works);
- a universal **consumption aggregation** rule (consumption arrives already
  normalized as `c_norm`);
- a universal **floor of 1.0** (the only floor is the documented EPS
  log-stability floor, not a hard 1.0);
- any **DE-specific earnings** handling (DE specifics live in the app adapter,
  not the core contract).

## Mandatory adapter requirement (current)

Until core draw canonicalization is separately fixed and gated (see
[known_limitations.md](known_limitations.md)), **every adapter that uses the
current opportunity generator MUST apply an equivalent post-draw non-employment
canonicalization** and validate the complete non-employment state (wage,
`working`, `yemse`, and other employment-conditional fields) before pricing or
assembly. The DE adapter's `canonicalize_post_draws` is the proven reference.

## Claims discipline

When documenting a new configuration against this contract, keep claims scoped:

- **DE** proves **one** non-FR end-to-end configuration, not universal country
  support.
- **Fixed-wage dimension dropping** is recovery-proven for the **identified
  20-free** configuration only (see
  [validation/dimension_drop_hours_only_fw.md](validation/dimension_drop_hours_only_fw.md)).
- **`fw` and standard `vw`** are JAX-validated.
- **`loc_empirical` and `vw_occupation`** must not be used for JAX estimation
  until dedicated implementations are added and gated.
- The **certified FR regression anchor** is `238504.6360973987`; any change that
  must preserve certified behavior is checked against it.
