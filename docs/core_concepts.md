# Core concepts and architecture

This page explains the model families, the likelihood index, the spec-driven
block structure, and the core/app split — the package as it exists today.

## RUM and RURO

The core supports two model families, both expressed through the same index and
the same `compute_index` entry point (`ruro=True|False`):

- **RUM** (Random Utility Model): fixed choice sets. The observed alternative is
  carried at column 0 of each choice group (the "chosen-first" convention), and
  the likelihood is a conditional logit over that fixed set.
- **RURO** (Random Utility / Random Opportunity): alternatives are *drawn* from
  an opportunity distribution, and the likelihood corrects for the sampling
  (proposal) density. This is the model used for the labour-supply opportunity
  draws.

Front-ends: `RUMModel` and `RUROModel` (both via `from_spec(spec)`), plus
`RUROModel.recover_synthetic(...)` for synthetic recovery gates.

## The likelihood index

For each alternative the linear index is:

```text
v = u + log_h + log_w + log_market - log_prior
```

- `u` — Box-Cox utility over consumption and leisure.
- `log_h` — hours-opportunity contribution (hours-band shifters).
- `log_w` — wage-density contribution (zero on the fixed-wage `fw` path).
- `log_market` — market-opportunity contribution. **Occupation contributions are
  currently folded into `log_market`**: occupation shifters are appended to the
  market-opportunity shifters during spec parsing, so they enter the index
  through the same `log_market` term rather than a separate occupation term.
- `log_prior` — the log proposal density of the drawn opportunity. Subtracting
  it is the importance-sampling correction that makes RURO consistent.

### RUM as the correction-null view — evaluation vs. fitting

"RUM" as a mathematical object is the **correction-null** view: every
opportunity term and the importance-sampling correction are zero, so `v = u`
over a fixed choice set. Two things in the code must be kept distinct:

- **`compute_index(spec, data, theta, ruro=False)` is the verified correction-
  null RUM *evaluation*.** It calls `build_rum_view`, which derives an
  evaluation view with `wage_spec="fw"` (so `log_w = 0`), no hours shifters (so
  `log_h = 0`), no market/occupation shifters (so `log_market = 0`), and
  `prior = 1` (so `log_prior = 0`) — i.e. `v = u`. It runs on the NumPy engine.
- **`RUMModel.fit()` does *not* create that view.** It builds the **same full
  JAX objective as `RUROModel.fit()`** from the supplied spec/data (both call
  the same `_build_objective`). It does **not** automatically null opportunity
  terms or set `prior = 1`. The `RUM`/`RURO` label only sets result metadata; it
  does not change the likelihood that is fit.

#### Getting a mathematical RUM *fit* from the current front-end

Because `fit()` does not null anything, a fit equals the correction-null RUM
only when the **inputs already satisfy** the null conditions:

- `wage_spec="fw"` (no wage density term);
- no `hours_opportunity` / `market_opportunity` / `occupation_opportunity`
  shifters;
- input `prior == 1` for every row.

Under those conditions the full objective reduces to `v = u`, matching
`compute_index(..., ruro=False)`. Otherwise `fit()` is fitting the full
RURO-style objective regardless of the `RUM` label. This is recorded as a
limitation in [known_limitations.md](known_limitations.md).

## Spec-driven optional blocks

The specification (`EstimationSpec.from_yaml`) drives which terms exist. Each
opportunity block is **optional**: if a block is absent from the YAML, its term
is genuinely absent (not a pinned-to-zero coefficient). Blocks:

- `utility` (required) — Box-Cox consumption and leisure, with leisure
  shifters (age, children, …).
- `hours_opportunity` — hours-band shifters feeding `log_h`.
- `wage_opportunity` — wage density feeding `log_w` (`wage_spec`: `fw`, `vw`,
  and the parser-recognized `vw_occupation` / `loc_empirical`).
- `market_opportunity` and `occupation_opportunity` — feed `log_market`
  (occupation folded in, as above).

The active variables of the chosen blocks determine which columns the
engine-ready loader requires (see the contract page).

## Core / app split

- **Core (`dclaborsupply`)** owns the science of estimation: spec parsing,
  likelihood (NumPy reference + JAX), solvers, standard errors, recovery gates,
  the engine-ready loader, and diagnostics. It imports without Java/JAX/GAMSPy.
- **App (`dclaborsupply_app`)** owns microsimulation and country specifics:
  EUROMOD integration, country data preparation (e.g. the DE adapter), welfare,
  and reporting. It builds the engine-ready bundles the core consumes.

### The engine-ready boundary

Data crosses from the app (or any country adapter) into the core as an
**engine-ready bundle** — harmonised parquet/DataFrames plus normalization
metadata. The core does not know about EUROMOD or any specific country; it only
requires that the bundle satisfies the documented contract. See
[engine_ready_contract.md](engine_ready_contract.md).

## Proven capability and limitations

Claims are deliberately scoped. See the linked notes for the evidence.

- **Certified FR baseline** reproduced exactly (negLL `238504.6360973987`).
- **One non-FR end-to-end configuration** (DE 2017 minimal) runs on the
  unchanged core — proof for one configuration, not universal country support.
- **Fixed-wage dimension dropping** is recovery-proven for one identified
  20-free configuration only — see
  [validation/dimension_drop_hours_only_fw.md](validation/dimension_drop_hours_only_fw.md).
- **JAX wage paths:** `fw` and standard `vw` are validated; `loc_empirical` and
  `vw_occupation` must not be used for JAX estimation until implemented and
  gated — see [known_limitations.md](known_limitations.md).

These are current facts about the implementation. Aspirational behavior (broader
country/spec agnosticism, additional wage paths) is not described here as if it
were implemented.
