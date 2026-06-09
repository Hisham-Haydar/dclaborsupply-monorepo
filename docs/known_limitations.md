# Known limitations

Documentation-only record of currently-known gaps. No code is modified by
this note; it states what is and is not validated, plus the required
workarounds, so callers fail safely.

## JAX wage-specification support

The spec parser recognizes several wage specifications, but the JAX
estimation engine does not implement all of them. Status as of commit
`99a727c`:

- `fw` (fixed wage): implemented and recovery-validated. See
  [the validation note](validation/dimension_drop_hours_only_fw.md).
- `vw` (standard log-normal variable wage): validated — the certified FR and
  DE paths reproduce.
- `loc_empirical`: parser-recognized, but the JAX engine has no dedicated
  implementation matching NumPy's distinct branch.
- `vw_occupation`: parser-recognized, but the JAX engine has no dedicated
  implementation matching NumPy's distinct branch.

`loc_empirical` and `vw_occupation` exist as distinct branches in the NumPy
engine, but the JAX engine has no dedicated implementations matching those
branches. Commit `99a727c` states this explicitly: "JAX loc_empirical and
vw_occupation remain unproven and out of scope."

**Do not use these specifications for JAX estimation until dedicated
implementations are added and separately gated.** This pass does not modify
any code to enforce that.

## Draw canonicalization (post-draw non-employment state)

The unchanged core opportunity generator can leave non-employment rows
internally inconsistent, specifically:

- a draw-zero alternative with a positive realized wage at `hours == 0`, and
- simulated non-employment rows that retain stale `working` / `yemse` fields
  from the employment draw.

DE's app-layer `canonicalize_post_draws` (in
`packages/dclaborsupply_app/.../de/draws_prep.py`) is proven to repair this
(idempotent; covered by `de/tests/test_draws_prep.py`).

Moving canonicalization into the core generator is formally deferred to a
separately gated Wave-5 core-enhancement step that requires FR reproduction
before it may land. It is intentionally not attempted here.

**Mandatory workaround until resolved:** every country adapter that uses the
current generator must apply an equivalent post-draw canonicalization and
validate the complete non-employment state (wage, `working`, `yemse`, and any
other employment-conditional fields) before pricing or assembly. Skipping
this yields inconsistent non-employment rows and is a correctness bug, not a
cosmetic one.

## `RUMModel.fit()` does not create the correction-null RUM view

This matters for anyone (humans or LLMs) reasoning about what a "RUM fit"
actually computes.

- The only API that automatically constructs the correction-null RUM view is
  `compute_index(spec, data, theta, ruro=False)`, which builds an evaluation
  view (`build_rum_view`) that nulls every opportunity term and the
  importance-sampling correction. This is an **evaluation** path.
- **`RUMModel.fit()` does NOT use that view.** It builds the **same full JAX
  objective as `RUROModel.fit()`** (both call the same `_build_objective`) from
  the supplied spec/data. It does **not** automatically null opportunity terms
  or set `prior == 1`. The `RUM`/`RURO` distinction at the front-end only
  affects result metadata, not the fitted likelihood.

**Consequence:** `RUMModel.fit()` equals a mathematical RUM fit **only when the
inputs already satisfy** the null conditions: `wage_spec="fw"`, no
hours/market/occupation opportunity shifters, and input `prior == 1`. Otherwise
it is fitting the full RURO-style objective despite the `RUM` label. There is
currently no front-end flag that forces the correction-null view during fitting;
use `compute_index(..., ruro=False)` for a verified correction-null RUM
*evaluation*. The shipped CLI smoke
(`docs/examples/synthetic_cli_smoke.yaml`) deliberately does **not** meet these
conditions (it has a non-uniform `prior`), so it is a front-end smoke, not a
canonical RUM demonstration.
