# Dimension-drop validation: identified hours-only fixed-wage RURO

**Status:** validated (scratch evidence, uncommitted). **Date:** 2026-06-09.
**Capability provenance:** commit `99a727c` — *feat(likelihood): support
fixed-wage JAX models*.

## What this proves

Commit `99a727c` added truthful JAX fixed-wage support: for `wage_spec="fw"`
the wage opportunity is genuinely optional — the JAX engine accesses no wage
arrays, no wage variance (sigma), and no wage parameters on the fixed-wage
path (rather than carrying a pinned-to-zero wage block).

On top of that capability, the package was shown to express, compute, and
recover an *identified* hours-only fixed-wage RURO configuration in which the
wage, occupation, and market dimensions are **absent rather than pinned**.
This is the narrow, concrete claim validated here — nothing broader.

## Identified recovery configuration

- 20 free parameters + 4 pinned leisure-curvature parameters
  (`theta_l_sm`, `theta_l_sf`, `theta_l_m`, `theta_l_f` pinned at -0.5).
- `wage_spec="fw"`; wage / occupation / market shifter blocks absent from the
  spec (not present as zeroed coefficients).
- Singles hours grid `[0, 10, 20, 25, 30, 35, 40, 50]`; couples a balanced
  full-factorial grid over `{0, 20, 30, 40, 50}` x `{0, 20, 30, 40, 50}`
  (male hours-state orthogonal to female hours-state).

The 20-free configuration was selected by a prior design audit (Step 1B):
pinning the four leisure-curvature parameters and enriching the hours grids
removes the leisure `beta_l0`/`theta_l` ridge and materially de-correlates the
FT/LH hours-band coefficients, yielding a full-rank expected Fisher
information.

## Recovery result: all seven pre-registered gates passed

- Free / pinned parameter count: 20 free + 4 pinned (dropped dims absent).
- Synthetic actual-choice per group: exactly one.
- NumPy actual-choice DGP vs JAX recovery objective at the truth: agree
  within 2.0e-11.
- Parameter-binding audit: zero silent drops.
- Optimizer: success; no bound-active free parameters; fitted negLL not worse
  than negLL at the truth.
- Fitted exact-JAX Hessian: PD, rank 20/20, minimum eigenvalue 3.155,
  condition number 585.2, all SEs finite and positive.
- Standardized recovery: max |theta_hat - theta_star| / SE = 1.587
  (all at or below 4.0).

**Regression safety:** the JAX fixed-wage change preserved the certified FR
variable-wage (`vw`) negative log-likelihood 238504.6360973987, a difference
of 3.99e-7 from the rounded certified target — the certified path is
unchanged.

## Scope boundaries (explicit)

This note proves a deliberately narrow thing. It does **not** prove more.

- **Identified config only.** Recovery is proven for the identified 20-free
  configuration, not the original 24-free configuration. The original 24-free
  configuration failed its pre-registered recovery gate because of
  design-induced weak local identification. It was not proven unrecoverable
  or structurally unidentified.
- **Drop-only flexibility.** It proves the engine handles dropping wage /
  occupation / market dimensions, not substituting them, swapping them, or
  making arbitrary dimension changes.
- **JAX fixed-wage path only.** It proves the JAX `fw` path. It does not
  validate every wage specification (see `../known_limitations.md` —
  `loc_empirical` and `vw_occupation` remain unsupported/unvalidated for JAX).
- **Not universal dimension-agnosticism.** A single identified drop-only
  configuration recovering does not establish that the package is agnostic to
  arbitrary dimension sets.

## Scratch evidence (referenced, not committed)

Under `…/EUROMOD-STORAGE/scratch/staging/hours_only_fw_recovery/`:

- `recover_1C.py`, `recover_1C_report.json` — the pre-registered 20-free
  recovery run (the seven-gate result above).
- `design_audit_1B.py` — the design / expected-Fisher audit that selected the
  20-free configuration (FT/LH de-correlation; full Fisher rank).
- `audit.py` — the recovery-objective consistency audit (NumPy vs JAX).

These scratch artifacts are intentionally not committed; they are recorded
here for traceability only.
