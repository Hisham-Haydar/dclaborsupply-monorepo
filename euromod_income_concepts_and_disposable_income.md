# EUROMOD Income Concepts and Disposable-Income Semantics — Reference

**Status:** Foundational reference (conceptual map + operational contract).
**Version/date boundary:** This document reflects the sources, EUROMOD versions, country
systems, datasets, and project files examined **as of 2026-06-06**. It does **not** claim
validity across every future EUROMOD release. See [§9 Revalidation triggers](#9-revalidation-triggers).

---

## 1. Purpose, scope, evidence categories, and version/date boundary

### 1.1 Purpose

This is a durable, authoritative reference explaining **how EUROMOD constructs and aggregates
income**, from raw input components (`yem`, `yse`, `yiv`, …) through standardized income lists
(`il*`, `ils*`) to **standard disposable income** (`ils_dispy`). The operational facts the
FR/DE labour-supply pipeline depends on are situated **within** that conceptual map. The intent
is that future users and projects can understand these concepts without repeating the underlying
investigation.

### 1.2 Scope

Covered: the income-concept and aggregation layer only —
(1) raw components → income lists → disposable income;
(2) the layered income-list construction, separating universal semantics from configuration;
(3) `ils_dispy` and the project-derived `ils_dispy_real`;
(4) the aggregation/replication convention (person-row → household);
(5) country parameterization (FR 35-hour split, DE `yemse` identity, system/dataset pairing);
(6) non-positive disposable income and the downstream strictly-positive consumption floor.

Out of scope: the full tax-benefit spine, individual policy implementations, the labour-supply
utility/estimation specification, welfare/decomposition stages, and draw generation. These are
referenced only where they consume disposable income.

### 1.3 Evidence categories

Every substantive claim below is tagged with one of:

- **[AUTH]** Authoritative documentation — official EUROMOD documentation/Help, or the official
  EU-SILC methodological guidelines.
- **[CFG]** Country or dataset configuration — established by a specific country system, dataset,
  or EUROMOD configuration (e.g. FR `IlsDef_fr`, FR/DE datasets).
- **[CONTRACT]** Project pipeline contract — established by the FR/DE project code in this repo.
- **[EMP]** Empirical observation — observed in the supplied priced data.
- **[INF]** Inference — derived from evidence but not explicitly stated by an authoritative source.

Where sources are silent, conflict, or remain ambiguous, this is recorded rather than resolved by
force. Citations: documents by title + section/heading (+ page where available); CHM Help by help
file + topic file + heading; code by repo-relative `path:line`; data by file + grouping unit +
denominator + method.

### 1.4 Sources examined (full citations in [§10](#10-sources))

- **[AUTH-SILC]** *EU-SILC: Methodological guidelines with description of variables — 2021
  Operation (v8, accessible)* (PDF, 621 pp.). NOTE: this governs the **EU-SILC input survey
  data**, not EUROMOD's `ils_*` lists.
- **[AUTH-EM]** *EUROMOD Help* (`EUROMODHelp.chm`), topics: *EUROMOD terminology*, *DefIL*,
  *DefOutput*, *IlArithOp*, *ILVarOp* (EUROMOD release `J2.0+`, install under `C:\Program Files\EUROMOD\`).
- **[CFG-FR]** Project-extracted FR configuration references in `Data/documentation/` (input/output
  reference, input-variable index, output-variable index, standardized-income-concepts CSV, the
  `FR_2015_all_tables_compact.md` policy dump).
- **[CONTRACT]** FR/DE pipeline code under `scripts/bpool/`, `scripts/enhanced/`, and the DE
  pricing smoke under `EUROMOD-STORAGE/scratch/staging/de_2017_pricing_smoke/`.
- **[EMP]** FR priced-long parquets `EUROMOD-STORAGE/new_data/fr_p3a_bpool_priced__{2015,2016,2017}__{singles,couples}.parquet`.

---

## 2. EUROMOD income-concept hierarchy: components, income lists, and aggregates

### 2.1 The three layers

EUROMOD income is built in three conceptual layers:

1. **Raw variables** — input data variables and model-simulated variables.
2. **Income lists** (`il_*`, `ils_*`) — named aggregates of variables and/or other income lists.
3. **Standard disposable income** (`ils_dispy`) — the top standardized income concept.

**[AUTH-EM]** An **EUROMOD incomelist** is *"the aggregate of several EUROMOD variables and
possibly other incomelists. These components (in rare cases fractions or multiples of them) are
either added or subtracted to build the aggregate."* The term *"income list"* indicates the most
common applications are income concepts (disposable income, taxable income, …)
(*EUROMOD Help → EUROMOD terminology*, heading "EUROMOD incomelist"; `EM_BC_Terminology.htm`).

**[AUTH-EM]** Income-list names *"always start with `il_` or `ils_`, where `ils_` denotes system
or standard incomelists. These are incomelists, which must be defined for each country."*
(*EUROMOD Help → DefIL*; `EM_FC_DefIL.htm`). Thus the **`ils_` prefix marks the standardized,
cross-country concept**, while its **exact component list is country configuration** (it "must be
defined for each country").

**[AUTH-EM]** EUROMOD knows four variable types: (1) input-data variables, (2) model-simulated
variables (postfix `_s`), (3) intermediate variables, (4) special-purpose variables
(*EM_BC_Terminology.htm*, "EUROMOD variables"). The `_s` postfix on a component (e.g. `tin_s`,
`bsawk_s`) therefore means "simulated by the model," as opposed to taken from input data.

### 2.2 Raw input income components (illustrative; FR)

**[CFG-FR]** Raw monetary input components in the FR DRD include (label / description):

| Variable | Meaning |
|---|---|
| `yem` | INCOME: Employment — gross employee cash or near-cash income |
| `yse` | INCOME: Self-Employment income |
| `yiy` | INCOME: Investment (interest, dividends, profit from capital) |
| `ypp` | INCOME: Private Pension |
| `ypr` | INCOME: Property (rent) |
| `ypt` | INCOME: Private Transfers received |

(`Data/documentation/euromod_fr_2015_2017_input_output_reference.md`, "Selected input variables";
fuller index in `euromod_fr_2015_2017_input_variables.csv`.)

**[INF]** `yiv` named in the task brief corresponds to EUROMOD's investment-income family; in the
FR DRD the realized variable is `yiy` (interest/dividends/capital). The two refer to the same
concept layer (capital/investment income); the precise mnemonic is country/dataset-dependent. The
priced files additionally carry a draw-side wage variable `yivwg` (the drawn hourly/period wage),
which is **not** an income-list component but a pricing input (see [§5.4](#54-de-yemse--yem--yse-identity)).

### 2.3 The standardized income-list families (FR realization)

**[CFG-FR]** The FR systems realize the standard `ils_*` chain as follows (signs are EUROMOD
`DefIL` signs `+`/`−`/`n/a`; from `IlsDef_fr`, baseline `FR_2015`):

```text
ils_earns   = + yse + yem00 + yemxp            ( + yemmc_s [n/a] )
ils_origy   = + ils_earns + ypp + yiy + ypr + yot + ypt − xmp
ils_ben     = + ils_pen + ils_benmt + ils_bennt
ils_tax     = + tin_s + tpr + tscxc_s + tscdf_s + tsckt_s + tinto_s + twl + tmu
ils_sicdy   = (all social-insurance contributions paid by (self-)employed and others)
ils_dispy   = + ils_origy + ils_ben − ils_sicdy − ils_tax
```

(`Data/documentation/euromod_fr_2015_2017_standard_income_concepts.csv`;
`euromod_fr_2015_2017_input_output_reference.md`, "Standardized income concepts".)

**[EMP]** The FR priced-long output carries the full `ils_*` family on every person row, including
`ils_earns, ils_origy, ils_pen, ils_bennt, ils_benmt, ils_ben, ils_tax, ils_sicdy, ils_dispy`, plus
`ils_b1_*`/`ils_b2_*` benefit sub-lists, `ils_base_*` tax bases, and `ils_udb_*` UDB-mapped
components (parquet schema of `fr_p3a_bpool_priced__2016__singles.parquet`, 569 columns).

### 2.4 Universal vs. configured in the hierarchy

- **[AUTH-EM]** Universal: the **concept** "disposable income = original income + benefits − taxes
  − social insurance contributions," its name `ils_dispy`, and the `il_`/`ils_` naming/aggregation
  mechanics (add/subtract components; lists may nest).
- **[CFG]** Configured per country/system: **which** variables and sub-lists populate each `ils_*`
  list, the signs, and year-specific component switches. Example **[CFG-FR]**: the FR 2015 system
  includes `tinrf_s` (PPE working-tax-credit refund) and treats `bsawk_s` (PA activity allowance)
  as `n/a`, while FR 2016/2017 reverse this, inside `ils_benmt`/`ils_bensim`/`ils_b1_bsa`
  (`euromod_fr_2015_2017_input_output_reference.md`, "Year-specific differences"). The disposable-
  income **identity** is unchanged across the three years; only the means-tested benefit
  composition switches.

---

## 3. Disposable income: `ils_dispy`, `ils_dispy_real`, unit, and nominal-vs-real semantics

### 3.1 `ils_dispy` — definition and status

**[AUTH-EM]** `ils_dispy` is **EUROMOD standard disposable income**, an official standard concept:
*"In general the following components make up disposable income in EUROMOD (for each country and
system): original income … plus benefits … minus direct taxes … minus social insurance
contributions. As this income concept is standardised as far as possible over the countries
implemented in the model it is referred to as standard disposable income (and defined in the
incomelist `ils_dispy`)."* (*EM_BC_Terminology.htm*, "EUROMOD standard disposable income".)

**[AUTH-EM]** The canonical `DefIL` definition is shown in the Help itself:
`ils_dispy = ils_origy (+) ils_ben (+) ils_tax (−) ils_sicee (−) ils_sicse (−)`
(*EM_FC_DefIL.htm*, Example 1, "disposable income"). The FR realization
([§2.3](#23-the-standardized-income-list-families-fr-realization)) collapses the two SIC lists into
a single `ils_sicdy` but is the same concept. **[INF]** the partition of social-insurance
contributions into `ils_sicee`/`ils_sicse` (Help example) vs. a single `ils_sicdy` (FR) is a
country-configuration choice, not a semantic difference.

### 3.2 Unit and row-level meaning of `ils_dispy`

- **[AUTH-EM]** EUROMOD standard output writes *"one row for each person listed in the input
  data,"* containing model-calculated variables and income lists, *"most essentially EUROMOD
  standard disposable income"* (*EM_BC_Terminology.htm*, "Standard output"). So in person-level
  output **each row's `ils_dispy` is that person's value of the income list** — see
  [§4](#4-aggregation-and-replication-convention) for the additive interpretation.
- **[AUTH-EM]** Monetary period: EUROMOD outputs monetary amounts at the model's working
  periodicity; in this project the values are **monthly euros** (the FR/DE pipelines treat
  `ils_dispy` as a monthly amount — e.g. `WPM = 52/12` monthly-wage conversion in the DE pricing
  rule, `de_2017_pricing_smoke/smoke.py:25`, `:131-133`). **[INF]** monthly periodicity is a
  model/dataset configuration, not a universal EUROMOD constant; confirm per system before reuse.
- **[AUTH-SILC]** The disposable-income concept itself: *"the total income of a household that is
  available for spending or saving"* (EU-SILC guidelines, §5 Income data, "Equivalised disposable
  income", p. 39).

### 3.3 `ils_dispy_real` — project-derived CPI-deflated disposable income

**[CONTRACT]** `ils_dispy_real` is **not** a standard EUROMOD concept. It is a **project output**:
CPI-deflated disposable income, computed as

```text
ils_dispy_real = ils_dispy * phi_{data_year}      # base year 2016
phi_2015 = 1.0031 ,  phi_2016 = 1.0000 ,  phi_2017 = 0.9886
```

(`scripts/bpool/run_bpool_euromod.py:13-15` (docstring), `:68` (`_CPI`), `:469` (computation);
mirrored in `scripts/bpool/run_bpool_euromod_chunk.py:47,213` and the canary check
`scripts/bpool/assemble_bpool_priced.py:26,77`).

- **Unit / row meaning:** same unit and row-level meaning as `ils_dispy` (monthly euros, per-person
  in the priced files), only re-expressed in **2016-constant euros**. It is a scalar rescaling of
  the nominal value by a year-constant factor `phi`. **[CONTRACT]**
- **Nominal vs. real:** `ils_dispy` is **nominal** (the EUROMOD output in the prices of the data
  year); `ils_dispy_real` is the **CPI-deflated (2016-real)** counterpart. For 2016 the two are
  identical (`phi = 1.0000`). **[CONTRACT]**
- **Provenance guardrail:** wage deflation for the estimator's wage regressors is applied
  separately and *after* pricing inputs are no longer used, *"preventing double deflation of
  `ils_dispy_real`"* (`scripts/bpool/build_bpool_estimation_ready.py:298-317`, `deflate_wages_for_estimation`).
  So `ils_dispy_real` is deflated exactly once, at the pricing-runner stage. **[CONTRACT]**

### 3.4 Ambiguities recorded

- **[INF / ambiguity]** The EUROMOD Help does not fix a universal periodicity for `ils_dispy`;
  "monthly" here is established by the FR/DE pipeline/datasets, not by EUROMOD documentation.
- **[INF / ambiguity]** "Real" in `ils_dispy_real` means **CPI-deflated to 2016**, a project
  choice of base year and index. No authoritative EUROMOD source defines a "real disposable income"
  concept; do not promote `ils_dispy_real` to a EUROMOD standard.

---

## 4. Aggregation and replication convention

**Question:** On person rows, is `ils_dispy` an additive person-level contribution, a
household-level value replicated onto members, or configuration-dependent? And is summing person
rows the documented way to get household disposable income?

### 4.1 Authoritative documentation (resolves the convention)

**[AUTH-EM]** EUROMOD output level is controlled by the `DefOutput` parameter `TAX_UNIT`:

- With `TAX_UNIT` set to **individual level**, output is one row per person and the income list is
  *"the value of the incomelist (i.e. the sum of the comprised variables)"* **for that person**.
- With `TAX_UNIT` set to **household level**, *"output is aggregate on household level (i.e. one row
  for each household), where rules of aggregation are those defined in section … the assessment
  unit. That means employment income (`yem`) and disposable income (`ils_dispy`) are household …
  disposable income."*

(*EUROMOD Help → DefOutput*, Examples 1–2; `EM_FC_DefOutput.htm`.)

**[AUTH-EM]** The default aggregation rule over an assessment unit for monetary variables is the
**sum over its members** (the same topic notes that a non-monetary demographic like `dag` instead
takes the *head's* value, confirming that monetary lists like `ils_dispy` are summed, not
replicated). **[AUTH-SILC]** Independently, the EU-SILC guidelines describe the standard household
construction as *"summing up the individual incomes at the household level"* (EU-SILC guidelines,
§5 Income data, "Income types → Level", p. 39).

**Conclusion (authoritative): `ils_dispy` is computed at the chosen output assessment unit. In
person-level output it is the individual's additive contribution; household disposable income is
obtained by summing the person rows over the household — which is exactly the operation EUROMOD
itself performs when `TAX_UNIT` is set to household level.** This is therefore **not** "household
value replicated onto members"; it is **additive person-level contributions** whose **sum** is the
household value. **[AUTH-EM + INF]** (Inference element: applying the documented household
aggregation rule to the specific FR/DE person-level files.)

### 4.2 FR configuration that makes person rows additive

**[CFG-FR]** The FR standard output used by this project is configured at **individual level**:
policy `output_std_fr` is **on** with `TAX_UNIT = tu_individual_fr`, while the household-level
policy `output_std_hh_fr` (which would write one `ils*` row per household with `tu_household_fr`) is
**off** (`euromod_fr_2015_2017_input_output_reference.md`, "Standard output: common configuration";
`FR_2015_all_tables_compact.md`, `output_std_hh_fr / DefOutput`). Consequently the FR priced-long
files contain **individual-level** `ils_dispy`, and household disposable income must be obtained by
summing members.

### 4.3 FR empirical confirmation

**Method.** Read-only analysis of the raw priced-long files
`EUROMOD-STORAGE/new_data/fr_p3a_bpool_priced__{year}__couples.parquet`. Grouping unit = the
**household-alternative key** the pipeline itself documents and uses for couples:
`(stacked_hh_uid, draw_joint, is_chosen_joint)` within a year
(`scripts/bpool/build_bpool_estimation_ready.py:233-245`, `couples_dispy_lookup`). For each group,
member rows = persons sharing that key; the household-alternative value is
`sum(member ils_dispy_real)`. Denominator = number of groups.

**[EMP] Couples — member-value composition by household-alternative group:**

| Year | Households (`stacked_hh_uid`) | Alts/HH | Groups (= alternative units) | Member-count range | Groups identical | Groups distinct | Groups w/ missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 2,566 | 901 | 2,311,966 | 2–9 | 1,083 (0.047%) | 2,310,883 (99.953%) | 0 |
| 2016 | 2,577 | 901 | 2,321,877 | 2–8 | 987 (0.043%) | 2,320,890 (99.957%) | 0 |
| 2017 | 2,295 | 901 | 2,067,795 | 2–7 | 572 (0.028%) | 2,067,223 (99.972%) | 0 |
| **Total** | **7,438** | 901 | **6,701,638** | 2–9 | **2,642 (0.039%)** | **6,698,996 (99.961%)** | **0** |

Member-count distribution (rows per household-alternative group, all years pooled):
`{2: 2,099,330; 3: 1,587,562; 4: 2,163,301; 5: 702,780; 6: 119,833; 7: 25,228; 8: 1,802; 9: 1,802}`.

**Interpretation.** The "couples" households contain **2–9 persons** (the couple plus children/other
members), not just the two deciders, and the disposable-income column varies **across** members in
**99.96 %** of household-alternative groups. Distinct member values are inconsistent with a single
household figure replicated onto every member; they are consistent with **additive person-level
contributions** (each person carries their own original income, taxes, SIC, and whichever
household-assessed benefits EUROMOD assigns to them). Per the task's caution, identical values were
**not** treated as proof of replication — and here the overwhelming finding is the opposite
(distinct), so the additive reading is doubly supported. The household disposable income is the
**sum** over the member rows, which is precisely the pipeline's `groupby(...).sum()` operation. **[EMP]**

**[INF]** The rare identical-value groups (0.039 %) are most plausibly non-employment / structurally
symmetric alternatives where members happen to coincide; they do not indicate replication, because
the same key in other alternatives yields distinct values.

### 4.4 The pipeline's aggregation operation (project contract)

**[CONTRACT]** The FR build computes couples household disposable income exactly as the documented
sum:

```python
# scripts/bpool/build_bpool_estimation_ready.py:241-243
joint = pr.groupby(
    ["stacked_hh_uid", "draw_joint", "is_chosen_joint", "data_year"], as_index=False
)["ils_dispy_real"].sum()
```

For **singles**, the alternative value is the single decider's own row
(`ruro_decider == 1`), keyed `(stacked_hh_uid, draw, data_year)`
(`build_bpool_estimation_ready.py:221-230`, `singles_dispy_lookup`). The legacy enhanced pipeline
uses the same convention: singles consumption = `ils_dispy` (+ any `other_members_income`); couples
consumption = `ils_dispy.groupby([idhh, draw]).transform("sum")`
(`scripts/enhanced/enh_RURO_prep_mnl_basic.py:710-729`).

### 4.5 DE consistency evidence and its limitation

**[CONTRACT/EMP]** The DE 2017 pricing smoke computes household disposable income by the **same
sum** convention:

```python
# EUROMOD-STORAGE/scratch/staging/de_2017_pricing_smoke/smoke.py:24, 252
COUNTRY, SYSTEM, DATASET, DISPY = "DE", "DE_2016", "DE_2017_a2", "ils_dispy"
hh_disp = sim_out.groupby("idhh")[DISPY].sum()...
```

The smoke `REPORT.md` reports both person-level `ils_dispy` (min −359.33 / median 2358.89 /
max 16508.88; 12 negative) **and** household-summed disposable income (min −404.08 / median
6519.55 / max 31525.46; 3 negative), confirming person rows summed to household totals
(`de_2017_pricing_smoke/REPORT.md`, gate (e)).

**Limitation.** The smoke **persists only the household aggregate**, not the person-level output.
`alt_results.csv` is the merge of the alternative metadata with `hh_disp` and contains a single
`hh_dispy` column per alternative plus `log_prior`/`prior` — there is **no per-person `ils_dispy`
column** (`smoke.py:252-254`; verified by inspecting `alt_results.csv` header:
`orig_hh, alt_key, kind, new_idhh, log_q_*, log_prior, prior, hh_dispy`). The intermediate
`sim_out` (person-level) was not written to disk. **Therefore DE person-row semantics cannot be
independently re-verified from the persisted artifacts without another EUROMOD run, which this task
does not perform.** The DE evidence is consistent with, but does not independently re-prove, the
additive convention established authoritatively in [§4.1](#41-authoritative-documentation-resolves-the-convention)
and empirically for FR in [§4.3](#43-fr-empirical-confirmation). **[CONTRACT/EMP, with stated limitation]**

---

## 5. Country parameterization

### 5.1 Universal vs. configured (summary)

- **[AUTH-EM] Universal:** the `il_`/`ils_` mechanism; `ils_dispy` as standard disposable income;
  the add/subtract aggregation and assessment-unit sum rule; the `_s` simulated-variable postfix.
- **[CFG] Country-system rules:** the exact components/signs of each `ils_*` list, year switches,
  and which benefits are individual vs. household-assessed.
- **[CFG] Dataset conventions:** variable availability and `SetDefault` values per dataset.
- **[CONTRACT] Project transformations:** `ils_dispy_real`, the consumption floor, normalization,
  and the system/dataset pairing chosen for pricing.

### 5.2 FR 35-hour split: `yem00` / `yemxp`

**[CFG-FR]** In the FR system, employment income enters `ils_earns` **split into two components**:

```text
ils_earns = + yse + yem00 + yemxp        # ( + yemmc_s [n/a] )
```

where, per the FR input index, `yem00` = *"INCOME: Employment: Main — gross employee cash or
near-cash income"* and `yemxp` = *"INCOME: Employment: Extra pay — wages received for overtime pay"*
(`euromod_fr_2015_2017_input_variables.csv` rows for `yem00`, `yemxp`;
`euromod_fr_2015_2017_standard_income_concepts.csv`, `ils_earns`).

**[CFG-FR]** The FR `SetDefault_fr` policy sets the defaults `yem00 = yem` and `yemxp = 0` (i.e. in
the absence of an explicit overtime split, all employment income is "regular hours" and overtime is
zero) (`FR_2015_all_tables_compact.md`, `SetDefault_fr / SetDefault`, lines 2957 and 2982-2983).
The FR model also carries a separate overtime-treatment formula (e.g. a `0.985…*yem00` factor and a
"Non-exempted overtime pay (from 2019)" rule) in its policy spine
(`FR_2015_all_tables_compact.md`, lines 1384, 1389).

**Characterization.** The `yem00`/`yemxp` split is a **country-system construct** ([CFG-FR]): it is
how the FR system represents the French institutional distinction between regular-hours pay and
overtime pay (France's statutory 35-hour workweek). Calling the "35 hours" itself an institutional
constant is **[INF]** — it reflects French labour law context, not a value asserted by the EUROMOD
Help; what the evidence directly establishes is the **variable split and its FR `DefIL`/`SetDefault`
configuration**. It is **not** universal EUROMOD semantics: other countries (e.g. DE,
[§5.4](#54-de-yemse--yem--yse-identity)) carry a single employment-income variable with no
`yem00`/`yemxp` decomposition.

### 5.3 FR system/dataset pairing convention (pricing)

**[CONTRACT]** The FR pricing runner pairs each **data year** with a **policy system one year
earlier** and the matching dataset:

| `data_year` (long file) | EUROMOD system | Dataset |
|---|---|---|
| 2015 | `FR_2014` | `FR_2015_a2` |
| 2016 | `FR_2015` | `FR_2016_a3` |
| 2017 | `FR_2016` | `FR_2017_a2` |

with comment *"System pairing (opportunity_year = data_year − 1)"*
(`scripts/bpool/run_bpool_euromod.py:8-11`, `:62-67`). **[INF]** This `system = data_year − 1`
lag is a deliberate project modelling choice (aligning the counterfactual policy year to the
opportunity year); it is **not** an EUROMOD-imposed rule.

**Consequence worth noting [INF/ambiguity]:** because of the lag, the `ils_*` formulas that
actually applied when producing the priced files are those of the **`FR_2014/FR_2015/FR_2016`**
systems, whereas the project's standardized-income-concepts reference
(`euromod_fr_2015_2017_standard_income_concepts.csv`) is keyed to **`FR_2015/FR_2016/FR_2017`**.
The disposable-income identity is stable across these years, but the exact means-tested-benefit
composition (the PPE↔PA switch, [§2.4](#24-universal-vs-configured-in-the-hierarchy)) is
year-sensitive. When reconciling a specific priced value to a component formula, use the **system
that actually priced it** (`data_year − 1`), not the same-year `IlsDef`.

### 5.4 DE `yemse = yem + yse` identity

**[CONTRACT/CFG-DE]** For DE, the validated pricing rule uses a **single** employment-income
variable `yem` and enforces the identity **`yemse = yem + yse`** (combined employment + self-
employment), with `yem = wage * hours * 52/12` (monthly) on decider rows:

```python
# de_2017_pricing_smoke/smoke.py:131-133
hh.loc[m, "yem"]   = w * h * WPM
hh.loc[m, "yemse"] = w * h * WPM + yse
```

The smoke verifies the identity holds to machine precision on both input and output
(`max |yemse − (yem + yse)| = 0.00e+00`, gates (g); `smoke.py:219, 281`; `REPORT.md` gate (g)).
There is **no** `yem00`/`yemxp` 35-hour split for DE. **[CONTRACT/CFG-DE]** The DE system/dataset
pairing mirrors FR's lag: system `DE_2016` ↔ dataset `DE_2017_a2` (`smoke.py:24`).

**[INF]** `yemse` is a DE-system aggregate variable; the additive identity `yemse = yem + yse` is a
DE configuration fact, established here by the project's validated rule and by EUROMOD producing
zero income-identity warnings on it (`REPORT.md`, gate (d): "output 320×436, total warnings/errors
= 0").

### 5.5 System/dataset pairing — evidentiary basis

The pairing tables in [§5.3](#53-fr-systemdataset-pairing-convention-pricing) and
[§5.4](#54-de-yemse--yem--yse-identity) are **[CONTRACT]** facts (project code), not EUROMOD
documentation. EUROMOD permits running any system against any compatible dataset; the **specific
lag pairing** is the project's modelling decision and must be re-checked if the estimation design
or the set of priced years changes.

---

## 6. Non-positive disposable income and positive-consumption requirements

### 6.1 Why disposable income can be zero or negative (conceptual)

**[AUTH-EM/INF]** `ils_dispy = original income + benefits − taxes − social insurance
contributions`. For a person or household with little or no market income but non-zero taxes or
contributions (e.g. on capital/property income, or contributions not fully offset by transfers),
the subtractions can exceed the additions, producing **non-positive disposable income**. This is a
legitimate model output, not an error. **[EMP]** The priced data confirm small but non-zero shares
of negative disposable income, concentrated in **non-employment** decider states:

- Priced-file canary (`fr_p3a_bpool_priced__meta.json`): among **non-working decider rows**, e.g.
  2016 singles 190 negatives (median non-work dispy 875.64), 2016 couples 1,225 negatives; all
  files pass their canary. **[EMP]**
- DE smoke: 3 of the household-summed negatives are all **simulated non-employment** alternatives;
  **no chosen alternative is negative** (`de_2017_pricing_smoke/REPORT.md`, "Disposable-income
  notes"). **[EMP]**

### 6.2 Why downstream consumption must be strictly positive

**[CONTRACT/INF]** The labour-supply utility uses a **Box-Cox transform of consumption**
(`BC(c)`), and consumption enters in logs during normalization (`log(c_norm)`). Box-Cox and `log`
are defined only for strictly positive arguments, so a consumption input of `0` or a negative value
is mathematically inadmissible. The pipeline therefore requires **`consumption > 0`** at the
estimation stage. (The floor constant's own comment ties this to the original R code:
*"Floor for consumption/leisure (matches R code: `pmax(1, ils_dispy)`)"*,
`scripts/enhanced/enh_RURO_prep_mnl_basic.py:53`.)

### 6.3 FR pipeline treatment — exact floor value and stage

**[CONTRACT]** The floor is the constant **`DCM_MIN_POSITIVE = 1.0`** and consumption is the
**clipped real disposable income**:

```text
consumption = clip(ils_dispy_real, lower = 1.0)
```

- B-pool track — applied at the **engine-ready harmonisation** stage, **not** at pricing:
  - `scripts/bpool/harmonise_bpool_engine_ready.py:43-44` — `TOTAL_LEISURE_HOURS = 80.0`,
    `DCM_MIN_POSITIVE = 1.0`.
  - Singles: `cons = ils_dispy_real.clip(lower = DCM_MIN_POSITIVE)` (`:107`).
  - Couples: household **joint** `ils_dispy_real` (already summed over the tax unit upstream in
    `build_bpool_estimation_ready.py`) then `.clip(lower = DCM_MIN_POSITIVE)` (`:154`).
- Legacy enhanced track — same constant and clip
  (`scripts/enhanced/enh_RURO_prep_mnl_basic.py:53`, `:728` singles, `:1095-1096` couples).

**[CONTRACT] Raw disposable income is retained before the floor.** The priced files keep raw
`ils_dispy`/`ils_dispy_real` (nominal and CPI-deflated). The build stage copies `ils_dispy_real`
into `consumption` *without* flooring (`build_bpool_estimation_ready.py:336, 362`); the **floor is
introduced only at engine-ready harmonisation**, so the raw value is preserved through pricing and
estimation-ready and is clipped only in the engine-facing `consumption`. The pricing stage applies
**no** floor.

**[CONTRACT] Alternatives are floored, not excluded.** `clip(lower=1.0)` rewrites sub-floor values
to `1.0` and **changes neither the row count nor the alternative-key set**; every alternative
(including non-positive ones) is retained.

### 6.4 Empirical counts (floor incidence)

**Method.** Read-only over the FR priced-long files. **Alternative unit** = the model-relevant unit
the pipeline floors:
- **Singles:** one observation per alternative = the decider row (`ruro_decider == 1`), keyed
  `(stacked_hh_uid, draw)`; value = `ils_dispy_real`. (101 alternatives/HH.)
- **Couples:** the household-alternative value = `sum(member ils_dispy_real)` over the documented
  key `(stacked_hh_uid, draw_joint, is_chosen_joint)`, evaluated **before** the floor. (901
  alternatives/HH.)

Floor = `1.0`. "Floored" = raw value `< 1.0` (these become exactly `1.0` after `clip`). Denominator
= total alternative units in that stratum. Chosen alternative = `is_chosen == 1` (singles) /
`is_chosen_joint == 1` (couples).

**[EMP] Singles (alternative unit = decider row; 101/HH):**

| Year | Alt units | Missing | Raw ≤ 0 | 0 < raw < 1 | Raw < 1 (floored→1.0) | Chosen units | Chosen floored |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 168,569 | 0 | 384 | 43 | 427 (0.2533%) | 1,669 | 1 |
| 2016 | 169,276 | 0 | 370 | 55 | 425 (0.2511%) | 1,676 | 2 |
| 2017 | 167,862 | 0 | 248 | 37 | 285 (0.1698%) | 1,662 | 0 |
| **Total** | **505,707** | **0** | **1,002** | **135** | **1,137 (0.2248%)** | **5,007** | **3** |

**[EMP] Couples (alternative unit = household-alternative sum; 901/HH):**

| Year | Alt units | Missing | Raw ≤ 0 | 0 < raw < 1 | Raw < 1 (floored→1.0) | Chosen units | Chosen floored |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 2,311,966 | 0 | 908 | 40 | 948 (0.0410%) | 2,566 | 2 |
| 2016 | 2,321,877 | 0 | 639 | 47 | 686 (0.0295%) | 2,577 | 1 |
| 2017 | 2,067,795 | 0 | 171 | 11 | 182 (0.0088%) | 2,295 | 0 |
| **Total** | **6,701,638** | **0** | **1,718** | **98** | **1,816 (0.0271%)** | **7,438** | **3** |

**[EMP] Overall (singles + couples alternative units):**

| Metric | Value |
|---|---:|
| Total alternative units | 7,207,345 |
| Missing raw values | 0 |
| Raw ≤ 0 | 2,720 |
| 0 < raw < 1 | 233 |
| Raw < 1 (changed by the 1.0 floor) | 2,953 (**0.0410 %**) |
| = 1.0 after flooring | 2,953 |
| Chosen alternatives floored | 6 (3 singles + 3 couples) |

**Caveat honored.** "Non-positive" is **not** the same as "all values affected by the floor":
flooring at `1.0` changes every value `< 1.0`, which is the union of `raw ≤ 0` (2,720) and the small
sliver `0 < raw < 1` (233) = 2,953. The floor touches **0.041 %** of alternative units overall; it
is a guard for a rare tail, not a routine transformation. Row counts and alternative-key sets are
**unchanged** by flooring (it is a value clip, not a filter). **[EMP/CONTRACT]**

---

## 7. Operational contracts (for the labour-supply pipeline)

These are the binding facts a downstream consumer must honor. All are **[CONTRACT]** unless tagged.

1. **Valid aggregation operation.** Household disposable income for couples =
   `sum(member ils_dispy_real)` over `(stacked_hh_uid, draw_joint, is_chosen_joint[, data_year])`;
   singles = the decider row (`ruro_decider == 1`). Summing person rows is the **documented**
   household operation (**[AUTH-EM]** assessment-unit sum rule). Do not average or replicate.
2. **Raw-output preservation.** Raw `ils_dispy` (nominal) and `ils_dispy_real` (2016-real) are
   retained through pricing and estimation-ready. The floor does **not** overwrite them; it is
   applied only to the engine-facing `consumption`.
3. **Floor value and stage.** `consumption = clip(ils_dispy_real, lower = 1.0)`, with
   `DCM_MIN_POSITIVE = 1.0`, applied at **engine-ready harmonisation**
   (`harmonise_bpool_engine_ready.py`), never at pricing.
4. **Floor flag/count requirement.** Because flooring silently rewrites `< 1.0` values to `1.0`,
   the count of floored alternatives (and, ideally, a per-row floor flag) must be reported so that a
   floored consumption is never mistaken for a genuine €1 outcome. Empirically this is a rare tail
   (≈0.04 %), so it is easy to overlook — report it explicitly. (**[INF]** the explicit flag is a
   recommended practice; the pipeline currently logs floored counts, e.g.
   `enh_RURO_prep_mnl_basic.py:1098-1104`.)
5. **Source-identity preservation for clustering.** The original household id `idorighh` is
   preserved as `cluster_id` and is the clustered-SE key — **not** the synthetic stacked id
   (`build_bpool_estimation_ready.py:347-348, 499-509`; `harmonise_bpool_engine_ready.py:132, 280`).
   Aggregation/flooring must not destroy `idorighh`.
6. **Alternatives floored, not excluded.** Non-positive alternatives are clipped and kept; row
   counts and alternative-key sets are invariant under flooring.

---

## 8. Open questions, source conflicts, and documentation ambiguities

1. **Periodicity is configuration, not documented universally.** EUROMOD Help does not pin a
   universal period for `ils_dispy`. "Monthly" is established by the FR/DE datasets/pipeline. **[INF]**
2. **`ils_sicee`/`ils_sicse` vs. `ils_sicdy`.** The Help's `DefIL` example subtracts two SIC lists;
   the FR realization subtracts a single `ils_sicdy`. Same concept, different country
   configuration; not a conflict, but a naming difference to be aware of when matching formulas. **[CFG/INF]**
3. **System-year vs. reference-year for `IlsDef`.** Priced files were produced with `system =
   data_year − 1`, but the project's income-concept CSV is keyed to the same-year systems. For
   exact component reconciliation use the pricing system, especially across the FR PPE↔PA switch. **[INF/ambiguity]**
4. **DE row-level semantics not independently re-verifiable** from persisted artifacts (only the
   household aggregate was saved); the additive convention rests on EUROMOD documentation + FR
   empirics, with DE consistent but not re-proved. **[stated limitation]**
5. **`yiv` vs. `yiy`.** The brief's `yiv` maps to the FR investment-income variable `yiy`; mnemonics
   for capital/investment income vary by dataset. **[INF]**
6. **"35-hour" as institutional constant.** The `yem00`/`yemxp` split is firmly FR configuration;
   attributing it specifically to the 35-hour statutory week is contextual inference, not an EUROMOD
   Help statement. **[INF]**
7. **Equivalisation.** EU-SILC's *equivalised* disposable income (modified-OECD scale, p. 39) is a
   distinct concept from EUROMOD `ils_dispy`; this project uses **unequivalised** household sums for
   consumption. Do not conflate. **[AUTH-SILC/INF]**

---

## 9. Revalidation triggers

Re-run the relevant parts of this investigation and update this document if any of the following
change:

- **EUROMOD release** changes (the install here is `J2.0+`): `DefIL`/`DefOutput` semantics or the
  `ils_dispy` standard definition could change.
- **Country system or dataset** changes (new FR/DE systems, a new country, or new dataset vintages):
  `ils_*` component lists, `SetDefault` values, the `yem00`/`yemxp` split, the `yemse` identity, and
  the negative-income incidence are all configuration-dependent.
- **System/dataset pairing** changes (e.g. dropping the `data_year − 1` lag, or new priced years):
  re-derive [§5.3](#53-fr-systemdataset-pairing-convention-pricing)/[§5.4](#54-de-yemse--yem--yse-identity).
- **Output level** changes from individual to household (`output_std_hh_fr` switched on, or a
  different `TAX_UNIT`): the additive-vs-aggregated reading and the summation step change.
- **Floor policy** changes (`DCM_MIN_POSITIVE` value, the clipped variable, or the stage at which it
  is applied), or a switch from flooring to exclusion: re-run the [§6.4](#64-empirical-counts-floor-incidence) counts.
- **Real-deflation choice** changes (base year ≠ 2016, different CPI index, or the
  `ils_dispy_real = ils_dispy × phi` rule).
- **A fresh EUROMOD run** is produced — especially a DE run that **persists person-level output** —
  which would allow independent re-verification of DE row-level semantics ([§4.5](#45-de-consistency-evidence-and-its-limitation)).
- Any change to the **estimation utility** that alters the strict-positivity requirement on
  consumption ([§6.2](#62-why-downstream-consumption-must-be-strictly-positive)).

---

## 10. Sources

### 10.1 Authoritative documentation

- **[AUTH-EM]** *EUROMOD Help*, `C:\Program Files\EUROMOD\Help\EUROMODHelp.chm` (EUROMOD release
  `J2.0+`). Topics cited (decompiled topic files):
  - *EUROMOD Basic Concepts → EUROMOD terminology* (`EM_BC_Terminology.htm`): headings "EUROMOD
    incomelist", "EUROMOD assessment unit (tax unit)", "Standard output", "EUROMOD standard
    disposable income", "EUROMOD variables".
  - *EUROMOD Functions → DefIL* (`EM_FC_DefIL.htm`): Example 1 (`ils_dispy` definition; `il_`/`ils_`
    naming; `_s` simulated postfix).
  - *EUROMOD Functions → DefOutput* (`EM_FC_DefOutput.htm`): Examples 1–2 (`TAX_UNIT` individual vs.
    household; household aggregation = one row per household; `dag` = head's value).
  - *EUROMOD Functions → IlArithOp* (`EM_FC_IlArithOp.htm`), *ILVarOp* (`EM_FC_IlVarOp.htm`):
    income-list arithmetic operate at variable level / individual level.
- **[AUTH-SILC]** *EU-SILC: Methodological guidelines with description of variables — 2021
  Operation (v8, accessible)*, PDF (621 pp.),
  `EUROMOD-STORAGE/Data/Methodological guidelines 2021 operation v8 (accessible).pdf`. Cited: §2
  "Standardised and core variables" (pp. 17-18); §5 "Income data" (pp. 39-40) — disposable-income
  definition, income-types "Level" (household = sum of individual incomes), income reference period,
  gross income components (PY010G, PY050G, …). (Governs **EU-SILC input data**, not EUROMOD `ils_*`.)
- Model documentation directory consulted for scope/provenance:
  `EUROMOD-STORAGE/Euromod_model/Documentation/` (e.g. `EM_data_codebook_J2.0+.xlsm`,
  `EUROMOD policy parameters J2.0+.xlsx`, `EM_Whatsnew_J2.0+.pdf`). No conflicting income-concept
  statements were found there within scope.

### 10.2 Country / dataset configuration (FR)

- `Data/documentation/euromod_fr_2015_2017_input_output_reference.md` — FR input/output reference;
  standardized income concepts; standard-output configuration (individual-level on, household-level
  off); year-specific PPE/PA switch.
- `Data/documentation/euromod_fr_2015_2017_standard_income_concepts.csv` — `IlsDef_fr` component
  table for `ils_earns/origy/ben/tax/sicdy/dispy` and sub-lists.
- `Data/documentation/euromod_fr_2015_2017_input_variables.csv` — DRD input-variable index
  (`yem`, `yse`, `yiy`, `yem00`, `yemxp`, …).
- `Data/documentation/euromod_fr_2015_2017_output_variable_index.csv` — standard-output policy config.
- `Data/documentation/FR_2015_all_tables_compact.md` — FR policy dump: `output_std_hh_fr` (lines
  2941-2946); `SetDefault_fr` (`yem00 = yem` line 2957; `yemxp = 0` line 2982; `yem00 = yem` line
  2983); overtime formula (lines 1384, 1389).

### 10.3 Project pipeline contract (code; repo-relative)

- `scripts/bpool/run_bpool_euromod.py` — system/dataset pairing (`:8-11`, `:62-67`); CPI base 2016
  (`:13-15`, `:68`); `ils_dispy_real = ils_dispy * phi` (`:469`).
- `scripts/bpool/run_bpool_euromod_chunk.py` — chunked `ils_dispy_real` (`:47`, `:213`).
- `scripts/bpool/assemble_bpool_priced.py` — CPI (`:26`); deflation-consistency canary (`:77`).
- `scripts/bpool/build_bpool_estimation_ready.py` — singles decider lookup (`:221-230`); couples
  joint sum (`:233-245`); `consumption = ils_dispy_real` unfloored (`:336`, `:362`); wage-deflation
  guardrail (`:298-317`); `cluster_id = idorighh` (`:347-348`, `:499-509`).
- `scripts/bpool/harmonise_bpool_engine_ready.py` — `TOTAL_LEISURE_HOURS=80`,
  `DCM_MIN_POSITIVE=1.0` (`:43-44`); singles floor (`:107`); couples joint floor (`:154`); cluster
  key (`:132`, `:280`).
- `scripts/enhanced/enh_RURO_prep_mnl_basic.py` — floor constant + R-code lineage (`:52-53`);
  singles consumption + floor (`:710-729`); couples per-person floor + sum (`:1090-1096`,
  `:1294-1309`).
- `EUROMOD-STORAGE/scratch/staging/de_2017_pricing_smoke/smoke.py` — DE pricing rule, `yemse=yem+yse`
  (`:24`, `:131-133`, `:219`, `:281`); household-summed dispy (`:252-254`).
- `EUROMOD-STORAGE/scratch/staging/de_2017_pricing_smoke/REPORT.md` — DE smoke gates, negative-income
  notes, persisted-artifacts list.

### 10.4 Empirical data

- `EUROMOD-STORAGE/new_data/fr_p3a_bpool_priced__{2015,2016,2017}__{singles,couples}.parquet` — raw
  priced-long person rows (singles ~239k–244k rows/file; couples ~6.8M–7.6M rows/file).
- `EUROMOD-STORAGE/new_data/fr_p3a_bpool_priced__meta.json` — per-file canary (non-null,
  non-work-income negatives, CPI consistency).
- `EUROMOD-STORAGE/scratch/staging/de_2017_pricing_smoke/alt_results.csv` — DE per-alternative
  household-summed disposable income (no person-level column).

Empirical counts in [§4.3](#43-fr-empirical-confirmation) and
[§6.4](#64-empirical-counts-floor-incidence) were produced by read-only in-memory analysis of the
parquets (grouping units and denominators stated inline). No EUROMOD execution occurred; no analysis
files were written into the repository.

---

## 11. Appendix: other directly relevant EUROMOD facts discovered during the research

- **[AUTH-EM] Assessment unit (tax unit) range.** EUROMOD assessment units span from a single
  individual to the whole household, with intermediate (family) units; "tax unit" is the common
  (loose) term for these (`EM_BC_Terminology.htm`, "EUROMOD assessment unit"). The output `TAX_UNIT`
  parameter selects which unit the income list is reported on.
- **[AUTH-EM] VOID sentinel.** Simulated variables are initialised to `VOID = 0.0000000000001`;
  using a VOID-valued simulated variable (other than as output) raises an error
  (`EM_FC_DefIL.htm`, footnote [1]). Relevant when reasoning about why a simulated component might
  appear as a tiny non-zero value.
- **[AUTH-EM] `il` vs. `DefIL` printing.** In `DefOutput`, parameter `il <name>` prints the income
  list's **value** (the sum of its components), while `DefIL <name>` prints **each component
  entry** separately (`EM_FC_DefOutput.htm`, Example 1). Useful for interpreting EUROMOD output
  columns.
- **[CFG-FR] Rich `ils_*` surface in priced output.** Beyond the headline lists, the FR priced
  files expose benefit sub-lists `ils_b1_*` (per-benefit: child `bcb`/`bfa`, education `bed`,
  old-age `boa`, survivor `bsu`, disability `bdi`, unemployment `bun`, housing `bho`/`bhl`, social
  assistance `bsa`, in-work `bwk`), `ils_b2_*` combinations, tax bases `ils_base_*`, and UDB-mapped
  components `ils_udb_*` (incl. `ils_udb_yem`, `ils_udb_yse`). These allow decomposition of
  `ils_dispy` into policy contributions if needed.
- **[CONTRACT] Two parallel FR tracks.** A legacy "enhanced" MNL-prep track
  (`scripts/enhanced/enh_RURO_prep_mnl_basic.py`) and the current "B-pool" track
  (`scripts/bpool/*`) implement the **same** floor (`DCM_MIN_POSITIVE = 1.0`), leisure
  (`80 − hours`), and normalization conventions; the B-pool harmoniser explicitly mirrors the
  enhanced prep (`harmonise_bpool_engine_ready.py:9-13`). Either can be cited for the floor contract;
  they agree.
- **[CONTRACT] Leisure floor shares the same constant.** `leisure = clip(80 − hours, lower = 1.0)`
  uses the same `DCM_MIN_POSITIVE = 1.0` as consumption (`harmonise_bpool_engine_ready.py:103`),
  so the strict-positivity guard is symmetric across both Box-Cox arguments.

---

*End of reference. Reflects sources examined as of 2026-06-06; see [§9](#9-revalidation-triggers)
for when to revalidate.*
