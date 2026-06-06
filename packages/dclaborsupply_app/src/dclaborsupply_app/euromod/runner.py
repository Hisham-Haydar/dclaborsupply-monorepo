"""Generic, connector-injected EUROMOD pricing runner (app layer; no EUROMOD import).

Replicates each labour-supply *alternative* as an isolated synthetic household, applies
an injected country :class:`EarningsMutationPolicy` to the decider rows, prices via an
injected :class:`~dclaborsupply_app.euromod.connector.PricingConnector`, and returns raw
per-person output + provenance + a SEPARATE tax-unit totals table.

Scope (deliberately NOT here): engine-ready assembly, CPI deflation, consumption
flooring/normalization, the singles/couples aggregation asymmetry, and engine-ready
``cluster_id`` restoration. The tax-unit totals are a uniform per-alternative SUM of the
raw nominal ``ils_dispy`` over all members — never relabelled "consumption" and never an
``ils_dispy_real``. No core/france/MNL imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from .connector import PricingConnector, PricingConnectorResult

WEEKS_PER_MONTH = 52.0 / 12.0


class EarningsMutationPolicy(Protocol):
    """Country labour-earnings mutation. Given a decider's baseline row and the
    alternative's (hours, wage), return the column→value overrides to apply to that
    decider before pricing. MUST NOT mutate ``member`` in place."""

    def __call__(self, member: Mapping[str, Any], *, hours: float, wage: float,
                 weeks_per_month: float) -> Dict[str, float]:
        ...


@dataclass
class PricingColumns:
    """Configurable ID / relationship-reference / state column names."""
    hh: str = "idhh"                 # baseline household id (original)
    person: str = "idperson"
    partner: str = "idpartner"
    father: str = "idfather"
    mother: str = "idmother"
    orig_hh: str = "idorighh"
    orig_person: str = "idorigperson"
    dgn: str = "dgn"
    decider_flag: str = "ruro_decider"   # baseline flag marking expected deciders
    # alternatives-table columns
    source_hh: str = "source_idhh"   # links an alternative to its baseline household
    decider_person: str = "decider_idperson"
    hours: str = "hours"
    wage: str = "wage"

    def relationship_cols(self) -> List[str]:
        return [self.partner, self.father, self.mother]

    def protected_cols(self, alt_key_cols: Sequence[str]) -> set:
        """Columns a country policy MUST NOT override (identity / provenance / keys)."""
        return {
            self.hh, self.person, self.orig_hh, self.orig_person, self.dgn,
            *self.relationship_cols(),
            "source_idhh", "source_idorighh", "source_idperson",
            "ruro_decider", "data_year", *alt_key_cols,
        }


@dataclass
class PricingResult:
    """Output of :meth:`EuromodPricingRunner.price`."""
    output: pd.DataFrame              # raw per-person EUROMOD output + provenance; raw nominal ils_dispy
    reverse_mapping: pd.DataFrame     # synthetic (idhh, idperson) -> source ids + alt keys
    taxunit_totals: pd.DataFrame      # alt keys (+ source_idhh) -> ils_dispy_taxunit_sum
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


_INT64_MAX = np.iinfo(np.int64).max


class EuromodPricingRunner:
    """Connector- and policy-injected pricing runner.

    Parameters
    ----------
    connector : PricingConnector   (e.g. EuromodConnector, or a fake for tests)
    policy : EarningsMutationPolicy (e.g. DE de_earnings_policy)
    columns : PricingColumns       column-name configuration
    hh_base, person_mult : synthetic-id scheme. Each alternative gets
        ``new_idhh = hh_base + alt_index``; member k gets ``new_idhh*person_mult + k``.
        ``person_mult`` bounds members/household; ranges are validated for int64 safety.
    """

    def __init__(self, connector: PricingConnector, policy: EarningsMutationPolicy, *,
                 columns: Optional[PricingColumns] = None,
                 hh_base: int = 900_000_000, person_mult: int = 1000) -> None:
        self.connector = connector
        self.policy = policy
        self.c = columns if columns is not None else PricingColumns()  # no shared mutable default
        self.hh_base = int(hh_base)
        self.person_mult = int(person_mult)

    # -- build ---------------------------------------------------------------
    def build_inputs(
        self,
        alternatives: pd.DataFrame,
        baseline: pd.DataFrame,
        *,
        alt_key_cols: Sequence[str],
        weeks_per_month: float = WEEKS_PER_MONTH,
        data_year: Optional[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and assemble (euromod_input, provenance). No pricing."""
        c = self.c
        alt_key_cols = list(alt_key_cols)
        for col in [c.source_hh, c.decider_person, c.hours, c.wage, *alt_key_cols]:
            if col not in alternatives.columns:
                raise ValueError(f"alternatives missing required column '{col}'.")
        for col in [c.hh, c.person]:
            if col not in baseline.columns:
                raise ValueError(f"baseline missing required column '{col}'.")

        # duplicate alternative keys (same decider twice within one alternative)
        key_full = [c.source_hh, *alt_key_cols, c.decider_person]
        dups = alternatives.duplicated(subset=key_full)
        if dups.any():
            ex = alternatives.loc[dups, key_full].head(5).to_dict("records")
            raise ValueError(f"duplicate alternative/decider keys ({int(dups.sum())}); e.g. {ex}")

        rel_cols = [r for r in c.relationship_cols() if r in baseline.columns]
        base_by_hh = {hh: g for hh, g in baseline.groupby(c.hh, sort=False)}
        protected = c.protected_cols(alt_key_cols)

        # Expected-decider contract: when the baseline carries the decider flag, the
        # decider set of EVERY alternative must EXACTLY equal the flagged baseline
        # deciders of its household (no missing, no extra). ruro_decider provenance is
        # taken from this baseline flag, never derived from whichever rows appear.
        has_flag = c.decider_flag in baseline.columns
        expected_deciders: Dict[Any, set] = {}
        if has_flag:
            flagged = baseline[pd.to_numeric(baseline[c.decider_flag], errors="coerce").fillna(0) == 1]
            for hhid, g in flagged.groupby(c.hh, sort=False):
                expected_deciders[hhid] = set(g[c.person].astype("int64"))

        # group alternatives -> one alternative per (source_hh, *alt_key_cols)
        alt_groups = alternatives.groupby([c.source_hh, *alt_key_cols], sort=False)
        n_alts = alt_groups.ngroups
        # unsafe-range checks
        max_members = int(baseline.groupby(c.hh).size().max())
        if max_members >= self.person_mult:
            raise ValueError(f"max members/household ({max_members}) >= person_mult ({self.person_mult}); unsafe ids.")
        max_new_hh = self.hh_base + n_alts
        if max_new_hh * self.person_mult + max_members > _INT64_MAX:
            raise ValueError("synthetic id range exceeds int64; reduce hh_base/person_mult or n_alts.")

        em_rows: List[pd.DataFrame] = []
        prov_rows: List[pd.DataFrame] = []
        rev_rows: List[dict] = []

        for alt_idx, (_key, grp) in enumerate(alt_groups):
            src_hh = grp[c.source_hh].iloc[0]
            if src_hh not in base_by_hh:
                raise ValueError(f"incomplete household restoration: baseline has no household {src_hh}.")
            hh = base_by_hh[src_hh].copy().reset_index(drop=True)
            n = len(hh)
            if n == 0:
                raise ValueError(f"incomplete household restoration: household {src_hh} has no members.")
            orig_pids = hh[c.person].astype("int64").tolist()
            # unresolved relationship refs (must point within this household)
            pidset = set(orig_pids)
            for rcol in rel_cols:
                ref = pd.to_numeric(hh[rcol], errors="coerce").fillna(0).astype("int64")
                bad = ref[(ref != 0) & (~ref.isin(pidset))]
                if len(bad):
                    raise ValueError(
                        f"unresolved relationship reference in household {src_hh}, column '{rcol}': "
                        f"{bad.tolist()[:5]} not in {sorted(pidset)}.")

            new_hh = self.hh_base + alt_idx
            new_ids = {old: new_hh * self.person_mult + (k + 1) for k, old in enumerate(orig_pids)}

            src_idhh = hh[c.hh].astype("int64").to_numpy()
            if c.orig_hh in hh.columns:
                src_orighh = hh[c.orig_hh].astype("int64").to_numpy()
                if pd.Series(src_orighh).nunique() > 1:
                    raise ValueError(
                        f"source_idorighh not household-consistent for household {src_hh}: "
                        f"{sorted(set(src_orighh.tolist()))}.")
            else:
                src_orighh = src_idhh.copy()
            src_person = np.array(orig_pids, dtype="int64")
            orighh_of = dict(zip(orig_pids, src_orighh.tolist()))
            dgn_vals = (pd.to_numeric(hh[c.dgn], errors="coerce").to_numpy()
                        if c.dgn in hh.columns else np.full(n, np.nan))

            # expected-decider completeness: actual decider set must equal the flagged set
            actual_deciders = set()
            for _, drow in grp.iterrows():
                dpid = int(drow[c.decider_person])
                if dpid not in set(orig_pids):
                    raise ValueError(
                        f"missing decision-maker: decider idperson {dpid} not a member of household {src_hh}.")
                actual_deciders.add(dpid)
            if has_flag:
                exp = expected_deciders.get(src_hh, set())
                if actual_deciders != exp:
                    raise ValueError(
                        f"decider-set mismatch for household {src_hh} alternative {_key}: "
                        f"expected (flagged) {sorted(exp)} != actual {sorted(actual_deciders)} "
                        f"(missing={sorted(exp - actual_deciders)}, extra={sorted(actual_deciders - exp)}).")
                decider_prov = exp
            else:
                decider_prov = actual_deciders

            # remap ids + relationships (0 stays 0)
            hh[c.hh] = new_hh
            if c.orig_hh in hh.columns:
                hh[c.orig_hh] = new_hh
            hh[c.person] = [new_ids[p] for p in orig_pids]
            for rcol in rel_cols:
                ref = pd.to_numeric(hh[rcol], errors="coerce").fillna(0).astype("int64")
                hh[rcol] = [new_ids.get(int(x), 0) for x in ref]
            if c.orig_person in hh.columns:
                hh[c.orig_person] = hh[c.person]

            # apply policy to each decider in this alternative (identity columns protected)
            for _, drow in grp.iterrows():
                dpid = int(drow[c.decider_person])
                pos = orig_pids.index(dpid)
                member = base_by_hh[src_hh].reset_index(drop=True).iloc[pos].to_dict()
                overrides = self.policy(member, hours=float(drow[c.hours]),
                                        wage=float(drow[c.wage]), weeks_per_month=weeks_per_month)
                for col, val in overrides.items():
                    if col in protected:
                        raise ValueError(
                            f"policy may not override protected identity/provenance/key column '{col}'.")
                    if col not in hh.columns:
                        raise ValueError(f"policy override column '{col}' not in baseline schema.")
                    # cast integer baseline columns to float before writing a float override
                    if hh[col].dtype.kind != "f":
                        hh[col] = hh[col].astype("float64")
                    hh.loc[pos, col] = float(val)

            # provenance (kept OUT of the EUROMOD input; merged back onto output).
            # ruro_decider comes from the EXPECTED (baseline-flagged) decider set.
            prov = pd.DataFrame({
                c.hh: hh[c.hh].to_numpy(), c.person: hh[c.person].to_numpy(),
                "source_idhh": src_idhh, "source_idorighh": src_orighh,
                "source_idperson": src_person,
                "ruro_decider": [1 if p in decider_prov else 0 for p in orig_pids],
                "dgn": dgn_vals,
            })
            for kc in alt_key_cols:
                prov[kc] = grp[kc].iloc[0]
            if data_year is not None:
                prov["data_year"] = data_year
            prov_rows.append(prov)

            for old, new in new_ids.items():
                rev_rows.append(dict(new_idhh=int(new_hh), new_idperson=int(new),
                                     source_idhh=int(src_hh), source_idorighh=int(orighh_of[old]),
                                     source_idperson=int(old),
                                     **{kc: grp[kc].iloc[0] for kc in alt_key_cols}))
            em_rows.append(hh)

        euromod_input = pd.concat(em_rows, ignore_index=True)
        # the decider flag is a control column, not a EUROMOD input variable -> don't send it
        euromod_input = euromod_input.drop(columns=[c.decider_flag], errors="ignore")
        provenance = pd.concat(prov_rows, ignore_index=True)

        # complete-membership + cross-household integrity verification
        for new_hh, g in euromod_input.groupby(c.hh):
            src = int(provenance.loc[provenance[c.hh] == new_hh, "source_idhh"].iloc[0])
            if len(g) != len(base_by_hh[src]):
                raise ValueError(f"incomplete household restoration for synthetic hh {new_hh}.")
            pids = set(g[c.person].astype("int64"))
            for rcol in rel_cols:
                ref = pd.to_numeric(g[rcol], errors="coerce").fillna(0).astype("int64")
                if int(((ref != 0) & (~ref.isin(pids))).sum()) > 0:
                    raise ValueError(f"cross-household relationship reference in synthetic hh {new_hh}.")

        self._reverse_mapping = pd.DataFrame(rev_rows)
        return euromod_input, provenance

    # -- price ---------------------------------------------------------------
    def price(
        self,
        alternatives: pd.DataFrame,
        baseline: pd.DataFrame,
        *,
        country: str,
        system: str,
        dataset: str,
        alt_key_cols: Sequence[str],
        weeks_per_month: float = WEEKS_PER_MONTH,
        data_year: Optional[int] = None,
        dispy_col: str = "ils_dispy",
    ) -> PricingResult:
        """Build isolated alternatives, price via the connector, attach provenance,
        and compute the separate uniform tax-unit totals."""
        c = self.c
        alt_key_cols = list(alt_key_cols)
        euromod_input, provenance = self.build_inputs(
            alternatives, baseline, alt_key_cols=alt_key_cols,
            weeks_per_month=weeks_per_month, data_year=data_year)

        res: PricingConnectorResult = self.connector.run(
            euromod_input, country=country, system=system, dataset=dataset)
        out = res.output.copy()
        if dispy_col not in out.columns:
            raise ValueError(f"connector output missing '{dispy_col}'.")

        # --- connector-output one-to-one validation (before attaching provenance) ---
        join = [c.hh, c.person]
        for k in join:
            if k not in out.columns:
                raise ValueError(f"connector output missing synthetic id column '{k}'.")
            out[k] = pd.to_numeric(out[k], errors="coerce").astype("int64")
            provenance[k] = pd.to_numeric(provenance[k], errors="coerce").astype("int64")
        if out.duplicated(subset=join).any():
            n = int(out.duplicated(subset=join).sum())
            raise ValueError(f"connector output has {n} duplicate synthetic (idhh, idperson) rows.")
        expected = set(map(tuple, provenance[join].itertuples(index=False, name=None)))
        got = set(map(tuple, out[join].itertuples(index=False, name=None)))
        missing, extra = expected - got, got - expected
        if missing:
            raise ValueError(f"connector output missing {len(missing)} expected member(s); e.g. {sorted(missing)[:5]}.")
        if extra:
            raise ValueError(f"connector output has {len(extra)} unexpected member(s); e.g. {sorted(extra)[:5]}.")

        # attach provenance (only columns not already produced by EUROMOD), by synthetic id.
        # set-equality + no duplicates + validate="one_to_one" together guarantee complete
        # one-to-one matching.
        add_cols = [col for col in provenance.columns if col not in out.columns or col in join]
        out = out.merge(provenance[add_cols], on=join, how="left", validate="one_to_one")

        # SEPARATE tax-unit totals: uniform sum of raw nominal ils_dispy over all members
        gcols = ["source_idhh", *alt_key_cols]
        taxunit = (out.groupby(gcols, sort=False)[dispy_col].sum()
                   .rename("ils_dispy_taxunit_sum").reset_index())

        return PricingResult(
            output=out, reverse_mapping=self._reverse_mapping, taxunit_totals=taxunit,
            warnings=list(res.warnings), errors=list(res.errors))
