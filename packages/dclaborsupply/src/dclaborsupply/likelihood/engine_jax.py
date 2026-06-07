"""JAX likelihood engine for the RURO joint MNL (lifted, migration matrix Wave 1.2).

Lifted from the research repo with copy + import/laziness adaptation ONLY — no
math change:
  - builders + primitives  <- MNL/scripts/bpool/jax_ll_probe.py
  - joint assembler        <- MNL/scripts/bpool/jax_joint_hessian.py (build_joint_neg_ll)

The probe/CLI drivers (main()) and the estimation_spec_parser / estimation_utils /
estimation_engine / joint_recovery_test imports + sys.path hacks from the sources
were NOT lifted: the builders are self-contained, taking ``data`` and ``spec``
(an ``dclaborsupply.spec.parser.EstimationSpec``) as arguments. This module has
ZERO old-repo / MNL imports.

JAX is an OPTIONAL extra and is imported LAZILY (never at module import time):
``import dclaborsupply`` and ``import dclaborsupply.likelihood.engine_jax`` both
succeed with jax NOT installed. jax/jnp are bound by ``_load_jax()`` only when a
builder is first called. float64 (``jax_enable_x64``) is enabled in the loader —
required to reproduce the certified negLL.

Index construction is unchanged:
    V = u + log_h + log_w + log_market - log_prior
with worker-gated wage / market / occupation terms. The Wave-0.1 invariant holds:
non-working H/W/Occ contributions are identically zero (worker gate), matching the
prep-side ``working`` mask. ``jbox_cox`` and the ``- log_prior`` handling are
untouched.
"""
from __future__ import annotations

import numpy as np

# jax / jnp are bound lazily by _load_jax() (see module docstring). They live at
# module scope so the primitives below can reference `jnp`/`jax` once a builder
# has triggered the import — nothing here imports jax at module load time.
jax = None
jnp = None


def _load_jax():
    """Import jax lazily (float64), binding the module-level ``jax``/``jnp``.

    Keeps the package importable with jax absent: nothing runs at module import
    time. Mirrors the sources' ``jax.config.update("jax_enable_x64", True)``
    (MUST precede heavy jnp use; required to hit the certified negLL tolerance).
    """
    global jax, jnp
    if jax is None:
        try:
            import jax as _jax
        except ImportError as exc:  # pragma: no cover - exercised only w/o jax
            raise ImportError(
                "JAX backend requested but jax is not installed. "
                "Install with `pip install dclaborsupply[jax]`."
            ) from exc
        _jax.config.update("jax_enable_x64", True)
        import jax.numpy as _jnp
        jax = _jax
        jnp = _jnp
    return jax, jnp


# ---------------------------------------------------------------------------
# JAX primitives (faithful to estimation_engine / estimation_utils)
# ---------------------------------------------------------------------------
def jbox_cox(x, theta):
    """BC(x;θ) = (x^θ - 1)/θ for |θ|>=1e-8 else log(x). Matches box_cox_transform."""
    # jnp.where keeps both branches finite for autodiff; the eps guard mirrors
    # the engine's abs(theta) < 1e-8 -> log(x) limit.
    safe_theta = jnp.where(jnp.abs(theta) < 1e-8, 1.0, theta)
    powered = (jnp.power(x, safe_theta) - 1.0) / safe_theta
    return jnp.where(jnp.abs(theta) < 1e-8, jnp.log(x), powered)


def jgroup_logsumexp(V, n_groups, n_alts, actual_choice=None):
    """LSE per group; data laid out as (n_groups, n_alts) row-major.

    Observed-choice term:
      - actual_choice=None  -> V_obs = column 0 (the numpy-engine convention,
        valid for REAL data where draw==0 is the observed row, 100% at col 0).
      - actual_choice given  -> V_obs = sum_j(actual_choice_j * V_j) per group
        (the GAMSPy-engine convention; REQUIRED for SYNTHETIC data where the
        Gumbel-max draw lands the chosen alt anywhere, ~0% at col 0).
    """
    Vg = V.reshape(n_groups, n_alts)
    mx = jnp.max(Vg, axis=1, keepdims=True)
    lse = (mx[:, 0] + jnp.log(jnp.sum(jnp.exp(Vg - mx), axis=1)))
    if actual_choice is None:
        V_obs = Vg[:, 0]
    else:
        acg = actual_choice.reshape(n_groups, n_alts)
        V_obs = jnp.sum(acg * Vg, axis=1)
    return V_obs, lse


def _center_proposal(log_market, prior, n_groups, n_alts):
    """Proposal-weighted within-group centering (matches _center_within_choice_set)."""
    lm = log_market.reshape(n_groups, n_alts)
    w = prior.reshape(n_groups, n_alts)
    denom = jnp.sum(w, axis=1, keepdims=True) + 1e-12  # EPS as in engine
    mean_val = jnp.sum(lm * w, axis=1, keepdims=True) / denom
    return (lm - mean_val).reshape(-1)


def build_jax_singles_ll(data, spec, is_male, use_actual_choice=False,
                         per_group=False, gender_split=None):
    """Return a jit-compiled negLL(theta) for one singles group, plus the
    param-name->index map used to slot the 49-vector into named scalars.

    use_actual_choice=True -> observed term uses data.actual_choice (REQUIRED
    for synthetic recovery); False -> column-0 (validated real-data path).

    per_group=True -> instead of the summed negLL scalar, return a fn giving the
    per-group POSITIVE log-likelihood VECTOR (shape (n_groups,)). jax.jacrev of
    that vector is the per-choice-set score matrix (rows = groups), the meat for
    the clustered sandwich. The summed path is unchanged (negLL = -sum(vector)).

    gender_split: optional set of HOURS-shifter base coef names (e.g.
    {"beta_E","beta_h_pt2"}) to relax male-vs-female for the LR pooling test.
    For each base coef in the set, this builder reads `coef + "_m"` (male group)
    or `coef + "_f"` (female group) instead of the shared `coef`. Default None
    -> baseline path is byte-identical. The names must exist in the spec.

    All data arrays are captured as float64 jnp constants (device-resident).
    """
    _load_jax()
    suffix = "_sm" if is_male else "_sf"
    _gsplit = set(gender_split or ())
    _gsuf_split = "_m" if is_male else "_f"
    pidx = {n: spec.get_param_index(n) for n in spec.all_param_names}
    _ac = (jnp.asarray(data.actual_choice, dtype=jnp.float64)
           if use_actual_choice else None)
    n_groups = int(data.n_groups)
    n_alts = int(data.n_obs // data.n_groups)

    # --- capture data as device constants (float64) ---
    leisure = jnp.asarray(data.leisure, dtype=jnp.float64)
    consumption = jnp.asarray(data.consumption, dtype=jnp.float64)
    working = jnp.asarray(data.working, dtype=jnp.float64)
    prior = jnp.asarray(data.prior, dtype=jnp.float64)
    log_prior = jnp.log(prior)

    def _arr(name):
        v = getattr(data, name, None)
        return None if v is None else jnp.asarray(v, dtype=jnp.float64)

    # leisure shifters (age_norm, age_norm2, n_children[female only])
    leis_shifters = []
    for sh in spec.utility_leisure_shifters:
        var = sh["variable"]
        coef = sh["coefficient"]
        gs = sh.get("gender_specific", False)
        if gs and var == "n_children" and is_male:
            continue  # n_children is female-only
        arr = _arr(var)
        if arr is None:
            continue
        leis_shifters.append((coef + suffix, arr))

    beta_l0_name = spec.utility_leisure_intercept + suffix
    theta_l_name = (spec.utility_leisure_theta + suffix) if spec.utility_leisure_theta else None
    singles_group = "singles_male" if is_male else "singles_female"
    theta_c_name = spec.theta_c_param_name(singles_group)
    beta_c_fixed = getattr(spec, "utility_consumption_coef_fixed", None)

    # hours shifters
    gsuf_h = "_male" if is_male else "_female"
    hours = []
    for sh in spec.hours_shifters:
        var = sh["variable"]
        coef = sh["coefficient"]
        arr = _arr(var)
        if arr is None:
            continue
        inter = sh.get("interaction", None)
        use_working = (inter == "working") or (isinstance(inter, (list, tuple)) and "working" in inter)
        coef_use = (coef + _gsuf_split) if coef in _gsplit else coef
        hours.append((coef_use, arr, use_working))

    # wage mean shifters + sigma (vw/loc_empirical only). For fixed wages (wage_spec=="fw")
    # there is NO wage density: do not touch log_wage / wage shifters / wage_variance_param.
    _wage_fixed = (getattr(spec, "wage_spec", "fw") == "fw")
    wage_terms = []
    sigma_name = None
    log_wage = None
    if not _wage_fixed:
        for sh in spec.wage_mean_shifters:
            var = sh["variable"]
            coef = sh["coefficient"]
            if var == "intercept":
                wage_terms.append((coef, None))
            else:
                arr = _arr(var)
                if arr is not None:
                    wage_terms.append((coef, arr))
        sigma_name = spec.wage_variance_param
        log_wage = _arr("log_wage")

    # market shifters (+ scales, + interaction working) and centering flag
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}
    mkt = []
    for sh in (getattr(spec, "market_opportunity_shifters", None) or []):
        var = sh["variable"]
        coef = sh["coefficient"]
        applies = str(sh.get("applies_to", "both")).strip().lower()
        # singles routing (post-58d0dba): skip only cm/cf; male/female honoured
        if applies in {"cm", "cf"}:
            continue
        if applies in {"male", "sm"} and not is_male:
            continue
        if applies in {"female", "sf"} and is_male:
            continue
        arr = _arr(var)
        if arr is None:
            continue
        arr = arr * float(scale_map.get(var, 1.0))
        inter = sh.get("interaction", None)
        use_working = (inter == "working") or (isinstance(inter, (list, tuple)) and "working" in inter)
        mkt.append((coef, arr, use_working))
    do_center = bool(getattr(spec, "market_opportunity_center_within_choice_set", False))
    center_prop = (getattr(spec, "market_opportunity_center_weights", None) == "proposal")

    LOG2PI = float(np.log(2 * np.pi))

    _fixed = dict(getattr(spec, "fixed_params", {}) or {})

    def neg_ll(theta):
        def P(name):
            # fixed_params take precedence (pinned, not in theta / pidx)
            if name in _fixed:
                return _fixed[name]
            return theta[pidx[name]]

        # ---- utility ----
        theta_l = P(theta_l_name) if theta_l_name else 0.0
        theta_c = P(theta_c_name) if theta_c_name else 0.0
        bc_l = jbox_cox(leisure, theta_l)
        bc_c = jbox_cox(consumption, theta_c)
        beta_l_coeff = P(beta_l0_name)
        for cname, arr in leis_shifters:
            beta_l_coeff = beta_l_coeff + P(cname) * arr
        beta_c = beta_c_fixed if beta_c_fixed is not None else P(spec.utility_consumption_coef + suffix)
        u = beta_l_coeff * bc_l + beta_c * bc_c  # beta_cl=0 for singles

        # ---- hours opportunity ----
        log_h = jnp.zeros_like(u)
        for cname, arr, uw in hours:
            x = arr * working if uw else arr
            log_h = log_h + P(cname) * x

        # ---- wage opportunity ----
        if _wage_fixed:
            log_w = jnp.zeros_like(u)   # fw: fixed wages, no wage density (no log_wage/sigma/wage params)
        else:
            mu = jnp.zeros_like(u)
            for cname, arr in wage_terms:
                mu = mu + (P(cname) if arr is None else P(cname) * arr)
            sigma = P(sigma_name)
            resid = (log_wage - mu) / sigma
            log_w_full = -0.5 * resid**2 - jnp.log(sigma) - 0.5 * LOG2PI - log_wage
            log_w = jnp.where(working > 0, log_w_full, 0.0)

        # ---- market opportunity (+ centering) ----
        log_market = jnp.zeros_like(u)
        for cname, arr, uw in mkt:
            x = arr * working if uw else arr
            log_market = log_market + P(cname) * x
        if do_center:
            if center_prop:
                log_market = _center_proposal(log_market, prior, n_groups, n_alts)
            else:
                lm = log_market.reshape(n_groups, n_alts)
                log_market = (lm - jnp.mean(lm, axis=1, keepdims=True)).reshape(-1)

        # ---- composite V and grouped LL ----
        V = u + log_h + log_w + log_market - log_prior
        V_obs, lse = jgroup_logsumexp(V, n_groups, n_alts, _ac)
        per = V_obs - lse                 # per-group positive log-likelihood
        if per_group:
            return per                    # vector (n_groups,) — jacrev => scores
        return -jnp.sum(per)              # negative LL (matches engine convention)

    return jax.jit(neg_ll), pidx


def build_jax_couples_ll(data, spec, use_actual_choice=False,
                        per_group=False, gender_split=None):
    """Return a jit-compiled negLL(theta) for the couples group.

    gender_split: optional set of HOURS-shifter base coef names to relax
    male-vs-female (LR pooling test). For each base coef in the set the male
    leg reads coef+"_m" and the female leg coef+"_f"; default None -> shared
    coef (baseline path byte-identical). See build_jax_singles_ll.

    use_actual_choice=True -> observed term uses data.actual_choice (REQUIRED
    for synthetic recovery); False -> column-0 (validated real-data path).

    Faithful to compute_likelihood_couples:
      u = beta_l_coeff_m*BC(l_m) + beta_l_coeff_f*BC(l_f) + beta_c*BC(c)
          + beta_ll*BC(l_m)*BC(l_f)           [consumption added ONCE]
      log_h = sum over genders of hours shifters (shared coefs, gender data)
      log_w = sum over genders of log-normal wage density (worker-gated)
      log_market = gsur(both) + region/year/urb(household) + occ(male/female),
                   then proposal-weighted within-group centering
      V = u + log_h + log_w + log_market - log(prior); grouped LSE.
    """
    _load_jax()
    pidx = {n: spec.get_param_index(n) for n in spec.all_param_names}
    n_groups = int(data.n_groups)
    n_alts = int(data.n_obs // data.n_groups)
    _ac = (jnp.asarray(data.actual_choice, dtype=jnp.float64)
           if use_actual_choice else None)
    _gsplit = set(gender_split or ())

    def _arr(name):
        v = getattr(data, name, None)
        return None if v is None else jnp.asarray(v, dtype=jnp.float64)

    leisure_m = _arr("leisure_male")
    leisure_f = _arr("leisure_female")
    consumption = _arr("consumption")
    working_m = _arr("working_male")
    working_f = _arr("working_female")
    prior = _arr("prior")
    log_prior = jnp.log(prior)
    n_children = _arr("n_children")

    # couples Box-Cox exponents: theta_l_m / theta_l_f; theta_c fixed (couples)
    theta_l_m_name = (spec.utility_leisure_theta + "_m") if spec.utility_leisure_theta else None
    theta_l_f_name = (spec.utility_leisure_theta + "_f") if spec.utility_leisure_theta else None
    couples_theta_c_fixed = getattr(spec, "utility_consumption_theta_couples_fixed", None)
    beta_c_fixed = getattr(spec, "utility_consumption_coef_fixed", None)
    beta_l0_m_name = spec.utility_leisure_intercept + "_m"
    beta_l0_f_name = spec.utility_leisure_intercept + "_f"
    interaction_name = spec.couples_interaction_coef  # beta_ll

    # leisure shifters: male -> _male data + _m coef; female -> _female data + _f coef;
    # n_children -> household-level data, female-only, _f coef
    leis_m, leis_f = [], []
    for sh in spec.utility_leisure_shifters:
        var, coef = sh["variable"], sh["coefficient"]
        if var == "n_children":
            if n_children is not None:
                leis_f.append((coef + "_f", n_children))
            continue
        am = _arr(var + "_male"); af = _arr(var + "_female")
        if am is not None:
            leis_m.append((coef + "_m", am))
        if af is not None:
            leis_f.append((coef + "_f", af))

    # hours: shared coef (fallback), gender data + gender working interaction.
    # gender_split: for base coefs in the set, the male leg reads coef+"_m" and
    # the female leg coef+"_f" (LR pooling test). Default None -> shared coef
    # (baseline path byte-identical).
    def _hours(suffix, working, coef_gsuf):
        terms = []
        for sh in spec.hours_shifters:
            var, coef = sh["variable"], sh["coefficient"]
            arr = _arr(var + suffix)
            if arr is None:
                continue
            inter = sh.get("interaction", None)
            uw = (inter == "working") or (isinstance(inter, (list, tuple)) and "working" in inter)
            coef_use = (coef + coef_gsuf) if coef in _gsplit else coef
            terms.append((coef_use, arr, uw, working))
        return terms
    hours_m = _hours("_male", working_m, "_m")
    hours_f = _hours("_female", working_f, "_f")

    # wage: shared coef, gender data, worker-gated (vw/loc_empirical only). For fixed wages
    # (wage_spec=="fw") there is NO wage density: skip log_wage / shifters / wage_variance_param.
    _wage_fixed = (getattr(spec, "wage_spec", "fw") == "fw")
    def _wage(suffix, working):
        terms = []
        lw = _arr("log_wage" + suffix)
        for sh in spec.wage_mean_shifters:
            var, coef = sh["variable"], sh["coefficient"]
            if var == "intercept":
                terms.append((coef, None))
            else:
                arr = _arr(var + suffix)
                if arr is not None:
                    terms.append((coef, arr))
        return lw, working, terms
    sigma_name = None
    wage_m = wage_f = None
    if not _wage_fixed:
        wage_m = _wage("_male", working_m)
        wage_f = _wage("_female", working_f)
        sigma_name = spec.wage_variance_param

    # market: gsur(both: male+female), region/year/urb(household: var * (wm+wf)),
    # occupation loc4(male/female). gsur scaled by 10.
    scale_map = getattr(spec, "market_opportunity_variable_scales", None) or {}
    mkt_terms = []  # list of (coef, contribution_array)
    wfsum = (working_m + working_f)
    for sh in (getattr(spec, "market_opportunity_shifters", None) or []):
        var, coef = sh["variable"], sh["coefficient"]
        applies = str(sh.get("applies_to", "both")).strip().lower()
        scale = float(scale_map.get(var, 1.0))
        inter = sh.get("interaction", None)
        uw = (inter == "working") or (isinstance(inter, (list, tuple)) and "working" in inter)
        if applies == "household":
            arr = _arr(var)
            if arr is None:
                continue
            contrib = arr * scale
            if uw:
                contrib = contrib * wfsum
            mkt_terms.append((coef, contrib))
        else:
            # male/cm/both -> male var * working_male ; female/cf/both -> female var * working_female
            if applies in ("male", "cm", "both"):
                am = _arr(var + "_male")
                if am is not None:
                    c = am * scale
                    if uw:
                        c = c * working_m
                    mkt_terms.append((coef, c))
            if applies in ("female", "cf", "both"):
                af = _arr(var + "_female")
                if af is not None:
                    c = af * scale
                    if uw:
                        c = c * working_f
                    mkt_terms.append((coef, c))
    do_center = bool(getattr(spec, "market_opportunity_center_within_choice_set", False))
    center_prop = (getattr(spec, "market_opportunity_center_weights", None) == "proposal")

    LOG2PI = float(np.log(2 * np.pi))
    _fixed = dict(getattr(spec, "fixed_params", {}) or {})

    def neg_ll(theta):
        def P(name):
            if name in _fixed:
                return _fixed[name]
            return theta[pidx[name]]

        theta_l_m = P(theta_l_m_name) if theta_l_m_name else 0.0
        theta_l_f = P(theta_l_f_name) if theta_l_f_name else 0.0
        theta_c = (float(couples_theta_c_fixed) if couples_theta_c_fixed is not None else 0.0)
        bc_l_m = jbox_cox(leisure_m, theta_l_m)
        bc_l_f = jbox_cox(leisure_f, theta_l_f)
        bc_c = jbox_cox(consumption, theta_c)

        blc_m = P(beta_l0_m_name)
        for cn, arr in leis_m:
            blc_m = blc_m + P(cn) * arr
        blc_f = P(beta_l0_f_name)
        for cn, arr in leis_f:
            blc_f = blc_f + P(cn) * arr
        beta_c = beta_c_fixed if beta_c_fixed is not None else P(spec.utility_consumption_coef)
        beta_ll = P(interaction_name) if interaction_name else 0.0
        u = (blc_m * bc_l_m + blc_f * bc_l_f + beta_c * bc_c
             + beta_ll * bc_l_m * bc_l_f)

        log_h = jnp.zeros_like(u)
        for cn, arr, uw, wk in (hours_m + hours_f):
            x = arr * wk if uw else arr
            log_h = log_h + P(cn) * x

        if _wage_fixed:
            log_w = jnp.zeros_like(u)   # fw: fixed wages, no wage density (no log_wage/sigma/wage params)
        else:
            log_w = jnp.zeros_like(u)
            for (lw, wk, terms) in (wage_m, wage_f):
                mu = jnp.zeros_like(u)
                for cn, arr in terms:
                    mu = mu + (P(cn) if arr is None else P(cn) * arr)
                sigma = P(sigma_name)
                resid = (lw - mu) / sigma
                lwd = -0.5 * resid**2 - jnp.log(sigma) - 0.5 * LOG2PI - lw
                log_w = log_w + jnp.where(wk > 0, lwd, 0.0)

        log_market = jnp.zeros_like(u)
        for cn, contrib in mkt_terms:
            log_market = log_market + P(cn) * contrib
        if do_center:
            if center_prop:
                log_market = _center_proposal(log_market, prior, n_groups, n_alts)
            else:
                lm = log_market.reshape(n_groups, n_alts)
                log_market = (lm - jnp.mean(lm, axis=1, keepdims=True)).reshape(-1)

        V = u + log_h + log_w + log_market - log_prior
        V_obs, lse = jgroup_logsumexp(V, n_groups, n_alts, _ac)
        per = V_obs - lse                 # per-group positive log-likelihood
        if per_group:
            return per                    # vector (n_groups,) — jacrev => scores
        return -jnp.sum(per)

    return jax.jit(neg_ll), pidx


def build_joint_neg_ll(spec, data_sm, data_sf, data_cou, gender_split=None):
    """joint negLL(theta) = singles_male + singles_female + couples, one shared
    theta-vector. Each sub-builder is the machine-precision-validated JAX fn.

    gender_split: optional set of hours-shifter base coef names to relax
    male-vs-female (coef -> coef_m / coef_f on the respective legs); default
    None -> all callers unchanged (baseline path byte-identical)."""
    _load_jax()
    f_sm, _ = build_jax_singles_ll(data_sm, spec, is_male=True,
                                   gender_split=gender_split)
    f_sf, _ = build_jax_singles_ll(data_sf, spec, is_male=False,
                                   gender_split=gender_split)
    f_cou, _ = build_jax_couples_ll(data_cou, spec, gender_split=gender_split)

    def joint(theta):
        return f_sm(theta) + f_sf(theta) + f_cou(theta)

    return jax.jit(joint)
