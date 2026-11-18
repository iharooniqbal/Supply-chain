"""
Supply Chain Disruption & Financial Loss Simulator
==================================================
One perishable lot: supplier -> reefer transport -> cold storage -> sale.

Uncertainty
  * Transport delay : lognormal base delay + compound-Poisson heavy-tailed (Pareto) disruptions,
                      with a normal/disrupted regime that multiplies the disruption rate.
  * Temperature     : Ornstein-Uhlenbeck around the setpoint (exact discretisation) plus
                      reefer-failure jumps (temperature drifts toward ambient until repair).
  * Deterioration   : dlnQ = -k(T) dt, Arrhenius k(T). Spoiled if Q < q_spoil at sale.
  * Market price    : mean-reverting jump-diffusion on log-price (Schwartz / Merton style).

Outputs: expected loss, VaR/CVaR (empirical and GPD tail fit), avoided spoilage,
net benefit vs baseline (with paired CIs), break-even analysis, sensitivity.

Run:  python sc_simulator.py --paths 20000 --out results
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
from scipy import stats

R_GAS = 8.314


# --------------------------------------------------------------------------- parameters
@dataclass
class Params:
    # lot
    volume_t: float = 20.0           # tonnes
    price0: float = 4000.0           # $/tonne, spot at dispatch (also long-run mean)
    # transport delay (hours)
    planned_transit_h: float = 72.0
    d0_median_h: float = 6.0
    d0_sigma: float = 0.6
    lam_per_day: float = 0.15        # disruption rate, normal regime
    p_disrupted: float = 0.20        # prob. shipment is in disrupted regime
    regime_mult: float = 4.0         # rate multiplier in disrupted regime
    jump_alpha: float = 1.5          # Pareto tail index
    jump_xm_h: float = 12.0          # Pareto scale (min jump, hours)
    max_delay_h: float = 240.0       # truncation of total delay
    # temperature (deg C, per hour)
    t_set: float = 4.0
    theta: float = 0.5
    sigma_T: float = 0.6
    t_amb: float = 30.0
    theta_fail: float = 0.08
    fail_rate_per_h: float = 0.0004
    fail_mean_dur_h: float = 12.0
    # cold storage
    hold_h: float = 24.0
    t_store: float = 3.0
    theta_store: float = 0.5
    sigma_store: float = 0.2
    # quality
    q0: float = 0.90
    shelf_life_h: float = 240.0      # hours from Q=1 to q_spoil at t_ref
    q_spoil: float = 0.50
    t_ref: float = 4.0
    Ea: float = 70_000.0             # J/mol
    value_slope: float = 0.8         # price factor f(Q) = 1 - slope*(1-Q)
    salvage: float = 0.10            # fraction of spot price recovered if spoiled
    # price (log-price, per day)
    kappa_day: float = 0.10
    sigma_day: float = 0.04
    jump_rate_day: float = 0.05
    jump_mean: float = -0.02
    jump_sd: float = 0.12
    # contract
    late_pen_per_day: float = 0.015  # fraction of lot value per day late
    late_pen_cap: float = 0.30

    @property
    def k_ref(self) -> float:        # per hour at t_ref
        return np.log(1.0 / self.q_spoil) / self.shelf_life_h


@dataclass
class Policy:
    name: str
    buffer_h: float = 0.0
    buffer_cost_t_h: float = 0.8            # $/tonne/hour of early dispatch
    expedite_threshold_h: Optional[float] = None
    expedite_recovery: float = 0.6          # share of excess delay recovered
    expedite_cost: float = 0.0              # $ per lot when triggered
    cold_delta: float = 0.0                 # setpoint reduction, deg C
    energy_cost_c_t_h: float = 0.15         # $/tonne per deg C per transit hour
    dual: bool = False
    dual_premium_frac: float = 0.015        # extra freight, fraction of lot value
    route_b_extra_h: float = 12.0
    hedge_frac: float = 0.0
    hedge_discount: float = 0.01            # forward price = spot0 * (1 - discount)


def default_policies() -> list[Policy]:
    return [
        Policy("Baseline"),
        Policy("Safety buffer (36h)", buffer_h=36),
        Policy("Expedite if delay>36h", expedite_threshold_h=36, expedite_cost=2000),
        Policy("Colder setpoint (-2C)", cold_delta=2.0),
        Policy("Dual sourcing 50/50", dual=True),
        Policy("Forward-sell 50%", hedge_frac=0.5),
        Policy("Expedite + colder", expedite_threshold_h=36, expedite_cost=2000, cold_delta=2.0),
    ]


# --------------------------------------------------------------------------- random streams
class Stream:
    """Seeded RNG stream; optional antithetic pairing (second half mirrors the first).
    Using one stream per random source with a fixed number of draws per step gives
    common random numbers across policies."""

    def __init__(self, seed: int, antithetic: bool):
        self.rng = np.random.default_rng(seed)
        self.anti = antithetic

    def normal(self, n: int) -> np.ndarray:
        if self.anti:
            z = self.rng.standard_normal(n // 2)
            return np.concatenate([z, -z])
        return self.rng.standard_normal(n)

    def uniform(self, *shape: int) -> np.ndarray:
        n = shape[0]
        if self.anti:
            u = self.rng.random((n // 2,) + tuple(shape[1:]))
            return np.concatenate([u, 1.0 - u])
        return self.rng.random(shape)


# --------------------------------------------------------------------------- components
def draw_delay(p: Params, n: int, seed: int, anti: bool, K: int = 25) -> np.ndarray:
    """Total delay (hours) = lognormal base + compound Poisson(Pareto) disruptions."""
    s = Stream(seed, anti)
    d0 = p.d0_median_h * np.exp(p.d0_sigma * s.normal(n))
    disrupted = s.uniform(n) < p.p_disrupted
    lam = p.lam_per_day * (p.planned_transit_h / 24.0) * np.where(disrupted, p.regime_mult, 1.0)
    u_cnt = np.clip(s.uniform(n), 1e-12, 1 - 1e-12)
    n_j = np.minimum(stats.poisson.ppf(u_cnt, lam).astype(int), K)
    u_j = np.clip(s.uniform(n, K), 1e-12, 1 - 1e-12)
    jumps = p.jump_xm_h * (1.0 - u_j) ** (-1.0 / p.jump_alpha)
    mask = np.arange(K)[None, :] < n_j[:, None]
    return np.minimum(d0 + (jumps * mask).sum(axis=1), p.max_delay_h)


def simulate_lot(p: Params, t_set: float, arrival_h: np.ndarray, deadline_h: float,
                 seed_T: int, seed_price: int, anti: bool) -> dict:
    """Hourly simulation of temperature, quality and price for one lot."""
    n = len(arrival_h)
    sale_h = np.maximum(arrival_h, deadline_h) + p.hold_h
    sale_idx = np.ceil(sale_h).astype(int)
    n_steps = int(sale_idx.max())

    sT, sF, sP = Stream(seed_T, anti), Stream(seed_T + 1, anti), Stream(seed_price, anti)
    T = np.full(n, t_set)
    logQ = np.full(n, np.log(p.q0))
    fail_left = np.zeros(n)
    X = np.full(n, np.log(p.price0))
    X_sale = np.zeros(n)
    m_log = np.log(p.price0)
    dt_d = 1.0 / 24.0
    k_ref, Tref_K = p.k_ref, p.t_ref + 273.15
    max_T = t_set

    for t in range(n_steps):
        # ---- price (shared across policies by construction)
        zP = sP.normal(n)
        uJ = sP.uniform(n)
        zJ = sP.normal(n)
        X = (X + p.kappa_day * (m_log - X) * dt_d + p.sigma_day * np.sqrt(dt_d) * zP
             + (uJ < p.jump_rate_day * dt_d) * (p.jump_mean + p.jump_sd * zJ))

        # ---- reefer failure & temperature
        in_transit = t < arrival_h
        uF, uD = sF.uniform(n), sF.uniform(n)
        new_fail = in_transit & (fail_left <= 0) & (uF < p.fail_rate_per_h)
        fail_left = np.where(new_fail, -p.fail_mean_dur_h * np.log(np.clip(uD, 1e-12, 1.0)), fail_left)
        failing = in_transit & (fail_left > 0)
        mean_T = np.where(in_transit, np.where(failing, p.t_amb, t_set), p.t_store)
        th = np.where(in_transit, np.where(failing, p.theta_fail, p.theta), p.theta_store)
        sg = np.where(in_transit, p.sigma_T, p.sigma_store)
        a = np.exp(-th)
        sd = sg * np.sqrt((1 - a ** 2) / (2 * th))
        T = mean_T + (T - mean_T) * a + sd * sT.normal(n)
        fail_left = np.maximum(fail_left - 1.0, 0.0)

        # ---- deterioration (only until sale)
        active = t < sale_h
        k = k_ref * np.exp(p.Ea / R_GAS * (1.0 / Tref_K - 1.0 / (T + 273.15)))
        logQ = logQ - np.where(active, k, 0.0)
        max_T = max(max_T, float(T[in_transit].max()) if in_transit.any() else max_T)

        hit = sale_idx == t + 1
        X_sale[hit] = X[hit]

    return dict(Q=np.exp(logQ), price=np.exp(X_sale), arrival=arrival_h, sale=sale_h)


# --------------------------------------------------------------------------- policy run
def run_policy(p: Params, pol: Policy, n: int = 20000, seed: int = 12345,
               antithetic: bool = True) -> dict:
    if antithetic and n % 2:
        n += 1
    pe = replace(p, t_set=p.t_set - pol.cold_delta)
    deadline = pol.buffer_h + p.planned_transit_h
    lots = [(1.0 if not pol.dual else 0.5, 0.0, seed + 100, seed + 200)]
    if pol.dual:
        lots.append((0.5, pol.route_b_extra_h, seed + 300, seed + 400))

    P_forward = p.price0 * (1 - pol.hedge_discount)
    revenue = np.zeros(n)
    penalty = np.zeros(n)
    costs = np.zeros(n)
    spoiled_t = np.zeros(n)
    spoil_val = np.zeros(n)
    late_h = np.zeros(n)

    for w, extra_h, seed_delay, seed_T in lots:
        vol = w * p.volume_t
        delay = draw_delay(p, n, seed_delay, antithetic)
        if pol.expedite_threshold_h is not None:
            thr = pol.expedite_threshold_h
            trig = delay > thr
            delay = np.where(trig, thr + (1 - pol.expedite_recovery) * (delay - thr), delay)
            costs += w * pol.expedite_cost * trig
        arrival = p.planned_transit_h + extra_h + delay
        sim = simulate_lot(pe, pe.t_set, arrival, deadline, seed_T, seed + 500, antithetic)

        spoiled = sim["Q"] < p.q_spoil
        f = 1.0 - p.value_slope * (1.0 - sim["Q"])
        ok_rev = f * ((1 - pol.hedge_frac) * sim["price"] + pol.hedge_frac * P_forward)
        rev_per_t = np.where(spoiled, p.salvage * sim["price"], ok_rev)
        revenue += vol * rev_per_t

        late = np.maximum(0.0, sim["arrival"] - deadline)
        late_h += w * late
        frac = np.minimum(p.late_pen_cap, p.late_pen_per_day * late / 24.0)
        penalty += frac * vol * p.price0

        spoiled_t += vol * spoiled
        spoil_val += vol * spoiled * sim["price"] * (1 - p.salvage)
        costs += vol * pol.buffer_h * pol.buffer_cost_t_h
        costs += vol * pol.cold_delta * pol.energy_cost_c_t_h * sim["arrival"]
        if pol.dual:
            costs += w * pol.dual_premium_frac * p.volume_t * p.price0

    # reference = revenue under the nominal plan (no delay, no failures, spot price unchanged)
    q_plan = p.q0 * np.exp(-p.k_ref * (p.planned_transit_h + p.hold_h))
    ref_rev = p.volume_t * p.price0 * (1.0 - p.value_slope * (1.0 - q_plan))
    loss = ref_rev - revenue + penalty + costs
    return dict(loss=loss, spoiled_t=spoiled_t, spoil_val=spoil_val, late_h=late_h,
                policy_cost=costs, penalty=penalty, anti=antithetic, n=n)


# --------------------------------------------------------------------------- risk metrics
def _se(x: np.ndarray, anti: bool) -> float:
    if anti:
        h = len(x) // 2
        x = 0.5 * (x[:h] + x[h:])
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def var_cvar(x: np.ndarray, q: float) -> tuple[float, float]:
    v = np.quantile(x, q)
    return float(v), float(x[x >= v].mean())


def gpd_cvar(x: np.ndarray, q: float = 0.99, u_q: float = 0.90) -> tuple[float, float, float]:
    """Peaks-over-threshold GPD estimate of VaR_q and ES_q (McNeil-Frey). Returns (VaR, ES, xi)."""
    u = np.quantile(x, u_q)
    y = x[x > u] - u
    if len(y) < 50:
        return np.nan, np.nan, np.nan
    try:
        xi, _, beta = stats.genpareto.fit(y, floc=0)
        if xi >= 1:
            return np.nan, np.nan, float(xi)
        ratio = (1 - q) / (1 - u_q)
        var = u + beta / xi * (ratio ** (-xi) - 1) if abs(xi) > 1e-6 else u - beta * np.log(ratio)
        es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
        if not np.isfinite(es) or es > 1.1 * x.max():   # loss is bounded -> reject unstable fit
            return np.nan, np.nan, float(xi)
        return float(var), float(es), float(xi)
    except Exception:
        return np.nan, np.nan, np.nan


def summarize(res: dict, base: Optional[dict] = None) -> dict:
    L = res["loss"]
    v95, c95 = var_cvar(L, 0.95)
    v99, c99 = var_cvar(L, 0.99)
    _, c99g, xi = gpd_cvar(L, 0.99)
    out = dict(
        mean=L.mean(), se=_se(L, res["anti"]), VaR95=v95, CVaR95=c95, VaR99=v99, CVaR99=c99,
        CVaR99_GPD=c99g, xi=xi, spoil_prob=(res["spoiled_t"] > 0).mean(),
        spoiled_t=res["spoiled_t"].mean(), spoil_val=res["spoil_val"].mean(),
        late_h=res["late_h"].mean(), policy_cost=res["policy_cost"].mean(),
    )
    if base is not None:
        d = base["loss"] - L                      # paired difference thanks to CRN
        out["net_benefit"] = d.mean()
        out["net_benefit_se"] = _se(d, res["anti"])
        out["avoided_spoiled_t"] = base["spoiled_t"].mean() - res["spoiled_t"].mean()
        out["avoided_spoil_val"] = base["spoil_val"].mean() - res["spoil_val"].mean()
    return out


# --------------------------------------------------------------------------- analyses
def compare_policies(p: Params, policies: list[Policy], n: int, seed: int):
    results = {pol.name: run_policy(p, pol, n, seed) for pol in policies}
    base = results[policies[0].name]
    table = {name: summarize(r, None if name == policies[0].name else base)
             for name, r in results.items()}
    return results, table


def print_table(table: dict):
    hdr = (f"{'Policy':<26}{'E[loss]':>10}{'±95%CI':>8}{'VaR95':>10}{'CVaR95':>10}{'CVaR99':>10}"
           f"{'GPD99':>10}{'P(spoil)':>9}{'spoil t':>8}{'netBen':>9}{'±CI':>7}{'avoid $':>9}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in table.items():
        nb = r.get("net_benefit", float("nan"))
        nbs = 1.96 * r.get("net_benefit_se", float("nan"))
        av = r.get("avoided_spoil_val", float("nan"))
        gpd = f"{r['CVaR99_GPD']:,.0f}" if np.isfinite(r["CVaR99_GPD"]) else "n/a"
        print(f"{name:<26}{r['mean']:>10,.0f}{1.96 * r['se']:>8,.0f}{r['VaR95']:>10,.0f}"
              f"{r['CVaR95']:>10,.0f}{r['CVaR99']:>10,.0f}{gpd:>10}"
              f"{r['spoil_prob']:>9.3f}{r['spoiled_t']:>8.2f}{nb:>9,.0f}{nbs:>7,.0f}{av:>9,.0f}")


def breakeven_lambda(p: Params, pol: Policy, lams, n: int, seed: int):
    out = []
    for lam in lams:
        pl = replace(p, lam_per_day=lam)
        b = run_policy(pl, Policy("Baseline"), n, seed)
        r = run_policy(pl, pol, n, seed)
        d = b["loss"] - r["loss"]
        out.append((lam, d.mean(), 1.96 * _se(d, True),
                    var_cvar(b["loss"], 0.95)[1] - var_cvar(r["loss"], 0.95)[1]))
    return out


def sensitivity(p: Params, n: int, seed: int):
    base_s = summarize(run_policy(p, Policy("Baseline"), n, seed))
    specs = [("lam_per_day", 0.5, 1.5), ("jump_alpha", 1.25, 2.0), ("sigma_T", 0.5, 2.0),
             ("Ea", 0.8, 1.2), ("fail_rate_per_h", 0.5, 2.0), ("sigma_day", 0.5, 2.0),
             ("d0_sigma", 0.5, 1.5)]
    out = {m: (base_s[m], []) for m in ("mean", "CVaR95")}
    for name, lo, hi in specs:
        sm = []
        for mult in (lo, hi):
            v = getattr(p, name) * mult if name != "jump_alpha" else mult   # alpha given as absolute value
            sm.append(summarize(run_policy(replace(p, **{name: v}), Policy("Baseline"), n, seed)))
        for m in out:
            out[m][1].append((name, lo, hi, sm[0][m], sm[1][m]))
    return out


def check_analytical(p: Params) -> tuple[float, float]:
    """Sanity check: with no randomness and no disruptions, Q at sale must equal
    q0*exp(-k(T_set)*t_sale) analytically."""
    q = replace(p, d0_sigma=0.0, lam_per_day=0.0, sigma_T=0.0, sigma_store=0.0,
                fail_rate_per_h=0.0, t_store=p.t_set)
    n = 1000
    arrival = np.full(n, q.planned_transit_h + q.d0_median_h)
    sim = simulate_lot(q, q.t_set, arrival, q.planned_transit_h, 1, 2, False)
    t_sale = np.ceil(max(arrival[0], q.planned_transit_h) + q.hold_h)
    analytic = q.q0 * np.exp(-q.k_ref * t_sale)   # k(T_set)=k_ref because T_set=T_ref
    return float(sim["Q"].mean()), float(analytic)


# --------------------------------------------------------------------------- plots
def make_plots(out, p, policies, results, table, conv_loss, breakeven, sens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    names = list(results)
    base = results[names[0]]["loss"]

    # 1. loss histograms
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    hi = np.quantile(base, 0.999)
    bins = np.linspace(min(base.min(), -15000), hi, 70)
    for ax, nm in zip(axes.ravel(), names[1:]):
        ax.hist(base, bins=bins, alpha=0.45, color="grey", label="Baseline", density=True)
        ax.hist(results[nm]["loss"], bins=bins, alpha=0.55, color="tab:blue", label=nm, density=True)
        ax.axvline(table[names[0]]["CVaR95"], color="grey", ls="--", lw=1)
        ax.axvline(table[nm]["CVaR95"], color="tab:blue", ls="--", lw=1)
        ax.set_title(nm, fontsize=10)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.supxlabel("Loss per lot ($)  (dashed = CVaR95)")
    fig.suptitle("Loss distributions vs baseline (log density)")
    fig.tight_layout()
    fig.savefig(f"{out}/loss_histograms.png", dpi=130)
    plt.close(fig)

    # 2. risk metric bars
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names)); w = 0.2
    for i, (k, c) in enumerate([("mean", "tab:green"), ("VaR95", "tab:orange"),
                                ("CVaR95", "tab:red"), ("CVaR99", "darkred")]):
        ax.bar(x + (i - 1.5) * w, [table[n][k] for n in names], w, label=k, color=c)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("$ per lot"); ax.legend(); ax.set_title("Expected loss and tail risk by policy")
    fig.tight_layout(); fig.savefig(f"{out}/risk_metrics.png", dpi=130); plt.close(fig)

    # 3. convergence
    rng = np.random.default_rng(0)
    perm = rng.permutation(conv_loss)
    ns = np.unique(np.logspace(2.5, np.log10(len(perm)), 25).astype(int))
    means = np.array([perm[:k].mean() for k in ns])
    ses = np.array([perm[:k].std(ddof=1) / np.sqrt(k) for k in ns])
    cv = np.array([var_cvar(perm[:k], 0.95)[1] for k in ns])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ns, means); ax[0].fill_between(ns, means - 1.96 * ses, means + 1.96 * ses, alpha=0.25)
    ax[0].set_xscale("log"); ax[0].set_title("Baseline E[loss] (95% CI)"); ax[0].set_xlabel("paths")
    ax[1].plot(ns, cv, color="tab:red"); ax[1].set_xscale("log")
    ax[1].set_title("Baseline CVaR95"); ax[1].set_xlabel("paths")
    fig.tight_layout(); fig.savefig(f"{out}/convergence.png", dpi=130); plt.close(fig)

    # 4. break-even
    if breakeven:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for nm, rows in breakeven.items():
            lam = [r[0] for r in rows]; nb = np.array([r[1] for r in rows]); ci = np.array([r[2] for r in rows])
            ax.errorbar(lam, nb, yerr=ci, marker="o", capsize=3, label=f"{nm}: E[loss] saved")
            ax.plot(lam, [r[3] for r in rows], ls=":", marker="s", label=f"{nm}: CVaR95 reduction")
        ax.axhline(0, color="k", lw=1); ax.set_xlabel("disruption rate λ (per day)")
        ax.set_ylabel("net benefit vs baseline ($/lot)"); ax.legend(fontsize=8)
        ax.set_title("Break-even disruption rate")
        fig.tight_layout(); fig.savefig(f"{out}/breakeven.png", dpi=130); plt.close(fig)

    # 5. tornado (expected loss and CVaR95)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (metric, (base_val, rows)) in zip(axes, sens.items()):
        rows = sorted(rows, key=lambda r: -abs(r[4] - r[3]))
        for i, (nm, lo, hi_, v_lo, v_hi) in enumerate(rows[::-1]):
            ax.barh(i, v_lo - base_val, left=base_val, color="tab:blue", alpha=0.7)
            ax.barh(i, v_hi - base_val, left=base_val, color="tab:red", alpha=0.7)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
        ax.axvline(base_val, color="k", lw=1)
        ax.set_xlabel(f"Baseline {metric} ($/lot)")
    fig.suptitle("One-at-a-time sensitivity (blue = low, red = high parameter value)")
    fig.tight_layout(); fig.savefig(f"{out}/tornado.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast-extras", type=int, default=6000, help="paths for sensitivity / break-even runs")
    args = ap.parse_args()
    p = Params()
    pols = default_policies()
    t0 = time.time()

    q_sim, q_ana = check_analytical(p)
    print(f"Sanity check (no randomness): simulated Q = {q_sim:.5f}, analytic Q = {q_ana:.5f}\n")

    results, table = compare_policies(p, pols, args.paths, args.seed)
    print(f"{args.paths} paths per policy, common random numbers + antithetic variates\n")
    print_table(table)
    print(f"\nGPD tail index xi (baseline): {table['Baseline']['xi']:.3f}")

    lams = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]
    be = {pl.name: breakeven_lambda(p, pl, lams, args.fast_extras, args.seed)
          for pl in pols if pl.name in ("Safety buffer (36h)", "Expedite if delay>36h", "Dual sourcing 50/50")}
    print("\nBreak-even (net benefit in E[loss], $/lot) by disruption rate lambda:")
    for nm, rows in be.items():
        print(f"  {nm}: " + ", ".join(f"λ={r[0]:.2f}: {r[1]:+,.0f}" for r in rows))

    sens = sensitivity(p, args.fast_extras, args.seed)
    for metric, (bv, rows) in sens.items():
        print(f"\nSensitivity of baseline {metric} (base {bv:,.0f}):")
        for nm, lo, hi, vlo, vhi in sorted(rows, key=lambda r: -abs(r[4] - r[3])):
            print(f"  {nm:<16} low {lo:<5} -> {vlo:>9,.0f}   high {hi:<5} -> {vhi:>9,.0f}")

    make_plots(args.out, p, pols, results, table, results["Baseline"]["loss"], be, sens)

    import csv
    with open(f"{args.out}/policy_summary.csv", "w", newline="") as f:
        keys = sorted({k for r in table.values() for k in r})
        w = csv.writer(f); w.writerow(["policy"] + keys)
        for nm, r in table.items():
            w.writerow([nm] + [r.get(k, "") for k in keys])
    print(f"\nPlots + CSV written to ./{args.out}/   (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
