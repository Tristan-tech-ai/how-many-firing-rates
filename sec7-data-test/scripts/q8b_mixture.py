"""
Q8b per PREREG_Q8b.md: within-condition mixture test on powered (unit, W) pairs from results_q8.json.
M_mix: NB mixture at the CERTIFIED levels (shared), pi_c free.  M_single: NB(mu_c, F_u).
3-fold CV within condition; Delta_u = mean over conditions of per-trial held-out LL difference.
K4 power: simulate discrete-smoothed and smooth for each pair; powered if Delta>0 for (i) and <=0 for (ii)
in >= 80% of NSIM. Deterministic (seed 1). Run: py -u q8b_mixture.py > q8b_log.txt
"""

import numpy as np, json, os, sys, time
from scipy.stats import nbinom, poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "..", "mc_maze_test", "mc_maze_cache.npz"), allow_pickle=True)
sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
ends = np.append(sidx[1:], len(sp))
conds = np.unique(tt)
NC = len(conds)
R8 = json.load(open(os.path.join(HERE, "results_q8.json")))
FLOOR = 1e-2
FOLDS = 3
NSIM = 5
EM_IT = 100
rng = np.random.default_rng(1)


def counts_for(k, W):
    s = sp[sidx[k] : ends[k]]
    c = np.searchsorted(s, mo + W) - np.searchsorted(s, mo)
    return [c[tt == t].astype(float) for t in conds]


def logpmf(x, mu, F):
    mu = max(mu, FLOOR)
    if F <= 1.05:
        return poisson.logpmf(x, mu)
    return nbinom.logpmf(x, mu / (F - 1.0), 1.0 / F)


def fit_eval(train, test, levels, F):
    """returns (LL_mix_per_trial, LL_single_per_trial) on test."""
    Lp = np.column_stack([logpmf(train, L, F) for L in levels])
    pi = np.ones(len(levels)) / len(levels)
    for _ in range(EM_IT):
        lw = Lp + np.log(pi)[None, :]
        lw -= lw.max(1, keepdims=True)
        r = np.exp(lw)
        r /= r.sum(1, keepdims=True)
        pn = r.mean(0)
        if np.abs(pn - pi).max() < 1e-10:
            pi = pn
            break
        pi = pn
    Lt = np.column_stack([logpmf(test, L, F) for L in levels]) + np.log(np.maximum(pi, 1e-300))[None, :]
    m = Lt.max(1)
    ll_mix = m + np.log(np.exp(Lt - m[:, None]).sum(1))
    ll_single = logpmf(test, train.mean(), F)
    return ll_mix, ll_single


def delta_for(groups, levels, F, rng):
    d = []
    for g in groups:
        n = len(g)
        if n < 6:
            continue
        idx = rng.permutation(n)
        folds = np.array_split(idx, FOLDS)
        dd = []
        for f in folds:
            te = g[f]
            tr = g[np.setdiff1d(idx, f)]
            a, b = fit_eval(tr, te, levels, F)
            dd.append(float((a - b).mean()))
        d.append(np.mean(dd))
    return float(np.mean(d)) if d else float("nan")


def sim_counts(mus, ns, F, rng):
    out = []
    for mu, n in zip(mus, ns):
        mu = max(mu, 1e-9)
        out.append(
            rng.poisson(mu, n).astype(float)
            if F <= 1.05
            else rng.negative_binomial(mu / (F - 1.0), 1.0 / F, n).astype(float)
        )
    return out


pairs = []
for u in R8["units"]:
    for W, r in u["W"].items():
        if r.get("powered"):
            pairs.append((u["unit"], float(W), r))
print(f"powered pairs from Q8: {len(pairs)}")
t0 = time.time()
rows = []
for i, (k, W, r) in enumerate(pairs):
    groups = counts_for(k, W)
    ns = [len(g) for g in groups]
    F = r["fano"]
    levels = [p + r["lam"] for p in r["Nstar_pts"]]
    d_data = delta_for(groups, levels, F, rng)
    # K4 power
    ok_i, ok_ii = 0, 0
    for s in range(NSIM):
        # (i) discrete-smoothed: per trial a level drawn with condition-specific Dirichlet weights
        gi = []
        for n in ns:
            pi = rng.dirichlet(np.ones(len(levels)))
            lv = np.array(levels)[rng.choice(len(levels), n, p=pi)]
            gi.append(np.concatenate([sim_counts([l], [1], F, rng)[0] for l in lv]) if n else np.zeros(0))
        gii = sim_counts([g.mean() for g in groups], ns, F, rng)
        ok_i += delta_for(gi, levels, F, rng) > 0
        ok_ii += delta_for(gii, levels, F, rng) <= 0
    powered = (ok_i >= 0.8 * NSIM) and (ok_ii >= 0.8 * NSIM)
    rows.append(
        {
            "unit": k,
            "W": W,
            "A": r["A"],
            "lam": r["lam"],
            "fano": F,
            "Nstar": r["Nstar"],
            "delta": d_data,
            "K4_i": ok_i / NSIM,
            "K4_ii": ok_ii / NSIM,
            "powered_b": bool(powered),
        }
    )
    if i % 20 == 0:
        print(
            f"pair {i:3d}/{len(pairs)} [{time.time()-t0:.0f}s] unit {k} W{int(W*1000)} A={r['A']:.2f} N*={r['Nstar']} delta={d_data:+.4f} K4=({ok_i}/{NSIM},{ok_ii}/{NSIM}) powered_b={powered}",
            flush=True,
        )
pw = [x for x in rows if x["powered_b"]]
frac_open = float(np.mean([x["delta"] > 0 for x in pw])) if pw else float("nan")
units_pw = len(set(x["unit"] for x in pw))
if units_pw < 20:
    v = f"Q8b-UNDERPOWERED ({units_pw} powered units)"
elif frac_open >= 0.7:
    v = "HATCH OPEN — within-condition mixtures at the certified levels beat a single NB in >= 70% of powered units"
elif frac_open <= 0.3:
    v = "HATCH CLOSED — single NB beats the certified-level mixture in >= 70% of powered units; the smooth tuning is not a smoothed discrete code"
else:
    v = "Q8b-MIXED"
print(
    f"\npowered_b pairs {len(pw)} over {units_pw} units; Delta>0 in {100*frac_open:.0f}%; median Delta {np.median([x['delta'] for x in pw]) if pw else float('nan'):+.4f} nats/trial"
)
print(f"VERDICT: {v}")
json.dump(
    {"rows": rows, "verdict": v, "frac_open": frac_open, "units_powered": units_pw},
    open(os.path.join(HERE, "results_q8b.json"), "w"),
    indent=1,
)
print("[saved] results_q8b.json")
