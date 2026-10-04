"""
Q8 test per PREREG_Q8.md: K1 shuffle control -> K2 power simulation -> data N_cv(u, W, m) -> verdict.
Deterministic (seed 0). Run: py q8_levels.py   (writes results_q8.json, progress to stdout)
"""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
from support_exact import support

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "..", "mc_maze_test", "mc_maze_cache.npz"), allow_pickle=True)
sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
ends = np.append(sidx[1:], len(sp))
conds = np.unique(tt)
NC = len(conds)
WS = (0.1, 0.2, 0.4)
MS = (8, 16, 32, 10**6)
NMAX = 8
R_DATA = 20
R_SIM = 6
NSIM = 8
FLOOR = 1e-2
rng = np.random.default_rng(0)


def counts_for(k, W):
    s = sp[sidx[k] : ends[k]]
    c = np.searchsorted(s, mo + W) - np.searchsorted(s, mo)
    return [c[tt == t].astype(float) for t in conds]


def quant(means, N, iters=40):
    if N >= len(np.unique(means)):
        return means.copy()
    cen = np.quantile(means, (np.arange(N) + 0.5) / N)
    for _ in range(iters):
        a = np.abs(means[:, None] - cen[None, :]).argmin(1)
        cnt = np.bincount(a, minlength=N)
        sm = np.bincount(a, weights=means, minlength=N)
        new = np.where(cnt > 0, sm / np.maximum(cnt, 1), cen)
        if np.allclose(new, cen):
            break
        cen = new
    a = np.abs(means[:, None] - cen[None, :]).argmin(1)
    return cen[a]


def ncv(groups, m, R, rng):
    """N_cv and held-out LL curve for one unit-window: groups = per-condition count arrays."""
    LL = np.zeros(NMAX + 1)  # index 0 = free model
    for _ in range(R):
        means = np.zeros(NC)
        S = np.zeros(NC)
        n_te = np.zeros(NC)
        for i, g in enumerate(groups):
            n = min(m, len(g))
            idx = rng.permutation(len(g))[:n]
            h = n // 2
            tr = g[idx[:h]]
            te = g[idx[h:]]
            means[i] = tr.mean() if h else 0.0
            S[i] = te.sum()
            n_te[i] = len(te)

        def ll(rates):
            r = np.maximum(rates, FLOOR)
            return float((S * np.log(r) - n_te * r).sum())

        LL[0] += ll(means)
        for N in range(1, NMAX + 1):
            LL[N] += ll(quant(means, N))
    LL /= R
    return int(np.argmax(LL[1:]) + 1), LL


def simulate(levels_per_cond, ncounts, fano, rng):
    out = []
    for mu, n in zip(levels_per_cond, ncounts):
        mu = max(mu, 1e-9)
        if fano <= 1.05:
            out.append(rng.poisson(mu, n).astype(float))
        else:
            r = mu / (fano - 1.0)
            p = 1.0 / fano
            out.append(rng.negative_binomial(r, p, n).astype(float))
    return out


res = {
    "config": {
        "W": WS,
        "m": MS,
        "NMAX": NMAX,
        "R_DATA": R_DATA,
        "R_SIM": R_SIM,
        "NSIM": NSIM,
        "FLOOR": FLOOR,
    },
    "units": [],
}
t0 = time.time()
k1_pass = 0
k1_tot = 0
for k in range(182):
    rec = {"unit": k, "W": {}}
    for W in WS:
        groups = counts_for(k, W)
        ncounts = [len(g) for g in groups]
        means = np.array([g.mean() for g in groups])
        lam = float(means.min())
        A = float(means.max() - lam)
        ok = means > 0.1
        fano = (
            float(np.mean([g.var(ddof=1) / g.mean() for g in groups if g.mean() > 0.1])) if ok.any() else 1.0
        )
        row = {"lam": lam, "A": A, "fano": fano}
        if A < 0.05:
            row["skip"] = "A<0.05"
            rec["W"][str(W)] = row
            continue
        # K1 shuffle
        allc = np.concatenate(groups)
        perm = rng.permutation(len(allc))
        cuts = np.cumsum([0] + ncounts)
        shuf = [allc[perm[cuts[i] : cuts[i + 1]]] for i in range(NC)]
        n_sh, _ = ncv(shuf, 10**6, 10, rng)
        row["K1_shuffle_Ncv"] = n_sh
        k1_tot += 1
        k1_pass += n_sh == 1
        # N* and K2 power
        S = support(A, lam)
        row["Nstar"] = S["N"]
        row["Nstar_pts"] = S["pts"]
        nlev = np.maximum(1, np.round(np.array(S["w"]) * NC)).astype(int)
        while nlev.sum() > NC:
            nlev[nlev.argmax()] -= 1
        while nlev.sum() < NC:
            nlev[nlev.argmax()] += 1
        disc_levels = np.repeat(np.array(S["pts"]) + lam, nlev)
        sim = {"disc": [], "smooth": []}
        for s in range(NSIM):
            lv = rng.permutation(disc_levels)
            sm = lam + A * rng.random(NC)
            for tag, levels in (("disc", lv), ("smooth", sm)):
                g = simulate(levels, ncounts, fano, rng)
                sim[tag].append([ncv(g, m, R_SIM, rng)[0] for m in MS])
        d = np.array(sim["disc"])
        s_ = np.array(sim["smooth"])
        sep = float(np.mean(np.abs(d[:, -1] - s_[:, -1]) >= 1))
        row["K2_sep_frac"] = sep
        row["powered"] = bool(sep >= 0.8)
        row["sim_disc_Ncv_by_m"] = d.mean(0).tolist()
        row["sim_smooth_Ncv_by_m"] = s_.mean(0).tolist()
        # data
        curve = []
        LLs = []
        for m in MS:
            n_cv, LL = ncv(groups, m, R_DATA, rng)
            curve.append(n_cv)
            LLs.append(LL.tolist())
        row["Ncv_by_m"] = curve
        row["LL_by_m"] = LLs
        rec["W"][str(W)] = row
    res["units"].append(rec)
    if k % 10 == 0:
        print(
            f"unit {k:3d}  [{time.time()-t0:.0f}s]  K1 pass {k1_pass}/{k1_tot}   "
            + "  ".join(
                f"W{int(float(W)*1000)}: A={rec['W'][str(W)].get('A',0):.2f} N*={rec['W'][str(W)].get('Nstar','-')} pow={rec['W'][str(W)].get('powered','-')} Ncv={rec['W'][str(W)].get('Ncv_by_m','-')}"
                for W in WS
            ),
            flush=True,
        )
res["K1"] = {"pass": k1_pass, "total": k1_tot}
json.dump(res, open(os.path.join(HERE, "results_q8.json"), "w"), indent=1)
print(f"\n[saved] results_q8.json   K1 shuffle: N_cv==1 in {k1_pass}/{k1_tot}   [{time.time()-t0:.0f}s]")
