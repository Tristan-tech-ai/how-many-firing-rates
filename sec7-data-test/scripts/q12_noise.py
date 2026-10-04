"""PREREG_Q12.md: modulated-Poisson (gain-fluctuation) noise. Per unit sigma_G^2 from var_c - mu_c ~ sigma_G^2 mu_c^2.
P1: continuous (uniform-map) simulation growth exponents vs data. P2: transduction model (k = 0.75) N_cv curves vs data.
Seed 19. Usage: py q12_noise.py"""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(19)
K = 0.75
NMAX = 16
R = 10
FLOOR = 1e-2


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


def ncv(groups, m):
    nc = len(groups)
    LL = np.zeros(NMAX + 1)
    for _ in range(R):
        means = np.zeros(nc)
        S = np.zeros(nc)
        n_te = np.zeros(nc)
        for i, g in enumerate(groups):
            n = min(m, len(g))
            idx = rng.permutation(len(g))[:n]
            h = n // 2
            tr = g[idx[:h]]
            te = g[idx[h:]]
            means[i] = tr.mean() if h else 0.0
            S[i] = te.sum()
            n_te[i] = len(te)

        def ll(r):
            r = np.maximum(r, FLOOR)
            return float((S * np.log(r) - n_te * r).sum())

        LL[0] += ll(means)
        for N in range(1, NMAX + 1):
            LL[N] += ll(quant(means, N))
    return int(np.argmax(LL[1:] / R) + 1)


def g(x):
    return np.clip((np.exp(K * x) - np.exp(-K)) / (np.exp(K) - np.exp(-K)), 0, 1)


def mop_counts(rc, ncounts, sG2):
    """modulated Poisson: per trial gain G ~ Gamma(mean 1, var sG2); count ~ Poisson(G mu)."""
    out = []
    for mu, n in zip(rc, ncounts):
        mu = max(mu, 1e-9)
        G = rng.gamma(1.0 / sG2, sG2, n) if sG2 > 1e-6 else np.ones(n)
        out.append(rng.poisson(G * mu).astype(float))
    return out


def unit_stats(cache, resfile, W):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
    ends = np.append(starts[1:], len(sp))
    R_ = json.load(open(resfile))
    conds = np.unique(tt)
    out = []
    for u in R_["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)
        groups = [c[tt == cc].astype(float) for cc in conds]
        mu = np.array([gg.mean() for gg in groups])
        var = np.array([gg.var(ddof=1) for gg in groups])
        m = mu > 0.05
        sG2 = (
            float(max(0.0, np.sum((var[m] - mu[m]) * mu[m] ** 2) / max(np.sum(mu[m] ** 4), 1e-12)))
            if m.sum() >= 3
            else 0.0
        )
        out.append(
            {
                "A": r["A"],
                "lam": r["lam"],
                "fano": r["fano"],
                "sG2": sG2,
                "ncounts": [len(gg) for gg in groups],
                "Ncv_data": r["Ncv_by_m"],
                "sim_cont_nb": r["sim_smooth_Ncv_by_m"],
            }
        )
    return out


DS = [
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", (8, 16, 10**6), [8, 16, 22]),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", (8, 16, 10**6), [8, 16, 23]),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", (8, 16, 10**6), [8, 16, 49]),
    (
        "M1",
        "../mc_maze_test/mc_maze_cache.npz",
        "results_q8_rep.json",
        "0.2",
        (8, 16, 32, 10**6),
        [8, 16, 32, 64],
    ),
]
res = {}
t0 = time.time()
for name, cache, resfile, W, MS, mlab in DS:
    units = unit_stats(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    Nd = np.array([u["Ncv_data"] for u in units], float)
    Nc_nb = np.array([u["sim_cont_nb"] for u in units], float)
    Nc_mop, Nt_mop = [], []
    for u in units:
        nc = len(u["ncounts"])
        rc_cont = u["lam"] + u["A"] * rng.random(nc)
        Nc_mop.append([ncv(mop_counts(rc_cont, u["ncounts"], u["sG2"]), m) for m in MS])
        rc_tr = u["lam"] + u["A"] * g(rng.standard_normal(nc))
        Nt_mop.append([ncv(mop_counts(rc_tr, u["ncounts"], u["sG2"]), m) for m in MS])
    Nc_mop = np.array(Nc_mop, float)
    Nt_mop = np.array(Nt_mop, float)
    lm = np.log(mlab)
    gexp = lambda M: np.polyfit(lm, np.log(np.maximum(M.mean(0), 1)), 1)[0]
    A = np.array([u["A"] for u in units])
    aexp = lambda M: np.polyfit(np.log(A), np.log(np.maximum(M[:, -1], 1)), 1)[0]
    sG = np.array([u["sG2"] for u in units])
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s]  sigma_G^2 median {np.median(sG):.3f} (>0 in {100*np.mean(sG>0):.0f}%)"
    )
    print(
        f"   P1 growth exponent: data {gexp(Nd):+.2f}  continuous NB {gexp(Nc_nb):+.2f}  continuous MoP {gexp(Nc_mop):+.2f}   A-exponent: data {aexp(Nd):+.2f} cont-NB {aexp(Nc_nb):+.2f} cont-MoP {aexp(Nc_mop):+.2f}"
    )
    print(
        f"   P2 transduction+MoP N_cv curve {np.round(Nt_mop.mean(0),2).tolist()} vs data {np.round(Nd.mean(0),2).tolist()}; max |diff| {np.abs(Nt_mop.mean(0)-Nd.mean(0)).max():.2f}; A-exp model {aexp(Nt_mop):+.2f}; growth model {gexp(Nt_mop):+.2f}",
        flush=True,
    )
    res[name] = {
        "n": len(units),
        "sG2_median": float(np.median(sG)),
        "growth_data": float(gexp(Nd)),
        "growth_nb": float(gexp(Nc_nb)),
        "growth_mop": float(gexp(Nc_mop)),
        "curve_transd_mop": Nt_mop.mean(0).tolist(),
        "curve_data": Nd.mean(0).tolist(),
        "maxdiff": float(np.abs(Nt_mop.mean(0) - Nd.mean(0)).max()),
    }
json.dump(res, open(os.path.join(HERE, "results_q12.json"), "w"), indent=1)
print("[saved] results_q12.json")
