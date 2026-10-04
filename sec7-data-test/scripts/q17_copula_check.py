"""Control: is the Q16 copula sampler (a = rho = 0) distributionally the same as the Q15/Q17 numpy sampler?
Area 2, 3 seeds each, mean N_cv curve. Seeds 301-303."""

import numpy as np, json, os, sys, time
from scipy.stats import norm, nbinom, poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (301, 302, 303)
rng = None


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

        for N in range(1, NMAX + 1):
            LL[N] += ll(quant(means, N))
    return int(np.argmax(LL[1:]) + 1)


def load(cache, resfile, W):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
    ends = np.append(starts[1:], len(sp))
    R_ = json.load(open(resfile))
    conds = np.unique(tt)
    units = []
    for u in R_["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = (np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)).astype(float)
        units.append([c[tt == cc] for cc in conds])
    return units


def sim_numpy(groups):
    out = []
    for g in groups:
        mu = max(g.mean(), 1e-9)
        var = g.var(ddof=1) if len(g) > 1 else mu
        out.append(
            rng.negative_binomial(mu * mu / (var - mu), mu / var, len(g)).astype(float)
            if var > mu * 1.02
            else rng.poisson(mu, len(g)).astype(float)
        )
    return out


def sim_copula(groups):
    out = []
    for g in groups:
        mu = max(g.mean(), 1e-9)
        var = g.var(ddof=1) if len(g) > 1 else mu
        u = np.clip(norm.cdf(rng.standard_normal(len(g))), 1e-9, 1 - 1e-9)
        out.append(nbinom.ppf(u, mu * mu / (var - mu), mu / var) if var > mu * 1.02 else poisson.ppf(u, mu))
    return out


units = load(
    os.path.join(HERE, "area2/area2_bump_cache.npz"), os.path.join(HERE, "results_q8_area2.json"), "0.2"
)
t0 = time.time()
A = []
B = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    A.append(np.array([[ncv(sim_numpy(g), m) for m in MS] for g in units], float).mean(0))
    B.append(np.array([[ncv(sim_copula(g), m) for m in MS] for g in units], float).mean(0))
A = np.array(A)
B = np.array(B)
print(
    f"area 2 (n={len(units)}) [{time.time()-t0:.0f}s] numpy sampler {np.round(A.mean(0),2).tolist()} sd {np.round(A.std(0,ddof=1),2).tolist()} | copula sampler {np.round(B.mean(0),2).tolist()} sd {np.round(B.std(0,ddof=1),2).tolist()}"
)
json.dump(
    {"numpy": A.tolist(), "copula": B.tolist()},
    open(os.path.join(HERE, "results_q17_copula.json"), "w"),
    indent=1,
)
