"""Q17 instrument check: seed spread of the mean N_cv curves (data and per-condition matched independent sim).
Same code as q15_matched.py; 5 seeds. Usage: py q17_seeds.py"""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (101, 102, 103, 104, 105)
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
    starts = sidx
    ends = np.append(sidx[1:], len(sp))
    R_ = json.load(open(resfile))
    conds = np.unique(tt)
    units = []  # Q8-consistent reading (documented unit shift)
    for u in R_["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = (np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)).astype(float)
        if ends[k] > starts[k]:
            units.append([c[tt == cc] for cc in conds])
    return units


def nb(mu, var, n):
    mu = max(mu, 1e-9)
    if var > mu * 1.02:
        return rng.negative_binomial(mu * mu / (var - mu), mu / var, n).astype(float)
    return rng.poisson(mu, n).astype(float)


def sim_matched(groups):
    return [nb(g.mean(), g.var(ddof=1) if len(g) > 1 else g.mean(), len(g)) for g in groups]


DS = [("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2")]

res = {}
t0 = time.time()
for name, cache, resfile, W in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    Dcur = []
    Mcur = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        Nd = np.array([[ncv(g, m) for m in MS] for g in units], float)
        Nm = np.array([[ncv(sim_matched(g), m) for m in MS] for g in units], float)
        Dcur.append(Nd.mean(0))
        Mcur.append(Nm.mean(0))
    Dcur = np.array(Dcur)
    Mcur = np.array(Mcur)
    diff = Mcur - Dcur
    res[name] = {
        "n": len(units),
        "data_mean": Dcur.mean(0).tolist(),
        "data_sd": Dcur.std(0, ddof=1).tolist(),
        "matched_mean": Mcur.mean(0).tolist(),
        "matched_sd": Mcur.std(0, ddof=1).tolist(),
        "diff_mean": diff.mean(0).tolist(),
        "diff_sd": diff.std(0, ddof=1).tolist(),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s] m=8/16/32/all: data {np.round(Dcur.mean(0),2).tolist()} sd {np.round(Dcur.std(0,ddof=1),2).tolist()} | matched {np.round(Mcur.mean(0),2).tolist()} sd {np.round(Mcur.std(0,ddof=1),2).tolist()} | matched-data {np.round(diff.mean(0),2).tolist()} sd {np.round(diff.std(0,ddof=1),2).tolist()}",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q17_m1fix.json"), "w"), indent=1)
print("[saved]")
