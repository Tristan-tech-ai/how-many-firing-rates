"""PREREG_Q19.md: shrinkage-seeded comparator vs data vs plug-in comparator. Seeds 401-403."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (401, 402, 403)
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


def load(cache, resfile, W, q8_reading):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    if q8_reading:
        starts = sidx
        ends = np.append(sidx[1:], len(sp))
    else:
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
        if ends[k] <= starts[k]:
            continue
        s = sp[starts[k] : ends[k]]
        c = (np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)).astype(float)
        units.append([c[tt == cc] for cc in conds])
    return units


def nb(mu, var, n):
    mu = max(mu, 1e-9)
    if var > mu * 1.02:
        return rng.negative_binomial(mu * mu / (var - mu), mu / var, n).astype(float)
    return rng.poisson(mu, n).astype(float)


def seeds_plugin(groups):
    return [(g.mean(), g.var(ddof=1) if len(g) > 1 else g.mean(), len(g)) for g in groups]


def seeds_shrunk(groups):
    mu = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) if len(g) > 1 else g.mean() for g in groups])
    n = np.array([len(g) for g in groups], float)
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    mt = gm + (mu - gm) * shr
    return [(max(mt[i], 0.0), v[i], int(n[i])) for i in range(len(groups))]


def simulate(seeds):
    return [nb(m, v, n) for m, v, n in seeds]


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2", True),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", False),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", False),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", False),
]
res = {}
t0 = time.time()
for name, cache, resfile, W, q8r in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W, q8r)
    D = []
    P = []
    S = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d = []
        p = []
        s = []
        for g in units:
            d.append([ncv(g, m) for m in MS])
            p.append([ncv(simulate(seeds_plugin(g)), m) for m in MS])
            s.append([ncv(simulate(seeds_shrunk(g)), m) for m in MS])
        D.append(np.array(d, float).mean(0))
        P.append(np.array(p, float).mean(0))
        S.append(np.array(s, float).mean(0))
    D = np.array(D)
    P = np.array(P)
    S = np.array(S)
    ds = S - D
    dp = P - D
    res[name] = {
        "n": len(units),
        "data": D.mean(0).tolist(),
        "shrunk": S.mean(0).tolist(),
        "plugin": P.mean(0).tolist(),
        "shrunk_minus_data": ds.mean(0).tolist(),
        "sd": ds.std(0, ddof=1).tolist(),
        "plugin_minus_data": dp.mean(0).tolist(),
        "sd_plugin": dp.std(0, ddof=1).tolist(),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s] m=8/16/32/all: data {np.round(D.mean(0),2).tolist()} | shrunk {np.round(S.mean(0),2).tolist()} -> shrunk-data {np.round(ds.mean(0),2).tolist()} sd {np.round(ds.std(0,ddof=1),2).tolist()} | plugin-data {np.round(dp.mean(0),2).tolist()} sd {np.round(dp.std(0,ddof=1),2).tolist()}",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q19.json"), "w"), indent=1)
print("[saved]")
