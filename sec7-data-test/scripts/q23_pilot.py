"""PREREG_Q23.md: predict N_cv at m=32 and m=all from a 16-trial pilot (shrunk and plug-in seeds). Seeds 701, 702."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
PILOT = 16
SEEDS = (701, 702)
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


def pilot_seeds(groups):
    pil = [g[rng.permutation(len(g))[: min(PILOT, len(g))]] for g in groups]
    mu = np.array([p.mean() for p in pil])
    v = np.array([p.var(ddof=1) if len(p) > 1 else p.mean() for p in pil])
    n = np.array([len(p) for p in pil])
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    mt = np.maximum(gm + (mu - gm) * shr, 0.0)
    nfull = [len(g) for g in groups]
    return mu, mt, v, nfull


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2", True),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", False),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", False),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", False),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05", False),
]
MS = (32, 10**6)
res = {}
t0 = time.time()
for name, cache, resfile, W, q8r in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W, q8r)
    D = []
    S = []
    P = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d = []
        s_ = []
        p_ = []
        for g in units:
            mu, mt, v, nfull = pilot_seeds(g)
            gs = [nb(mt[i], v[i], nfull[i]) for i in range(len(g))]
            gp = [nb(mu[i], v[i], nfull[i]) for i in range(len(g))]
            d.append([ncv(g, m) for m in MS])
            s_.append([ncv(gs, m) for m in MS])
            p_.append([ncv(gp, m) for m in MS])
        D.append(np.array(d, float).mean(0))
        S.append(np.array(s_, float).mean(0))
        P.append(np.array(p_, float).mean(0))
    D = np.array(D)
    S = np.array(S)
    P = np.array(P)
    res[name] = {
        "n": len(units),
        "data": D.mean(0).tolist(),
        "pred_shrunk": S.mean(0).tolist(),
        "pred_plugin": P.mean(0).tolist(),
        "err_shrunk": (S - D).mean(0).tolist(),
        "err_plugin": (P - D).mean(0).tolist(),
        "sd_shrunk": (S - D).std(0, ddof=1).tolist(),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s] m=32/all: data {np.round(D.mean(0),2).tolist()} | pilot-16 shrunk prediction {np.round(S.mean(0),2).tolist()} (err {np.round((S-D).mean(0),2).tolist()}) | plug-in {np.round(P.mean(0),2).tolist()} (err {np.round((P-D).mean(0),2).tolist()})",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q23.json"), "w"), indent=1)
print("[saved]")
