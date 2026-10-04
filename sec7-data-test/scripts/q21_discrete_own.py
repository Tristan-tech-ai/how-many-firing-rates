"""PREREG_Q21.md: discrete-own (N* levels on the unit's shrunken map) vs continuous-own (shrunken map), matched noise. Seeds 501-503."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (501, 502, 503)
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
        units.append(([c[tt == cc] for cc in conds], int(r.get("Nstar", 2))))
    return units


def nb(mu, var, n):
    mu = max(mu, 1e-9)
    if var > mu * 1.02:
        return rng.negative_binomial(mu * mu / (var - mu), mu / var, n).astype(float)
    return rng.poisson(mu, n).astype(float)


def shrunk(groups):
    mu = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) if len(g) > 1 else g.mean() for g in groups])
    n = np.array([len(g) for g in groups])
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    return np.maximum(gm + (mu - gm) * shr, 0.0), v, n


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2", True),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", False),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", False),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", False),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05", False),
]
res = {}
t0 = time.time()
for name, cache, resfile, W, q8r in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W, q8r)
    D = []
    C = []
    Q = []
    near_all = []
    near_32 = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d = []
        c = []
        q = []
        for groups, nstar in units:
            mt, v, n = shrunk(groups)
            md = quant(mt, max(nstar, 1))
            gc = [nb(mt[i], v[i], n[i]) for i in range(len(groups))]
            gd = [nb(md[i], v[i], n[i]) for i in range(len(groups))]
            d.append([ncv(groups, m) for m in MS])
            c.append([ncv(gc, m) for m in MS])
            q.append([ncv(gd, m) for m in MS])
        d = np.array(d, float)
        c = np.array(c, float)
        q = np.array(q, float)
        D.append(d.mean(0))
        C.append(c.mean(0))
        Q.append(q.mean(0))
        for j, store in ((3, near_all), (2, near_32)):
            dc = np.abs(c[:, j] - d[:, j])
            dq = np.abs(q[:, j] - d[:, j])
            store.append(float(np.mean(dc < dq) + 0.5 * np.mean(dc == dq)))
    D = np.array(D).mean(0)
    C = np.array(C).mean(0)
    Q = np.array(Q).mean(0)
    nstars = np.array([u[1] for u in units])
    res[name] = {
        "n": len(units),
        "Nstar_median": float(np.median(nstars)),
        "data": D.tolist(),
        "cont_own": C.tolist(),
        "disc_own": Q.tolist(),
        "near_cont_all": float(np.mean(near_all)),
        "near_cont_32": float(np.mean(near_32)),
    }
    print(
        f"{name} (n={len(units)}, N* median {np.median(nstars):.0f}) [{time.time()-t0:.0f}s] m=8/16/32/all: data {np.round(D,2).tolist()} | continuous-own {np.round(C,2).tolist()} | discrete-own {np.round(Q,2).tolist()} | |cont-data| at 32/all {abs(C[2]-D[2]):.2f}/{abs(C[3]-D[3]):.2f} vs |disc-data| {abs(Q[2]-D[2]):.2f}/{abs(Q[3]-D[3]):.2f} | units nearer continuous: {100*np.mean(near_all):.0f}% (all), {100*np.mean(near_32):.0f}% (32)",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q21.json"), "w"), indent=1)
print("[saved]")
