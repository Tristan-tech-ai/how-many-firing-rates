"""PREREG_Q21.md Amendment 1: own-map discrete vs continuous comparators for the retina data. Seeds 601, 602."""

import numpy as np, json, os, sys, time
import scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (601, 602)
NC = 36
WS = (1, 3, 6)
WNAME = {1: "0.0167", 3: "0.05", 6: "0.1"}
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


def draw(mu, var, n, bern):
    mu = max(mu, 1e-9)
    if bern:
        return (rng.random(n) < min(mu, 1.0)).astype(float)
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


M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
B = M["binned"]
nreps = M["nreps"].ravel().astype(int)
NCELL, NMOV, NBIN = B.shape[2], B.shape[3], B.shape[1]
Q8 = json.load(open(os.path.join(HERE, "results_q8_retina.json")))
flags = {(u["cell"], u["movie"]): u["W"] for u in Q8["units"]}
res = {}
t0 = time.time()
for w in WS:
    wn = WNAME[w]
    bern = w == 1
    D = []
    C = []
    Q = []
    near_all = []
    near_32 = []
    nunits = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        d = []
        c = []
        q = []
        for cell in range(NCELL):
            for mov in range(NMOV):
                row = flags.get((cell, mov), {}).get(wn, {})
                if not (row.get("powered") and "Ncv_by_m" in row and row["Ncv_by_m"][-1] >= 2):
                    continue
                nr = nreps[mov]
                X = B[:nr, :, cell, mov].astype(float)
                bins = np.sort(rng.choice(NBIN - 6, NC, replace=False))
                Cm = np.stack([X[:, b : b + w].sum(1) for b in bins], 1)
                groups = [Cm[:, i] for i in range(NC)]
                mt, v, n = shrunk(groups)
                mt = np.array([g.mean() for g in groups])
                md = quant(mt, max(int(row.get("Nstar", 2)), 1))
                gc = [draw(mt[i], v[i], n[i], bern) for i in range(NC)]
                gd = [draw(md[i], v[i], n[i], bern) for i in range(NC)]
                d.append([ncv(groups, m) for m in MS])
                c.append([ncv(gc, m) for m in MS])
                q.append([ncv(gd, m) for m in MS])
        d = np.array(d, float)
        c = np.array(c, float)
        q = np.array(q, float)
        nunits = len(d)
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
    res[wn] = {
        "n": nunits,
        "data": D.tolist(),
        "cont_own": C.tolist(),
        "disc_own": Q.tolist(),
        "near_cont_all": float(np.mean(near_all)),
        "near_cont_32": float(np.mean(near_32)),
    }
    print(
        f"retina W={wn}s (n={nunits}) [{time.time()-t0:.0f}s] m=8/16/32/all: data {np.round(D,2).tolist()} | continuous-own {np.round(C,2).tolist()} | discrete-own {np.round(Q,2).tolist()} | |cont-data| 32/all {abs(C[2]-D[2]):.2f}/{abs(C[3]-D[3]):.2f} vs |disc-data| {abs(Q[2]-D[2]):.2f}/{abs(Q[3]-D[3]):.2f} | units nearer continuous: {100*np.mean(near_all):.0f}% (all), {100*np.mean(near_32):.0f}% (32)",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q21_retina_plugin.json"), "w"), indent=1)
print("[saved]")
