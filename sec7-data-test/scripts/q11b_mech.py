"""PREREG_Q11.md: threshold-exponential transduction of Gaussian input; compare shape, N_cv curves, Q10 class with data. Seed 15."""

import numpy as np, json, os, sys, time
from scipy.stats import skew

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(16)
K_GAIN = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
DS_SEL = sys.argv[2].split(",") if len(sys.argv) > 2 else None
src = open(os.path.join(HERE, "q10_shape.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(HERE, "q10_shape.py")}
exec(src.split("HYPS = [")[0], ns)
ns["rng"] = rng
classify = ns["classify"]
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
    return np.clip((np.exp(K_GAIN * x) - np.exp(-K_GAIN)) / (np.exp(K_GAIN) - np.exp(-K_GAIN)), 0, 1)


def counts_from_cache(path):
    z = np.load(path, allow_pickle=True)
    tt = z["trial_type"]
    u, c = np.unique(tt, return_counts=True)
    return list(c)


DS = [
    (
        "M1",
        "results_q8_rep.json",
        (8, 16, 32, 10**6),
        counts_from_cache(os.path.join(HERE, "..", "mc_maze_test", "mc_maze_cache.npz")),
    ),
    (
        "Area2",
        "results_q8_area2.json",
        (8, 16, 10**6),
        counts_from_cache(os.path.join(HERE, "area2", "area2_bump_cache.npz")),
    ),
    (
        "DMFC",
        "results_q8_dmfc.json",
        (8, 16, 10**6),
        counts_from_cache(os.path.join(HERE, "area2", "dmfc_rsg_cache.npz")),
    ),
    (
        "V1",
        "results_q8_v1.json",
        (8, 16, 10**6),
        counts_from_cache(os.path.join(HERE, "v1", "v1_static36_cache.npz")),
    ),
]
out = {}
t0 = time.time()
for name, fn, MS, ncounts in [d for d in DS if DS_SEL is None or d[0] in DS_SEL]:
    Rj = json.load(open(os.path.join(HERE, fn)))
    rows = []
    for u in Rj["units"]:
        for W, r in u["W"].items():
            if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
                continue
            nc = len(ncounts)
            x = rng.standard_normal(nc)
            rc = r["lam"] + r["A"] * g(x)
            groups = [
                (
                    rng.poisson(max(mu, 1e-9), n)
                    if r["fano"] <= 1.05
                    else rng.negative_binomial(max(mu, 1e-9) / (r["fano"] - 1), 1 / r["fano"], n)
                ).astype(float)
                for mu, n in zip(rc, ncounts)
            ]
            m = np.array([gr.mean() for gr in groups])
            lowq = float(np.mean(m < m.min() + 0.25 * (m.max() - m.min())))
            rows.append(
                {
                    "W": W,
                    "A": r["A"],
                    "skew_rate": float(skew(m)),
                    "skew_log": float(skew(np.log(m + 0.05))),
                    "lowq": lowq,
                    "Ncv_model": [ncv(groups, mm) for mm in MS],
                    "Ncv_data": r["Ncv_by_m"],
                    "q10": classify(m)[0],
                }
            )
    out[name] = rows
    A = np.array([x["A"] for x in rows])
    Nm = np.array([x["Ncv_model"] for x in rows], float)
    Nd = np.array([x["Ncv_data"] for x in rows], float)
    sr = np.array([x["skew_rate"] for x in rows])
    sl = np.array([x["skew_log"] for x in rows])
    lq = np.array([x["lowq"] for x in rows])
    em = np.polyfit(np.log(A), np.log(np.maximum(Nm[:, -1], 1)), 1)[0]
    ed = np.polyfit(np.log(A), np.log(np.maximum(Nd[:, -1], 1)), 1)[0]
    gm = np.polyfit(np.log(MS[:-1] + (max(ncounts),)), np.log(np.maximum(Nm.mean(0), 1)), 1)[0]
    gd = np.polyfit(np.log(MS[:-1] + (max(ncounts),)), np.log(np.maximum(Nd.mean(0), 1)), 1)[0]
    q10 = {h: 100 * np.mean([x["q10"] == h for x in rows]) for h in ("sqrt", "rate", "log", "two")}
    print(
        f"{name} (n={len(rows)}) [{time.time()-t0:.0f}s]\n   P1 shape: rate skew>0 in {100*np.mean(sr>0):.0f}% (median {np.median(sr):+.2f}); log-skew median {np.median(sl):+.2f}; lowest-quarter {100*np.median(lq):.0f}% (data 44-48%)"
    )
    print(
        f"   P2 levels: model N_cv curve {np.round(Nm.mean(0),2).tolist()} vs data {np.round(Nd.mean(0),2).tolist()}; max |diff| {np.abs(Nm.mean(0)-Nd.mean(0)).max():.2f}; A-exponent model {em:+.2f} vs data {ed:+.2f}; growth exponent model {gm:+.2f} vs data {gd:+.2f}"
    )
    print(
        f"   P3 Q10 class of model maps: sqrt {q10['sqrt']:.0f}%  rate {q10['rate']:.0f}%  log {q10['log']:.0f}%  two {q10['two']:.0f}%",
        flush=True,
    )
json.dump(out, open(os.path.join(HERE, f"results_q11b_k{K_GAIN}.json"), "w"))
print("[saved]")
