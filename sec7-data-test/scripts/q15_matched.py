"""PREREG_Q15.md: per-condition moment-matched noise vs unit-wide Fano vs data. Seed 22."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(22)
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)


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
        units.append(([c[tt == cc] for cc in conds], r["fano"]))
    return units


def nb(mu, var, n):
    mu = max(mu, 1e-9)
    if var > mu * 1.02:
        return rng.negative_binomial(mu * mu / (var - mu), mu / var, n).astype(float)
    return rng.poisson(mu, n).astype(float)


def sim_matched(groups):
    return [nb(g.mean(), g.var(ddof=1) if len(g) > 1 else g.mean(), len(g)) for g in groups]


def sim_unitfano(groups, fano):
    return [nb(g.mean(), fano * max(g.mean(), 1e-9), len(g)) for g in groups]


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2"),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25"),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05"),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2"),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2"),
]
res = {}
t0 = time.time()
for name, cache, resfile, W in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    Nd = []
    Nm = []
    Nf = []
    neff = []
    for groups, fano in units:
        gm = sim_matched(groups)
        gf = sim_unitfano(groups, fano)
        Nd.append([ncv(groups, m) for m in MS])
        Nm.append([ncv(gm, m) for m in MS])
        Nf.append([ncv(gf, m) for m in MS])
        neff.append([np.mean([min(m, len(g)) for g in groups]) for m in MS])
    Nd = np.array(Nd, float)
    Nm = np.array(Nm, float)
    Nf = np.array(Nf, float)
    ne = np.array(neff).mean(0)
    keep = np.concatenate([[True], np.diff(ne) > 0.5])
    lm = np.log(ne[keep])
    gexp = lambda M: float(np.polyfit(lm, np.log(np.maximum(M.mean(0)[keep], 1)), 1)[0])
    res[name] = {
        "n": len(units),
        "n_eff": ne.tolist(),
        "data": Nd.mean(0).tolist(),
        "matched": Nm.mean(0).tolist(),
        "unitfano": Nf.mean(0).tolist(),
        "maxdiff_matched": float(np.abs(Nm.mean(0) - Nd.mean(0)).max()),
        "maxdiff_unitfano": float(np.abs(Nf.mean(0) - Nd.mean(0)).max()),
        "gexp_data": gexp(Nd),
        "gexp_matched": gexp(Nm),
        "gexp_unitfano": gexp(Nf),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s] n_eff {np.round(ne,1).tolist()}: N_cv data {np.round(Nd.mean(0),2).tolist()} | matched {np.round(Nm.mean(0),2).tolist()} (max diff {np.abs(Nm.mean(0)-Nd.mean(0)).max():.2f}) | unit-Fano {np.round(Nf.mean(0),2).tolist()} (max diff {np.abs(Nf.mean(0)-Nd.mean(0)).max():.2f}); growth exp data {gexp(Nd):+.2f} matched {gexp(Nm):+.2f} unit-Fano {gexp(Nf):+.2f}",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q15.json"), "w"), indent=1)
print("[saved]")
