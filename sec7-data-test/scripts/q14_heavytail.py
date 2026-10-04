"""PREREG_Q14.md: heavy-tail diagnostic. (a) kurtosis data vs NB sim; (b) mean vs trimmed-mean N_cv, paired. Seed 21."""

import numpy as np, json, os, sys, time
from scipy.stats import kurtosis

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(21)
NMAX = 16
R = 40
FLOOR = 1e-2
MS = (8, 16, 32)


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


def tmean(x, cut=0.25):
    x = np.sort(x)
    k = int(np.floor(cut * len(x)))
    return x[k : len(x) - k].mean() if len(x) - 2 * k > 0 else x.mean()


def ncv_pair(groups, m):
    """returns (N_cv with mean, N_cv with trimmed mean) on the same subsamples."""
    nc = len(groups)
    LLa = np.zeros(NMAX + 1)
    LLb = np.zeros(NMAX + 1)
    for _ in range(R):
        ma = np.zeros(nc)
        mb = np.zeros(nc)
        S = np.zeros(nc)
        n_te = np.zeros(nc)
        for i, g in enumerate(groups):
            n = min(m, len(g))
            idx = rng.permutation(len(g))[:n]
            h = n // 2
            tr = g[idx[:h]]
            te = g[idx[h:]]
            ma[i] = tr.mean() if h else 0.0
            mb[i] = tmean(tr) if h else 0.0
            S[i] = te.sum()
            n_te[i] = len(te)

        def ll(r):
            r = np.maximum(r, FLOOR)
            return float((S * np.log(r) - n_te * r).sum())

        for N in range(1, NMAX + 1):
            LLa[N] += ll(quant(ma, N))
            LLb[N] += ll(quant(mb, N))
    return int(np.argmax(LLa[1:]) + 1), int(np.argmax(LLb[1:]) + 1)


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


def simulate(groups, fano):
    return [
        (
            rng.poisson(max(g.mean(), 1e-9), len(g))
            if fano <= 1.05
            else rng.negative_binomial(max(g.mean(), 1e-9) / (fano - 1), 1 / fano, len(g))
        ).astype(float)
        for g in groups
    ]


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2"),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25"),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05"),
]
res = {}
t0 = time.time()
for name, cache, resfile, W in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    Kd = []
    Ks = []
    Dd = []
    Ds = []
    for groups, fano in units:
        simg = simulate(groups, fano)
        kd = [kurtosis(g, fisher=True, bias=False) for g in groups if len(g) >= 16 and g.std() > 0]
        ks = [kurtosis(g, fisher=True, bias=False) for g in simg if len(g) >= 16 and g.std() > 0]
        if kd and ks:
            Kd.append(np.mean(kd))
            Ks.append(np.mean(ks))
        dd = []
        ds = []
        for m in MS:
            a, b = ncv_pair(groups, m)
            dd.append(b - a)
            a2, b2 = ncv_pair(simg, m)
            ds.append(b2 - a2)
        Dd.append(dd)
        Ds.append(ds)
    Kd = np.array(Kd)
    Ks = np.array(Ks)
    Dd = np.array(Dd, float)
    Ds = np.array(Ds, float)
    D = Dd.mean(0) - Ds.mean(0)
    res[name] = {
        "n": len(units),
        "frac_kurt_above": float(np.mean(Kd > Ks)),
        "kurt_data_median": float(np.median(Kd)),
        "kurt_sim_median": float(np.median(Ks)),
        "trim_minus_mean_data": Dd.mean(0).tolist(),
        "trim_minus_mean_sim": Ds.mean(0).tolist(),
        "D": D.tolist(),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s]: (a) kurtosis data>sim in {100*np.mean(Kd>Ks):.0f}% of units (medians data {np.median(Kd):.2f} vs sim {np.median(Ks):.2f});  (b) trimmed−mean N_cv at m=8/16/32: data {np.round(Dd.mean(0),2).tolist()} control {np.round(Ds.mean(0),2).tolist()} -> D {np.round(D,2).tolist()}",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q14.json"), "w"), indent=1)
print("[saved]")
