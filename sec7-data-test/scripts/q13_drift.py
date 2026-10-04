"""PREREG_Q13.md: contiguous vs random trial subsampling. Seed 20."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20)
NMAX = 16
R = 20
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


def ncv(groups, m, contiguous):
    """groups: per-condition count arrays in SESSION ORDER."""
    nc = len(groups)
    LL = np.zeros(NMAX + 1)
    for _ in range(R):
        means = np.zeros(nc)
        S = np.zeros(nc)
        n_te = np.zeros(nc)
        for i, g in enumerate(groups):
            n = min(m, len(g))
            if contiguous:
                off = rng.integers(0, len(g) - n + 1)
                sel = np.arange(off, off + n)
                sel = rng.permutation(sel)
            else:
                sel = rng.permutation(len(g))[:n]
            h = n // 2
            tr = g[sel[:h]]
            te = g[sel[h:]]
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


def load(cache, resfile, W):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
    ends = np.append(starts[1:], len(sp))
    R_ = json.load(open(resfile))
    conds = np.unique(tt)
    order = np.argsort(mo)
    tt_o = tt[order]
    mo_o = mo[order]
    units = []
    for u in R_["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = (np.searchsorted(s, mo_o + float(W)) - np.searchsorted(s, mo_o)).astype(float)
        groups = [c[tt_o == cc] for cc in conds]
        units.append((groups, r["fano"]))
    return units


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2"),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25"),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05"),
]
res = {}
t0 = time.time()
for name, cache, resfile, W in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    D = []
    Dsim = []
    for groups, fano in units:
        d = []
        ds = []
        # simulated control: same means, independent NB/Poisson noise, session order irrelevant
        simg = [
            (
                rng.poisson(max(g.mean(), 1e-9), len(g))
                if fano <= 1.05
                else rng.negative_binomial(max(g.mean(), 1e-9) / (fano - 1), 1 / fano, len(g))
            ).astype(float)
            for g in groups
        ]
        for m in MS:
            d.append(ncv(groups, m, True) - ncv(groups, m, False))
            ds.append(ncv(simg, m, True) - ncv(simg, m, False))
        D.append(d)
        Dsim.append(ds)
    D = np.array(D, float)
    Dsim = np.array(Dsim, float)
    res[name] = {
        "n": len(units),
        "diff_mean": D.mean(0).tolist(),
        "diff_sim": Dsim.mean(0).tolist(),
        "frac_pos_m8": float(np.mean(D[:, 0] > 0)),
        "frac_neg_m8": float(np.mean(D[:, 0] < 0)),
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s]: contiguous − random N_cv at m=8/16/32: data {np.round(D.mean(0),2).tolist()}  simulated control {np.round(Dsim.mean(0),2).tolist()};  units contiguous>random at m=8: {100*np.mean(D[:,0]>0):.0f}%, <: {100*np.mean(D[:,0]<0):.0f}%",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q13.json"), "w"), indent=1)
print("[saved]")
