"""PREREG_Q11d.md: normalised map shape vs dynamic range, data vs two models. Seed 18."""

import numpy as np, json, os, sys, scipy.io as sio
from scipy.stats import skew

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = "."
rng = np.random.default_rng(18)
K = 0.75


def g(x):
    return np.clip((np.exp(K * x) - np.exp(-K)) / (np.exp(K) - np.exp(-K)), 0, 1)


def noisy(rc, ncounts, fano):
    return np.array(
        [
            (
                rng.poisson(max(mu, 1e-9), n).mean()
                if fano <= 1.05
                else rng.negative_binomial(max(mu, 1e-9) / (fano - 1), 1 / fano, n).mean()
            )
            for mu, n in zip(rc, ncounts)
        ]
    )


def ks(a, b):
    a = np.sort(a)
    b = np.sort(b)
    allv = np.concatenate([a, b])
    Fa = np.searchsorted(a, allv, side="right") / len(a)
    Fb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.abs(Fa - Fb).max())


def norm(m):
    return (m - m.min()) / max(m.max() - m.min(), 1e-9)


def cortex(cache, resfile, W):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
    ends = np.append(starts[1:], len(sp))
    R = json.load(open(resfile))
    conds = np.unique(tt)
    out = []
    for u in R["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)
        groups = [c[tt == cc].astype(float) for cc in conds]
        out.append(
            (np.array([gg.mean() for gg in groups]), [len(gg) for gg in groups], r["fano"], r["A"], r["lam"])
        )
    return out


def retina():
    M = sio.loadmat("retina/movieBinnedSpiking.mat")
    B = M["binned"]
    nreps = M["nreps"].ravel().astype(int)
    R = json.load(open("results_q8_retina.json"))
    rb = np.random.default_rng(3)
    out = []
    for u in R["units"]:
        bins = np.sort(rb.choice(B.shape[1] - 6, 36, replace=False))
        r = u["W"].get("0.1", {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        nr = nreps[u["movie"]]
        X = B[:nr, :, u["cell"], u["movie"]].astype(float)
        C = np.stack([X[:, b : b + 6].sum(1) for b in bins], 1)
        out.append((C.mean(0), [nr] * 36, r["fano"], r["A"], r["lam"]))
    return out


DS = {
    "M1": cortex("../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2"),
    "Area2": cortex("area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2"),
    "DMFC": cortex("area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2"),
    "V1": cortex("v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25"),
    "Retina": retina(),
}
res = {}
print(
    f"{'dataset':<8}{'n':>5}  {'tercile: low-A | mid | high-A  (sparse<0.25 / top>0.75 / skew)':<70}  KS(low,high): data  transd  lognull"
)
for name, units in DS.items():
    A = np.array([u[3] for u in units])
    t = np.digitize(A, np.quantile(A, [1 / 3, 2 / 3]))
    Yd, Yt, Yl = [[], [], []], [[], [], []], [[], [], []]
    stats = []
    for (m, nc, fano, a, lam), ti in zip(units, t):
        yd = norm(m)
        Yd[ti].append(yd)
        nc_ = len(m)
        yt = norm(noisy(lam + a * g(rng.standard_normal(nc_)), nc, fano))
        Yt[ti].append(yt)
        q = np.minimum(1.0, rng.lognormal(0, 1, nc_) / np.exp(1.645))
        yl = norm(noisy(lam + a * q, nc, fano))
        Yl[ti].append(yl)
    desc = " | ".join(
        f"{100*np.median([np.mean(y<0.25) for y in Yd[i]]):.0f}%/{100*np.median([np.mean(y>0.75) for y in Yd[i]]):.0f}%/{np.median([skew(y) for y in Yd[i]]):+.2f}"
        for i in range(3)
    )
    kd = ks(np.concatenate(Yd[0]), np.concatenate(Yd[2]))
    kt = ks(np.concatenate(Yt[0]), np.concatenate(Yt[2]))
    kl = ks(np.concatenate(Yl[0]), np.concatenate(Yl[2]))
    res[name] = {"n": len(units), "ks_data": kd, "ks_transd": kt, "ks_lognull": kl, "desc": desc}
    print(
        f"{name:<8}{len(units):>5}  {desc:<70}  {kd:.3f}  {kt:.3f}  {kl:.3f}   {'DEPENDENT' if kd - max(kt, kl) >= 0.10 else 'invariant'}"
    )
json.dump(res, open("results_q11d.json", "w"), indent=1)
