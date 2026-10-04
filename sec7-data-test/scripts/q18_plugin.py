"""PREREG_Q18.md: plug-in artefact check. stage1 = matched sim from data; stage2 = matched sim from stage1's estimates.
M1 read the Q8-consistent way. Seeds 201, 202."""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
SEEDS = (201, 202)
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


def sim_matched(groups):
    return [nb(g.mean(), g.var(ddof=1) if len(g) > 1 else g.mean(), len(g)) for g in groups]


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2", True),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", False),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", False),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", False),
]
q17 = (
    json.load(open(os.path.join(HERE, "results_q17.json")))
    if os.path.exists(os.path.join(HERE, "results_q17.json"))
    else {}
)
q17m1 = (
    json.load(open(os.path.join(HERE, "results_q17_m1fix.json")))
    if os.path.exists(os.path.join(HERE, "results_q17_m1fix.json"))
    else {}
)
res = {}
t0 = time.time()
for name, cache, resfile, W, q8r in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W, q8r)
    S1 = []
    S2 = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n1 = []
        n2 = []
        for g in units:
            s1 = sim_matched(g)
            s2 = sim_matched(s1)
            n1.append([ncv(s1, m) for m in MS])
            n2.append([ncv(s2, m) for m in MS])
        S1.append(np.array(n1, float).mean(0))
        S2.append(np.array(n2, float).mean(0))
    S1 = np.array(S1)
    S2 = np.array(S2)
    d = S2 - S1
    ref = (q17m1.get("M1") if name == "M1" else q17.get(name)) or {}
    deficit = ref.get("diff_mean", [float("nan")] * 4)
    res[name] = {
        "n": len(units),
        "stage1": S1.mean(0).tolist(),
        "stage2": S2.mean(0).tolist(),
        "stage2_minus_stage1": d.mean(0).tolist(),
        "sd": d.std(0, ddof=1).tolist(),
        "q17_deficit": deficit,
    }
    print(
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s] m=8/16/32/all: stage1 {np.round(S1.mean(0),2).tolist()} stage2 {np.round(S2.mean(0),2).tolist()} -> stage2-stage1 {np.round(d.mean(0),2).tolist()} sd {np.round(d.std(0,ddof=1),2).tolist()} | Q17 matched-data deficit {np.round(deficit,2).tolist()}",
        flush=True,
    )
json.dump(res, open(os.path.join(HERE, "results_q18.json"), "w"), indent=1)
print("[saved]")
