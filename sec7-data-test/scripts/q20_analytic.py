"""PREREG_Q20.md: one-constant analytic resolution rule. Fit c on M1 (Q19 data curve), test on V1/DMFC/Area2 (Q19) and A1 (Q15)."""

import numpy as np, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
NMAX = 16
MS = (8, 16, 32, 10**6)


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


def unit_stats(groups):
    mu = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) if len(g) > 1 else g.mean() for g in groups])
    n = np.array([len(g) for g in groups], float)
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    mt = np.maximum(gm + (mu - gm) * shr, 0.0)
    return mt, v, n


def rule_count(mt, v, n, m, c):
    h = np.maximum(np.minimum(m, n) // 2, 1)
    s = np.sqrt(np.maximum(v, 1e-9) / h)
    order = np.argsort(mt)
    mu = mt[order]
    s = s[order]
    groups = 1
    gmin = mu[0]
    sprev = s[0]
    for i in range(1, len(mu)):
        if mu[i] - gmin > c * 0.5 * (s[i] + sprev):
            groups += 1
            gmin = mu[i]
        sprev = s[i]
    return min(groups, NMAX, len(np.unique(np.round(mu, 9))))


def dataset_curve(units, c):
    return np.array([[rule_count(mt, v, n, m, c) for m in MS] for (mt, v, n) in units], float).mean(0)


DS = [
    ("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2", True),
    ("V1", "v1/v1_static36_cache.npz", "results_q8_v1.json", "0.25", False),
    ("DMFC", "area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", "0.2", False),
    ("Area2", "area2/area2_bump_cache.npz", "results_q8_area2.json", "0.2", False),
    ("A1", "a1/a1_cache.npz", "results_q8_a1.json", "0.05", False),
]
q19 = json.load(open(os.path.join(HERE, "results_q19.json")))
q15 = json.load(open(os.path.join(HERE, "results_q15.json")))
data = {k: np.array(q19[k]["data"]) for k in q19}
data["A1"] = np.array(q15["A1"]["data"])
stats = {}
for name, cache, resfile, W, q8r in DS:
    stats[name] = [
        unit_stats(g) for g in load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W, q8r)
    ]
grid = np.arange(0.5, 6.001, 0.05)
errs = [np.sum((dataset_curve(stats["M1"], c) - data["M1"]) ** 2) for c in grid]
c_star = float(grid[int(np.argmin(errs))])
print(
    f"fit on M1: c* = {c_star:.2f}  (M1 rule {np.round(dataset_curve(stats['M1'], c_star),2).tolist()} vs data {np.round(data['M1'],2).tolist()})"
)
res = {"c_star": c_star}
for name in ("V1", "DMFC", "Area2", "A1"):
    cur = dataset_curve(stats[name], c_star)
    d = data[name]
    diff = cur - d
    res[name] = {
        "rule": cur.tolist(),
        "data": d.tolist(),
        "diff": diff.tolist(),
        "maxabs": float(np.abs(diff).max()),
    }
    print(
        f"{name} (n={len(stats[name])}): rule {np.round(cur,2).tolist()} vs data {np.round(d,2).tolist()} -> diff {np.round(diff,2).tolist()} max|diff| {np.abs(diff).max():.2f}"
    )
json.dump(res, open(os.path.join(HERE, "results_q20.json"), "w"), indent=1)
print("[saved]")
