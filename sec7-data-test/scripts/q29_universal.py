"""PREREG_Q29.md: cross-dataset universality of the normalised within-neuron rate map. Seed 1201."""

import numpy as np, json, os, sys, itertools
import scipy.io as sio
from scipy.stats import skew

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(1201)
M_EQ = 13
NSPLIT = 200


def load_cache(cache, resfile, W, q8_reading, wkey=None):
    z = np.load(os.path.join(HERE, cache), allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    if q8_reading:
        starts = sidx
        ends = np.append(sidx[1:], len(sp))
    else:
        starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
        ends = np.append(starts[1:], len(sp))
    conds = np.unique(tt)
    units = []
    if resfile:
        R_ = json.load(open(os.path.join(HERE, resfile)))
        for u in R_["units"]:
            r = u["W"].get(wkey or str(W), {})
            if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
                continue
            k = u["unit"]
            if ends[k] <= starts[k]:
                continue
            s = sp[starts[k] : ends[k]]
            c = (np.searchsorted(s, mo + W) - np.searchsorted(s, mo)).astype(float)
            units.append([c[tt == cc] for cc in conds])
    return units


def load_q26(cache, resfile, W=0.5):
    z = np.load(os.path.join(HERE, cache))
    sp, sidx, code, st = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    ends = np.append(sidx[1:], len(sp))
    R_ = json.load(open(os.path.join(HERE, resfile)))
    units = []
    for u in R_["units"]:
        m = u["W"]["0.5"]["map36"]
        if not (m and m["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[sidx[k] : ends[k]]
        c = (np.searchsorted(s, st + W) - np.searchsorted(s, st)).astype(float)
        units.append([c[code == j] for j in range(36)])
    return units


def load_retina(w=3, NC=36):
    M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
    B = M["binned"]
    nreps = M["nreps"].ravel().astype(int)
    NCELL, NMOV, NBIN = B.shape[2], B.shape[3], B.shape[1]
    Q8 = json.load(open(os.path.join(HERE, "results_q8_retina.json")))
    flags = {(u["cell"], u["movie"]): u["W"] for u in Q8["units"]}
    units = []
    for cell in range(NCELL):
        for mov in range(NMOV):
            row = flags.get((cell, mov), {}).get("0.05", {})
            if not (row.get("powered") and "Ncv_by_m" in row and row["Ncv_by_m"][-1] >= 2):
                continue
            nr = nreps[mov]
            X = B[:nr, :, cell, mov].astype(float)
            bins = np.sort(rng.choice(NBIN - 6, NC, replace=False))
            Cm = np.stack([X[:, b : b + w].sum(1) for b in bins], 1)
            units.append([Cm[:, i] for i in range(NC)])
    return units


def norm_map(groups, m):
    g = [x[rng.permutation(len(x))[: min(m, len(x))]] if m else x for x in groups]
    mu = np.array([x.mean() for x in g])
    v = np.array([x.var(ddof=1) if len(x) > 1 else x.mean() for x in g])
    n = np.array([len(x) for x in g])
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    mt = gm + (mu - gm) * shr
    rngm = mt.max() - mt.min()
    return (mt - mt.min()) / rngm if rngm > 1e-9 else None


DS = {
    "M1": lambda: load_cache("../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", 0.2, True, "0.2"),
    "Area2": lambda: load_cache("area2/area2_bump_cache.npz", "results_q8_area2.json", 0.2, False, "0.2"),
    "DMFC": lambda: load_cache("area2/dmfc_rsg_cache.npz", "results_q8_dmfc.json", 0.2, False, "0.2"),
    "V1static": lambda: load_cache("v1/v1_static36_cache.npz", "results_q8_v1.json", 0.25, False, "0.25"),
    "V1drift": lambda: load_cache("v1/v1_drifting_cache.npz", "results_q8_v1d.json", 0.5, False, "0.5"),
    "V1contrast1": lambda: load_q26("v1/v1_fc_contrast_cache.npz", "results_q26.json"),
    "V1contrast2": lambda: load_q26("v1/v1_fc_contrast_cache_s2.npz", "results_q26_s2.json"),
    "Retina": lambda: load_retina(),
}
res = {"per_dataset": {}, "pairs": {}}
for mode, m in (("equalised13", M_EQ), ("all", None)):
    print(f"\n===== {mode} =====")
    X = {}
    Q = {}
    for name, loader in DS.items():
        units = loader()
        maps = [nm for nm in (norm_map(g, m) for g in units) if nm is not None]
        if not maps:
            print(name, "no units")
            continue
        pooled = np.concatenate(maps)
        X[name] = (maps, pooled)
        ks = []
        for _ in range(NSPLIT):
            idx = rng.permutation(len(maps))
            a = np.concatenate([maps[i] for i in idx[: len(maps) // 2]])
            b = np.concatenate([maps[i] for i in idx[len(maps) // 2 :]])
            grid = np.linspace(0, 1, 201)
            ks.append(
                np.max(
                    np.abs(
                        np.searchsorted(np.sort(a), grid, side="right") / len(a)
                        - np.searchsorted(np.sort(b), grid, side="right") / len(b)
                    )
                )
            )
        Q[name] = float(np.percentile(ks, 95))
        summ = {
            "n_units": len(maps),
            "median_x": float(np.median(pooled)),
            "sparse_frac_below_0.25": float(np.mean(pooled < 0.25)),
            "skew": float(skew(pooled)),
            "q95_split_KS": Q[name],
        }
        res["per_dataset"][f"{mode}|{name}"] = summ
        print(
            f"{name:12s} units {len(maps):3d}: median x {summ['median_x']:.2f}, frac < 0.25 {summ['sparse_frac_below_0.25']:.2f}, skew {summ['skew']:+.2f}, split-half q95 KS {Q[name]:.3f}"
        )
    grid = np.linspace(0, 1, 201)
    names = list(X)
    same = 0
    total = 0
    lines = []
    for a, b in itertools.combinations(names, 2):
        Fa = np.searchsorted(np.sort(X[a][1]), grid, side="right") / len(X[a][1])
        Fb = np.searchsorted(np.sort(X[b][1]), grid, side="right") / len(X[b][1])
        ks = float(np.max(np.abs(Fa - Fb)))
        ok = ks <= 1.5 * max(Q[a], Q[b]) or ks <= 0.08
        same += ok
        total += 1
        res["pairs"][f"{mode}|{a}|{b}"] = {"KS": ks, "same": bool(ok)}
        lines.append(f"  {a:12s} vs {b:12s}: KS {ks:.3f} {'same' if ok else 'DIFFERENT'}")
    print("\n".join(lines))
    frac = same / total
    verdict = "P-universal" if frac >= 0.75 else ("P-area-specific" if frac <= 0.4 else "P-mixed")
    sp = [res["per_dataset"][f"{mode}|{n}"]["sparse_frac_below_0.25"] for n in names]
    pooled_sp = float(np.mean(np.concatenate([X[n][1] for n in names]) < 0.25))
    print(
        f"pairs same shape: {same}/{total} ({100*frac:.0f}%); sparseness pooled {pooled_sp:.2f}, per dataset within 0.10: {sum(abs(s - pooled_sp) <= 0.10 for s in sp)}/{len(sp)} -> {verdict}"
    )
    res[f"verdict|{mode}"] = {
        "same": same,
        "total": total,
        "frac": frac,
        "verdict": verdict,
        "pooled_sparseness": pooled_sp,
    }
json.dump(res, open(os.path.join(HERE, "results_q29.json"), "w"), indent=1)
print("[saved] results_q29.json")
