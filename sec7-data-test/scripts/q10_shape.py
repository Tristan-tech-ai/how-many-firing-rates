"""PREREG_Q10.md: shape of the continuous map. Per tuned powered pair: KS to four hypotheses; K2 confusion; K1 shuffle. Seed 13."""

import numpy as np, json, os, sys, time, scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(13)
NSIM = 8


def transforms(m):
    return {"sqrt": np.sqrt(np.maximum(m, 0)), "rate": m, "log": np.log(np.maximum(m, 0) + 0.05)}


def ks_uniform(y):
    y = np.sort(y)
    lo, hi = y[0], y[-1]
    if hi - lo < 1e-9:
        return 1.0
    F = (y - lo) / (hi - lo)
    n = len(y)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def ks_two(m):
    y = np.sort(m)
    c = np.array([np.quantile(y, 0.25), np.quantile(y, 0.75)])
    for _ in range(30):
        a = np.abs(y[:, None] - c[None, :]).argmin(1)
        c = np.array([y[a == j].mean() if (a == j).any() else c[j] for j in range(2)])
    p1 = np.mean(np.abs(y - c[0]) < np.abs(y - c[1]))
    n = len(y)
    i = np.arange(1, n + 1)
    Ftheory = np.where(y < c[1], np.where(y >= c[0], p1, 0.0), 1.0)  # step CDF of the two-point distribution
    return float(max(np.max(np.abs(i / n - Ftheory)), np.max(np.abs((i - 1) / n - Ftheory))))


def classify(m):
    d = {k: ks_uniform(v) for k, v in transforms(m).items()}
    d["two"] = ks_two(m)
    return min(d, key=d.get), d


def sim_means(hyp, lo, hi, nc, ncounts, fano, bern):
    if hyp == "sqrt":
        r = (np.sqrt(lo) + (np.sqrt(hi) - np.sqrt(lo)) * rng.random(nc)) ** 2
    elif hyp == "rate":
        r = lo + (hi - lo) * rng.random(nc)
    elif hyp == "log":
        r = np.exp(np.log(lo + 0.05) + (np.log(hi + 0.05) - np.log(lo + 0.05)) * rng.random(nc)) - 0.05
    else:
        r = np.where(rng.random(nc) < 0.5, lo, hi)
    out = np.zeros(nc)
    for i, (mu, n) in enumerate(zip(np.maximum(r, 1e-9), ncounts)):
        if bern:
            out[i] = (rng.random(n) < min(mu, 1)).mean()
        elif fano <= 1.05:
            out[i] = rng.poisson(mu, n).mean()
        else:
            out[i] = rng.negative_binomial(mu / (fano - 1), 1 / fano, n).mean()
    return out


HYPS = ["sqrt", "rate", "log", "two"]


def analyse(name, pairs):
    """pairs: list of (label, means array, ncounts, fano, bern)"""
    rows = []
    K1 = {h: 0 for h in HYPS}
    for label, m, ncounts, fano, bern in pairs:
        win, d = classify(m)
        lo, hi = float(m.min()), float(m.max())
        conf = {h: {g: 0 for g in HYPS} for h in HYPS}
        for h in HYPS:
            for _ in range(NSIM):
                w, _ = classify(sim_means(h, lo, hi, len(m), ncounts, fano, bern))
                conf[h][w] += 1
        rec = np.array([conf[h][h] / NSIM for h in HYPS])
        powered = bool(rec.min() >= 0.6)
        K1[classify(rng.permutation(m))[0]] += 1
        rows.append(
            {
                "label": label,
                "win": win,
                "ks": d,
                "powered": powered,
                "recovery": dict(zip(HYPS, rec.tolist())),
            }
        )
    pw = [r for r in rows if r["powered"]]
    wins = {h: np.mean([r["win"] == h for r in pw]) if pw else float("nan") for h in HYPS}
    print(
        f"{name}: pairs {len(rows)}, powered {len(pw)}; WIN fractions (powered): "
        + "  ".join(f"{h} {100*wins[h]:.0f}%" for h in HYPS)
        + f"; mean KS: "
        + "  ".join(f"{h} {np.mean([r['ks'][h] for r in pw]):.3f}" for h in HYPS)
        + f"; K1 shuffle wins: {K1}",
        flush=True,
    )
    return rows, wins


def cortex_pairs(cache, resfile, W_list, tag):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = np.concatenate([[0], sidx[:-1]]) if sidx[-1] == len(sp) else sidx
    ends = np.append(starts[1:], len(sp))
    R = json.load(open(resfile))
    conds = np.unique(tt)
    pairs = []
    for u in R["units"]:
        k = u["unit"]
        for W in W_list:
            r = u["W"].get(W, {})
            if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
                continue
            s = sp[starts[k] : ends[k]]
            c = np.searchsorted(s, mo + float(W)) - np.searchsorted(s, mo)
            groups = [c[tt == cc].astype(float) for cc in conds]
            m = np.array([g.mean() for g in groups])
            pairs.append((f"{tag}-u{k}-W{W}", m, [len(g) for g in groups], r["fano"], False))
    return pairs


out = {}
out["M1"] = analyse(
    "M1",
    cortex_pairs(
        os.path.join(HERE, "..", "mc_maze_test", "mc_maze_cache.npz"),
        "results_q8_rep.json",
        ["0.1", "0.2", "0.4"],
        "M1",
    ),
)
out["Area2"] = analyse(
    "Area2",
    cortex_pairs(
        os.path.join(HERE, "area2", "area2_bump_cache.npz"),
        "results_q8_area2.json",
        ["0.1", "0.2", "0.4"],
        "A2",
    ),
)
out["DMFC"] = analyse(
    "DMFC",
    cortex_pairs(
        os.path.join(HERE, "area2", "dmfc_rsg_cache.npz"),
        "results_q8_dmfc.json",
        ["0.1", "0.2", "0.4"],
        "DMFC",
    ),
)
# retina: rebuild 36-bin means with the seed-3 draw sequence as in q8_retina.py (per-unit draw order)
M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
B = M["binned"]
nreps = M["nreps"].ravel().astype(int)
NBIN = B.shape[1]
R = json.load(open("results_q8_retina.json"))
rb = np.random.default_rng(3)
pairs = []
for u in R["units"]:
    bins = np.sort(rb.choice(NBIN - 6, 36, replace=False))
    nr = nreps[u["movie"]]
    X = B[:nr, :, u["cell"], u["movie"]].astype(float)
    for W, w in (("0.0167", 1), ("0.05", 3), ("0.1", 6)):
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        C = np.stack([X[:, b : b + w].sum(1) for b in bins], 1)
        pairs.append((f"R-u{u['unit']}-W{W}", C.mean(0), [nr] * 36, r["fano"], w == 1))
out["Retina"] = analyse("Retina", pairs)
json.dump({k: {"rows": v[0], "wins": v[1]} for k, v in out.items()}, open("results_q10.json", "w"), indent=1)
print("[saved] results_q10.json")
