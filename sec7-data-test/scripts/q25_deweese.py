"""PREREG_Q25.md (+ Amendments 1-2): binary spiking vs low-rate Poisson in CRCNS ac-2 (anaesthetised rat A1, whole-cell).
Outputs results_q25.json. Seed 801."""

import numpy as np, json, os, re, glob, sys
import scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "crcns_ac2", "crcns", "ac-2", "Anesthetized_wholecell_and_LFP_data")
rng = np.random.default_rng(801)
FS = 4000
ONSET = 60
WINS = {"50ms": (ONSET, ONSET + 200), "100ms": (ONSET, ONSET + 400)}
BASE = (0, 60)
NT = 32
NMAX = 16
R = 20
FLOOR = 1e-2


def params(cell):
    p = {"spikethresh": 50.0, "first_cycle": None, "last_cycle": None}
    f = os.path.join(ROOT, cell, "cell_specific_parameters.m")
    if os.path.exists(f):
        for line in open(f, errors="replace"):
            m = re.match(r"\s*(spikethresh|first_cycle|last_cycle)\s*=\s*([-\d.]+)", line)
            if m:
                p[m.group(1)] = float(m.group(2))
    return p


def spikes(r, th):
    d = np.diff(r, axis=1)
    pk = (r[:, 1:-1] >= th) & (d[:, :-1] >= 5) & (r[:, 1:-1] >= r[:, 2:]) & (r[:, 1:-1] > r[:, :-2])
    out = np.zeros_like(r, dtype=bool)
    out[:, 1:-1] = pk
    return out


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

        def ll(x):
            x = np.maximum(x, FLOOR)
            return float((S * np.log(x) - n_te * x).sum())

        for N in range(1, NMAX + 1):
            LL[N] += ll(quant(means, N))
    return int(np.argmax(LL[1:]) + 1)


def nb(mu, var, n):
    mu = max(mu, 1e-9)
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


cells = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
res = {"cells": {}, "pooled": {}}
pool = {
    ep: {w: {th: {"obs2": 0.0, "exp2": 0.0, "pairs": []} for th in ("author", 25, 35, 45)} for w in WINS}
    for ep in ("all", "stationary")
}
for cell in cells:
    M = sio.loadmat(os.path.join(ROOT, cell, "rmat_and_lfpmat.mat"))
    r = M["r"].astype(float)
    p = params(cell)
    nrow = r.shape[0]
    tone = np.arange(nrow) % NT
    cyc = np.arange(nrow) // NT
    stat = np.ones(nrow, bool)
    if p["first_cycle"] is not None and p["last_cycle"] is not None:
        stat = (cyc >= p["first_cycle"] - 1) & (cyc <= p["last_cycle"] - 1)
    rec = {
        "rows": int(nrow),
        "repeats_per_tone": nrow / NT,
        "spikethresh": p["spikethresh"],
        "stationary_rows": int(stat.sum()),
        "windows": {},
    }
    for th_name, th in (("author", p["spikethresh"]), (25, 25.0), (35, 35.0), (45, 45.0)):
        sp = spikes(r, th)
        for w, (a, b) in WINS.items():
            cnt = sp[:, a:b].sum(1).astype(float)
            base = sp[:, BASE[0] : BASE[1]].sum(1).astype(float) * ((b - a) / (BASE[1] - BASE[0]))
            for ep, mask in (("all", np.ones(nrow, bool)), ("stationary", stat)):
                groups = [cnt[mask & (tone == t)] for t in range(NT)]
                bgroups = [base[mask & (tone == t)] for t in range(NT)]
                mu = np.array([g.mean() if len(g) else 0 for g in groups])
                n = np.array([len(g) for g in groups])
                bmu = np.array([g.mean() if len(g) else 0 for g in bgroups])
                se = np.array([g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.inf for g in groups])
                resp = (mu >= 0.15) & (mu - bmu >= 2 * se)
                key = f"{w}|{ep}|{th_name}"
                entry = {
                    "total_spikes": float(sum(g.sum() for g in groups)),
                    "n_responsive_tones": int(resp.sum()),
                    "best_mu": float(mu.max()),
                    "best_n": int(n[mu.argmax()]),
                }
                if resp.any():
                    b_ = int(np.argmax(np.where(resp, mu, -1)))
                    g = groups[b_]
                    m_ = g.mean()
                    exp2 = 1 - np.exp(-m_) * (1 + m_)
                    obs2 = float((g >= 2).mean())
                    entry.update(
                        {
                            "best_tone_mu": float(m_),
                            "best_tone_fano": float(g.var(ddof=1) / m_) if m_ > 0 else None,
                            "best_tone_R2": float(obs2 / exp2) if exp2 > 0 else None,
                            "best_tone_P2": obs2,
                        }
                    )
                    for t in np.where(resp)[0]:
                        gg = groups[t]
                        mm = gg.mean()
                        e2 = 1 - np.exp(-mm) * (1 + mm)
                        pool[ep][w][th_name]["obs2"] += float((gg >= 2).sum())
                        pool[ep][w][th_name]["exp2"] += float(len(gg) * e2)
                        pool[ep][w][th_name]["pairs"].append((cell, int(t), float(mm), int(len(gg))))
                if (
                    th_name == "author"
                    and w == "50ms"
                    and ep == "stationary"
                    and resp.any()
                    and n.min() >= 15
                    and mu.max() - mu.min() >= 0.1
                ):
                    mt, v, nn = shrunk(groups)
                    md = quant(mt, 2)
                    gc = [nb(mt[i], v[i], nn[i]) for i in range(NT)]
                    gd = [nb(md[i], v[i], nn[i]) for i in range(NT)]
                    entry["levels"] = {
                        "data": [ncv(groups, m) for m in (8, 16, 10**6)],
                        "cont_own": [ncv(gc, m) for m in (8, 16, 10**6)],
                        "disc_own": [ncv(gd, m) for m in (8, 16, 10**6)],
                        "A": float(mu.max() - mu.min()),
                        "lam": float(mu.min()),
                    }
                rec["windows"][key] = entry
    res["cells"][cell] = rec
    e = rec["windows"]["50ms|stationary|author"]
    print(
        f"{cell}: rows {nrow} ({nrow/NT:.1f} rep/tone), stat rows {int(stat.sum())}, thresh {p['spikethresh']:.0f}: spikes(50ms) {e['total_spikes']:.0f}, responsive tones {e['n_responsive_tones']}, best mu {e['best_mu']:.2f}"
        + (
            f", best-tone R2 {e['best_tone_R2']:.2f} Fano {e['best_tone_fano']:.2f} P2 {e['best_tone_P2']:.3f}"
            if "best_tone_R2" in e and e["best_tone_R2"] is not None
            else ""
        )
        + (
            f", levels {e['levels']['data']} cont {e['levels']['cont_own']} disc {e['levels']['disc_own']}"
            if "levels" in e
            else ""
        ),
        flush=True,
    )
print("\n=== POOLED two-spike ratio R2 = obs / Poisson-expected (parametric bootstrap 95%)")
for ep in ("all", "stationary"):
    for w in WINS:
        for th in ("author", 25, 35, 45):
            P = pool[ep][w][th]
            pairs = P["pairs"]
            if P["exp2"] <= 0:
                print(f"{ep:10s} {w:5s} thr {str(th):6s}: no responsive pairs")
                continue
            boots = []
            for _ in range(2000):
                o = 0.0
                for cell, t, mm, nn in pairs:
                    o += (rng.poisson(mm, nn) >= 2).sum()
                boots.append(o / P["exp2"])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            R2 = P["obs2"] / P["exp2"]
            print(
                f"{ep:10s} {w:5s} thr {str(th):6s}: pairs {len(pairs):3d} (cells {len(set(c for c,_,_,_ in pairs)):2d}), obs >=2 trials {P['obs2']:.0f}, Poisson-expected {P['exp2']:.1f}, R2 = {R2:.2f}  [null 95% of R2 under Poisson: {lo:.2f}-{hi:.2f}]"
            )
            res["pooled"][f"{ep}|{w}|{th}"] = {
                "pairs": len(pairs),
                "cells": len(set(c for c, _, _, _ in pairs)),
                "obs2": P["obs2"],
                "exp2": P["exp2"],
                "R2": R2,
                "null95": [float(lo), float(hi)],
            }
json.dump(res, open(os.path.join(HERE, "results_q25.json"), "w"), indent=1)
print("[saved] results_q25.json")
