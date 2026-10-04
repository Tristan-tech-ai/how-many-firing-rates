"""PREREG_Q26.md (+ Amendments 1-2): grating contrast as the graded axis, Allen FC session 779839471, VISp good units.
(a) per-orientation contrast maps (9 contrasts x 15 repeats), (b) 36-cell contrast x orientation map.
Own-map comparators as in Q21 (shrunk seeds; plug-in seeds reported), certified N* from support_exact. Seeds 901-903.
"""

import numpy as np, json, os, sys, time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support_exact import support

NMAX = 16
R = 20
FLOOR = 1e-2
SEEDS = (901, 902, 903)
WS = (0.25, 0.5)
MS_A = (8, 10**6)
MS_B = (8, 10**6)
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


def seeds(groups, shrink):
    mu = np.array([g.mean() for g in groups])
    v = np.array([g.var(ddof=1) if len(g) > 1 else g.mean() for g in groups])
    n = np.array([len(g) for g in groups])
    if not shrink:
        return mu, v, n
    se2 = v / np.maximum(n, 1)
    tau2 = max(mu.var(ddof=1) - se2.mean(), 0.0)
    gm = mu.mean()
    shr = tau2 / (tau2 + se2) if tau2 > 0 else np.zeros_like(se2)
    return np.maximum(gm + (mu - gm) * shr, 0.0), v, n


def comparators(groups, nstar, ms):
    out = {}
    for tag, shrink in (("shrunk", True), ("plugin", False)):
        mt, v, n = seeds(groups, shrink)
        md = quant(mt, max(nstar, 1))
        gc = [nb(mt[i], v[i], n[i]) for i in range(len(groups))]
        gd = [nb(md[i], v[i], n[i]) for i in range(len(groups))]
        out[tag] = {"cont": [ncv(gc, m) for m in ms], "disc": [ncv(gd, m) for m in ms]}
    return out


Z = np.load(os.path.join(HERE, "v1", "v1_fc_contrast_cache.npz"))
sp, sidx, code, st, con, ori = (
    Z["spikes"],
    Z["sidx"],
    Z["trial_type"],
    Z["move_onset_time"],
    Z["contrast"],
    Z["orientation"],
)
ends = np.append(sidx[1:], len(sp))
NU = len(sidx)
cons = np.unique(con)
oris = np.unique(ori)
print(f"units {NU}, presentations {len(st)}, contrasts {cons.tolist()}, orientations {oris.tolist()}")
res = {"config": {"W": WS, "seeds": SEEDS, "R": R, "NMAX": NMAX}, "units": []}
t0 = time.time()
for k in range(NU):
    s = sp[sidx[k] : ends[k]]
    rec = {"unit": int(k), "W": {}}
    for W in WS:
        c = (np.searchsorted(s, st + W) - np.searchsorted(s, st)).astype(float)
        row = {"per_orientation": [], "map36": None}
        # (a) per-orientation contrast maps
        for o in oris:
            groups = [c[(ori == o) & (con == cc)] for cc in cons]
            mu = np.array([g.mean() for g in groups])
            lam = float(mu.min())
            A = float(mu.max() - lam)
            ent = {"orientation": float(o), "lam": lam, "A": A}
            if A < 0.05:
                ent["skip"] = "A<0.05"
                row["per_orientation"].append(ent)
                continue
            S = support(A, lam)
            nstar = int(S["N"])
            ent["Nstar"] = nstar
            allc = np.concatenate(groups)
            rng = np.random.default_rng(SEEDS[0])
            perm = rng.permutation(len(allc))
            sh = [allc[perm[i * 15 : (i + 1) * 15]] for i in range(len(groups))]
            ent["K1_shuffle_Ncv"] = ncv(sh, 10**6)
            D = []
            C = {"shrunk": {"cont": [], "disc": []}, "plugin": {"cont": [], "disc": []}}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                D.append([ncv(groups, m) for m in MS_A])
                cm = comparators(groups, nstar, MS_A)
                for tag in C:
                    for kind in ("cont", "disc"):
                        C[tag][kind].append(cm[tag][kind])
            ent["Ncv_by_m"] = np.mean(D, 0).tolist()
            ent["comp"] = {tag: {kind: np.mean(C[tag][kind], 0).tolist() for kind in C[tag]} for tag in C}
            row["per_orientation"].append(ent)
        # (b) 36-cell map
        groups = [c[code == j] for j in range(36)]
        mu = np.array([g.mean() for g in groups])
        lam = float(mu.min())
        A = float(mu.max() - lam)
        if A >= 0.05:
            S = support(A, lam)
            nstar = int(S["N"])
            D = []
            C = {"shrunk": {"cont": [], "disc": []}, "plugin": {"cont": [], "disc": []}}
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                D.append([ncv(groups, m) for m in MS_B])
                cm = comparators(groups, nstar, MS_B)
                for tag in C:
                    for kind in ("cont", "disc"):
                        C[tag][kind].append(cm[tag][kind])
            row["map36"] = {
                "lam": lam,
                "A": A,
                "Nstar": nstar,
                "Ncv_by_m": np.mean(D, 0).tolist(),
                "comp": {tag: {kind: np.mean(C[tag][kind], 0).tolist() for kind in C[tag]} for tag in C},
            }
        rec["W"][str(W)] = row
    res["units"].append(rec)
    if k % 10 == 0 or k == NU - 1:
        r5 = rec["W"]["0.5"]
        po = [e for e in r5["per_orientation"] if "Ncv_by_m" in e]
        print(
            f"unit {k:3d}/{NU} [{time.time()-t0:.0f}s] W=0.5: tuned orientations {len(po)}/4"
            + (
                f", e.g. A={po[0]['A']:.2f} N*={po[0]['Nstar']} data {po[0]['Ncv_by_m']} cont {po[0]['comp']['shrunk']['cont']} disc {po[0]['comp']['shrunk']['disc']}"
                if po
                else ""
            )
            + (
                f" | map36 A={r5['map36']['A']:.2f} N*={r5['map36']['Nstar']} data {r5['map36']['Ncv_by_m']} cont {r5['map36']['comp']['shrunk']['cont']} disc {r5['map36']['comp']['shrunk']['disc']}"
                if r5["map36"]
                else ""
            ),
            flush=True,
        )
json.dump(res, open(os.path.join(HERE, "results_q26.json"), "w"), indent=1)
print("[saved] results_q26.json")
