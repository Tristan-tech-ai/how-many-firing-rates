"""Q8 on salamander retina (PREREG_Q8_retina.md + Amendment 1). Output schema = results_q8*.json. Seed 3."""

import numpy as np, json, os, sys, time, scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
from support_exact import support

HERE = os.path.dirname(os.path.abspath(__file__))
M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
B = M["binned"]
nreps = M["nreps"].ravel().astype(int)
NCELL, NMOV = B.shape[2], B.shape[3]
NBIN = B.shape[1]
WS = (1, 3, 6)
WNAME = {1: "0.0167", 3: "0.05", 6: "0.1"}
MS = (8, 16, 32, 10**6)
NMAX = 16
R_DATA = 20
R_SIM = 6
NSIM = 8
FLOOR = 1e-2
NC = 36
rng = np.random.default_rng(3)
FULL = "--full" in sys.argv


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


def ncv(groups, m, R, rng):
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

        LL[0] += ll(means)
        for N in range(1, NMAX + 1):
            LL[N] += ll(quant(means, N))
    LL /= R
    return int(np.argmax(LL[1:]) + 1), LL


def simulate(levels, ncounts, fano, rng, bern):
    out = []
    for mu, n in zip(levels, ncounts):
        mu = max(mu, 1e-9)
        if bern:
            out.append((rng.random(n) < min(mu, 1.0)).astype(float))
        elif fano <= 1.05:
            out.append(rng.poisson(mu, n).astype(float))
        else:
            out.append(rng.negative_binomial(mu / (fano - 1.0), 1.0 / fano, n).astype(float))
    return out


res = {
    "config": {
        "W": [0.0167, 0.05, 0.1],
        "m": MS,
        "NMAX": NMAX,
        "R_DATA": R_DATA,
        "R_SIM": R_SIM,
        "NSIM": NSIM,
        "FLOOR": FLOOR,
        "NC": NC,
        "full": FULL,
    },
    "units": [],
}
t0 = time.time()
k1p = 0
k1t = 0
uid = 0
for cell in range(NCELL):
    for mov in range(NMOV):
        rec = {"unit": uid, "cell": cell, "movie": mov, "W": {}}
        nr = nreps[mov]
        X = B[:nr, :, cell, mov].astype(float)  # reps x bins
        bins = np.arange(NBIN - 6) if FULL else np.sort(rng.choice(NBIN - 6, NC, replace=False))
        for w in WS:
            C = np.stack([X[:, b : b + w].sum(1) for b in bins], 1)  # reps x conditions
            groups = [C[:, i] for i in range(C.shape[1])]
            ncounts = [len(g) for g in groups]
            means = C.mean(0)
            lam = float(means.min())
            A = float(means.max() - lam)
            ok = means > 0.1
            fano = (
                float(np.mean([g.var(ddof=1) / g.mean() for g in groups if g.mean() > 0.1]))
                if ok.any()
                else 1.0
            )
            row = {"lam": lam, "A": A, "fano": fano}
            if A < 0.05:
                row["skip"] = "A<0.05"
                rec["W"][WNAME[w]] = row
                continue
            bern = w == 1
            allc = C.T.ravel()
            perm = rng.permutation(len(allc))
            shuf = [allc[perm[i * nr : (i + 1) * nr]] for i in range(len(groups))]
            n_sh, _ = ncv(shuf, 10**6, 10, rng)
            row["K1_shuffle_Ncv"] = n_sh
            k1t += 1
            k1p += n_sh == 1
            S = support(A, lam)
            row["Nstar"] = S["N"]
            row["Nstar_pts"] = S["pts"]
            nlev = np.maximum(1, np.round(np.array(S["w"]) * len(groups))).astype(int)
            while nlev.sum() > len(groups):
                nlev[nlev.argmax()] -= 1
            while nlev.sum() < len(groups):
                nlev[nlev.argmax()] += 1
            disc_levels = np.repeat(np.array(S["pts"]) + lam, nlev)
            sim = {"disc": [], "smooth": []}
            for s in range(NSIM):
                for tag, lv in (
                    ("disc", rng.permutation(disc_levels)),
                    ("smooth", lam + A * rng.random(len(groups))),
                ):
                    sim[tag].append(
                        [ncv(simulate(lv, ncounts, fano, rng, bern), m, R_SIM, rng)[0] for m in MS]
                    )
            d = np.array(sim["disc"])
            s_ = np.array(sim["smooth"])
            sep = float(np.mean(np.abs(d[:, -1] - s_[:, -1]) >= 1))
            row["K2_sep_frac"] = sep
            row["powered"] = bool(sep >= 0.8)
            row["sim_disc_Ncv_by_m"] = d.mean(0).tolist()
            row["sim_smooth_Ncv_by_m"] = s_.mean(0).tolist()
            curve = []
            LLs = []
            for m in MS:
                n_cv, LL = ncv(groups, m, R_DATA, rng)
                curve.append(n_cv)
                LLs.append(LL.tolist())
            row["Ncv_by_m"] = curve
            row["LL_by_m"] = LLs
            rec["W"][WNAME[w]] = row
        res["units"].append(rec)
        uid += 1
        if uid % 25 == 0:
            print(
                f"unit {uid:3d}/465 [{time.time()-t0:.0f}s] K1 {k1p}/{k1t}  "
                + "  ".join(
                    f"{k}: A={v.get('A',0):.2f} N*={v.get('Nstar','-')} pow={v.get('powered','-')} Ncv={v.get('Ncv_by_m','-')}"
                    for k, v in rec["W"].items()
                ),
                flush=True,
            )
res["K1"] = {"pass": k1p, "total": k1t}
out = "results_q8_retina_full.json" if FULL else "results_q8_retina.json"
json.dump(res, open(os.path.join(HERE, out), "w"), indent=1)
print(f"[saved] {out}  K1 {k1p}/{k1t}  [{time.time()-t0:.0f}s]")
