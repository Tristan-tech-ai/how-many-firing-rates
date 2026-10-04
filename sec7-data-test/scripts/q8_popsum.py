"""PREREG_Q8_population.md: level count of the SUMMED population vs N*(A_pop). Seed 6. Output results_q8_pop.json"""

import numpy as np, json, os, sys, time, scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
from support_exact import support

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(6)
NMAX = 16
R = 10
NSIM = 4
R_SIM = 4
FLOOR = 1e-2


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


def ncv(groups, m, R):
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
    return int(np.argmax(LL[1:]) + 1)


def simulate(levels, ncounts, fano):
    out = []
    for mu, n in zip(levels, ncounts):
        mu = max(mu, 1e-9)
        out.append(
            rng.poisson(mu, n).astype(float)
            if fano <= 1.05
            else rng.negative_binomial(mu / (fano - 1.0), 1.0 / fano, n).astype(float)
        )
    return out


def analyse(C_units, cond, tag, Ks, MS):
    """C_units: (units x trials) counts; cond: trial condition codes."""
    conds = np.unique(cond)
    res = []
    for K in Ks:
        rows = []
        for s in range(10):
            sub = rng.choice(C_units.shape[0], K, replace=False)
            tot = C_units[sub].sum(0)
            groups = [tot[cond == c] for c in conds]
            ncounts = [len(g) for g in groups]
            means = np.array([g.mean() for g in groups])
            lam = float(means.min())
            A = float(means.max() - lam)
            fano = (
                float(np.mean([g.var(ddof=1) / g.mean() for g in groups if g.mean() > 0.1]))
                if (means > 0.1).any()
                else 1.0
            )
            Sx = support(min(A, 60.0), lam)
            Nstar = Sx["N"]
            nlev = np.maximum(1, np.round(np.array(Sx["w"]) * len(groups))).astype(int)
            while nlev.sum() > len(groups):
                nlev[nlev.argmax()] -= 1
            while nlev.sum() < len(groups):
                nlev[nlev.argmax()] += 1
            disc_levels = np.repeat(np.array(Sx["pts"]) + lam, nlev)
            data = [ncv(groups, m, R) for m in MS]
            d_sim = np.mean(
                [
                    [ncv(simulate(rng.permutation(disc_levels), ncounts, fano), m, R_SIM) for m in MS]
                    for _ in range(NSIM)
                ],
                0,
            )
            s_sim = np.mean(
                [
                    [ncv(simulate(lam + A * rng.random(len(groups)), ncounts, fano), m, R_SIM) for m in MS]
                    for _ in range(NSIM)
                ],
                0,
            )
            rows.append(
                {
                    "K": K,
                    "A": A,
                    "lam": lam,
                    "fano": fano,
                    "Nstar": Nstar,
                    "data": data,
                    "disc": d_sim.tolist(),
                    "smooth": s_sim.tolist(),
                }
            )
        D = np.array([r["data"] for r in rows])
        Ns = np.array([r["Nstar"] for r in rows])
        print(
            f"  {tag} K={K:3d}: A_pop {np.mean([r['A'] for r in rows]):6.2f}  N* {Ns.mean():.1f}  data N_cv(m) {np.round(D.mean(0),1).tolist()}  disc {np.round(np.mean([r['disc'] for r in rows],0),1).tolist()}  smooth {np.round(np.mean([r['smooth'] for r in rows],0),1).tolist()}  |data-N*|<=1: {100*np.mean(np.abs(D[:,-1]-Ns)<=1):.0f}%  growth: {100*np.mean(D[:,-1]>D[:,-2]):.0f}%",
            flush=True,
        )
        res.extend(rows)
    return res


out = {}
# ---- M1 ----
z = np.load(os.path.join(HERE, "..", "mc_maze_test", "mc_maze_cache.npz"), allow_pickle=True)
sp, endx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
starts = np.concatenate([[0], endx[:-1]])
C = np.zeros((len(endx), len(mo)))
for k in range(len(endx)):
    s = sp[starts[k] : endx[k]]
    C[k] = np.searchsorted(s, mo + 0.2) - np.searchsorted(s, mo)
print("M1 population sums, W = 200 ms")
out["M1"] = analyse(C, tt, "M1", (1, 2, 4, 8, 16, 32, 64, 128), (8, 16, 32, 10**6))
# ---- retina ----
M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
B = M["binned"]
nreps = M["nreps"].ravel().astype(int)
rb = np.random.default_rng(3)
for w, name in ((1, "retina 16.7 ms"), (6, "retina 100 ms")):
    rows_all = []
    for mov in range(5):
        nr = nreps[mov]
        bins = np.sort(rb.choice(B.shape[1] - 6, 36, replace=False))
        X = np.stack([B[:nr, b : b + w, :, mov].sum(1) for b in bins], 0)  # 36 bins x reps x cells
        Cc = X.transpose(2, 0, 1).reshape(93, -1)
        cond = np.repeat(np.arange(36), nr)
        print(f"{name}, movie {mov}")
        rows_all += analyse(Cc, cond, f"{name} m{mov}", (1, 4, 16, 93), (8, 16, 32, 10**6))
    out[name] = rows_all
json.dump(out, open(os.path.join(HERE, "results_q8_pop.json"), "w"), indent=1)
print("[saved] results_q8_pop.json")
