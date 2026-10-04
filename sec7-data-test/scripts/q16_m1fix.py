"""PREREG_Q16.md: residual ACF in session order + copula AR(1) dependence-matched simulation. Seed 23."""

import numpy as np, json, os, sys, time
from scipy.stats import norm, nbinom, poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(23)
NMAX = 16
R = 20
FLOOR = 1e-2
MS = (8, 16, 32, 10**6)
L = 20
NSH = 20


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


def load(cache, resfile, W):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    starts = sidx
    ends = np.append(sidx[1:], len(sp))
    R_ = json.load(open(resfile))
    conds = np.unique(tt)
    order = np.argsort(mo)
    tt_o = tt[order]
    mo_o = mo[order]
    cidx = np.searchsorted(conds, tt_o)
    units = []
    for u in R_["units"]:
        r = u["W"].get(W, {})
        if not (r.get("powered") and "Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[starts[k] : ends[k]]
        c = (np.searchsorted(s, mo_o + float(W)) - np.searchsorted(s, mo_o)).astype(float)
        if ends[k] > starts[k]:
            units.append((c, cidx, len(conds)))
    return units


def acf(z, L):
    z = z - z.mean()
    v = (z * z).sum()
    return np.array([(z[:-k] * z[k:]).sum() / v if v > 0 else 0.0 for k in range(1, L + 1)])


def residuals(c, cidx, nc):
    mu = np.array([c[cidx == j].mean() for j in range(nc)])
    sd = np.array([c[cidx == j].std(ddof=1) if (cidx == j).sum() > 1 else 1.0 for j in range(nc)])
    sd = np.where(sd > 0, sd, 1.0)
    return (c - mu[cidx]) / sd[cidx], mu, sd


def sim_dep(c, cidx, nc, rho, a):
    mu, sd = residuals(c, cidx, nc)[1:]
    var = sd**2
    T = len(c)
    z = np.zeros(T)
    z[0] = rng.standard_normal()
    for t in range(1, T):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho * rho) * rng.standard_normal()
    w = np.sqrt(a) * z + np.sqrt(1 - a) * rng.standard_normal(T)
    u = np.clip(norm.cdf(w), 1e-9, 1 - 1e-9)
    out = np.zeros(T)
    for j in range(nc):
        sel = cidx == j
        m_ = max(mu[j], 1e-9)
        v_ = var[j]
        out[sel] = (
            nbinom.ppf(u[sel], m_ * m_ / (v_ - m_), m_ / v_) if v_ > m_ * 1.02 else poisson.ppf(u[sel], m_)
        )
    return [out[cidx == j] for j in range(nc)]


DS = [("M1", "../mc_maze_test/mc_maze_cache.npz", "results_q8_rep.json", "0.2")]
q15 = json.load(open(os.path.join(HERE, "results_q15.json")))
res = {}
t0 = time.time()
for name, cache, resfile, W in DS:
    units = load(os.path.join(HERE, cache), os.path.join(HERE, resfile), W)
    A1s = []
    A210 = []
    above = []
    Nsim = []
    rhos = []
    As = []
    for c, cidx, nc in units:
        z = residuals(c, cidx, nc)[0]
        ac = acf(z, L)
        sh = np.array([acf(rng.permutation(z), L)[0] for _ in range(NSH)])
        A1s.append(ac[0])
        A210.append(ac[1:10].mean())
        above.append(ac[0] > np.quantile(sh, 0.975))
        if ac[0] > 0 and ac[1] > 0:
            rho = float(np.clip(ac[1] / ac[0], 0, 0.98))
            a = float(np.clip(ac[0] ** 2 / ac[1], 0, 0.98))
        elif ac[0] > 0:
            rho = 0.0
            a = float(np.clip(ac[0], 0, 0.98))
        else:
            rho = 0.0
            a = 0.0
        rhos.append(rho)
        As.append(a)
        if name != "A1":
            g = sim_dep(c, cidx, nc, rho, a)
            Nsim.append([ncv(g, m) for m in MS])
    A1s = np.array(A1s)
    sh_sd = float(np.std([acf(rng.permutation(residuals(*units[0])[0]), L)[0] for _ in range(200)]))
    out = {
        "n": len(units),
        "acf1_mean": float(A1s.mean()),
        "acf1_median": float(np.median(A1s)),
        "acf2_10_mean": float(np.mean(A210)),
        "frac_above_shuffle": float(np.mean(above)),
        "shuffle_sd_one_unit": sh_sd,
        "rho_median": float(np.median(rhos)),
        "a_median": float(np.median(As)),
    }
    line = (
        f"{name} (n={len(units)}) [{time.time()-t0:.0f}s]: lag-1 ACF mean {A1s.mean():+.3f} (median {np.median(A1s):+.3f}; shuffle sd ~{sh_sd:.3f}); "
        f"lags 2-10 mean {np.mean(A210):+.3f}; units above shuffle 97.5%: {100*np.mean(above):.0f}%; rho median {np.median(rhos):.2f}, a median {np.median(As):.2f}"
    )
    if Nsim:
        Nsim = np.array(Nsim, float).mean(0)
        d = np.array(q15[name]["data"])
        out.update({"sim_dep": Nsim.tolist(), "data": d.tolist(), "diff": (Nsim - d).tolist()})
        line += (
            f" | N_cv dep-sim {np.round(Nsim,2).tolist()} vs data {np.round(d,2).tolist()} (matched-indep {np.round(q15[name]['matched'],2).tolist()}); "
            f"diff at m=all {Nsim[-1]-d[-1]:+.2f}, at m=8/16 {Nsim[0]-d[0]:+.2f}/{Nsim[1]-d[1]:+.2f}"
        )
    print(line, flush=True)
    res[name] = out
json.dump(res, open(os.path.join(HERE, "results_q16_m1fix.json"), "w"), indent=1)
print("[saved]")
