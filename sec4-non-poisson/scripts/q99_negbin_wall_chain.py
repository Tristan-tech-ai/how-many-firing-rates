"""
Q99: the fifth family, over-dispersed: negative binomial NB(r, mean mu), variance mu + mu^2/r. 2026-09-06.
Fisher coordinate: I(mu) = 1/mu - 1/(mu + r) ... precisely I(mu) = r/(mu (mu + r)), so
tau(mu) = int sqrt(I) dmu = 2 sqrt(r) asinh(sqrt(mu/r)); degenerate endpoint at mu = 0 like Poisson;
skewness kappa_3 = (1 + 2 mu/r)/sqrt(mu (1 + mu/r)) -> 2/tau as mu -> 0 (Poisson profile) with an
over-dispersion correction. Prediction (Statement 14 sign rule): the wall is NARROWER than the Poisson
wall (sum of the gap excess below 0.6212 at g = 1.55), tending to it as r -> infinity.
Half-line chain as in Q98 (wall at mu = 0, 30 free atoms in tau, periodic tail). Output y = 0..ymax.
Usage: py q99_negbin_wall_chain.py <r list> [<g, default 1.55>]
"""

import os, sys, json
import numpy as np
from scipy.optimize import root
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
np.seterr(all="ignore")
m, n_t = 30, 10
g = float(sys.argv[2]) if len(sys.argv) > 2 else 1.55
_E75 = json.load(open(os.path.join(HERE, "results_q75_Eg.json")))
_E74 = json.load(open(os.path.join(HERE, "results_q74_halfline_long.json")))
E = _E74.get(str(g)) or _E75.get(str(g)) or _E75.get(f"{g:.2f}") or _E74.get(f"{g:.2f}")
tP0, wP0, uG0 = np.array(E["tP"]), np.array(E["wP"]), np.array(E["uG"])


def mu_of(tau, r):
    return r * np.sinh(np.asarray(tau, float) / (2 * np.sqrt(r))) ** 2


def logNB(r, mu, y):
    """log pmf of NB with mean mu, shape r: C(y+r-1, y) (r/(r+mu))^r (mu/(r+mu))^y."""
    mu = np.asarray(mu, float)[:, None]
    y = np.asarray(y, float)[None, :]
    lmu = np.where(mu > 0, np.log(np.maximum(mu, 1e-300)), -np.inf)
    out = gammaln(y + r) - gammaln(y + 1) - gammaln(r) + r * np.log(r) - (y + r) * np.log(r + mu) + y * lmu
    out = np.where((y == 0) & (mu == 0), 0.0, out)
    out = np.where((y > 0) & (mu == 0), -np.inf, out)
    return out


def chain_nb(r, g, t_init, w_init):
    tmax = t_init[-1] + g * n_t
    mumax = mu_of(tmax, r)
    ymax = int(mumax + 12 * np.sqrt(mumax + mumax**2 / r) + 60)
    y = np.arange(ymax + 1)
    tail = lambda tm: tm + g * np.arange(1, n_t + 1)

    def F(v):
        tt = v[:m]
        ww = v[m:]
        tt_all = np.concatenate([[0.0], tt, tail(tt[-1])])
        mm = mu_of(tt_all, r)
        allw = np.concatenate([ww, np.ones(n_t)])
        LP = logNB(r, mm, y)
        Q = allw @ np.exp(LP)
        lQ = np.log(np.maximum(Q, 1e-300))
        LPq = LP[: m + 2]
        Pq = np.exp(LPq)
        gg = np.where(Pq > 0, LPq - lQ[None, :], 0.0)
        D = (Pq * gg).sum(1)
        mq = mm[: m + 2][:, None]
        score_mu = np.where(mq > 0, y[None, :] / np.maximum(mq, 1e-300) - (y[None, :] + r) / (r + mq), 0.0)
        dmu_dt = np.sqrt(np.maximum(mq * (mq + r), 0)) / np.sqrt(r)  # dmu/dtau = 1/sqrt(I)
        Dp = (Pq * score_mu * dmu_dt * gg).sum(1)
        return np.concatenate([D[1 : m + 1] - D[0], Dp[1 : m + 1], [D[m + 1] - D[0]]])

    sol = root(
        F,
        np.concatenate([t_init, w_init]),
        method="lm",
        options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 20000},
    )
    return sol.x[:m], sol.x[m:], np.abs(F(sol.x)).max()


poisson_excess = np.diff(np.concatenate([[0], tP0])) - np.diff(np.concatenate([[0], uG0]))
say(
    f"g = {g}: Poisson wall excess (Q75): {np.round(poisson_excess[:8], 4).tolist()}, sum {poisson_excess.sum():.4f}"
)
out = {}
for r in [float(v) for v in sys.argv[1].split(",")]:
    tN, wN, res = chain_nb(r, g, tP0, wP0)
    gapsN = np.diff(np.concatenate([[0], tN]))
    exc = gapsN - np.diff(np.concatenate([[0], uG0]))
    say(
        f"r = {r:g}: residual {res:.1e}; NB gaps {np.round(gapsN[:7], 4).tolist()}; excess over Gaussian {np.round(exc[:8], 4).tolist()}; sum(excess) {exc.sum():.4f} (Poisson 0.6212); mu at atoms 1..4 {np.round(mu_of(tN[:4], r), 3).tolist()}"
    )
    out[str(r)] = {
        "residual": float(res),
        "gaps": gapsN.tolist(),
        "excess": exc.tolist(),
        "sum_excess": float(exc.sum()),
    }
json.dump(out, open(os.path.join(HERE, f"results_q99_negbin_wall_g{g}.json"), "w"), indent=1)
