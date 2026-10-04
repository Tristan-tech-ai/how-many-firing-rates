"""
Q101: the sixth family, generalized Poisson (Consul), dispersion index CONSTANT in mu. 2026-09-08.
GP(theta, lam): P(y) = theta (theta + lam y)^(y-1) exp(-theta - lam y) / y!, mean mu = theta/(1-lam),
variance theta/(1-lam)^3, so Var/Mean = 1/(1-lam)^2 =: 1 + Delta with Delta ~ 2 lam, the SAME for every mu.
To first order in Delta its standardised cumulants match the negative binomial at the same local Delta
(skewness (1 + 1.5 Delta)/sqrt(mu), excess kurtosis (1 + 5 Delta)/mu), so the per-gap law measured on
NB/binomial (Statement 16: delta_k = -c_k Delta_k excess_k, Delta_k = mu_k/r) predicts, with NO free
parameter, delta_k / Delta = -c_k excess_k for the GP chain, i.e. the same c_k profile, now against a
constant Delta. Half-line chain as in Q99 (wall at mu = 0, 30 free atoms, periodic tail g = 1.55 in the
GP Fisher coordinate computed by quadrature of the exact Fisher information).
Usage: py q101_genpoisson_wall_chain.py <lam list>
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
g = 1.55
E = json.load(open(os.path.join(HERE, "results_q74_halfline_long.json")))["1.55"]
tP0, wP0, uG0 = np.array(E["tP"]), np.array(E["wP"]), np.array(E["uG"])
gapsG = np.diff(np.concatenate([[0], uG0]))
excP = np.diff(np.concatenate([[0], tP0])) - gapsG
muP = (tP0 / 2) ** 2
N = json.load(open(os.path.join(HERE, "results_q99_negbin_wall_g1.55.json")))
c_k = (
    -(np.array(N["1000.0"]["excess"]) - excP) * 1000.0 / np.where(excP != 0, muP * excP, np.nan)
)  # NB profile


def gp(mu, y, lam):
    """log pmf and score d log p / d mu for GP with mean mu (theta = mu (1 - lam)); mu (n,), y (ny,)."""
    mu = np.asarray(mu, float)[:, None]
    y = np.asarray(y, float)[None, :]
    th = mu * (1 - lam)
    a = th + lam * y
    la = np.where(a > 0, np.log(np.maximum(a, 1e-300)), -np.inf)
    lth = np.where(th > 0, np.log(np.maximum(th, 1e-300)), -np.inf)
    lp = lth + (y - 1) * la - th - lam * y - gammaln(y + 1)
    lp = np.where((y == 0) & (mu == 0), 0.0, lp)
    lp = np.where((y > 0) & (mu == 0), -np.inf, lp)
    lp = np.where((a <= 0) & (y > 0), -np.inf, lp)
    # d/dtheta: 1/theta + (y-1)/a - 1 ; d theta/d mu = 1 - lam
    sc = np.where(th > 0, (1 / np.maximum(th, 1e-300) + (y - 1) / np.maximum(a, 1e-300) - 1) * (1 - lam), 0.0)
    return lp, sc


def tau_map(lam, mumax):
    """tau(mu) = int_0^mu sqrt(I) dmu' by the substitution mu = v^2 (dmu = 2 v dv): the integrand 2 (v sqrt(I) - 1)
    is smooth at v = 0, so tau = 2 v + int_0^v 2 (v' sqrt(I(v'^2)) - 1) dv'. (The trapezoid on a mu grid was wrong
    by -0.16 at the first node because of the 1/sqrt(mu) singularity; found 2026-09-08.)"""
    vg = np.linspace(1e-4, np.sqrt(mumax), 8001)
    grid = vg**2
    y = np.arange(int(mumax * 1.2 + 12 * np.sqrt(mumax) + 60) + 1)
    corr = np.zeros_like(vg)
    for a in range(0, len(vg), 500):
        mu = grid[a : a + 500]
        lp, sc = gp(mu, y, lam)
        p = np.exp(lp)
        I = (p * sc**2).sum(1)
        corr[a : a + 500] = 2 * (vg[a : a + 500] * np.sqrt(I) - 1)
    cum = np.concatenate([[0], np.cumsum(0.5 * (corr[1:] + corr[:-1]) * np.diff(vg))])
    taug = 2 * vg + cum
    return (lambda mu: 2 * np.sqrt(mu) + np.interp(np.sqrt(mu), vg, cum)), (
        lambda tau: np.interp(tau, taug, grid)
    )


def chain(lam):
    tmax = tP0[-1] + g * n_t
    mumax = (tmax / 2) ** 2 * 1.3
    y = np.arange(int(mumax * 1.2 + 12 * np.sqrt(mumax) + 60) + 1)
    tau_f, tau_inv = tau_map(lam, mumax)
    tail = lambda tm: 2 * np.sqrt(tau_inv(tau_f((tm / 2) ** 2) + g * np.arange(1, n_t + 1)))

    def F(v):
        tt = v[:m]
        ww = v[m:]
        tt_all = np.concatenate([[0.0], tt, tail(tt[-1])])
        mm = (tt_all / 2) ** 2
        allw = np.concatenate([ww, np.ones(n_t)])
        LP, SC = gp(mm, y, lam)
        P = np.exp(LP)
        Q = allw @ P
        lQ = np.log(np.maximum(Q, 1e-300))
        LPq = LP[: m + 2]
        Pq = P[: m + 2]
        gg = np.where(Pq > 0, LPq - lQ[None, :], 0.0)
        D = (Pq * gg).sum(1)
        Dp = (Pq * SC[: m + 2] * gg).sum(1) * np.sqrt(mm[: m + 2])
        return np.concatenate([D[1 : m + 1] - D[0], Dp[1 : m + 1], [D[m + 1] - D[0]]])

    sol = root(
        F, np.concatenate([tP0, wP0]), method="lm", options={"xtol": 1e-13, "ftol": 1e-13, "maxiter": 20000}
    )
    return sol.x[:m], sol.x[m:], np.abs(F(sol.x)).max(), tau_f


say(f"NB profile c_k (k = 1..12): {np.round(c_k[:12], 3).tolist()}")
out = {}
for lam in [float(v) for v in sys.argv[1].split(",")]:
    Delta = 1 / (1 - lam) ** 2 - 1
    tN, wN, res, tau = chain(lam)
    tauN = tau((tN / 2) ** 2)
    ex = np.diff(np.concatenate([[0], tauN])) - gapsG
    d = ex - excP
    pred = -c_k * Delta * excP
    flag = "" if res < 1e-9 else "  NOT CONVERGED"
    say(
        f"lam = {lam}: Delta = {Delta:.5f}; residual {res:.1e}{flag}; sum excess {ex.sum():.4f} (Poisson {excP.sum():.4f}); delta_total/Delta = {d.sum()/Delta:+.3f}, predicted {pred[:12].sum()/Delta:+.3f}"
    )
    say("   k, mu_k, delta_k/Delta measured, predicted (-c_k excess_k):")
    for k in range(12):
        say(f"     {k+1:2d} {muP[k]:7.2f}  {d[k]/Delta:+.4f}  {pred[k]/Delta:+.4f}")
    out[str(lam)] = {
        "Delta": Delta,
        "residual": float(res),
        "excess": ex.tolist(),
        "delta_over_Delta": (d / Delta).tolist(),
        "pred_over_Delta": (pred / Delta).tolist(),
    }
json.dump(out, open(os.path.join(HERE, "results_q101_genpoisson_wall_g1.55.json"), "w"), indent=1)
