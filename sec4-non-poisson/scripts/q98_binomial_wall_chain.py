"""
Q98: the binomial WALL by the half-line chain (the setting of Q74/Q75 with the binomial kernel). 2026-09-06.
Wall at p = 0 (tau = 0), m = 30 free atoms in tau = 2 sqrt(n) arcsin(sqrt p), frozen periodic tail of 10
atoms at gap g (weight 1 each; the KKT conditions are scale invariant in the weights). Output y = 0..n.
For n in the list, the chain is solved (scipy root, lm, from the Poisson chain of Q75 at the same g as the
initial guess) and the gap excess over the Gaussian chain at the same g is printed, gap by gap, next to the
Poisson excess 0.407, 0.091, 0.047, 0.031, 0.020, 0.013, 0.008 (g = 1.55). Prediction: the binomial excess
tends to the Poisson one as n -> infinity (the wall lies at np >> 1, p << 1), and differs at moderate n
in the outer gaps first. The sum of the excess is the binomial wall constant, to be compared with the
0.36 per wall found from the counts at n = 30..100 (Statement 12) and the Poisson 0.30.
Usage: py q98_binomial_wall_chain.py <n list> [<g, default 1.55>]
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


def logB(n, p, y):
    p = np.asarray(p, float)[:, None]
    y = np.asarray(y, float)[None, :]
    lp = np.where(p > 0, np.log(np.maximum(p, 1e-300)), -np.inf)
    l1p = np.where(p < 1, np.log(np.maximum(1 - p, 1e-300)), -np.inf)
    out = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1) + y * lp + (n - y) * l1p
    out = np.where((y == 0) & (p == 0), 0.0, out)
    out = np.where((y > 0) & (p == 0), -np.inf, out)
    return out


def chain_binom(n, g, t_init, w_init):
    ymax = int(min(n, 4 * ((t_init[-1] + g * n_t) / (2 * np.sqrt(n))) ** 2 * n + 30 * np.sqrt(n) + 60))
    y = np.arange(ymax + 1)
    tail = lambda tm: tm + g * np.arange(1, n_t + 1)

    def p_of(t):
        return np.sin(np.clip(t, 0, np.pi * np.sqrt(n)) / (2 * np.sqrt(n))) ** 2

    def F(v):
        tt = v[:m]
        ww = v[m:]
        tt_all = np.concatenate([[0.0], tt, tail(tt[-1])])
        pp = p_of(tt_all)
        allw = np.concatenate([ww, np.ones(n_t)])
        LP = logB(n, pp, y)
        Q = allw @ np.exp(LP)
        lQ = np.log(np.maximum(Q, 1e-300))
        LPq = LP[: m + 2]
        Pq = np.exp(LPq)
        gg = np.where(Pq > 0, LPq - lQ[None, :], 0.0)
        D = (Pq * gg).sum(1)
        pq = pp[: m + 2][:, None]
        score_p = np.where(
            (pq > 0) & (pq < 1),
            y[None, :] / np.maximum(pq, 1e-300) - (n - y[None, :]) / np.maximum(1 - pq, 1e-300),
            0.0,
        )
        dp_dt = np.sqrt(np.maximum(pq * (1 - pq), 0)) / np.sqrt(n)  # dp/dtau for tau = 2 sqrt(n) asin sqrt p
        Dp = (Pq * score_p * dp_dt * gg).sum(1)
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
for n in [int(v) for v in sys.argv[1].split(",")]:
    tB, wB, res = chain_binom(n, g, tP0, wP0)
    gapsB = np.diff(np.concatenate([[0], tB]))
    exc = gapsB - np.diff(np.concatenate([[0], uG0]))
    p_atoms = np.sin(tB / (2 * np.sqrt(n))) ** 2
    say(
        f"n = {n}: residual {res:.1e}; binomial gaps {np.round(gapsB[:7], 4).tolist()}; excess over Gaussian {np.round(exc[:8], 4).tolist()}; sum(excess) {exc.sum():.4f}; p at atoms 1..6 {np.round(p_atoms[:6], 4).tolist()}"
    )
    out[str(n)] = {
        "residual": float(res),
        "gaps": gapsB.tolist(),
        "excess": exc.tolist(),
        "sum_excess": float(exc.sum()),
        "p_atoms": p_atoms.tolist(),
    }
json.dump(out, open(os.path.join(HERE, f"results_q98_binomial_wall_g{g}.json"), "w"), indent=1)
