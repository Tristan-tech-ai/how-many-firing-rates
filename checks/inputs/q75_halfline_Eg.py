"""
Q75: the half-line excess E(g) by continuation in the bulk gap g, warm-started from the clean g = 1.55
solution of Q74 (results_q74_halfline_long.json). 2026-09-06.
Down: 1.55 -> 1.37 in steps of 0.02; up: 1.55 -> 1.75 in steps of 0.04. For each g and each kernel the
30-atom free chain with a 10-atom frozen periodic tail; accept only clean solutions (residual < 1e-9 and
every gap within 0.15 of a smooth relaxation, i.e. no junction collapse). Output E(g) = t_30 - u_30 and the
gap-by-gap excess. Prediction to test: c_infinity = E(g_infinity)/2 with g_infinity = 2 * (Gaussian
transition spacing) -> about 1.41; and c(K) = E(g_eff(K))/2.
"""
import os, sys, json
import numpy as np
from scipy.optimize import root
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import q34_poisson_dark as qp
import q33_gauss_support as qg
say = lambda *a: print(*a, flush=True)
m, n_t = 30, 10
prev = json.load(open(os.path.join(HERE, "results_q74_halfline_long.json")))["1.55"]


def chain_poisson(g, t_init, w_init):
    tail = lambda tm: tm + g * np.arange(1, n_t + 1)
    tmax = t_init[-1] + g * n_t + 6; y = np.arange(int((tmax / 2) ** 2 + 18 * (tmax / 2) + 60))

    def F(v):
        tt = v[:m]; ww = v[m:]
        tt_all = np.concatenate([[0.0], tt, tail(tt[-1])]); xx = (tt_all / 2) ** 2
        allw = np.concatenate([ww, np.ones(n_t)])
        Dv, Dp = qp.D_and_Dp(xx[:m + 2], xx, allw, 0.0, y)
        return np.concatenate([Dv[1:m + 1] - Dv[0], Dp[1:m + 1], [Dv[m + 1] - Dv[0]]])
    sol = root(F, np.concatenate([t_init, w_init]), method="lm", options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 20000})
    return sol.x[:m], sol.x[m:], np.abs(F(sol.x)).max()


def chain_gauss(g, u_init, w_init):
    tail = lambda um: um + g * np.arange(1, n_t + 1)
    umax = u_init[-1] + g * n_t + 8; y, dy = qg.grid(umax / 2 + 2); shift = umax / 2

    def F(v):
        uu = v[:m]; ww = v[m:]
        uu_all = np.concatenate([[0.0], uu, tail(uu[-1])]) - shift
        allw = np.concatenate([ww, np.ones(n_t)])
        Dv, Dp = qg.D_and_Dp(uu_all[:m + 2], uu_all, allw, y, dy)
        return np.concatenate([Dv[1:m + 1] - Dv[0], Dp[1:m + 1], [Dv[m + 1] - Dv[0]]])
    sol = root(F, np.concatenate([u_init, w_init]), method="lm", options={"xtol": 1e-12, "ftol": 1e-12, "maxiter": 20000})
    return sol.x[:m], sol.x[m:], np.abs(F(sol.x)).max()


def clean(pos, g, res):
    gaps = np.diff(np.concatenate([[0], pos]))
    return res < 1e-9 and np.all(np.diff(gaps[:12]) <= 0.02) and np.all(np.abs(gaps[12:] - g) < 0.15)


def rescale(pos, g_old, g_new):
    # keep the wall, stretch the bulk: positions beyond atom 8 move with the gap
    pos = np.array(pos, float); out = pos.copy()
    for k in range(8, m):
        out[k] = out[k - 1] + (pos[k] - pos[k - 1]) * (g_new / g_old)
    return out


out = {}
for direction, gs in (("down", np.round(np.arange(1.53, 1.36, -0.02), 2)), ("up", np.round(np.arange(1.59, 1.76, 0.04), 2))):
    tP, wP, uG, wG, g_old = np.array(prev["tP"]), np.array(prev["wP"]), np.array(prev["uG"]), np.array(prev["wG"]), 1.55
    for g in gs:
        tP2, wP2, rP = chain_poisson(g, rescale(tP, g_old, g), wP)
        uG2, wG2, rG = chain_gauss(g, rescale(uG, g_old, g), wG)
        okP, okG = clean(tP2, g, rP), clean(uG2, g, rG)
        if not (okP and okG):
            say(f"g = {g}: not clean (P {okP} res {rP:.0e}; G {okG} res {rG:.0e}); stop this direction"); break
        tP, wP, uG, wG, g_old = tP2, wP2, uG2, wG2, g
        dg = np.diff(np.concatenate([[0], tP])) - np.diff(np.concatenate([[0], uG]))
        E = tP[-1] - uG[-1]
        say(f"g = {g}: E = {E:.4f}, E/2 = {E/2:.4f}; gap excess {np.round(dg[:8], 4).tolist()}; bulk gap excess (13..30) mean {dg[12:].mean():+.5f}; t_1 = {tP[0]:.4f}, u_1 = {uG[0]:.4f}")
        out[str(g)] = {"E": float(E), "tP": tP.tolist(), "uG": uG.tolist(), "wP": wP.tolist(), "wG": wG.tolist(), "resP": rP, "resG": rG}
    json.dump(out, open(os.path.join(HERE, "results_q75_Eg.json"), "w"), indent=1)
say("done")
