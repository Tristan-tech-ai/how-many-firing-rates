"""Symmetric binomial channel Bin(n, p): continuation in n with the MIRROR-REDUCED KKT system (symmetry imposed,
so the antisymmetric soft mode of a central pitchfork is absent).  Unknowns: right-half interior positions
u_1 < ... < u_m (in p, > 1/2; the endpoint p = 1 pinned), right-half weights w_1..w_m plus the endpoint weight,
and the centre weight w_c for odd K.  Equations: D(u_j) = D(endpoint) for all right-half atoms and the centre,
D'(u_j) = 0 for the interior right-half atoms, mass = 1.  Births: odd K -> split of the centre into +-d (K+1);
even K -> insertion at the centre (K+1) when D(1/2) - C > 0.  usage: py q134 <n0> <K0> <n_max>
"""

import sys, numpy as np, json, os
from scipy.special import gammaln
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
n0, K0, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
say = lambda *a: print(*a, flush=True)


def kern(n, x):
    y = np.arange(int(n) + 1)
    mu = np.asarray(x, float)[:, None]
    pp = np.clip(mu / n, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (
            gammaln(n + 1)
            - gammaln(y + 1)[None, :]
            - gammaln(n - y + 1)[None, :]
            + np.where(y[None, :] > 0, y[None, :] * np.log(np.maximum(pp, 1e-300)), 0.0)
            + np.where((n - y)[None, :] > 0, (n - y)[None, :] * np.log(np.maximum(1 - pp, 1e-300)), 0.0)
        )
    L = np.where((pp >= 1) & (y[None, :] < n), -np.inf, L)
    L = np.where((pp <= 0) & (y[None, :] > 0), -np.inf, L)
    S = np.where(
        (mu > 0) & (mu < n),
        y[None, :] / np.maximum(mu, 1e-300) - (n - y)[None, :] / np.maximum(n - mu, 1e-300),
        0.0,
    )
    return L, S


def full(n, u, w, wc):
    """right half (interior u ascending in x, endpoint n) -> full configuration"""
    xr = np.concatenate([u, [float(n)]])
    xl = n - xr[::-1]
    x = np.concatenate([xl, [n / 2], xr]) if wc is not None else np.concatenate([xl, xr])
    ww = np.concatenate([w[::-1], [wc], w]) if wc is not None else np.concatenate([w[::-1], w])
    return x, ww


def stats(n, x, ww):
    L, S = kern(n, x)
    PP = np.exp(L)
    QQ = ww @ PP
    lQQ = np.log(np.maximum(QQ, 1e-300))
    gg = np.where(PP > 0, L - lQQ[None, :], 0.0)
    return (PP * gg).sum(1), (PP * S * gg).sum(1), lQQ


def solve(n, u0, w0, wc0):
    m = len(u0)
    odd = wc0 is not None

    def unpack(v):
        u = v[:m]
        w = v[m : 2 * m + 1]
        wc = v[2 * m + 1] if odd else None
        return u, w, wc

    def F(v):
        u, w, wc = unpack(v)
        x, ww = full(n, u, w, wc)
        D, Dp, _ = stats(n, x, ww)
        K = len(x)
        h = K // 2
        right = np.arange(h, K)  # centre (if odd) and right half
        eqD = D[right[:-1]] - D[right[-1]]  # D equal at the centre/right atoms and the right endpoint
        eqDp = Dp[right[(1 if odd else 0) : -1]]  # D' = 0 at the right-half interior atoms
        return np.concatenate([eqD, eqDp, [ww.sum() - 1]])

    v0 = np.concatenate([u0, w0, [wc0]]) if odd else np.concatenate([u0, w0])
    sol = root(F, v0, method="hybr", options={"xtol": 1e-14, "maxfev": 200000})
    res = np.abs(F(sol.x)).max()
    if res > 1e-11:
        sol = root(F, sol.x, method="lm", options={"xtol": 1e-15, "ftol": 1e-15, "maxiter": 100000})
        res = np.abs(F(sol.x)).max()
    u, w, wc = unpack(sol.x)
    return u, w, wc, res


def solve_held(n, u0, w0, wc_fixed):
    """centre weight held; the centre's D-equation dropped (the newborn is not at equilibrium yet)"""
    m = len(u0)

    def F(v):
        u = v[:m]
        w = v[m : 2 * m + 1]
        x, ww = full(n, u, w, wc_fixed)
        D, Dp, _ = stats(n, x, ww)
        K = len(x)
        h = K // 2
        right = np.arange(h + 1, K)
        return np.concatenate([D[right[:-1]] - D[right[-1]], Dp[right[:-1]], [ww.sum() - 1]])

    v0 = np.concatenate([u0, w0])
    sol = root(F, v0, method="hybr", options={"xtol": 1e-14, "maxfev": 200000})
    res = np.abs(F(sol.x)).max()
    if res > 1e-11:
        sol = root(F, sol.x, method="lm", options={"xtol": 1e-15, "ftol": 1e-15, "maxiter": 100000})
        res = np.abs(F(sol.x)).max()
    return sol.x[:m], sol.x[m : 2 * m + 1], wc_fixed, res


def analyse(n, u, w, wc):
    x, ww = full(n, u, w, wc)
    D, Dp, lQQ = stats(n, x, ww)
    C = D[0]
    pg = np.linspace(0.5, 1, 20001)
    Lg, _ = kern(n, pg * n)
    Pg = np.exp(Lg)
    Dg = (Pg * np.where(Pg > 0, Lg - lQQ[None, :], 0.0)).sum(1) - C
    tg = 2 * np.sqrt(n) * np.arcsin(np.sqrt(pg))
    tf = 2 * np.sqrt(n) * np.arcsin(np.sqrt(np.clip(x / n, 0, 1)))
    sub = 10
    Dc = Dg[::sub]
    tc = tg[::sub]
    loc = np.zeros_like(Dc, bool)
    loc[1:-1] = (Dc[1:-1] > Dc[:-2]) & (Dc[1:-1] > Dc[2:])
    for ta in tf:
        loc &= np.abs(tc - ta) > 0.05
    if loc.any():
        ic = int(np.argmax(np.where(loc, Dc, -np.inf)))
        i = ic * sub
        vmax, tpk, ppk = Dg[i], tg[i], pg[i]
    else:
        vmax, tpk, ppk = -np.inf, np.nan, np.nan
    curv = (Dg[20] - 2 * Dg[0] + Dg[20]) / (pg[20] - pg[0]) ** 2  # symmetric second difference at the centre
    return C, vmax, tpk, ppk, curv, tf, Dg[0]


WALL = [3.126, 2.260, 1.99, 1.82, 1.72, 1.65, 1.61, 1.59] + [1.58] * 40


def guess(n, K):
    Ltot = np.pi * np.sqrt(n)
    half = K // 2
    t = [0.0]
    for g in WALL[: half - 1]:
        t.append(t[-1] + g)
    t = np.array(t)
    pos = (
        np.concatenate([t, [Ltot / 2], Ltot - t[::-1]]) if K % 2 == 1 else np.concatenate([t, Ltot - t[::-1]])
    )
    pos = np.sort(pos) * (Ltot / pos[-1])
    x = n * np.sin(np.clip(pos / (2 * np.sqrt(n)), 0, np.pi / 2)) ** 2
    x[0] = 0.0
    x[-1] = n
    h = K // 2
    odd = K % 2 == 1
    u = x[h + (1 if odd else 0) : -1]
    w = np.ones(len(u) + 1) / K
    wc = (1.0 / K) if odd else None
    return u, w, wc


u, w, wc, res = solve(n0, *guess(n0, K0))
say(f"start n = {n0}, K = {K0}: residual {res:.1e}, min w {min(w.min(), wc if wc is not None else 1):.4f}")
if res > 1e-9:
    say("start invalid; stop")
    sys.exit(1)
out = []
n = n0
K = K0
n_split = None
d_last = None
while n < nmax:
    n1 = n + 1
    t = 2 * np.sqrt(n) * np.arcsin(np.sqrt(np.clip(u / n, 0, 1)))
    u1 = n1 * np.sin(np.clip(t / (2 * np.sqrt(n1)), 0, np.pi / 2)) ** 2
    if wc is None and n_split is not None and d_last is not None:
        # just above a pitchfork: the pair separation grows as sqrt(n - n*); put the guess on that law
        d_g = d_last * np.sqrt(max(n1 - n_split, 0.25) / max(n - n_split, 0.25))
        u1[0] = n1 * np.sin((np.pi * np.sqrt(n1) / 2 + d_g) / (2 * np.sqrt(n1))) ** 2
    u1, w1, wc1, res = solve(n1, u1, w, wc)
    if wc1 is None and n_split is not None:
        d_now = 2 * np.sqrt(n1) * np.arcsin(np.sqrt(min(u1[0] / n1, 1.0))) - np.pi * np.sqrt(n1) / 2
        if d_now < 0.5 * d_g:  # collapsed toward the merged pair: re-solve from a wider guess
            for fac in (1.5, 2.5, 4.0):
                ug = u1.copy()
                ug[0] = n1 * np.sin((np.pi * np.sqrt(n1) / 2 + fac * d_g) / (2 * np.sqrt(n1))) ** 2
                u2, w2, wc2, res2 = solve(n1, ug, w, wc)
                d2 = 2 * np.sqrt(n1) * np.arcsin(np.sqrt(min(u2[0] / n1, 1.0))) - np.pi * np.sqrt(n1) / 2
                if res2 < 1e-9 and w2.min() > 0 and d2 > 0.5 * d_g:
                    u1, w1, wc1, res = u2, w2, wc2, res2
                    say(f"   pair re-found at separation {d2:.4f} (guess {d_g:.4f}, factor {fac})")
                    break
        d_last = 2 * np.sqrt(n1) * np.arcsin(np.sqrt(min(u1[0] / n1, 1.0))) - np.pi * np.sqrt(n1) / 2
    u1 = np.sort(n1 / 2 + np.abs(u1 - n1 / 2))  # a mirrored solution is the same configuration
    wmin = min(w1.min(), wc1 if wc1 is not None else 1.0)
    if res > 2e-7 or wmin <= 0 or np.any(np.diff(u1) <= 0) or u1[0] <= n1 / 2:
        say(f"n = {n1}: K = {K} solve failed (res {res:.1e}, min w {wmin:.2e}); stop")
        break
    C, vmax, tpk, ppk, curv, tf, Dc = analyse(n1, u1, w1, wc1)
    born = ""
    if (
        wc1 is None and Dc > 1e-10
    ):  # even K: centre is a gap; insert the centre atom (held weight first, then release)
        u2, w2, wc2, res2 = solve_held(n1, u1, w1 * (1 - 2e-3), 2e-3)
        if res2 < 1e-8:
            u2, w2, wc2, res2 = solve(n1, u2, w2, wc2)
        if res2 < 2e-7 and w2.min() > 0 and wc2 > 0 and np.all(np.diff(u2) > 0):
            u1, w1, wc1 = u2, w2, wc2
            K += 1
            n_split = None
            d_last = None
            born = f" INSERTION at the centre (D(1/2)-C was {Dc:.1e}); newborn w = {wc2:.5f}"
        else:
            born = f" centre insertion failed (res {res2:.1e}, wc {wc2:.2e})"
    elif wc1 is not None and curv > 0:  # odd K: the centre atom splits into a mirror pair
        for dt in (0.3, 0.5, 0.8):
            ur = (
                n1 * np.sin(dt / (2 * np.sqrt(n1))) ** 2 * 0
                + n1 * np.sin((np.pi * np.sqrt(n1) / 2 + dt) / (2 * np.sqrt(n1))) ** 2
            )
            u2, w2, wc2, res2 = solve(n1, np.concatenate([[ur], u1]), np.concatenate([[wc1 / 2], w1]), None)
            if res2 < 1e-9 and w2.min() > 0 and u2[0] > n1 / 2 and np.all(np.diff(u2) > 0):
                u1, w1, wc1 = u2, w2, None
                K += 1
                n_split = n1 - 0.25
                d_last = 2 * np.sqrt(n1) * np.arcsin(np.sqrt(u2[0] / n1)) - np.pi * np.sqrt(n1) / 2
                born = f" SPLIT of the centre (curv {curv:.1e}); pair half-separation {d_last:.4f} Fisher; pair w = {w2[0]:.5f}"
                break
        if not born:
            born = f" split failed (curv {curv:.1e})"
    C, vmax, tpk, ppk, curv, tf, Dc = analyse(n1, u1, w1, wc1)
    u, w, wc, n = u1, w1, wc1, n1
    say(
        f"n = {n}: K = {K}, pair d = {(d_last if (wc is None and d_last is not None) else float('nan')):.4f}, C = {C:.6f}, min w {min(w.min(), wc if wc is not None else 1):.5f}, off-atom max D-C = {vmax:+.2e} at t = {tpk:.3f}, centre D-C = {Dc:+.2e}, curv = {curv:+.2e}, h = {np.pi*np.sqrt(n)/2:.3f}{born}"
    )
    out.append(
        {
            "n": n,
            "K": K,
            "C": C,
            "w_min": float(min(w.min(), wc if wc is not None else 1)),
            "centre_DC": float(Dc),
            "curv": float(curv),
            "gaps": np.round(np.diff(tf), 4).tolist(),
            "born": born,
        }
    )
    json.dump(out, open(os.path.join(HERE, f"results_q134_sym_bin_from{n0}.json"), "w"), indent=1)
