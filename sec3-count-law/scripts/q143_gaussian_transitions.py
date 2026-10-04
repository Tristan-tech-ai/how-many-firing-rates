"""Q143: transitions A_g(K+1) of the unit-noise Gaussian channel Y = X + Z on [-A, A] for large K, by a mirror-reduced
KKT solver (float, scipy hybr) with a structured guess (wall profile of the stored K = 25 Gaussian point of q84,
interior uniform) and a bisection in A of the K-point's centre signal (D(0) - C for even K, centre curvature for odd K).
Validation: the 25-point must give A_g(26) = 20.6403 (q84 unfolding).  Output results_q143_Ag.json.
usage: py q143_gaussian_transitions.py <K_lo> <K_hi> [A_start]
"""

import sys, os, json, time
import numpy as np
from scipy.optimize import root

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()
K_lo, K_hi = int(sys.argv[1]), int(sys.argv[2])
A_start = float(sys.argv[3]) if len(sys.argv) > 3 else None
H = 0.02
L2P = 0.5 * np.log(2 * np.pi)


def grid(A):
    return np.arange(-A - 9.0, A + 9.0 + 1e-9, H)


def full(A, u, w, wc):
    xr = np.concatenate([u, [A]])
    xl = -xr[::-1]
    x = np.concatenate([xl, [0.0], xr]) if wc is not None else np.concatenate([xl, xr])
    ww = np.concatenate([w[::-1], [wc], w]) if wc is not None else np.concatenate([w[::-1], w])
    return x, ww


def stats(A, x, ww, y=None):
    y = grid(A) if y is None else y
    L = -0.5 * (y[None, :] - x[:, None]) ** 2 - L2P
    P = np.exp(L)
    Q = ww @ P
    lQ = np.log(np.maximum(Q, 1e-300))
    G = L - lQ[None, :]
    D = H * (P * G).sum(1)
    Dp = H * (P * (y[None, :] - x[:, None]) * G).sum(1)
    return D, Dp, lQ, y


def probe(A, x, ww):
    D, Dp, lQ, y = stats(A, x, ww)
    C = D[-1]
    xs = np.array([-0.05, 0.0, 0.05])
    L = -0.5 * (y[None, :] - xs[:, None]) ** 2 - L2P
    P = np.exp(L)
    Dv = H * (P * (L - lQ[None, :])).sum(1) - C
    return Dv[1], (Dv[0] - 2 * Dv[1] + Dv[2]) / 0.05**2, C, D


def solve(A, u0, w0, wc0):
    m = len(u0)
    odd = wc0 is not None

    def F(v):
        u = v[:m]
        w = v[m : 2 * m + 1]
        wc = v[2 * m + 1] if odd else None
        x, ww = full(A, u, w, wc)
        D, Dp, _, _ = stats(A, x, ww)
        K = len(x)
        h = K // 2
        right = np.arange(h, K)
        eqD = D[right[:-1]] - D[right[-1]]
        eqDp = Dp[right[(1 if odd else 0) : -1]]
        return np.concatenate([eqD, eqDp, [ww.sum() - 1]])

    v0 = np.concatenate([u0, w0] + ([[wc0]] if odd else []))
    sol = root(F, v0, method="hybr", options={"xtol": 1e-13, "maxfev": 20000})
    res = np.abs(F(sol.x)).max()
    if res > 1e-10:
        sol2 = root(F, sol.x, method="lm", options={"xtol": 1e-14, "ftol": 1e-14, "maxiter": 3000})
        res2 = np.abs(F(sol2.x)).max()
        if res2 < res:
            sol, res = sol2, res2
    v = sol.x
    u = v[:m]
    w = v[m : 2 * m + 1]
    wc = v[2 * m + 1] if odd else None
    ok = (
        res < 3e-8
        and np.all(np.diff(u) > 0)
        and u[0] > 0
        and u[-1] < A
        and w.min() > 0
        and (wc is None or wc > 0)
    )
    return u, w, wc, res, ok


# ---- reference wall profile from the stored Gaussian 25-point (q84)
ref = json.load(open(os.path.join(HERE, "results_q84_gauss_chain_20_34.json")))
xr = np.array([float(v) for v in ref["gauss_K25_us"]])
wr = np.array([float(v) for v in ref["gauss_K25_ws"]])
A_ref = -xr[0]
gaps_r = np.diff(xr)
NW = 8
wall_gaps = gaps_r[:NW]
wall_w = wr[: NW + 1]
w_int_ref = float(np.median(wr[NW + 1 : 25 - NW - 1]))
say(
    f"reference Gaussian K = 25 at A = {A_ref:.4f}: wall gaps {np.round(wall_gaps, 4).tolist()}, wall weights {np.round(wall_w, 4).tolist()}, interior weight {w_int_ref:.4f}"
)


def guess(A, K):
    m = K - 2 * (NW + 1)
    L = 2 * A
    Lw = wall_gaps.sum()
    g = (L - 2 * Lw) / (m + 1)
    t = np.concatenate([[0.0], np.cumsum(wall_gaps)])
    t = np.concatenate([t, t[-1] + g * np.arange(1, m + 1)])
    t = np.concatenate([t, L - t[: NW + 1][::-1]])
    t = np.sort(t)
    x = t - A
    w = np.concatenate(
        [wall_w * (25 / K) ** 0.5, np.full(m, w_int_ref * (25 / K) ** 0.8), wall_w[::-1] * (25 / K) ** 0.5]
    )
    w /= w.sum()
    h = K // 2
    odd = K % 2 == 1
    return x[h + (1 if odd else 0) : -1], w[h + (1 if odd else 0) :], (w[h] if odd else None)


LAST = {}


def offatom_max(A, x, ww):
    D, Dp, lQ, y = stats(A, x, ww)
    C = D[-1]
    xg = np.linspace(-A, A, int(2 * A / 0.02) + 1)
    L = -0.5 * (y[None, :] - xg[:, None]) ** 2 - L2P
    P = np.exp(L)
    Dg = H * (P * (L - lQ[None, :])).sum(1) - C
    keep = np.ones(len(xg), bool)
    for xa in x:
        keep &= np.abs(xg - xa) > 0.05
    return float(Dg[keep].max()) if keep.any() else -np.inf


def signal(A, K):
    st = LAST.get(K)
    if st is not None:  # continuation from the last converged K-point (positions scaled with A)
        u0 = st[0] * (A / st[3])
        w0 = st[1]
        wc0 = st[2]
        u, w, wc, res, ok = solve(A, u0, w0, wc0)
    else:
        ok = False
    if (
        not ok and (K - 1) in LAST
    ):  # birth guess from the last converged (K-1)-point: insert a centre atom or split the centre
        up, wp, wcp, Ap = LAST[K - 1]
        sc = A / Ap
        if wcp is None:
            u0 = up * sc
            w0 = wp * (1 - 0.012)
            wc0 = 0.012  # even -> odd: insertion at the centre
        else:
            u0 = np.concatenate([[0.30], up * sc])
            w0 = np.concatenate([[wcp / 2], wp])
            wc0 = None  # odd -> even: split of the centre
        u, w, wc, res, ok = solve(A, u0, w0, wc0)
    if not ok:
        u0, w0, wc0 = guess(A, K)
        u, w, wc, res, ok = solve(A, u0, w0, wc0)
    x, ww = full(A, u, w, wc)
    Dc, curv, C, D = probe(A, x, ww)
    if ok:
        vmax = offatom_max(A, x, ww)
        if vmax > 1e-6:
            ok = False
            res = vmax  # a K-point on a wrong branch violates KKT away from the centre
    if ok:
        LAST[K] = (u, w, wc, A)
    return (Dc if wc is None else curv), ok, res, C


out = {}
A = A_start
for K in range(K_lo, K_hi + 1):
    if A is None:
        A = 0.68 * K + 2.6 + 0.45  # above the K-th birth (A_g(K) ~ 0.68 K + 2.9), where the K-point is mature
    # find a bracket: scan upward in steps of 0.25 from a point where the K-point is clean and the signal negative
    tries = 0
    lo = None
    downs = 0
    while tries < 80:
        s, ok, res, C = signal(A, K)
        tries += 1
        if not ok:
            say(f"   K = {K}: A = {A:.3f} solve not clean (res {res:.1e}); stepping up")
            A += 0.1
            continue
        if s < 0:
            lo = (A, s)
            A += 0.1
            continue
        if lo is None:
            downs += 1
            A -= 0.05
            if downs > 40:
                break
            continue
        hi = (A, s)
        break
    if tries >= 60 or lo is None:
        say(f"K = {K}: no bracket; stop")
        break
    a, sa = lo
    b, sb = hi
    for it in range(20):
        c = 0.5 * (a + b)
        s, ok, res, C = signal(c, K)
        if not ok:
            c = a + 0.37 * (b - a)
            s, ok, res, C = signal(c, K)  # try an off-centre point before giving up
            if not ok:
                say(f"   bisection solve not clean at A = {c:.4f} (res {res:.1e})")
                break
        if s < 0:
            a, sa = c, s
        else:
            b, sb = c, s
        if b - a < 2e-4:
            break
    A_star = a + (-sa) * (b - a) / (sb - sa)
    typ = "insertion" if K % 2 == 0 else "split"
    out[K + 1] = A_star
    say(f"A_g({K + 1}) = {A_star:.5f} ({typ}; bracket {a:.4f} .. {b:.4f})   [{time.time()-t0:.0f}s]")
    json.dump(out, open(os.path.join(HERE, f"results_q143_Ag_K{K_lo}_{K_hi}.json"), "w"), indent=1)
    A = A_star + 0.12  # transitions can come 0.2 apart (K = 42, 43): start close above the last one
say(f"done [{time.time()-t0:.0f}s]")
