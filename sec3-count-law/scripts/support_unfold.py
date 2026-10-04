"""
Unfolding the birth of an atom: KKT Newton with the NEW atom's weight fixed at eps and the amplitude A free.
2026-09-07. Idea test for the odd-K ceiling (K = 15 resisted five guesses at fixed A).

At a transition the K-atom KKT system is singular: the new atom's weight is zero and its Jacobian column
vanishes ("atoms cannot be inserted" in the record). The standard cure for such a bifurcation is to change
the parametrisation: hold the new weight at a small eps and let A move. Unknowns: the K - 1 other weights,
the K - 2 interior positions, and x_{K-1} = A; equations: D(x_i) = D(x_0) for i = 1..K-1, sum w = 1, and
D'(x_i) = 0 at the interior atoms; 2K - 2 of each. The Jacobian is the analytic one of support_mp3 with one
extra column, d/dA, which acts through the last atom: dD(x_i)/dx_{K-1} - dD(x_0)/dx_{K-1} plus D'(x_{K-1})
on the row i = K - 1, and dD'(x_i)/dx_{K-1} on the derivative rows. Increasing eps in steps moves A up the
new branch; a fixed-A Newton (support_mp3, pin_ends) then finishes.
Control: from the certified 13-atom input at A = 152 the unfolding must reproduce the 13 -> 14 birth at
A_14 = 152.5075 (eps -> 0) and hand over to a 14-atom fixed-A solution with the known boundary pattern
(first atom at t = 3.126).
"""
import sys
from mpmath import mp, mpf, exp, log, loggamma, matrix, lu_solve, nstr
from support_mp3 import _logP, _P, _tables, D_of, Dp_of, Dpp_of


def _entries(x, w, P, Q, ny):
    def dD_dw(xi, k):
        s = mpf(0)
        for y in range(ny):
            p = _P(xi, y)
            if p > 0 and Q[y] > 0:
                s -= p * P[k][y] / Q[y]
        return s

    def dD_dx(xi, k):
        if x[k] <= 0:
            return mpf(0)
        s = mpf(0)
        for y in range(ny):
            p = _P(xi, y)
            if p > 0 and Q[y] > 0:
                s -= p * w[k] * P[k][y] * (mpf(y) / x[k] - 1) / Q[y]
        return s

    def dDp_dw(xi, k):
        if xi <= 0:
            return mpf(0)
        s = mpf(0)
        for y in range(ny):
            p = _P(xi, y)
            if p > 0 and Q[y] > 0:
                s -= p * (mpf(y) / xi - 1) * P[k][y] / Q[y]
        return s

    def dDp_dx(xi, k):
        if xi <= 0 or x[k] <= 0:
            return mpf(0)
        s = mpf(0)
        for y in range(ny):
            p = _P(xi, y)
            if p > 0 and Q[y] > 0:
                s -= p * (mpf(y) / xi - 1) * w[k] * P[k][y] * (mpf(y) / x[k] - 1) / Q[y]
        return s
    return dD_dw, dD_dx, dDp_dw, dDp_dx


def solve_unfold(A0, pts, wts, fixed_idx, eps, dps=40, newton_steps=40, tol_pow=8, ny=None, verbose=False, fix_position=False, max_dx=None, pure_below=None, feasible_step=False):
    """pts include the boundary atoms (pts[0] = 0, pts[-1] = A0); wts[fixed_idx] is replaced by eps and held.
    Returns dict with x (incl. the new A as x[-1]), w, C, newton_resid, hist."""
    mp.dps = dps
    if ny is None:
        Ab = float(A0) * 1.2
        ny = int(Ab + 18 * Ab ** 0.5 + 60)
    x = [mpf(p) if isinstance(p, str) else mpf(repr(float(p))) for p in pts]
    w = [mpf(v) if isinstance(v, str) else mpf(repr(float(v))) for v in wts]
    K = len(x)
    w[fixed_idx] = mpf(eps)
    t = sum(w); w = [v / t for v in w]; w[fixed_idx] = mpf(eps)      # renormalise the others, hold eps
    others = [k for k in range(K) if k != fixed_idx]
    scale = (1 - mpf(eps)) / sum(w[k] for k in others); w = [w[k] * scale if k != fixed_idx else mpf(eps) for k in range(K)]
    interior = list(range(1, K - 1))
    if fix_position:                      # stage 1: the new atom's position is held, its D' row dropped
        interior = [i for i in interior if i != fixed_idx]
    n = (K - 1) + len(interior) + 1
    vec = [w[k] for k in others] + [x[i] for i in interior] + [x[K - 1]]
    hist = []

    def unpack(v):
        ww = list(w); xx = list(x)
        for j, k in enumerate(others):
            ww[k] = v[j]
        ww[fixed_idx] = mpf(eps)
        for j, i in enumerate(interior):
            xx[i] = v[(K - 1) + j]
        xx[K - 1] = v[-1]
        return xx, ww

    def resid(v):
        xx, ww = unpack(v)
        P, Q = _tables(xx, ww, ny)
        Dv = [D_of(xi, Q, ny) for xi in xx]
        r = [Dv[i] - Dv[0] for i in range(1, K)] + [sum(ww) - 1] + [Dp_of(xx[i], Q, ny) for i in interior]
        return r, xx, ww, P, Q, Dv

    for it in range(newton_steps):
        r, xx, ww, P, Q, Dv = resid(vec)
        nr = max(abs(v) for v in r)
        hist.append(nstr(nr, 4))
        if verbose:
            print(f"      unfold it {it}: resid {nstr(nr, 4)}  A = {nstr(xx[-1], 10)}", flush=True)
        if nr < mpf(10) ** (-dps + tol_pow):
            break
        dD_dw, dD_dx, dDp_dw, dDp_dx = _entries(xx, ww, P, Q, ny)
        Dp_at = {i: Dp_of(xx[i], Q, ny) for i in list(interior) + [K - 1]}
        Dpp_at = {i: Dpp_of(xx[i], Q, ny) for i in interior}
        J = matrix(n, n)
        # rows 0..K-2: D(x_i) - D(x_0), i = 1..K-1
        for row, i in enumerate(range(1, K)):
            for j, k in enumerate(others):
                J[row, j] = dD_dw(xx[i], k) - dD_dw(xx[0], k)
            for j, m_ in enumerate(interior):
                v = dD_dx(xx[i], m_) - dD_dx(xx[0], m_)
                if m_ == i:
                    v += Dp_at[i]
                J[row, (K - 1) + j] = v
            v = dD_dx(xx[i], K - 1) - dD_dx(xx[0], K - 1)
            if i == K - 1:
                v += Dp_at[K - 1]
            J[row, n - 1] = v
        # row K-1: sum w - 1
        for j in range(K - 1):
            J[K - 1, j] = mpf(1)
        # rows K..: D'(x_i), interior
        for a, i in enumerate(interior):
            row = K + a
            for j, k in enumerate(others):
                J[row, j] = dDp_dw(xx[i], k)
            for j, m_ in enumerate(interior):
                v = dDp_dx(xx[i], m_)
                if m_ == i:
                    v += Dpp_at[i]
                J[row, (K - 1) + j] = v
            J[row, n - 1] = dDp_dx(xx[i], K - 1)
        try:
            dv = lu_solve(J, matrix(r))
        except Exception:
            break

        def ok(cand):
            xx2, ww2 = unpack(cand)
            if min(ww2) <= 0 or any(xx2[i + 1] <= xx2[i] for i in range(K - 1)):
                return None
            return max(abs(v) for v in resid(cand)[0])

        if max_dx is not None:            # cap the position moves (flat D near a light atom gives huge steps)
            big = max([abs(dv[(K - 1) + j]) for j in range(len(interior))] + [abs(dv[n - 1])] + [mpf(0)])
            if big > mpf(max_dx):
                dv = [v * mpf(max_dx) / big for v in dv]
        if feasible_step:
            # full Newton step, scaled back only as far as feasibility requires (no residual test):
            # weights stay above a tenth of their value, gaps above a fifth of theirs
            alpha = mpf(1)
            for j, k in enumerate(others):
                if dv[j] > 0:
                    alpha = min(alpha, mpf("0.9") * vec[j] / dv[j])
            xs_now = list(xx)
            for j, i in enumerate(interior):
                xs_now[i] = vec[(K - 1) + j]
            xs_now[K - 1] = vec[-1]
            dxs = [mpf(0)] * K
            for j, i in enumerate(interior):
                dxs[i] = dv[(K - 1) + j]
            dxs[K - 1] = dv[-1]
            for i in range(K - 1):
                gap = xs_now[i + 1] - xs_now[i]; dgap = dxs[i + 1] - dxs[i]      # new gap = gap - alpha*dgap
                if dgap > 0:
                    alpha = min(alpha, mpf("0.8") * gap / dgap)
            step = [vec[i] - alpha * dv[i] for i in range(n)]
            hist[-1] = hist[-1] + "@f" + nstr(alpha, 3)
            vec = step
            continue
        step = [vec[i] - dv[i] for i in range(n)]
        val = ok(step); lam_used = "1"
        pure = (pure_below is not None) and (nr < mpf(pure_below)) and (val is not None)
        if (not pure) and (val is None or val > nr):
            found = False
            for lam in ("0.5", "0.25", "0.1", "0.03", "0.01"):
                trial = [vec[i] - mpf(lam) * dv[i] for i in range(n)]
                vt = ok(trial)
                if vt is not None and vt < nr:
                    step = trial; found = True; lam_used = lam; break
            if not found:
                hist[-1] = hist[-1] + "@stuck"
                break
        hist[-1] = hist[-1] + "@" + lam_used
        vec = step
    r, xx, ww, P, Q, Dv = resid(vec)
    C = sum(wi * di for wi, di in zip(ww, Dv))
    return {"x": xx, "w": ww, "C": C, "A": xx[-1], "newton_resid": max(abs(v) for v in r), "hist": hist, "ny": ny}
