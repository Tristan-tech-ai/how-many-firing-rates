"""
KKT Newton with an ANALYTIC JACOBIAN (2026-09-05).

Why. The finite-difference Jacobian has a hard ceiling at K = 13. Fourteen atoms cannot be reached by any route
tried: multi-start stalls at K = 14, 15, 16 even mid-plateau, and insertion from a converged violating thirteen
stalls at every seed weight from 1e-4 to 1e-1, at A = 155 and at A = 165 alike, always with residuals in the
1e-4 to 1e-2 range. The failure follows K, not A. That is the signature of a conditioning limit in the Jacobian
rather than anything about the channel, and it is the same signature as the nested-difference bug fixed earlier
tonight, one level up.

Everything needed is in closed form. With L(x,y) = y log x - x - log y!, P(x,y) = exp L,
Q(y) = sum_k w_k P(x_k,y) and g(x,y) = L(x,y) - log Q(y):
    D(x)    = sum_y P(x,y) g(x,y)
    Dp(x)   = sum_y P(x,y) (y/x - 1) g(x,y)
    Dpp(x)  = sum_y P(x,y) [(y/x - 1)^2 - y/x^2] g(x,y) + 1/x
and the parameter derivatives, which enter only through Q:
    dD(x)/dw_k   = -sum_y P(x,y) P(x_k,y)/Q(y)
    dD(x)/dx_k   = -sum_y P(x,y) w_k P(x_k,y)(y/x_k - 1)/Q(y)
    dDp(x)/dw_k  = -sum_y P(x,y)(y/x - 1) P(x_k,y)/Q(y)
    dDp(x)/dx_k  = -sum_y P(x,y)(y/x - 1) w_k P(x_k,y)(y/x_k - 1)/Q(y)
with the direct terms Dp(x) and Dpp(x) added on the diagonal when the evaluation point IS the atom being varied.

Weights are carried unnormalised with sum w = 1 as its own residual row, which keeps the Jacobian exact; the
earlier solver normalised inside the residual, which would have put an extra term in every entry.
"""

import sys
from mpmath import mp, mpf, exp, log, loggamma, matrix, lu_solve, nstr


def _setup(A, dps):
    mp.dps = dps
    A = float(A)
    return int(A + 18 * A**0.5 + 50)


def _logP(x, y):
    if x <= 0:
        return mpf(0) if y == 0 else mpf("-1e400")
    return y * log(x) - x - loggamma(y + 1)


def _P(x, y):
    v = _logP(x, y)
    return exp(v) if v > -400 else mpf(0)


def _tables(xs, ws, ny):
    P = [[_P(x, y) for y in range(ny)] for x in xs]
    Q = [sum(ws[k] * P[k][y] for k in range(len(xs))) for y in range(ny)]
    return P, Q


def _g(x, y, Q):
    q = Q[y] if Q[y] > 0 else mpf("1e-400")
    return _logP(x, y) - log(q)


def D_of(x, Q, ny):
    s = mpf(0)
    for y in range(ny):
        p = _P(x, y)
        if p > 0:
            s += p * _g(x, y, Q)
    return s


def Dp_of(x, Q, ny):
    if x <= 0:
        return mpf(0)
    s = mpf(0)
    for y in range(ny):
        p = _P(x, y)
        if p > 0:
            s += p * (mpf(y) / x - 1) * _g(x, y, Q)
    return s


def Dpp_of(x, Q, ny):
    if x <= 0:
        return mpf(0)
    s = mpf(0)
    for y in range(ny):
        p = _P(x, y)
        if p > 0:
            t = mpf(y) / x
            s += p * ((t - 1) ** 2 - t / x) * _g(x, y, Q)
    return s + 1 / x


def _resid(xs_full, interior, vec, K, ny):
    w = list(vec[:K])
    x = list(xs_full)
    for j, i in enumerate(interior):
        x[i] = vec[K + j]
    P, Q = _tables(x, w, ny)
    Dv = [D_of(xi, Q, ny) for xi in x]
    r = [Dv[i] - Dv[0] for i in range(1, K)] + [sum(w) - 1] + [Dp_of(x[i], Q, ny) for i in interior]
    return r, x, w, P, Q, Dv


def solve(
    A, pts, wts, dps=40, newton_steps=30, tol_pow=12, pin_ends=False, pure_below=None, feasible_step=False
):
    """pin_ends=True: the first and last atoms are boundary atoms wherever they sit (input interval [x_0, x_K-1],
    e.g. the dark-current problem shifted to [lambda, A + lambda]); only atoms 1..K-2 move."""
    ny = _setup(A, dps)
    A = mpf(A)
    xs = [mpf(repr(float(p))) if not isinstance(p, str) else mpf(p) for p in pts]
    ws = [mpf(repr(float(v))) if not isinstance(v, str) else mpf(v) for v in wts]
    t = sum(ws)
    ws = [v / t for v in ws]
    K = len(xs)
    interior = (
        list(range(1, K - 1))
        if pin_ends
        else [i for i in range(K) if 1e-12 < float(xs[i]) < float(A) - 1e-12]
    )
    n = K + len(interior)
    vec = list(ws) + [xs[i] for i in interior]
    hist = []

    for _ in range(newton_steps):
        r, x, w, P, Q, Dv = _resid(xs, interior, vec, K, ny)
        nr = max(abs(v) for v in r)
        hist.append(nstr(nr, 4))
        if nr < mpf(10) ** (-dps + tol_pow):
            break

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

        Dp_at = [Dp_of(xi, Q, ny) for xi in x]
        Dpp_at = [Dpp_of(xi, Q, ny) for xi in x]
        J = matrix(n, n)
        for row_i, i in enumerate(range(1, K)):
            for k in range(K):
                J[row_i, k] = dD_dw(x[i], k) - dD_dw(x[0], k)
            for j, m_ in enumerate(interior):
                v = dD_dx(x[i], m_) - dD_dx(x[0], m_)
                if m_ == i:
                    v += Dp_at[i]
                if m_ == 0:
                    v -= Dp_at[0]
                J[row_i, K + j] = v
        for k in range(K):
            J[K - 1, k] = mpf(1)
        for j in range(len(interior)):
            J[K - 1, K + j] = mpf(0)
        for a, i in enumerate(interior):
            row = K + a
            for k in range(K):
                J[row, k] = dDp_dw(x[i], k)
            for j, m_ in enumerate(interior):
                v = dDp_dx(x[i], m_)
                if m_ == i:
                    v += Dpp_at[i]
                J[row, K + j] = v
        try:
            dv = lu_solve(J, matrix(r))
        except Exception:
            break

        def ok_state(cand):
            wc = list(cand[:K])
            xc = list(xs)
            for j, i in enumerate(interior):
                xc[i] = cand[K + j]
            if min(wc) <= 0:
                return None
            if any(xc[i + 1] <= xc[i] for i in range(K - 1)):
                return None
            rc = _resid(xs, interior, cand, K, ny)[0]
            return max(abs(v) for v in rc)

        if feasible_step:
            # full Newton step, scaled back only as far as feasibility requires (weights keep a tenth of
            # their value, gaps a fifth of theirs); no residual test
            alpha = mpf(1)
            for k in range(K):
                if dv[k] > 0:
                    alpha = min(alpha, mpf("0.9") * vec[k] / dv[k])
            xs_now = list(xs)
            dxs = [mpf(0)] * K
            for j, i in enumerate(interior):
                xs_now[i] = vec[K + j]
                dxs[i] = dv[K + j]
            for i in range(K - 1):
                gap = xs_now[i + 1] - xs_now[i]
                dgap = dxs[i + 1] - dxs[i]
                if dgap > 0:
                    alpha = min(alpha, mpf("0.8") * gap / dgap)
            vec = [vec[i] - alpha * dv[i] for i in range(n)]
            hist[-1] = hist[-1] + "@f" + nstr(alpha, 3)
            continue
        step = [vec[i] - dv[i] for i in range(n)]
        val = ok_state(step)
        lam_used = "1"
        pure = (pure_below is not None) and (nr < mpf(pure_below)) and (val is not None)
        if (not pure) and (val is None or val > nr):
            found = False
            for lam in ("0.5", "0.25", "0.1", "0.03", "0.01"):
                trial = [vec[i] - mpf(lam) * dv[i] for i in range(n)]
                vt = ok_state(trial)
                if vt is not None and vt < nr:
                    step = trial
                    found = True
                    lam_used = lam
                    break
            if not found:
                hist[-1] = hist[-1] + "@stuck"
                break
        hist[-1] = hist[-1] + "@" + lam_used
        vec = step

    r, x, w, P, Q, Dv = _resid(xs, interior, vec, K, ny)
    C = sum(wi * di for wi, di in zip(w, Dv))
    return {"x": x, "w": w, "Q": Q, "C": C, "ny": ny, "newton_resid": max(abs(v) for v in r), "hist": hist}
