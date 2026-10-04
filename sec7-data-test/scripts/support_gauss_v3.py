"""
support_gauss_v3: certified support of the capacity-achieving input of the amplitude-constrained AWGN channel
Y = x + Z, Z ~ N(0, 1), 0 <= x <= a. Same add-violator-refine-certify loop as support_exact_v3, with densities on a
fine y-grid (trapezoid integration). By symmetry the optimal input is symmetric about a/2; we do not impose it.
"""

import numpy as np, sys
from scipy.optimize import minimize

np.seterr(all="ignore")


def _grid(a, dy=0.02):
    return np.arange(-7.0, a + 7.0 + dy, dy)


def _P(xs, ys):
    xs = np.asarray(xs, float)[:, None]
    return np.exp(-0.5 * (ys[None, :] - xs) ** 2) / np.sqrt(2 * np.pi)


def _mi(xs, w, ys, dy):
    P = _P(xs, ys)
    Q = w @ P
    lq = np.log(np.where(Q > 0, Q, 1e-300))
    lP = np.log(np.where(P > 0, P, 1e-300))
    return float(w @ (((P * lP).sum(1) - P @ lq) * dy))


def _solve_w(PS, HS, dy):
    n = len(PS)

    def negC(w):
        Q = w @ PS
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        return -float(w @ (HS - (PS @ lq) * dy))

    r = minimize(
        negC,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0, 1)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"ftol": 1e-15, "maxiter": 500},
    )
    w = np.clip(r.x, 0, 1)
    w /= w.sum()
    return w, -negC(w)


def _merge(x, w, frac):
    mp, mw = [], []
    for p, ww in zip(x, w):
        if mp and (p - mp[-1]) < frac:
            tot = mw[-1] + ww
            mp[-1] = (mp[-1] * mw[-1] + p * ww) / tot
            mw[-1] = tot
        else:
            mp.append(float(p))
            mw.append(float(ww))
    w = np.array(mw)
    return np.array(mp), w / w.sum()


def _refine(x, w, a, ys, dy):
    n = len(x)
    free = np.ones(n, bool)
    free[0] = x[0] > 1e-9
    free[-1] = x[-1] < a - 1e-9
    nf = int(free.sum())

    def obj(z):
        xx = x.copy()
        xx[free] = z[:nf]
        ww = np.clip(z[nf:], 0, 1)
        ww = ww / ww.sum()
        return -_mi(xx, ww, ys, dy)

    z0 = np.concatenate([x[free], w])
    bounds = [(0.0, a)] * nf + [(0, 1)] * n
    r = minimize(
        obj,
        z0,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda z: z[nf:].sum() - 1}],
        options={"ftol": 1e-14, "maxiter": 600},
    )
    xx = x.copy()
    xx[free] = r.x[:nf]
    ww = np.clip(r.x[nf:], 0, 1)
    ww /= ww.sum()
    order = np.argsort(xx)
    xx, ww = xx[order], ww[order]
    keep = ww > 1e-7
    xx, ww = xx[keep], ww[keep]
    return xx, ww / ww.sum()


def support(a, tol=1e-6, coarse=20, fine=100, merge_frac=0.25, max_outer=30, maxit=200):
    a = float(a)
    ys = _grid(a)
    dy = ys[1] - ys[0]
    ngrid = max(401, int(coarse * a))
    xs = np.linspace(0.0, a, ngrid)
    P = _P(xs, ys)
    lP = np.log(np.where(P > 0, P, 1e-300))
    Hx = (P * lP).sum(1) * dy
    S = [0, ngrid - 1]
    for _ in range(maxit):
        w, C = _solve_w(P[S], Hx[S], dy)
        Q = w @ P[S]
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        D = Hx - (P @ lq) * dy
        k = int(D.argmax())
        if D[k] - C <= tol or k in S:
            break
        S.append(k)
    x = np.array([xs[i] for i in S])
    w = np.array(w)
    order = np.argsort(x)
    x, w = x[order], w[order]
    keep = w > 1e-7
    x, w = x[keep], w[keep]
    w /= w.sum()
    x, w = _merge(x, w, merge_frac)
    xf = np.linspace(0.0, a, int(fine * a) + 1)
    Pf = _P(xf, ys)
    lPf = np.log(np.where(Pf > 0, Pf, 1e-300))
    Hf = (Pf * lPf).sum(1) * dy
    for outer in range(max_outer):
        x, w = _refine(x, w, a, ys, dy)
        x, w = _merge(x, w, merge_frac)
        C = _mi(x, w, ys, dy)
        Q = w @ _P(x, ys)
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        Df = Hf - (Pf @ lq) * dy
        k = int(Df.argmax())
        slack = float(Df[k] - C)
        if slack <= tol:
            break
        xn = xf[k]
        if np.min(np.abs(x - xn)) < 0.5 * merge_frac:
            w = 0.98 * w + 0.02 / len(w)
            continue
        x = np.append(x, xn)
        w = np.append(w * 0.98, 0.02)
        order = np.argsort(x)
        x, w = x[order], w[order]
    return {
        "N": int(len(x)),
        "pts": x.tolist(),
        "w": w.tolist(),
        "C": float(C),
        "slack": float(slack),
        "certified": bool(slack <= tol),
        "spacing": np.diff(x).tolist(),
        "outer_rounds": outer + 1,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for a in (1.0, 1.6, 1.7, 2.5, 3.0, 4.0):
        r = support(a)
        print(
            f"a={a:5.2f} N*={r['N']} certified={r['certified']} slack={r['slack']:+.1e} pts={np.round(r['pts'],3).tolist()} w={np.round(r['w'],3).tolist()}"
        )
