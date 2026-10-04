"""
support_exact_v3: certified support of the capacity-achieving input of Y ~ Poisson(lam + x), 0 <= x <= A, for large A.
Outer loop: (cutting-plane on a coarse grid) -> merge by noise scale -> continuous SLSQP refinement of locations+weights
-> KKT certificate on a FINE grid -> if the fine-grid slack exceeds tol, ADD the fine-grid violator as a new point and
repeat. Terminates when slack <= tol (certified) or after max_outer rounds (reported uncertified). Deterministic.
"""

import numpy as np, sys
from scipy.special import gammaln
from scipy.optimize import minimize

np.seterr(all="ignore")


def _P(xs, lam, ny):
    mu = (np.asarray(xs, float) + lam)[:, None]
    y = np.arange(ny)[None, :]
    return np.exp(y * np.log(np.where(mu > 0, mu, 1e-300)) - mu - gammaln(y + 1))


def _mi(xs, w, lam, ny):
    P = _P(xs, lam, ny)
    Q = w @ P
    lq = np.log(np.where(Q > 0, Q, 1e-300))
    lP = np.log(np.where(P > 0, P, 1e-300))
    return float(w @ ((P * lP).sum(1) - P @ lq))


def _solve_w(PS, HS):
    n = len(PS)

    def negC(w):
        Q = w @ PS
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        return -float(w @ (HS - PS @ lq))

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
        if mp and (p - mp[-1]) < frac * (1.0 + np.sqrt(max(mp[-1], 0.0))):
            tot = mw[-1] + ww
            mp[-1] = (mp[-1] * mw[-1] + p * ww) / tot
            mw[-1] = tot
        else:
            mp.append(float(p))
            mw.append(float(ww))
    w = np.array(mw)
    return np.array(mp), w / w.sum()


def _refine(x, w, A, lam, ny):
    n = len(x)
    free = np.ones(n, bool)
    free[0] = x[0] > 1e-9
    free[-1] = x[-1] < A - 1e-9
    nf = int(free.sum())

    def obj(z):
        xx = x.copy()
        xx[free] = z[:nf]
        ww = np.clip(z[nf:], 0, 1)
        ww = ww / ww.sum()
        return -_mi(xx, ww, lam, ny)

    z0 = np.concatenate([x[free], w])
    bounds = [(0.0, A)] * nf + [(0, 1)] * n
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


def support(A, lam=0.0, tol=1e-8, coarse=6, fine=40, merge_frac=0.25, max_outer=30, maxit=200):
    A = float(A)
    ny = int((A + lam) * 3 + 60)
    ngrid = max(1201, int(coarse * A))
    xs = np.linspace(0.0, A, ngrid)
    P = _P(xs, lam, ny)
    logP = np.log(np.where(P > 0, P, 1e-300))
    Hx = (P * logP).sum(1)
    S = [0, ngrid - 1]
    for _ in range(maxit):
        w, C = _solve_w(P[S], Hx[S])
        Q = w @ P[S]
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        D = Hx - P @ lq
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
    xf = np.linspace(0.0, A, int(fine * A) + 1)
    Pf = _P(xf, lam, ny)
    lPf = np.log(np.where(Pf > 0, Pf, 1e-300))
    Hf = (Pf * lPf).sum(1)
    history = []
    for outer in range(max_outer):
        x, w = _refine(x, w, A, lam, ny)
        x, w = _merge(x, w, merge_frac)
        C = _mi(x, w, lam, ny)
        Q = w @ _P(x, lam, ny)
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        Df = Hf - Pf @ lq
        k = int(Df.argmax())
        slack = float(Df[k] - C)
        history.append((len(x), slack))
        if slack <= tol:
            break
        xn = xf[k]
        if np.min(np.abs(x - xn)) < 0.5 * merge_frac * (1.0 + np.sqrt(xn)):
            # violator sits on an existing point: perturb weights slightly to escape and continue
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
        "history": history,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for A in (3.36, 3.37, 10.0, 20.0, 50.0):
        r = support(A)
        print(
            f"A={A:6.2f} N*={r['N']} certified={r['certified']} slack={r['slack']:+.1e} rounds={r['outer_rounds']} pts={np.round(r['pts'],2).tolist()}"
        )
