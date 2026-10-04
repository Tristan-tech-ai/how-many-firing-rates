"""
Certified support size N*(A, lam) of the capacity-achieving input of Y ~ Poisson(lam + x), 0 <= x <= A.
Cutting-plane: solve the small concave problem over the current support with SLSQP, add the worst KKT
violator on a fine grid, repeat; adjacent support points merged. Deterministic.
K3 control: N*(A, 0) = 2 for A < 3.3679 and 3 at A = 3.37.
"""

import numpy as np, json, sys
from scipy.special import gammaln
from scipy.optimize import minimize

np.seterr(all="ignore")


def _P(xs, lam, ny):
    mu = (xs + lam)[:, None]
    y = np.arange(ny)[None, :]
    return np.exp(y * np.log(np.where(mu > 0, mu, 1e-300)) - mu - gammaln(y + 1))


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


def support(A, lam=0.0, ngrid=1201, tol=1e-8, maxit=40):
    xs = np.linspace(0.0, A, ngrid)
    ny = int((A + lam) * 3 + 60)
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
    keep = sorted((i, wi) for i, wi in zip(S, w) if wi > 1e-6)
    cl = []
    for i, wi in keep:
        if cl and i - cl[-1][-1][0] <= 3:
            cl[-1].append((i, wi))
        else:
            cl.append([(i, wi)])
    pts = [float(np.mean([xs[i] for i, _ in c])) for c in cl]
    wts = [float(sum(wi for _, wi in c)) for c in cl]
    return {"N": len(pts), "pts": pts, "w": wts, "C": float(C), "slack": float(D.max() - C)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import time

    t0 = time.time()
    print("K3 control (lam = 0):")
    for A in (2.0, 3.0, 3.36, 3.37, 3.5, 5.0):
        r = support(A)
        print(
            f"   A={A:5.2f}  N*={r['N']}  pts={np.round(r['pts'],3).tolist()}  w={np.round(r['w'],3).tolist()}  slack={r['slack']:+.1e}"
        )
    lo, hi = 3.0, 4.0
    for _ in range(16):
        m = 0.5 * (lo + hi)
        if support(m)["N"] == 2:
            lo = m
        else:
            hi = m
    print(
        f"   threshold A* = {0.5*(lo+hi):.4f}   (literature 3.3679)  {'PASS' if abs(0.5*(lo+hi)-3.3679)<2e-3 else '*** FAIL ***'}   [{time.time()-t0:.0f}s]"
    )
    print("\nstaircase N*(A, lam):")
    print("    A    lam=0   lam=0.3  lam=1.0  lam=3.0")
    rows = []
    for A in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20):
        Ns = [support(A, l)["N"] for l in (0.0, 0.3, 1.0, 3.0)]
        rows.append({"A": A, "N": Ns})
        print(f"  {A:4d}     {Ns[0]}        {Ns[1]}        {Ns[2]}        {Ns[3]}")
    json.dump(rows, open("results_support_exact.json", "w"), indent=1)
    print(f"[saved]  [{time.time()-t0:.0f}s]")
