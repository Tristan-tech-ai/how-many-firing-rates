"""N*(A, lam, Fano) by cutting-plane + KKT on the COM-Poisson family. Control: Fano = 1 must reproduce support_exact."""

import numpy as np, json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontier"))
from core import comp_P
from scipy.optimize import minimize
from support_exact import support as support_poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")


def support_F(A, lam, F, ngrid=601, tol=1e-8, maxit=40, T=1.0):
    xs = np.linspace(0.0, A, ngrid)
    P = comp_P(xs + lam, F, T)  # T=1 -> mean = xs + lam counts
    P = P / P.sum(1, keepdims=True)
    Hx = (P * np.log(np.where(P > 0, P, 1e-300))).sum(1)
    S = [0, ngrid - 1]
    for _ in range(maxit):
        PS, HS = P[S], Hx[S]
        n = len(S)

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
        C = -negC(w)
        Q = w @ PS
        D = Hx - P @ np.log(np.where(Q > 0, Q, 1e-300))
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
    return len(cl)


print("control: Fano = 1 vs support_exact (Poisson)")
for A, lam in ((2.0, 0.0), (3.5, 0.0), (5.0, 1.0), (8.0, 0.3)):
    a, b = support_F(A, lam, 1.0), support_poisson(A, lam)["N"]
    print(f"   A={A} lam={lam}: COM(F=1) N*={a}  Poisson N*={b}  {'OK' if a == b else '*** MISMATCH ***'}")
print("\nN*(A, lam, F):   rows A, cols Fano")
FS = (0.7, 0.9, 1.0, 1.13, 1.3, 1.5)
out = {}
for lam in (0.0, 0.3, 1.0):
    print(f"  lam = {lam}:      " + "".join(f"F={f:<6}" for f in FS))
    for A in (0.5, 1.0, 2.0, 3.0, 3.5, 5.0, 8.0, 12.0):
        Ns = [support_F(A, lam, f) for f in FS]
        out[f"{lam},{A}"] = Ns
        print(f"     A={A:5.1f}:    " + "".join(f"{n:<8}" for n in Ns))
json.dump(out, open("results_noiseNstar.json", "w"), indent=1)
print("[saved]")
