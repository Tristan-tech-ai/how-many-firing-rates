import numpy as np, json, sys
from scipy.stats import poisson
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
Y = np.arange(80)


def pmf(x):
    return poisson.pmf(Y, x)


def D(P, Q):
    m = P > 0
    return float((P[m] * (np.log(P[m]) - np.log(Q[m]))).sum())


def cert(A, nx=8001):
    def negC(p):
        Q = (1 - p) * pmf(0.0) + p * pmf(A)
        return -((1 - p) * D(pmf(0.0), Q) + p * D(pmf(A), Q))

    p = minimize_scalar(negC, bounds=(1e-6, 1 - 1e-6), method="bounded", options={"xatol": 1e-13}).x
    Q = (1 - p) * pmf(0.0) + p * pmf(A)
    C = -negC(p)
    xs = np.linspace(0.0, A, nx)[1:-1]  # INTERIOR only; endpoints have D = C by KKT equality
    sl = np.array([D(pmf(x), Q) - C for x in xs])
    k = sl.argmax()
    return p, C, float(sl[k]), float(xs[k])


print("interior KKT slack = max_{0<x<A} [D(P_x||Q) - C]  (binary optimal iff <= 0)")
print("   A       p*        C          slack        x_worst")
for A in (2.0, 2.5, 2.6, 2.718, 3.0, 3.3, 3.36, 3.37, 3.4, 3.5):
    p, C, s, xw = cert(A)
    print(f"  {A:5.3f}   {p:.5f}   {C:.6f}   {s:+.3e}   {xw:.3f}")
lo, hi = 3.0, 3.5
for _ in range(30):
    mid = 0.5 * (lo + hi)
    if cert(mid)[2] <= 1e-9:
        lo = mid
    else:
        hi = mid
Astar = 0.5 * (lo + hi)
print(
    f"\n  A* (binary -> graded) = {Astar:.4f}    literature (Cao-Hranilovic-Chen 2014 via Dytso 2024): 3.3679"
)
print(
    f"  at T = 50 ms: L* = {Astar/0.05:.1f} Hz    (earlier BA-support onset: 50-52 Hz = A 2.5-2.6  => ARTEFACT)"
)
print(
    f"  T*(L) = A*/L:  L=50: {Astar/50*1000:.1f} ms   L=60: {Astar/60*1000:.1f} ms   L=100: {Astar/100*1000:.1f} ms   L=150: {Astar/150*1000:.1f} ms   L=300: {Astar/300*1000:.1f} ms"
)
print(
    f"  Kostal-Shinomoto cap L=50 Hz at 50 ms: A=2.5, i.e. {(1-2.5/Astar)*100:.0f}% BELOW the onset (not 4%)"
)
json.dump({"A_star": Astar, "L_star_50ms": Astar / 0.05}, open("results_binary_cert.json", "w"), indent=1)
