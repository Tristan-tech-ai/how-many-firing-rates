"""
Krawczyk existence test: an EXACT KKT point lies within rho of the numerical one. 2026-09-06.

The certificate proves max_x D(x; F) <= I(F) + eta for the DECIMAL input F, with eta its KKT residual. The
audit's caveat 1 is the step from that to an exact capacity-achieving input with the same K atoms. This is the
first half of closing it: the KKT equality system
    G(w, x) = [ D(x_i) - D(x_0) for i = 1..K-1,  sum_k w_k - 1,  D'(x_i) for interior i ]  (2K - 2 equations)
in the unknowns (w_0..w_{K-1}, interior x_i), with x_0 = 0 and x_{K-1} = A fixed, has a zero in the box
X = x~ +- rho if the Krawczyk operator
    K(X) = x~ - Y G(x~) + (I - Y J(X)) (X - x~),   Y an approximate inverse of J(x~), J(X) the interval Jacobian,
satisfies K(X) strictly inside X (Krawczyk 1969; Moore 1977; then the zero is unique in X and J is nonsingular
there). The Jacobian entries are the analytic ones of support_mp3, evaluated in interval arithmetic with the
parameters as intervals; G(x~) is evaluated with degenerate parameter intervals. The second half (the certificate
run with the box as parameters, so that every enclosure covers the exact point) is q32_certificate_v4.py.

Control: the same test with rho too small must FAIL (the K(X) image cannot fit), and with rho far too large the
contraction factor must exceed 1; the accepted rho sits between, and the contraction factor is reported.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import iv, mp, mpf, mpi, matrix, inverse, nstr, loggamma, log, exp, norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def mid(I):
    return (mpf(I.a) + mpf(I.b)) / 2


def rad(I):
    return (mpf(I.b) - mpf(I.a)) / 2


class KKT:
    def __init__(self, A, W, X, ny, LF, interior=None):
        """W, X lists of intervals (weights, all atom positions incl. 0 and A); builds P, Q tables.
        interior: indices of the free atoms, fixed once from the point solution with a tolerance, so the box
        system has the same unknowns as the point system."""
        self.A = A
        self.W = W
        self.X = X
        self.K = len(X)
        self.ny = ny
        self.LF = LF
        if interior is None:
            interior = [
                i for i in range(self.K) if mpf(X[i].a) > mpf("1e-12") and mpf(X[i].b) < A - mpf("1e-12")
            ]
        self.interior = interior
        self.n = self.K + len(self.interior)
        self.lX = [iv.log(x) if mpf(x.a) > 0 else None for x in X]
        self.P = [[self._P(k, y) for y in range(ny)] for k in range(self.K)]
        self.Q = []
        for y in range(ny):
            q = iv.mpf(0)
            for k in range(self.K):
                q = q + W[k] * self.P[k][y]
            self.Q.append(q)
        self.logQ = [iv.log(q) for q in self.Q]
        self.R = [[self.P[k][y] / self.Q[y] for y in range(ny)] for k in range(self.K)]  # P_k / Q
        self.U = [
            [(y / X[k] - 1) if self.lX[k] is not None else None for y in range(ny)] for k in range(self.K)
        ]

    def _L(self, k, y):
        return y * self.lX[k] - self.X[k] - self.LF[y]

    def _P(self, k, y):
        if self.lX[k] is None:
            return iv.mpf(1) if y == 0 else iv.mpf(0)
        return iv.exp(self._L(k, y))

    def D(self, k):
        if self.lX[k] is None:
            return -self.logQ[0]
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s + self.P[k][y] * (self._L(k, y) - self.logQ[y])
        return s

    def Dp(self, k):
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s + self.P[k][y] * self.U[k][y] * (self._L(k, y) - self.logQ[y])
        return s

    def Dpp(self, k):
        s = iv.mpf(0)
        for y in range(self.ny):
            u = self.U[k][y] + 1
            s = s + self.P[k][y] * ((u - 1) ** 2 - u / self.X[k]) * (self._L(k, y) - self.logQ[y])
        return s + 1 / self.X[k]

    def dD_dw(self, i, k):
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s - self.P[i][y] * self.R[k][y]
        return s

    def dD_dx(self, i, m):
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s - self.P[i][y] * self.W[m] * self.R[m][y] * self.U[m][y]
        return s

    def dDp_dw(self, i, k):
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s - self.P[i][y] * self.U[i][y] * self.R[k][y]
        return s

    def dDp_dx(self, i, m):
        s = iv.mpf(0)
        for y in range(self.ny):
            s = s - self.P[i][y] * self.U[i][y] * self.W[m] * self.R[m][y] * self.U[m][y]
        return s

    def G(self):
        Dv = [self.D(k) for k in range(self.K)]
        g = [Dv[i] - Dv[0] for i in range(1, self.K)]
        t = iv.mpf(0)
        for w in self.W:
            t = t + w
        g.append(t - 1)
        g += [self.Dp(i) for i in self.interior]
        return g

    def J(self):
        K = self.K
        n = self.n
        inter = self.interior
        Dp_at = {i: self.Dp(i) for i in inter}
        Dpp_at = {i: self.Dpp(i) for i in inter}
        Jm = [[iv.mpf(0)] * n for _ in range(n)]
        for row, i in enumerate(range(1, K)):
            for k in range(K):
                Jm[row][k] = self.dD_dw(i, k) - self.dD_dw(0, k)
            for j, m in enumerate(inter):
                v = self.dD_dx(i, m) - self.dD_dx(0, m)
                if m == i:
                    v = v + Dp_at[i]
                Jm[row][K + j] = v
        for k in range(K):
            Jm[K - 1][k] = iv.mpf(1)
        for a, i in enumerate(inter):
            row = K + a
            for k in range(K):
                Jm[row][k] = self.dDp_dw(i, k)
            for j, m in enumerate(inter):
                v = self.dDp_dx(i, m)
                if m == i:
                    v = v + Dpp_at[i]
                Jm[row][K + j] = v
        return Jm


def krawczyk(A, xs, ws, rho, dps=50, quiet=False):
    mp.dps = dps
    iv.dps = dps
    A = mpf(
        repr(float(A))
    )  # the DECIMAL amplitude, identical to the stored last atom (a float 39.612 is not)
    ny = int(float(A) + 18 * float(A) ** 0.5 + 50)
    LF = [iv.loggamma(iv.mpf(y + 1)) for y in range(ny)]
    xt = [mpf(v) for v in xs]
    wt = [mpf(v) for v in ws]
    K = len(xt)
    rho = mpf(rho)
    t0 = time.time()
    # point system (degenerate intervals)
    S0 = KKT(A, [iv.mpf(v) for v in wt], [iv.mpf(v) for v in xt], ny, LF)
    G0 = S0.G()
    J0 = S0.J()
    n = S0.n
    Jmid = matrix(n, n)
    for i in range(n):
        for j in range(n):
            Jmid[i, j] = mid(J0[i][j])
    Y = inverse(Jmid)
    # box system
    inter = S0.interior
    Wb = [mpi(w - rho, w + rho) for w in wt]
    Xb = [iv.mpf(x) for x in xt]
    for i in inter:
        Xb[i] = mpi(xt[i] - rho, xt[i] + rho)
    SB = KKT(A, Wb, Xb, ny, LF, interior=inter)
    assert SB.n == n
    JX = SB.J()
    # K(X) - x~ = -Y G0 + (I - Y JX) [-rho, rho]^n
    YG = [sum((iv.mpf(Y[i, l]) * G0[l] for l in range(n)), iv.mpf(0)) for i in range(n)]
    worst_ratio = mpf(0)
    rows = []
    for i in range(n):
        M_row = []
        for j in range(n):
            s = iv.mpf(1 if i == j else 0)
            for l in range(n):
                s = s - iv.mpf(Y[i, l]) * JX[l][j]
            M_row.append(s)
        contraction = sum(max(abs(mpf(m.a)), abs(mpf(m.b))) for m in M_row)
        shift = max(abs(mpf(YG[i].a)), abs(mpf(YG[i].b)))
        ratio = (shift + contraction * rho) / rho
        worst_ratio = max(worst_ratio, ratio)
        rows.append((shift, contraction, ratio))
    ok = worst_ratio < 1
    if not quiet:
        print(
            f"A = {nstr(A,6)}, K = {K}, n = {n}, rho = {nstr(rho,3)}, dps = {dps}   [{time.time()-t0:.0f}s]"
        )
        print(
            f"   |G(x~)| max = {nstr(max(max(abs(mpf(g.a)), abs(mpf(g.b))) for g in G0), 3)}   |Y| (max row sum) = {nstr(max(sum(abs(Y[i,j]) for j in range(n)) for i in range(n)), 3)}"
        )
        print(
            f"   worst row: shift |Y G| = {nstr(max(r[0] for r in rows), 3)}, contraction sum |I - Y J(X)| = {nstr(max(r[1] for r in rows), 3)}"
        )
        print(
            f"   max_i (|Y G|_i + contraction_i rho) / rho = {nstr(worst_ratio, 5)}  ->  K(X) {'STRICTLY INSIDE X: exact zero exists and is unique in X' if ok else 'NOT inside X'}"
        )
    return {
        "ok": bool(ok),
        "rho": float(rho),
        "worst_ratio": float(worst_ratio),
        "shift_max": float(max(r[0] for r in rows)),
        "contraction_max": float(max(r[1] for r in rows)),
        "n": n,
        "K": K,
        "A": float(A),
        "seconds": time.time() - t0,
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "100"
    d = json.load(open(os.path.join(HERE, f"results_q32_A{which}_exact40.json")))
    out = {}
    for rho in ("1e-36", "1e-30", "1e-20", "1e-10", "1e-3"):
        out[rho] = krawczyk(d["A"], d["x"], d["w"], rho)
    json.dump(out, open(os.path.join(HERE, f"results_q32_krawczyk_A{which}.json"), "w"), indent=1)
    print(
        "\nreading: the test must fail for rho below the shift and for rho so large that the contraction exceeds 1,"
    )
    print("and pass in between; the accepted box is the smallest passing rho.")
