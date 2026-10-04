"""c60: full-chain solver for the amplitude-constrained Gaussian CAPACITY on [-A, A] (symmetric), and bisection-quality relocation of its
transitions. Run with `py` (flint).
Model: P(y) = sum w phin(y - x), phin normalised; g(x) = -int phin(y - x) log P(y) dy over the WHOLE line (folded:
y in [0, A + L], kernel phin(y - x) + phin(y + x)); C fixed at 0 (bulk value for unit mass density; the KKT positions and the signals below
are invariant under the mass scaling, which only shifts g by a constant).
Atoms: wall atoms FIXED at +-A (mass unknown, weight condition g(A) = C), interior pairs +-p_j (free: g = C, g' = 0), and for odd K a centre
atom (mass unknown, g(0) = C). Analytic Jacobian (arb_mat products), backtracking Newton.
Transition signal (K = count before the event): K even: s = g(0) - C (birth when it reaches 0); K odd: s = g''(0) (split when it reaches 0).
Start: interior depths from the capacity law (d0, nu0) = (0.02998, 0.56292) (refit on a half-line chain) plus the sharp profile deltaM(x/v)
near the centre; masses 1/rho, wall mass 3.496 (the half-line's). Scan A in steps DA from A0 until the signal changes sign on two
CONVERGED states (|E| below tol), then false position / bisection in A to 1e-10.
Usage: py c60_cap_fullchain.py K A0 DA PREC TAG"""

import sys, json, time, math
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import mpmath as mp
from flint import arb, arb_mat, ctx

say = lambda *a: print(*a, flush=True)
t0 = time.time()
K = int(sys.argv[1])
A0 = float(sys.argv[2])
DA = float(sys.argv[3])
PREC = int(sys.argv[4])
TAG = sys.argv[5]
ctx.prec = PREC
mp.mp.prec = PREC + 20
L = math.sqrt(2 * 0.69315 * PREC) + 1.0
s0 = 1 / (16 * np.pi**2)
LAW = (0.02998, 0.56292)


def rho_of(d, d0=LAW[0]):
    return brentq(
        lambda r: r**3 - 3 * r / (8 * np.pi**2) - 1 / (32 * np.pi**2 * r) - 3 * s0 * (d + d0), 0.02, 400
    )


def N_of(d):
    r = rho_of(d)
    return 4 * np.pi**2 * r**4 - r**2 + np.log(r) / 6 + LAW[1]


def dM(xi):
    xi = abs(xi)
    return 1 / 12 if xi == 0 else quad(lambda t: t * (1 / np.tanh(np.pi * t) - 1), xi, np.inf, limit=200)[0]


xs_, ws_ = [], []
n = 24
for i in range(1, n + 1):
    x = mp.cos(mp.pi * (i - mp.mpf(1) / 4) / (n + mp.mpf(1) / 2))
    for _ in range(200):
        p0, p1 = mp.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        dx = p1 / dp
        x -= dx
        if abs(dx) < mp.mpf(2) ** (-mp.mp.prec + 8):
            break
    xs_.append(x)
    ws_.append(2 / ((1 - x * x) * dp * dp))
XG = [arb(mp.nstr(x, int(PREC / 3.2) + 10)) for x in xs_]
WG = [arb(mp.nstr(w, int(PREC / 3.2) + 10)) for w in ws_]
NRM = 1 / arb(2 * mp.pi).sqrt()
phi = lambda t: NRM * (-(t * t) / 2).exp()
CENTRE = K % 2 == 1
NP = (K - 2 - int(CENTRE)) // 2  # interior positive atoms


class Chain:
    def setup(self, A):
        self.A = arb(A)
        XE = A + L
        npan = int(math.ceil(XE / 0.5))
        h = arb(XE) / npan
        self.Y = []
        self.W = []
        for k in range(npan):
            c0 = h * k + h / 2
            for t, w in zip(XG, WG):
                self.Y.append(c0 + h / 2 * t)
                self.W.append(h / 2 * w)
        NY = len(self.Y)
        self.Bw = [phi(y - self.A) + phi(y + self.A) for y in self.Y]
        self.B0 = [phi(y) for y in self.Y]

    def evaluate(self, u, jac=True, sig=False):
        p = u[:NP]
        m = u[NP : 2 * NP]
        mw = u[2 * NP]
        m0 = u[2 * NP + 1] if CENTRE else None
        Y = self.Y
        NY = len(Y)
        Em = [[phi(y - pj) for pj in p] for y in Y]
        Ep = [[phi(y + pj) for pj in p] for y in Y]
        B = [[Em[i][k] + Ep[i][k] for k in range(NP)] for i in range(NY)]
        Bd = [[(Y[i] - p[k]) * Em[i][k] - (Y[i] + p[k]) * Ep[i][k] for k in range(NP)] for i in range(NY)]
        B2 = [
            [((Y[i] - p[k]) ** 2 - 1) * Em[i][k] + ((Y[i] + p[k]) ** 2 - 1) * Ep[i][k] for k in range(NP)]
            for i in range(NY)
        ]
        cols = [B[i] + [self.Bw[i]] + ([self.B0[i]] if CENTRE else []) for i in range(NY)]
        mv = arb_mat([[mk] for mk in m] + [[mw]] + ([[m0]] if CENTRE else []))
        Pv = arb_mat(cols) * mv
        P = [Pv[i, 0] for i in range(NY)]
        S = [-self.W[i] * P[i].log() for i in range(NY)]
        T = [-self.W[i] / P[i] for i in range(NY)]
        # evaluation kernels: g at p_j (B), g' at p_j (Bd), g at A (Bw), g at 0 (2 B0); g'' at p_j (B2), g'' at 0
        K0 = arb_mat([B[i] + [self.Bw[i], 2 * self.B0[i]] for i in range(NY)])  # columns: p_1..p_n, A, 0
        Sv = arb_mat([[s] for s in S])
        g = K0.transpose() * Sv
        g1 = arb_mat(Bd).transpose() * Sv
        g2 = arb_mat(B2).transpose() * Sv
        g2c = sum(2 * ((Y[i] * Y[i] - 1) * self.B0[i]) * S[i] for i in range(NY))
        gA = g[NP, 0]
        gc = g[NP + 1, 0]
        E = (
            [g[j, 0] for j in range(NP)] + [g1[j, 0] for j in range(NP)] + [gA] + ([gc] if CENTRE else [])
        )  # C = 0
        out = {"E": E, "s": (g2c if CENTRE else gc)}
        if not jac:
            return out
        Tcols = arb_mat([[c * T[i] for c in cols[i]] for i in range(NY)])  # N x (NP + 1 [+1]) : T * basis
        Tdcol = arb_mat([[Bd[i][k] * T[i] for k in range(NP)] for i in range(NY)])
        rowsK = [
            B[i] + [self.Bw[i]] + ([2 * self.B0[i]] if CENTRE else []) for i in range(NY)
        ]  # g-equation kernels (p_j, A, [0])
        Kg = arb_mat(rowsK).transpose()
        Kd = arb_mat(Bd).transpose()
        Mg_m = Kg * Tcols
        Mg_p = Kg * Tdcol
        Md_m = Kd * Tcols
        Md_p = Kd * Tdcol
        nu = 2 * NP + 1 + int(CENTRE)
        J = arb_mat(nu, nu)
        ng = NP + 1 + int(CENTRE)  # number of g-equations
        grow = list(range(NP)) + [2 * NP] + ([2 * NP + 1] if CENTRE else [])  # their row indices in E
        for r_, rr in enumerate(grow):
            for k in range(NP):
                J[rr, k] = (m[k] * Mg_p[r_, k]).mid()
            for k in range(NP + 1 + int(CENTRE)):
                J[rr, NP + k] = Mg_m[r_, k].mid()
        for j in range(NP):
            rr = NP + j
            for k in range(NP):
                J[rr, k] = (m[k] * Md_p[j, k]).mid()
            for k in range(NP + 1 + int(CENTRE)):
                J[rr, NP + k] = Md_m[j, k].mid()
        for j in range(NP):
            J[j, j] = (J[j, j] + g1[j, 0]).mid()
            J[NP + j, j] = (J[NP + j, j] + g2[j, 0]).mid()
        out["J"] = J
        return out

    def solve(self, u, tol, itmax=40):
        r = None
        for it in range(itmax):
            o = self.evaluate(u)
            r = max(abs(e.mid()) for e in o["E"])
            if r < tol:
                return u, float(r), it, o
            st = o["J"].solve(arb_mat([[(-e).mid()] for e in o["E"]]))
            lam = arb(1)
            ok = False
            for _ in range(25):
                un = [arb((u[i] + lam * st[i, 0]).mid()) for i in range(len(u))]
                if min(float(x.mid()) for x in un[NP:]) <= 0:
                    lam /= 2
                    continue
                rn = max(abs(e.mid()) for e in self.evaluate(un, jac=False)["E"])
                if rn < r:
                    u, ok = un, True
                    break
                lam /= 2
            if not ok:
                break
        o = self.evaluate(u, jac=False)
        return u, float(max(abs(e.mid()) for e in o["E"])), it, o


def guess(A):
    v = 4 * np.pi * rho_of(A)
    ps = []
    for j in range(2, NP + 2):
        d = brentq(lambda d: N_of(d) - (j - 0.5), 1e-6, 1e4)
        x = A - d
        try:
            x = brentq(lambda x: N_of(A - x) - dM(x / v) - (j - 0.5), max(1e-6, x - 1), min(A - 1e-6, x + 1))
        except ValueError:
            pass
        ps.append(x)
    ps = ps[::-1]  # innermost first
    ms = [1 / rho_of(A - x) for x in ps]
    return (
        [arb(x) for x in ps]
        + [arb(x) for x in ms]
        + [arb(3.496285)]
        + ([arb(1 / rho_of(A))] if CENTRE else [])
    )


ch = Chain()
tol = math.exp(-4 * math.pi**2 * rho_of(A0) ** 2) * 1e-10
A = A0
ch.setup(A)
u = guess(A)
u, r, it, o = ch.solve(u, tol)
hist = []


def rec(A, u, r, it, o):
    s = float(o["s"].mid())
    conv = r < tol * 1e3
    hist.append(dict(A=A, s=s, res=r, it=it, conv=conv, p=[float(t.mid()) for t in u[:NP]]))
    say(
        f"   A = {A:.10f}  s = {s:+.4e}  |E| = {r:.1e} (tol {tol:.0e})  it {it}  conv {conv}   [{time.time() - t0:.0f}s]"
    )
    json.dump(dict(K=K, hist=hist), open(f"c60_{TAG}.json", "w"), indent=1)
    return s, conv


say(
    f"c60: K = {K} ({'centre split' if CENTRE else 'centre birth'} signal), {NP} interior pairs, A0 = {A0}, prec {PREC}"
)
s, conv = rec(A, u, r, it, o)
prev = (A, s, u) if conv else None
for step in range(30):
    A = round(A + DA, 12)
    ch.setup(A)
    u, r, it, o = ch.solve(u, tol)
    s, conv = rec(A, u, r, it, o)
    if not conv:
        say("STOP: unconverged")
        break
    if prev is not None and prev[1] < 0 <= s:
        lo, hi = prev, (A, s, u)
        for _ in range(60):
            if hi[0] - lo[0] < 1e-10:
                break
            Am = lo[0] - lo[1] * (hi[0] - lo[0]) / (hi[1] - lo[1])
            if not (lo[0] + 0.1 * (hi[0] - lo[0]) < Am < hi[0] - 0.1 * (hi[0] - lo[0])):
                Am = (lo[0] + hi[0]) / 2
            ch.setup(Am)
            um, rm, itm, om = ch.solve(lo[2], tol)
            sm, cm = rec(Am, um, rm, itm, om)
            if not cm:
                say("STOP: unconverged in the bracket")
                break
            if sm < 0:
                lo = (Am, sm, um)
            else:
                hi = (Am, sm, um)
        At = lo[0] - lo[1] * (hi[0] - lo[0]) / (hi[1] - lo[1])
        say(f"RESULT {TAG}: K = {K} -> {K + 1} at A_t = {At:.10f} (bracket [{lo[0]:.10f}, {hi[0]:.10f}])")
        json.dump(dict(K=K, A_t=At, lo=lo[0], hi=hi[0], hist=hist), open(f"c60_{TAG}.json", "w"), indent=1)
        break
    prev = (A, s, u)
say(f"done [{time.time() - t0:.0f}s]")
