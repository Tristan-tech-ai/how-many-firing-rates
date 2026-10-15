"""q1945_cmp_finite_checks.py (15 Oct 2026, task P2CMPFINITE): numerical checks of each step of the finiteness proof in
paper_neuron/CMP_FINITE_PROOF_2026-10-15.md for the Conway-Maxwell-Poisson counting channel in mean form.
Natural parameter theta, u = exp(theta); W_u(y) = u^y (y!)^-zeta / Z(u). Input rates [lam, lam + A] <-> u in [u0, u1] (u0 = 0 at lam = 0).
Checks, for (A, lam, zeta) in CASES (fibre-like values; lam = 0 is the endpoint case of six fibres):
 C1 mean m(u) strictly increasing on a grid of u from 1e-6 to 1e3 (step 1);
 C2 the optimal input on a grid of 801 rates (cutting plane as in q1939) and its output law q satisfy the two tail bounds of
    step 2: q(y) >= p_a u_a^y (y!)^-zeta / Z(u1) and q(y) <= u1^y (y!)^-zeta, for every y of the truncated range;
 C3 KKT on the input grid: max_u i(u) - C <= 1e-7 (the solver's own tolerance is 1e-8);
 C4 growth (step 3): for theta > theta1, i(theta) >= int_{theta1}^{theta} (m(theta) - m(s)) ds - log Z(theta1), and i(u) -> large;
 C5 endpoint (step 4, lam = 0 only): i(u) - C = u log u + a1 u + o(u) with i(u) < C for small u > 0.
Prints one line per check and ALL CHECKS PASS / SOME CHECK FAILED as last line (a reading aid for the proof, not a registered test)."""
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support_exact import _solve_w  # noqa: E402

YMAX = 400
Y = np.arange(YMAX + 1)
LF = gammaln(Y + 1)
CASES = [(8.0, 0.5, 3.0), (15.0, 1.0, 2.99), (8.0, 0.0, 3.0), (5.0, 0.0, 2.33)]


def logw(theta, zeta):
    lw = theta * Y - zeta * LF
    lz = np.logaddexp.reduce(lw)
    return lw - lz, lz


def mean_theta(theta, zeta):
    lw, _ = logw(theta, zeta)
    return float(np.exp(lw) @ Y)


def theta_of(r, zeta):
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_theta(mid, zeta) > r:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def rows(rs, zeta):
    P = np.zeros((len(rs), len(Y)))
    for i, r in enumerate(rs):
        if r == 0:
            P[i, 0] = 1.0
        else:
            lw, _ = logw(theta_of(r, zeta), zeta)
            P[i] = np.exp(lw)
    return P


def optimum(A, lam, zeta, ngrid=801, tol=1e-8, maxit=60):
    rs = np.linspace(lam, lam + A, ngrid); P = rows(rs, zeta)
    logP = np.log(np.where(P > 0, P, 1e-300)); Hx = (P * logP).sum(1)
    S = [0, ngrid - 1]
    for _ in range(maxit):
        w, C = _solve_w(P[S], Hx[S]); q = w @ P[S]; lq = np.log(np.where(q > 0, q, 1e-300)); D = Hx - P @ lq
        k = int(D.argmax())
        if D[k] - C <= tol or k in S:
            break
        S.append(k)
    return rs, P, S, np.asarray(w), C, q, D


def i_of_theta(theta, zeta, lq):
    lw, _ = logw(theta, zeta); w = np.exp(lw)
    return float(w @ (lw - lq))


def main():
    allok = True
    for A, lam, zeta in CASES:
        tag = "A %.1f lam %.1f zeta %.2f" % (A, lam, zeta)
        us = np.exp(np.linspace(np.log(1e-6), np.log(1e3), 400))
        ms = np.array([mean_theta(np.log(u), zeta) for u in us])
        ok1 = bool(np.all(np.diff(ms) > 0)); allok &= ok1
        print("C1 %s: mean increasing on 400 u from 1e-6 to 1e3: %s (m from %.3g to %.3g)" % (tag, ok1, ms[0], ms[-1]))
        rs, P, S, w, C, q, D = optimum(A, lam, zeta)
        pts = rs[S]; ok3 = bool(D.max() - C <= 1e-7); allok &= ok3
        print("C3 %s: C %.6f nats, support size %d, max i - C on grid %.2e: %s" % (tag, C, int(np.sum(w > 1e-6)), D.max() - C, ok3))
        th1 = theta_of(lam + A, zeta); u1 = np.exp(th1)
        _, lz1 = logw(th1, zeta)
        pos = [(r, wi) for r, wi in zip(pts, w) if r > 0 and wi > 1e-6]
        r_a = min(r for r, _ in pos); u_a = np.exp(theta_of(r_a, zeta)); p_a = sum(wi for r, wi in pos if r >= r_a)
        ylim = int(np.max(np.nonzero(q > 1e-250)[0]))
        yy = Y[: ylim + 1]
        lower = np.log(p_a) + yy * np.log(u_a) - zeta * LF[: ylim + 1] - lz1
        upper = yy * np.log(u1) - zeta * LF[: ylim + 1]
        lqv = np.log(q[: ylim + 1])
        ok2 = bool(np.all(lqv >= lower - 1e-9) and np.all(lqv <= upper + 1e-9)); allok &= ok2
        print("C2 %s: tail bounds hold for y = 0..%d: %s (min slack lower %.3g, upper %.3g)"
              % (tag, ylim, ok2, float(np.min(lqv - lower)), float(np.min(upper - lqv))))
        lq = np.log(np.where(q > 0, q, 1e-300))
        ths = th1 + np.linspace(0.05, 3.0, 30); ok4 = True; last = None
        sgrid = np.linspace(th1, ths[-1], 3001); msg = np.array([mean_theta(s, zeta) for s in sgrid])
        for th in ths:
            iv = i_of_theta(th, zeta, lq); mth = mean_theta(th, zeta)
            sel = sgrid <= th; integ = np.trapezoid(mth - msg[sel], sgrid[sel])
            bound = integ - lz1
            ok4 &= iv >= bound - 1e-6
            last = (th - th1, iv, bound)
        ok4 &= last[1] > C + 1.0; allok &= ok4
        print("C4 %s: i >= growth bound for 30 theta above theta1: %s (at theta1 + %.2f: i %.3f, bound %.3f, C %.3f)"
              % (tag, ok4, last[0], last[1], last[2], C))
        if lam == 0:
            th = np.log(np.array([1e-3, 1e-4, 1e-5, 1e-6]))
            dv = np.array([i_of_theta(t, zeta, lq) for t in th]) - C
            u = np.exp(th); a1 = (dv - u * np.log(u)) / u
            ok5 = bool(np.all(dv < 0) and np.ptp(a1[1:]) < 1e-2); allok &= ok5
            print("C5 %s: i - C at u = 1e-3..1e-6: %s; (i - C - u log u)/u = %s: %s"
                  % (tag, " ".join("%.3e" % v for v in dv), " ".join("%.4f" % v for v in a1), ok5))
    print("ALL CHECKS PASS" if allok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
