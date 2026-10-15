"""q1908 (15 Oct 2026): task P2R1. Ball-arithmetic certificate for branch (a) of Proposition 4.3 of paper_neuron/THM_C_PROOF.md
(POISLB 3.2 (a), floating-point scan pl6_g2scan.py: largest admissible lam over y1 <= 3000 is 1.7897).

Claim certified: for every integer y1 in [1, 3000], g(y1) > -LAM_TEST (LAM_TEST = 1.8 by default). (y1 = 0 has g(0) = 1.)
Logic: suppose g(y1) = -lam with lam >= LAM_TEST. Every chord bound
  b_R(y) = (1 - theta)(-lam + phi(y1)) + theta (lamM + phi(y_end)) - phi(y),  theta = |y - y1|/R, y between y1 and y_end = y1 +- R,
is an upper bound on g(y) (convexity of G = g + phi + const and g(y_end) <= lamM), and is nonincreasing in lam; lamM is an upper
bound everywhere (Proposition 4.2). So with U(y) = min of any chosen chord bounds at lam = LAM_TEST (or lamM),
E_{y1} g(Y) <= sum_{y in window} p(y | y1) U(y) + lamM (1 - sum_{y in window} p(y | y1)) =: EU.
The KKT inequality at the feasible point y1 gives E_{y1} g(Y) >= delta0(y1). So EU - delta0(y1) < 0 (an upper bound computed in
arb, delta0 = e^{-x} - x E1(x) in arb) is a contradiction, hence lam < LAM_TEST. The chords are CHOSEN in floating point (argmin of the float bounds); only the chosen bound is
evaluated in arb with outward rounding, which keeps the argument rigorous (any chord is valid).
Output: paper_neuron/q1908_out.txt; last line READING with the count of certified y1 and the largest arb upper bound of EU - delta0.
"""
import sys
import time
import numpy as np
from scipy.special import digamma, gammaln
from flint import arb, ctx

ctx.prec = 128
LAM_TEST = float(sys.argv[1]) if len(sys.argv) > 1 else 1.8
Y1MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
t0 = time.time()
psi2 = arb(1) - arb.const_euler()
lamM_a = 1 - psi2 + psi2.exp()
lamM = float(lamM_a.upper())                                  # float copy, rounded up (used only to choose chords)
YTOP = Y1MAX + int(14 * np.sqrt(Y1MAX + 1) + 40)
phi_a = [arb(y) * (arb(y) + 1).digamma() - arb(y) for y in range(YTOP + 1)]
phi_f = np.array([y * digamma(y + 1.0) - y for y in range(YTOP + 1)])
lam_a = arb(LAM_TEST)
worst = None; nfail = 0
for y1 in range(1, Y1MAX + 1):
    s = np.sqrt(y1 + 1.0); Rmax = int(12 * s + 20)
    lo = max(0, int(y1 - 14 * s - 30)); hi = int(y1 + 14 * s + 30)
    yy = np.arange(lo, hi + 1); i1 = y1 - lo
    U = np.full(len(yy), lamM); choice = [None] * len(yy)
    U[i1] = -LAM_TEST; choice[i1] = ("self", 0)
    for R in range(1, Rmax + 1):
        for side in (1, -1):
            j = i1 + side * R
            if j < 0 or j >= len(yy) or (side == -1 and R > y1):
                continue
            a_, b_ = (i1, j) if side == 1 else (j, i1)
            seg = np.arange(a_, b_ + 1)
            th = np.abs(yy[seg] - y1) / R
            bnd = (1 - th) * (-LAM_TEST + phi_f[y1]) + th * (lamM + phi_f[yy[j]]) - phi_f[yy[seg]]
            better = bnd < U[seg]
            for k in seg[better]:
                choice[k] = (side, R)
            U[seg] = np.minimum(U[seg], bnd)
    # arb evaluation
    ly1 = arb(y1).log(); mass = arb(0); EU = arb(0)
    for k, y in enumerate(yy):
        p = (arb(-y1) + arb(int(y)) * ly1 - arb(int(y) + 1).lgamma()).exp()
        mass += p
        c = choice[k]
        if c is None:
            u = lamM_a
        elif c[0] == "self":
            u = -lam_a
        else:
            side, R = c; ye = y1 + side * R
            th = arb(abs(int(y) - y1)) / R
            u = (1 - th) * (-lam_a + phi_a[y1]) + th * (lamM_a + phi_a[ye]) - phi_a[int(y)]
        EU += p * u
    EU += lamM_a * (1 - mass)
    EU = EU - ((-arb(y1)).exp() + arb(y1) * (-arb(y1)).ei())          # minus delta0(y1) = e^{-x} - x E1(x), E1(x) = -Ei(-x)
    ub = float(EU.upper())
    if not (ub < 0):
        nfail += 1
        print("NOT CERTIFIED at y1 = %d: EU upper bound %.3e" % (y1, ub), flush=True)
    if worst is None or ub > worst[0]:
        worst = (ub, y1)
    if y1 % 500 == 0:
        print("y1 %d done, worst EU upper bound so far %.4e at y1 = %d   [%.0fs]" % (y1, worst[0], worst[1], time.time() - t0), flush=True)
print("READING q1908: LAM_TEST = %.4f, y1 = 1..%d, certified %d, not certified %d, largest arb upper bound of EU - delta0 %.4e at y1 = %d   [%.0fs]"
      % (LAM_TEST, Y1MAX, Y1MAX - nfail, nfail, worst[0], worst[1], time.time() - t0))
