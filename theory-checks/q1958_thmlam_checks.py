"""q1958_thmlam_checks.py (16 Oct 2026, task P2THMLAM): numerical checks of the steps that extend Theorem 3 of paper 2 from
spontaneous count 0 to a fixed spontaneous count lam > 0 (draft paper_neuron/THMLAM_PROOF_2026-10-16.md).

Channel Y ~ Poisson(X), X in [lam, lam + A]. Optimum from support_exact.support (cutting plane, KKT tol 1e-8; its x is the
offset above lam). Reference r(y) = exp(-(C+1) + phi(y)) / y!, phi(y) = y psi(y+1) - y, g = log(m / r).
Checks (each prints PASS or FAIL; a FAIL is a counterexample to a step of the draft or a solver defect, to be examined):
  C1  sup over y of g <= lambda_M = 1 - psi(2) + exp(psi(2))          (Proposition C-upper, extended)
  C2  g(0) <= 1 - m0 + m0 log(m1/m0), m0 = E[X|Y=0], m1 = E[X|Y=1]   (new lemma; holds for every lam >= 0)
  C3  if g attains its maximum at y = 0: g(0) <= 1 + 2 exp(-1 - gamma)  (new bound for that case)
  C4  min of g over integers y in [lam, lam + A] >= -lambda_ell = -2.777 (Proposition C-lower, feasible integers only)
  C5  m(0) <= M c_*, with M = exp(lambda_M), c_* = exp(-(C+1))       (edge term (i), replaces m(0) = e c_*)
  C6  sum over the optimum of the KKT residual |E_x g - delta_0(x)| at the mass points <= 1e-6 (solver sanity)
The slack 1e-6 on C1-C5 covers the solver tolerance; the margins are printed.
"""
import sys, math
import numpy as np
from scipy.special import gammaln, digamma, exp1
sys.path.insert(0, r"")
from support_exact import support

LAM_M = 1 - digamma(2) + math.exp(digamma(2))
LAM_L = 2.777
GAM = 0.5772156649015329
B0 = 1 + 2 * math.exp(-1 - GAM)
SL = 1e-6


def analyse(A, lam):
    s = support(A, lam, ngrid=2401, tol=1e-10, maxit=80)
    xs = lam + np.array(s["pts"]); w = np.array(s["w"]); C = s["C"]
    ny = int((A + lam) * 4 + 80); y = np.arange(ny)
    lp = y[None, :] * np.log(np.maximum(xs[:, None], 1e-300)) - xs[:, None] - gammaln(y + 1)[None, :]
    P = np.exp(lp); m = w @ P
    phi = y * digamma(y + 1) - y
    logr = -(C + 1) + phi - gammaln(y + 1)
    g = np.log(m) - logr
    post0 = w * np.exp(-xs); post0 /= post0.sum(); m0 = float(post0 @ xs)
    post1 = post0 * xs; post1 /= post1.sum(); m1 = float(post1 @ xs)
    out = {}
    lim = int(np.searchsorted(m, 1e-200, side="left"))
    gv = g[: max(10, int(A + lam + 20 * math.sqrt(A + lam + 1)))]
    out["C1"] = (float(gv.max()), LAM_M)
    out["C2"] = (float(g[0]), 1 - m0 + m0 * math.log(m1 / m0))
    out["C3"] = (float(g[0]), B0) if int(gv.argmax()) == 0 else None
    ys = np.arange(math.ceil(lam), int(math.floor(lam + A)) + 1)
    out["C4"] = (-LAM_L, float(g[ys].min()))
    out["C5"] = (float(m[0]), math.exp(LAM_M) * math.exp(-(C + 1)))
    d0 = np.exp(-xs) - xs * exp1(np.maximum(xs, 1e-300))
    kkt = (P * g[None, :]).sum(1) - d0
    out["C6"] = (float(np.abs(kkt).max()), 1e-6)
    return s["N"], out, int(gv.argmax()), m0, m1


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cases = [(A, lam) for A in (8, 30, 80, 200) for lam in (0.0, 0.1, 0.4, 1.0, 3.0)]
    nf = 0
    for A, lam in cases:
        N, o, am, m0, m1 = analyse(A, lam)
        line = []
        for k in ("C1", "C2", "C3", "C4", "C5", "C6"):
            v = o[k]
            if v is None:
                line.append(k + " n/a"); continue
            lhs, rhs = v
            ok = lhs <= rhs + SL
            nf += (not ok)
            line.append("%s %s (%.4g <= %.4g)" % (k, "PASS" if ok else "FAIL", lhs, rhs))
        print("A=%4g lam=%4.1f N=%3d argmax g=%d m0=%.4f m1=%.4f | %s" % (A, lam, N, am, m0, m1, " ; ".join(line)))
    print("checks failed: %d of %d cases" % (nf, len(cases)))
    print("VERDICT q1958 checks: %s" % ("PASS" if nf == 0 else "FAIL"))


if __name__ == "__main__":
    main()
