"""q1906 (15 Oct 2026): the two range constants of section 4 of paper_neuron/THM_C_PROOF.md, rechecked. Reading only.
 (U) lamM = f(1) = 1 - psi(2) + e^{psi(2)} in ball arithmetic (python-flint), and the elementary bound f(y) <= 3/2 + 9/(8y) for
     y >= 1 (from psi(y + 1) > log(y + 1/2), log(1 + u) >= u - u^2/2 and psi(y + 1) < log(y + 1)), which is below f(1) for y >= 2;
     a float check of f against that bound for y = 1 .. 10^5.
 (L) the large-y1 branch of the lower bound (POISLB 3.2 (b)) with c = 2.6 fixed: lam <= (lamM/c + Rb(y1))/(1 - 1/c) with the
     ceiling-free majorant Rb(y) = (c^2 y + 2 c sqrt(y) + 1)/(8 (y - c sqrt y)) of R^2/(8 (y - R + 1)), R = ceil(c sqrt y); Rb is
     checked to be decreasing on [3001, 10^7] and its value at 3001 gives the constant; the exact ceiling form is evaluated on
     y1 = 3001 .. 20000 for comparison (Amendment 658: max 2.775616 at 3026).
 Then the asymptotic constant c_P of Theorem C with ell = e^{-lam_l} for lam_l from (L), by q1903's formulas.
"""
import math
from flint import arb

# (U)
psi2 = arb(1) - arb.const_euler()                      # psi(2) = 1 - gamma
lamM = 1 - psi2 + psi2.exp()
print("(U) lamM = 1 - psi(2) + e^{psi(2)} =", lamM.str(12))
from scipy.special import digamma
worst = -1e9
for y in range(1, 100001):
    f = 1 + y * (math.log(y) - digamma(y + 1)) + math.exp(digamma(y + 1)) / y
    worst = max(worst, f - (1.5 + 9 / (8 * y)))
print("(U) max over y = 1..1e5 of f(y) - (3/2 + 9/(8y)) = %.3e (must be <= 0); bound at y = 2: %.4f < lamM" % (worst, 1.5 + 9 / 16))

# (L)
c = 2.6
lM = float(lamM.mid())
Rb = lambda y: (c * c * y + 2 * c * math.sqrt(y) + 1) / (8 * (y - c * math.sqrt(y)))
dec = all(Rb(y + 1) <= Rb(y) for y in range(3001, 10 ** 7, 997))
lam_b = (lM / c + Rb(3001)) / (1 - 1 / c)
exact = max(((lM / c + math.ceil(c * math.sqrt(y)) ** 2 / (8 * (y - math.ceil(c * math.sqrt(y)) + 1))) / (1 - 1 / c), y)
            for y in range(3001, 20001))
print("(L) Rb decreasing on [3001, 1e7] (sampled every 997): %s; lam bound from Rb(3001): %.5f; exact ceiling form max %.6f at y1 = %d"
      % (dec, lam_b, exact[0], exact[1]))
lam_l = math.ceil(lam_b * 1000) / 1000
print("(L) lam_l used: %.3f (the y1 <= 3000 branch gives at most 1.7897 in POISLB's floating-point scan, pl6)" % lam_l)

import q1903_poislb_constants as Q
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
for name, M, ell in [("POISLB (2.1034, 2.7553)", math.exp(2.1034), math.exp(-2.7553)),
                     ("corrected (lamM, %.3f)" % lam_l, math.exp(lM), math.exp(-lam_l))]:
    rho = Q.L1(M) / Q.L1(ell)

    def val(xi):
        phi = xi * math.log(xi) + 1 - xi; kL = Q.kappaL(xi)
        c0 = ((1 + xi / math.sqrt(kL)) * kL ** -0.25 * rho ** 0.5 / phi) ** (-2 / 3)
        I = quad(lambda f: math.sqrt(max(0.0, (xi * math.log(f) + 1 - f) / phi) / f), kL, 1, limit=400, epsabs=1e-13)[0]
        return math.sqrt(c0) * I / (2 * math.pi), c0, I
    r = minimize_scalar(lambda x: -val(x)[0], bounds=(0.1, 0.5), method="bounded", options={"xatol": 1e-8})
    v = val(r.x)
    print("c_P with %-26s rho %.4f  xi* %.4f  c0max %.5f  I %.5f  c_P %.5f" % (name, rho, r.x, v[1], v[2], v[0]))
print("READING q1906: done")
