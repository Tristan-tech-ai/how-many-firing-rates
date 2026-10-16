"""q1903 (15 Oct 2026): recompute the asymptotic constant of POISLB's Theorem K-Poisson from its formulas (POISLB_ATTEMPT.md
4.1, 4.4 (ii), 6, 7.5) as a function of the two range constants M (m <= M r) and ell (m >= ell r on [0, A]). Reading only.

  L(1, z) = (z - 1)/log z,  rho = L(1, M)/L(1, ell),
  phi_xi = xi log xi + 1 - xi,  kappa_L in (0, xi) solves log k + (1 - k)/xi = 0,
  Gam_xi = (1 + xi/sqrt(kappa_L)) kappa_L^{-1/4} rho^{1/2}/phi_xi,  c0max = Gam_xi^{-2/3},
  mm(f1, f2) = min_{f in [f1, f2]} (xi log f + 1 - f)/phi_xi,
  constant = max over xi, kappa_L < f1 < f2 < 1 of ((f2 - f1)/(4 pi)) (mm c0max/f2)^{1/2}.
Controls: POISLB's own constants (M = e^{2.1034}, ell = e^{-2.7553}) must give c0max(0.30) = 0.0720 and 0.0117; the "actual
constants" line (M = e, ell = 0.625) must give 0.0152.
"""
import math
from scipy.optimize import brentq


def L1(z):
    return 1.0 if abs(z - 1) < 1e-12 else (z - 1) / math.log(z)


def kappaL(xi):
    return brentq(lambda k: math.log(k) + (1 - k) / xi, 1e-12, xi * (1 - 1e-12))


def const(M, ell, xis=None):
    rho = L1(M) / L1(ell)
    best = (0, None)
    for xi in xis or [0.05 + 0.01 * i for i in range(60)]:
        phi = xi * math.log(xi) + 1 - xi
        kL = kappaL(xi)
        G = (1 + xi / math.sqrt(kL)) * kL ** -0.25 * rho ** 0.5 / phi
        c0 = G ** (-2 / 3)
        h = lambda f: (xi * math.log(f) + 1 - f) / phi
        for i in range(1, 60):
            for j in range(i + 1, 60):
                f1 = kL + (1 - kL) * i / 60; f2 = kL + (1 - kL) * j / 60
                mm = min(h(f1), h(f2))                               # h is concave, so its minimum is at an end
                if mm <= 0:
                    continue
                val = (f2 - f1) / (4 * math.pi) * (mm * c0 / f2) ** 0.5
                if val > best[0]:
                    best = (val, (round(xi, 3), round(kL, 5), round(c0, 4), round(f1, 3), round(f2, 3), round(mm, 3)))
    return rho, best


def main():
    print("control c0max(0.30) with POISLB constants:", end=" ")
    rho = L1(math.exp(2.1034)) / L1(math.exp(-2.7553)); xi = 0.30; kL = kappaL(xi); phi = xi * math.log(xi) + 1 - xi
    print(round(((1 + xi / math.sqrt(kL)) * kL ** -0.25 * rho ** 0.5 / phi) ** (-2 / 3), 4), "rho", round(rho, 3))
    cases = [("POISLB proved (M e^2.1034, ell e^-2.7553)", math.exp(2.1034), math.exp(-2.7553)),
             ("POISLB 'actual' (M e, ell 0.625)", math.e, 0.625),
             ("M = e^1.5 (bulk of P-a'), ell POISLB", math.exp(1.5), math.exp(-2.7553)),
             ("M = e (target of P-a), ell POISLB", math.e, math.exp(-2.7553)),
             ("M = e, ell = e^-1.1648 (Gaussian lambda_1)", math.e, math.exp(-1.1648)),
             ("M = e, ell = 1 (ideal)", math.e, 1.0)]
    for name, M, ell in cases:
        rho, (val, par) = const(M, ell)
        print("%-46s rho %7.3f  constant %.5f  (xi, kappa_L, c0max, f1, f2, mm) = %s" % (name, rho, val, par))
    print("READING q1903: constants printed above; the Gaussian chain of paper 1 reaches sqrt(c0max)/4 with no geometric factor")


if __name__ == "__main__":
    main()


def window_sum(M, ell):
    """Paper 1's assembly (sum over windows with the LOCAL decay rate) in the Fisher coordinate t = 2 sqrt(x):
    per unit t at least sqrt(Lambda(x))/(2 pi) atoms, Lambda(x) = Lambda0 h(x/A), h(f) = (xi log f + 1 - f)/phi_xi on (kappa_L, 1);
    dt = dx/sqrt(x) gives K >= (sqrt(c0max)/(2 pi)) I(xi) A^(2/3), I(xi) = int_{kappa_L}^1 sqrt(h(f)/f) df. A reading of what the
    assembly step would give if Lemma C-Poisson is applied window by window as in paper 1; not proved here."""
    from scipy.integrate import quad
    rho = L1(M) / L1(ell); best = (0, None)
    for xi in [0.05 + 0.01 * i for i in range(60)]:
        phi = xi * math.log(xi) + 1 - xi; kL = kappaL(xi)
        c0 = ((1 + xi / math.sqrt(kL)) * kL ** -0.25 * rho ** 0.5 / phi) ** (-2 / 3)
        I = quad(lambda f: math.sqrt(max(0.0, (xi * math.log(f) + 1 - f) / phi) / f), kL, 1, limit=200)[0]
        val = math.sqrt(c0) * I / (2 * math.pi)
        if val > best[0]:
            best = (val, (round(xi, 3), round(c0, 4), round(I, 4)))
    return best


if __name__ == "__main__":
    for name, M, ell in [("POISLB proved", math.exp(2.1034), math.exp(-2.7553)), ("M = e, ell POISLB", math.e, math.exp(-2.7553)),
                         ("M = e, ell = 0.625", math.e, 0.625), ("ideal M = e, ell = 1", math.e, 1.0)]:
        v, par = window_sum(M, ell)
        print("WINDOW SUM %-22s constant %.5f  (xi, c0max, I) = %s" % (name, v, par))
