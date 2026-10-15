"""q1921 (15 Oct 2026): ball-arithmetic enclosure of the constant of Theorem C at xi = 2411/10000 (task P2CPBALL). Reading.

c_P = max over xi of c0max(xi)^{1/2} I(xi)/(2 pi) >= the value at the fixed xi = 0.2411, so a rigorous lower bound at this xi is a
rigorous lower bound on c_P. Every quantity is an arb ball (python-flint, 256 bits):
  kappa_L: root of log k + (1 - k)/xi in (0, xi); the function is increasing there (derivative 1/k - 1/xi > 0), so bisection with
           certified signs gives an interval [k_lo, k_hi] that contains the root;
  varrho = L(1, M)/L(1, ell), L(1, z) = (z - 1)/log z, M = exp(1 - psi(2) + e^{psi(2)}), ell = exp(-2.777);
  c0max = [(1 + xi kappa^{-1/2}) kappa^{-1/4} varrho^{1/2}/phi_xi]^{-2/3} evaluated on the kappa ball;
  I = int_{kappa_L}^1 (h(f)/f)^{1/2} df, h(f) = (xi log f + 1 - f)/phi_xi, h = 0 at both ends and h <= h(xi) = 1:
      lower bound  I_lo = acb.integral over [k_hi + d, 1 - d] (h > 0 there, analytic square root), integrand >= 0 elsewhere;
      upper bound  I_hi = I_lo + (k_hi - k_lo + 2 d) max integrand, max integrand <= k_lo^{-1/2}.
Prints the enclosures and c_P_lo = (c0max lower)^{1/2} I_lo/(2 pi), c_P_hi likewise. Last line READING q1921.
"""
import sys

from flint import arb, acb, ctx

ctx.prec = 256
xi = arb(2411) / 10000
phi = xi * xi.log() + 1 - xi


def F(k):
    return k.log() + (1 - k) / xi


lo, hi = arb("1e-6"), xi * arb("0.999999")
assert F(lo) < 0 and F(hi) > 0
for _ in range(200):
    mid = (lo + hi) / 2
    mid = arb(mid.mid())
    v = F(mid)
    if v < 0:
        lo = mid
    elif v > 0:
        hi = mid
    else:
        break
k_lo, k_hi = lo, hi
kappa = arb((k_lo.mid() + k_hi.mid()) / 2, (k_hi.mid() - k_lo.mid()) / 2 + arb("1e-70").mid())
assert F(k_lo) < 0 and F(k_hi) > 0
psi2 = 1 - arb.const_euler()
lamM = 1 - psi2 + psi2.exp()
M = lamM.exp(); ell = (-arb("2.777")).exp()
L1 = lambda z: (z - 1) / z.log()
varrho = L1(M) / L1(ell)
c0 = ((1 + xi / kappa.sqrt()) * kappa ** arb("-0.25") * varrho.sqrt() / phi) ** (arb(-2) / 3)

d = arb("1e-8")


def integrand(z, analytic):
    hz = (acb(xi) * z.log() + 1 - z) / acb(phi)
    return (hz / z).sqrt(analytic=analytic)


I_lo_ball = acb.integral(integrand, acb(k_hi + d), acb(1 - d), rel_tol=arb("1e-30"))
I_lo = I_lo_ball.real.lower()
I_hi = I_lo_ball.real.upper() + (k_hi - k_lo + 2 * d) * (k_lo ** arb("-0.5")).upper()
two_pi = 2 * arb.pi()
cP_lo = (c0.lower().sqrt() * I_lo / two_pi).lower()
cP_hi = (c0.upper().sqrt() * arb(I_hi) / two_pi).upper()
print("kappa_L in [%s, %s] (width %s)" % (k_lo.str(20), k_hi.str(20), (k_hi - k_lo).str(3)))
print("lambda_M =", lamM.str(15), "; varrho =", varrho.str(15))
print("c0max =", c0.str(15))
print("I in [%s, %s]; imaginary part of the integral ball: %s" % (I_lo.str(15), arb(I_hi).str(15), I_lo_ball.imag.str(5)))
print("c_P(xi = 0.2411) in [%s, %s]" % (cP_lo.str(12), cP_hi.str(12)))
ok = cP_lo > arb("0.0560")
print("lower end above 0.0560: %s; float value 0.05606 inside [lo - 1e-5, hi + 1e-5]: %s"
      % (ok, bool(cP_lo - arb("1e-5") < arb("0.05606") < cP_hi + arb("1e-5"))))
print("READING q1921: done")
sys.exit(0 if ok else 2)
