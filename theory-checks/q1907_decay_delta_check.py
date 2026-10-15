"""q1907 (15 Oct 2026): check that the explicit coefficient delta' of the decay theorem (POISLB 4.2 Step 7, section 5 of
paper_neuron/THM_C_PROOF.md) tends to c0^{3/2} Gam_xi, so that delta' < 1 exactly when c0 < c0max(xi) = Gam_xi^{-2/3}.
Reading only. For xi = 0.2411, c0 = 0.9 c0max and A = 1e8 .. 1e40 (mpmath, 50 digits) it evaluates
  sigma = round(c0 A^{1/3} xi/(2 phi_xi)), t = 2 sigma/(xi A), Lambda0 = t A phi_xi, lam+- = 1 +- t/2, Delta = 1/lam- - 1/lam+,
  B^2 = lam+ (Lambda0 + 4 log A), K1 = A + sigma, x_L = kappa_L A,
  A1 = sup_Win |alpha_+ - alpha_-|, A2 = sup_Win alpha_- (on a log grid of y in [x_L, A], with Step 2's exact alpha formulas),
  delta1 = e^{Delta B^2/2} [A1 + A2 (t/lam-)((K1 + 1)^{1/2} B/lam-^{1/2} + 1) + A2 2 sigma B/(lam- (x_L - sigma + 1))^{1/2}],
  delta' = delta1 e^{sigma t/2} (L(1, M) r(x_L)/(L(1, ell) r(floor A)))^{1/2}, r(y)/c* = e^{phi(y)}/y!,
with M = e^{lamM}, ell = e^{-2.777}; and prints delta' against the limit 0.9^{3/2} = 0.8538.
"""
from mpmath import mp, mpf, log, exp, sqrt, loggamma, psi
import math
from scipy.optimize import brentq

mp.dps = 50
xi = mpf("0.2411"); phi = xi * log(xi) + 1 - xi
kL = mpf(brentq(lambda k: math.log(k) + (1 - k) / float(xi), 1e-12, float(xi) * (1 - 1e-12)))
lamM = 1 - psi(0, 2) + exp(psi(0, 2)); lam_l = mpf("2.777")
L1 = lambda z: (z - 1) / log(z)
rho = L1(exp(lamM)) / L1(exp(-lam_l))
Gam = (1 + xi / sqrt(kL)) * kL ** mpf(-0.25) * sqrt(rho) / phi
c0max = Gam ** (mpf(-2) / 3); c0 = mpf("0.9") * c0max
print("Gam = %s, c0max = %s, c0 = 0.9 c0max, limit of delta' = 0.9^{3/2} = %s" % (mp.nstr(Gam, 8), mp.nstr(c0max, 8), mp.nstr(mpf("0.9") ** 1.5, 6)))
lr = lambda y: y * psi(0, y + 1) - y - loggamma(y + 1)          # log(r(y)/c*)
for e in (8, 12, 16, 20, 30, 40):
    A = mpf(10) ** e
    sigma = int(mp.nint(c0 * A ** (mpf(1) / 3) * xi / (2 * phi))); t = 2 * mpf(sigma) / (xi * A); L0 = t * A * phi
    lp, lm = 1 + t / 2, 1 - t / 2; Dl = 1 / lm - 1 / lp
    B = sqrt(lp * (L0 + 4 * log(A))); K1 = A + sigma; xL = kL * A
    ap = lambda y: exp(-sigma * log(y) + t * y / 2 + loggamma(y + sigma + 1) - loggamma(y + 1) - (y + sigma) * log(lp))
    am = lambda y: exp(sigma * log(y) - t * y / 2 + loggamma(y - sigma + 1) - loggamma(y + 1) - (y - sigma) * log(lm))
    ys = [xL * (A / xL) ** (mpf(i) / 200) for i in range(201)]
    A1 = max(abs(ap(y) - am(y)) for y in ys); A2 = max(am(y) for y in ys)
    d1 = exp(Dl * B ** 2 / 2) * (A1 + A2 * (t / lm) * (sqrt(K1 + 1) * B / sqrt(lm) + 1) + A2 * 2 * sigma * B / sqrt(lm * (xL - sigma + 1)))
    dp = d1 * exp(sigma * t / 2) * sqrt(L1(exp(lamM)) * exp(lr(mp.floor(xL)) - lr(mp.floor(A))) / L1(exp(-lam_l)))
    print("A = 1e%2d: sigma %d, Lambda0 %s, A1 %s, A2 - 1 %s, delta' %s" % (e, sigma, mp.nstr(L0, 5), mp.nstr(A1, 3), mp.nstr(A2 - 1, 3), mp.nstr(dp, 6)))
print("READING q1907: done")
