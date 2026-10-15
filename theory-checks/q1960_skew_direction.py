"""q1960_skew_direction.py (16 Oct 2026, reading for amendment 2304): which direction of asymmetry do the three sigmoids of q1957 have?
A sigmoid F rising from 0 to 1 is a distribution function; the sign of the skewness of its density says where it is steep.
Positive skewness: short left tail, long right tail, so the curve leaves its floor abruptly and approaches its ceiling slowly.
Computed by quadrature on z in [-12, 12] for M_sq = Phi(z)^2, Gompertz = exp(-exp(-z)), complementary log-log = 1 - exp(-exp(z)),
and also the tail distances: the z-distance from the 10 to the 50 percent level against that from 50 to 90 percent."""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

z = np.linspace(-12, 12, 480001)
curves = {"M_sq Phi^2": norm.cdf(z) ** 2,
          "Gompertz exp(-exp(-z))": np.exp(-np.exp(-z)),
          "cll 1-exp(-exp(z))": 1 - np.exp(-np.exp(np.clip(z, -50, 50))),
          "logistic": 1 / (1 + np.exp(-z))}
for name, F in curves.items():
    f = np.gradient(F, z); f /= np.trapezoid(f, z)
    m = np.trapezoid(z * f, z); v = np.trapezoid((z - m) ** 2 * f, z); s = np.trapezoid((z - m) ** 3 * f, z) / v ** 1.5
    q = lambda p: float(np.interp(p, F, z))
    lo, hi = q(0.5) - q(0.1), q(0.9) - q(0.5)
    print("%-26s skewness %+.4f; z(50)-z(10) %.4f, z(90)-z(50) %.4f -> %s" % (
        name, s, lo, hi, "abrupt start, slow end" if hi > lo + 1e-9 else ("slow start, abrupt end" if lo > hi + 1e-9 else "symmetric")))
