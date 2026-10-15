"""q1918 (15 Oct 2026): instrument controls for route E of paper 2 (task P2FBL, plan paper_neuron/ROUTE_E_PLAN_2026-10-15.md), run
before the route E registration. Reading only; no on-off versus graded comparison is computed here.

DT bound (Polyanskiy, Poor, Verdu 2010, Theorem 17) for an i.i.d. input law P^n:
  eps <= DT(c) = E[ exp(-(S_n - c)^+) ],  S_n = i(X^n; Y^n) = sum of n i.i.d. letter densities, c = log((M - 1)/2).
Instrument (the one route E will use): the law of the letter density i(x, y) = log W(y|x) - log q(y) is put on a grid of step h;
rounding every value DOWN gives an upper bound on DT (g(s) = min(1, e^(c - s)) decreases in s), rounding UP a lower bound; the
n-fold law is an exact discrete convolution (FFT, negatives clipped); output values whose Poisson tail mass is below 1e-15 are cut and
their mass is added with g = 1 to the upper bound only.
Control 1: binary symmetric channel, crossover 0.11, n = 20, uniform input: the binned bracket against (a) the closed form over the
number of flips and (b) a brute force over all 2^20 outputs with i computed from W and q directly (no closed form).
Control 2: Poisson channel, peak count A = 5, no dark count, n = 10, on-off input with weight 0.4 on A: the bracket against a Monte
Carlo estimate with 10^6 blocks (pass if the estimate lies inside the bracket widened by 4 standard errors).
Control 3: largest log M with DT <= eps (bisection on c) for the BSC cell at eps = 0.1, against the closed form.
Last line: READING q1918: controls <pass/fail>.
"""
import itertools
import math
import sys

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import gammaln

H = 2e-5
TAIL = 1e-15


def letter_law(xs, px, W):
    """xs: input points, px: their weights, W(x) -> (y values, probs). Returns values v, probs p of i(X;Y), and cut mass."""
    ys_all = sorted(set(int(y) for x in xs for y in W(x)[0]))
    q = {y: 0.0 for y in ys_all}
    rows = []
    for x, p in zip(xs, px):
        ys, w = W(x)
        rows.append((p, ys, w))
        for y, wy in zip(ys, w):
            q[int(y)] += p * wy
    v, pr = [], []
    for p, ys, w in rows:
        for y, wy in zip(ys, w):
            if wy > 0 and p > 0:
                v.append(math.log(wy) - math.log(q[int(y)])); pr.append(p * wy)
    pr = np.array(pr); return np.array(v), pr, max(0.0, 1.0 - pr.sum())


def binned_sum_law(v, p, n, up):
    """Law of the sum of n letters on the grid H after rounding each letter value up (up=True) or down."""
    k = np.ceil(v / H).astype(np.int64) if up else np.floor(v / H).astype(np.int64)
    k0 = k.min(); arr = np.zeros(k.max() - k0 + 1); np.add.at(arr, k - k0, p)
    out = np.array([1.0]); base = arr.copy(); m = n; off = 0
    while m:
        if m & 1:
            out = np.clip(fftconvolve(out, base), 0, None)
        m >>= 1
        if m:
            base = np.clip(fftconvolve(base, base), 0, None)
    return (np.arange(len(out)) + n * k0) * H, out


def dt_from_law(s, pr, c):
    return float(np.sum(pr * np.minimum(1.0, np.exp(c - s))))


def dt_bracket(v, p, cut, n, c):
    s_lo, pr_lo = binned_sum_law(v, p, n, up=False)   # values rounded down -> DT upper
    s_hi, pr_hi = binned_sum_law(v, p, n, up=True)
    total_cut = 1.0 - (1.0 - cut) ** n
    return dt_from_law(s_hi, pr_hi, c), dt_from_law(s_lo, pr_lo, c) + total_cut


def bsc_W(d):
    return lambda x: (np.array([x, 1 - x]), np.array([1 - d, d]))


def poisson_W(x):
    if x == 0:
        return np.array([0]), np.array([1.0])
    ymax = int(x + 40 * math.sqrt(x + 1) + 60)
    y = np.arange(ymax + 1); w = np.exp(-x + y * math.log(x) - gammaln(y + 1))
    keep = w > TAIL * 1e-3
    return y[keep], w[keep]


def main():
    ok = True
    d, n = 0.11, 20
    v, p, cut = letter_law([0, 1], [0.5, 0.5], bsc_W(d))
    for c in (2.0, 5.0, 8.0):
        lo, hi = dt_bracket(v, p, cut, n, c)
        closed = sum(math.comb(n, k) * d ** k * (1 - d) ** (n - k)
                     * min(1.0, math.exp(c - (n * math.log(2) + k * math.log(d) + (n - k) * math.log(1 - d)))) for k in range(n + 1))
        # brute force: x^n = 0, all outputs, q(y^n) = sum over both inputs per letter computed from W
        terms = []
        for ybits in itertools.product((0, 1), repeat=n):
            k = sum(ybits)
            wy = d ** k * (1 - d) ** (n - k)
            qy = 1.0
            for b in ybits:
                qy *= 0.5 * (1 - d if b == 0 else d) + 0.5 * (d if b == 0 else 1 - d)
            terms.append(wy * min(1.0, math.exp(c - (math.log(wy) - math.log(qy)))))
        tot = math.fsum(terms)
        c1 = lo - 1e-12 <= closed <= hi + 1e-12 and abs(closed - tot) < 1e-12 * max(1.0, closed) and (hi - lo) < 1e-3 * hi
        ok &= c1
        print("Reading: control 1 BSC(0.11) n 20 c %.1f: bracket [%.10f, %.10f], closed form %.10f, brute force %.10f -> %s"
              % (c, lo, hi, closed, tot, "pass" if c1 else "FAIL"))
    # control 3: largest log M with DT <= 0.1
    def best_logM(dt, eps):
        a, b = -5.0, 50.0
        for _ in range(80):
            m = 0.5 * (a + b)
            (a, b) = (m, b) if dt(m) <= eps else (a, m)
        return math.log(2 * math.exp(a) + 1)
    closed_dt = lambda c: sum(math.comb(n, k) * d ** k * (1 - d) ** (n - k)
                              * min(1.0, math.exp(c - (n * math.log(2) + k * math.log(d) + (n - k) * math.log(1 - d)))) for k in range(n + 1))
    bin_dt = lambda c: dt_bracket(v, p, cut, n, c)[1]
    L1, L2 = best_logM(closed_dt, 0.1), best_logM(bin_dt, 0.1)
    c3 = L2 <= L1 + 1e-9 and L1 - L2 < 1e-2
    ok &= c3
    print("Reading: control 3 BSC eps 0.1: log M closed %.6f nats (%.4f bits), binned upper-bracket %.6f nats -> %s"
          % (L1, L1 / math.log(2), L2, "pass" if c3 else "FAIL"))
    # control 2: Poisson cell
    A, n2, wA = 5.0, 10, 0.4
    v, p, cut = letter_law([0.0, A], [1 - wA, wA], poisson_W)
    rng = np.random.default_rng(20261015)
    yA, wAy = poisson_W(A)
    q = {int(y): wA * w for y, w in zip(yA, wAy)}; q[0] = q.get(0, 0.0) + (1 - wA)
    for c in (1.0, 3.0):
        lo, hi = dt_bracket(v, p, cut, n2, c)
        N = 10 ** 6
        X = rng.random((N, n2)) < wA
        Y = np.where(X, rng.poisson(A, (N, n2)), 0)
        logW = np.where(X, -A + Y * math.log(A) - gammaln(Y + 1), 0.0)
        qv = np.vectorize(lambda y: q[int(y)])(Y)
        S = (logW - np.log(qv)).sum(1)
        g = np.minimum(1.0, np.exp(c - S)); est, se = g.mean(), g.std() / math.sqrt(N)
        c2 = lo - 4 * se <= est <= hi + 4 * se and (hi - lo) < 1e-3 * hi
        ok &= c2
        print("Reading: control 2 Poisson A 5 n 10 on-off w 0.4 c %.1f: bracket [%.8f, %.8f], Monte Carlo %.8f +- %.8f -> %s"
              % (c, lo, hi, est, se, "pass" if c2 else "FAIL"))
    print("READING q1918: controls %s" % ("pass" if ok else "fail"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
