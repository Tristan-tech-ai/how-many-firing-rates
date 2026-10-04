"""
Piece (i) of the certificate, with two gaps closed. 2026-09-06.

A review of the argument after Amendment 70 found two places where the monotone piece did not check what it
claimed:
    1. The left run started at x = 1e-8 and the decade sweep stopped at 1e-300, so (0, 1e-300) was never
       checked. It is closed here by a LEMMA with computed constants, valid on all of (0, x1] for x1 <= 1e-8:
           D'(x) = sum_y P(x,y) (y/x - 1) g(x,y),  g = y log x - x - log y! - log Q(y).
           y = 0: the term is e^{-x} (x + log Q(0)) <= 0 because Q(0) < 1 and x < -log Q(0).
           y = 1: the term is (1 - x) e^{-x} (log x - x - log Q(1)) <= (1 - x1) e^{-x1} (log x1 - log Q(1)_lo),
                  negative because x1 < Q(1)_lo; the factor lies in [(1 - x1)e^{-x1}, 1] and multiplies a
                  negative quantity bounded above by log x1 - log Q(1)_lo.
           y >= 2: |term| <= (x^{y-1}/(y-1)!) (y |log x| + x + log y! + |log Q(y)|); on (0, x1] with x1 < 1/e
                  the factor x^{y-1} |log x| is increasing, so every term is bounded by its value at x1; the
                  sum to ny uses the enclosed Q(y), and beyond ny |log Q(y)| <= log(1/w_A) + A + log y!, with
                  the ratio of consecutive terms at most 2 x1 / y, so the tail is a geometric series.
       The bound is evaluated in interval arithmetic and must be negative.
    2. The right grid started at A - h, and the cell (A - h, A] was exempt from the chord bound as lying
       above right_reach, so nobody checked it. The grid now starts at A itself.
Neither gap changed a verdict (D' is about -15 near 0 and the right margin is about 2), but the certificate
must check them, and now does. This module is used by q32_certificate_v3/v4 from now on, and by
q32_piece_i_fix.py to re-run piece (i) alone at every endpoint already certified.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mpf, mpi, iv, nstr, log, exp, loggamma
from support_iv import Channel
from q32_certificate_v3 import lo, hi, monotone_reach

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def left_sliver(ch, x1):
    """upper bound on D'(x) over (0, x1], by the lemma in the docstring; returns (bound, parts)"""
    X1 = iv.mpf(x1)
    Q0hi = hi(ch.Q[0])
    Q1lo = lo(ch.Q[1])
    assert Q0hi < 1 and mpf(x1) < -log(Q0hi), "y = 0 term needs Q(0) < 1 and x1 < -log Q(0)"
    assert mpf(x1) < Q1lo, "y = 1 term needs x1 < Q(1)"
    assert mpf(x1) < exp(-1)
    t0 = iv.mpf(0)
    t1 = (1 - X1) * iv.exp(-X1) * (iv.log(X1) - iv.log(iv.mpf(Q1lo)))
    A = ch.A
    ny = ch.ny
    wA = mpf(ch.w[-1].a)
    s = iv.mpf(0)
    absl = -iv.log(X1)
    for y in range(2, ny):
        lq = max(abs(log(lo(ch.Q[y]))), abs(log(hi(ch.Q[y]))))
        Ty = (X1 ** (y - 1) / iv.mpf(loggamma(y))).__class__  # placeholder to keep types uniform
        Ty = iv.exp((y - 1) * iv.log(X1) - iv.loggamma(iv.mpf(y))) * (
            y * absl + X1 + iv.loggamma(iv.mpf(y + 1)) + iv.mpf(lq)
        )
        s = s + Ty
    # tail y >= ny: |log Q(y)| <= log(1/w_A) + A + log y!, ratio of consecutive terms <= 2 x1 / ny
    y = ny
    Tny = iv.exp((y - 1) * iv.log(X1) - iv.loggamma(iv.mpf(y))) * (
        y * absl + X1 + 2 * iv.loggamma(iv.mpf(y + 1)) + iv.mpf(A) + iv.log(iv.mpf(1) / iv.mpf(wA))
    )
    ratio = 2 * X1 / ny
    assert hi(ratio) < 1
    tail = Tny / (1 - ratio)
    bound = t0 + t1 + s + tail
    return hi(bound), {"t1": hi(t1), "sum_2_to_ny": hi(s), "tail": hi(tail)}


def piece_i(ch, A_dec, x1=mpf(10) ** -8, n=140, T1=None, T2=None, say=print):
    """the monotone piece with both gaps closed; returns (left_reach, right_reach, ok, info)"""
    xs = ch.xp
    if T1 is None:
        T1 = ch.tail_bound(x1, 1)
    if T2 is None:
        T2 = ch.tail_bound(x1, 2)
    sl, parts = left_sliver(ch, x1)
    x_hi = xs[1] * mpf("0.9")
    left_pts = [x1 * (x_hi / x1) ** (mpf(j) / n) for j in range(n + 1)]
    left_reach, left_margin = monotone_reach(ch, left_pts, True, T1, T2)
    right_pts = [A_dec - (A_dec - xs[-2] * mpf("1.1")) * mpf(j) / n for j in range(0, n + 1)]  # starts AT A
    right_reach, right_margin = monotone_reach(ch, right_pts, False, T1, T2)
    ok = (
        (sl < 0)
        and (left_margin is not None and left_margin > 1)
        and (right_margin is not None and right_margin > 1)
        and right_pts[0] == A_dec
        and right_reach < A_dec
    )
    say(
        f"(i) sliver lemma: D' <= {nstr(sl, 4)} on (0, {nstr(x1, 2)}] (y=1 term {nstr(parts['t1'], 4)}, y>=2 sum {nstr(parts['sum_2_to_ny'], 2)}, tail {nstr(parts['tail'], 2)}); "
        f"grid: D' < 0 on [{nstr(x1, 2)}, {float(left_reach):.4f}] margin {nstr(left_margin, 4) if left_margin else 'n/a'}; "
        f"D' > 0 on [{float(right_reach):.4f}, A] margin {nstr(right_margin, 4) if right_margin else 'n/a'} (grid starts at A)  {'ok' if ok else 'FAIL'}"
    )
    return (
        left_reach,
        right_reach,
        ok,
        {
            "sliver_bound": float(sl),
            "left_reach": float(left_reach),
            "left_margin": float(left_margin) if left_margin else None,
            "right_reach": float(right_reach),
            "right_margin": float(right_margin) if right_margin else None,
        },
    )


if __name__ == "__main__":
    # control at A = 100 for both the float input (v3 status) and the Krawczyk box (v4 status)
    from mpmath import mp

    d = json.load(open(os.path.join(HERE, "results_q32_A100_mp2.json")))["11"]
    ch = Channel(100.0, d["x"], d["w"], 40)
    t0 = time.time()
    piece_i(ch, mpf(100))
    print(f"   [{time.time()-t0:.0f}s]")
    e = json.load(open(os.path.join(HERE, "results_q32_A100_exact40.json")))
    mp.dps = 40
    iv.dps = 40
    rho = mpf("1e-30")
    xt = [mpf(v) for v in e["x"]]
    wt = [mpf(v) for v in e["w"]]
    K = len(xt)
    Xb = [iv.mpf(0)] + [mpi(xt[i] - rho, xt[i] + rho) for i in range(1, K - 1)] + [iv.mpf(xt[-1])]
    Wb = [mpi(w - rho, w + rho) for w in wt]
    chb = Channel(100.0, Xb, Wb, 40)
    t0 = time.time()
    piece_i(chb, xt[-1])
    print(f"   [{time.time()-t0:.0f}s]")
