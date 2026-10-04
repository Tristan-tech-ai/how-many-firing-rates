"""
The certificate with PROVED constants. 2026-09-06.

Amendment 68 stated the one remaining limit of the repaired certificate (q32_certificate_v2.py): M2 and M3, the
bounds on |D''| and |D'''| that the three pieces use, were sampled maxima. This version replaces every sampled
quantity by an interval enclosure (support_iv.py): the constants come from the centered mean-value form
applied to depth 3 on each cell, the point values D(s), D'(s), D''(a) are degenerate intervals, C = I(F) is an
interval, and the truncation of the y-sum is bounded by tail_bound(). The argument itself is unchanged:
    (i)   D' < 0 on (0, left_reach] and D' > 0 on [right_reach, A), from the sign of D' at grid points with
          |D'(s)| > M2 h/2 on each cell; below x = 1e-8 the same sign test is run decade by decade to 1e-300;
    (ii)  at each interior atom a: D(x) <= D(a) + |D'(a)| r for |x - a| <= r, valid whenever r <= 3|D''(a)|/M3
          with M3 a proved bound on |D'''| over [a - r, a + r]; the largest certified r is taken from a
          halving sequence and refined upward by bisection (no fixed-point iteration, which oscillated);
    (iii) on every gap, on each cell [s0, s1] not inside a radius: D(x) <= max(D(s0), D(s1)) + M2 h^2/8;
          a cell whose slack is negative is split in two, to four levels, before it is called a failure.
Resolution is adaptive but the bounds never depend on samples: the pieces over a radius interval are refined
until the enclosure is within 1.5x of the sampled maximum, so the sampled values steer the cost and the
enclosure carries the proof.

What is proved, for the decimal input F. Let C = I(F). Then
    max_{x in [0, A]} D(x) <= C + eta,
where eta collects the KKT residual of the decimal input: eta = max over atoms of D(a) - C + |D'(a)| r,
and D(0) - C, D(A) - C at the boundary atoms, each taken at its interval upper end. By the standard bound
C(A) <= max_x D(x; F), F is within eta of capacity. eta is REPORTED, not assumed zero (5e-16 for inputs
stored as 17-digit floats); the step from an eta-optimal K-atom input to "the optimum has exactly K atoms"
is the audit's caveat 1 and is not made here.

History. The first proved version (depth 1, 32 fixed pieces) failed at A = 100 by 1e-8 to 1e-9 on five gaps:
M3 at the flat atoms was over-estimated 20 to 500 times, the radii collapsed (0.0055 at x = 27 against 2.30
sampled), and the cells next to those atoms lost. An instrument failure, diagnosed by the v2 values and by
control 3 of support_iv.py, and repaired by depth and resolution, not by any change to the argument.

Control: A = 100, K = 11 (v2 worst slacks 1.24e-6 and 1.11e-6) must pass; the proved constants are larger
than the sampled ones, so the slacks must shrink and must stay positive. That is the computation that fails
if the enclosures are still too loose.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mpf, mpi, iv, nstr, sqrt
from support_iv import Channel

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEPTH = 3


def lo(I):
    return mpf(I.a)


def hi(I):
    return mpf(I.b)


def absmax_iv(I):
    return max(abs(lo(I)), abs(hi(I)))


def monotone_reach(ch, pts, want_negative, T1, T2):
    reach = pts[0]
    worst = None
    dprev = None
    for j in range(len(pts) - 1):
        s0, s1 = pts[j], pts[j + 1]
        c0, c1 = (s0, s1) if s0 < s1 else (s1, s0)
        h = c1 - c0
        d0 = dprev if dprev is not None else ch.Dp_iv(iv.mpf(s0))
        d1 = ch.Dp_iv(iv.mpf(s1))
        dprev = d1
        if want_negative:
            u0, u1 = hi(d0) + T1, hi(d1) + T1
            ok = u0 < 0 and u1 < 0
            dmin = min(-u0, -u1)
        else:
            l0, l1 = lo(d0) - T1, lo(d1) - T1
            ok = l0 > 0 and l1 > 0
            dmin = min(l0, l1)
        if not ok:
            break
        M2 = absmax_iv(ch.enclose(mpi(c0, c1), 2, DEPTH)) + T2
        need = M2 * h / 2
        r = dmin / need if need > 0 else mpf("1e9")
        if r <= 1:
            break
        worst = r if worst is None else min(worst, r)
        reach = s1
    return reach, worst


def decades_negative(ch, top, bottom_exp=-300):
    """D' < 0 on [10^bottom_exp, top], by naive interval evaluation on each decade"""
    worst = None
    up = mpf(top)
    while up > mpf(10) ** bottom_exp:
        dn = max(up / 10, mpf(10) ** bottom_exp)
        ub = hi(ch.Dp_iv(mpi(dn, up)))
        if ub >= 0:
            return False, up, worst
        worst = ub if worst is None else max(worst, ub)
        up = dn
    return True, mpf(10) ** bottom_exp, worst


def m3_proved(ch, a, r, lo_lim, hi_lim, T3, d, pieces0=32, max_pieces=2048):
    # proved bound on the third derivative over [a - r, a + r] (clipped). Returns (Mi, Ms, pieces, certified,
    # genuine): certified = r <= 3d/Mi; genuine = the enclosure is within 1.5x of the sampled max, so a failure
    # is not the instrument. The over-estimate at depth 3 scales as p^4 in the piece width p, so the pieces
    # needed are predicted from the first evaluation in one jump instead of a doubling ladder.
    L, R = max(a - r, lo_lim), min(a + r, hi_lim)
    pieces = pieces0
    while True:
        Mi = ch.absmax(L, R, 3, pieces, DEPTH) + T3
        Ms = max(abs(ch.Dppp(L + (R - L) * mpf(j) / pieces)) for j in range(pieces + 1))
        cert = r <= 3 * d / Mi
        genuine = Mi <= mpf("1.5") * Ms
        if cert or genuine or pieces >= max_pieces:
            return Mi, Ms, pieces, cert, genuine
        over = Mi - Ms
        need = mpf("0.5") * Ms
        f = (over / need) ** mpf("0.25") if need > 0 else mpf(4)
        newp = pieces * 2
        while newp < pieces * f:
            newp *= 2
        pieces = min(max(newp, pieces * 2), max_pieces)


def radius(ch, a, lo_lim, hi_lim, T2, T3):
    d2 = ch.Dpp_iv(iv.mpf(a))
    d2u = hi(d2) + T2
    if d2u >= 0:
        return mpf(0), d2, None, 0
    d = -d2u
    rmax = min(a - lo_lim, hi_lim - a)
    best = None
    Mbest = None
    pbest = 0
    r = rmax / 4  # as v2 started; a candidate reaching the boundary atom is hopeless near x = 0
    for _ in range(16):
        Mi, Ms, pieces, cert, genuine = m3_proved(ch, a, r, lo_lim, hi_lim, T3, d)
        if cert:
            best, Mbest, pbest = r, Mi, pieces
            break
        r = r / 2
    if best is None:
        return mpf(0), d2, None, 0
    lo_r, hi_r = best, min(2 * best, rmax)
    if hi_r > lo_r:
        for _ in range(3):
            mid = (lo_r + hi_r) / 2
            Mi, Ms, pieces, cert, genuine = m3_proved(ch, a, mid, lo_lim, hi_lim, T3, d)
            if cert:
                lo_r, best, Mbest, pbest = mid, mid, Mi, pieces
            else:
                hi_r = mid
    return best, d2, Mbest, pbest


def chord_cell(ch, s0, s1, Dv, Ca, T0, T2, level=0, max_level=4):
    """worst slack C - bound over [s0, s1], splitting the cell when the slack is negative"""
    for s in (s0, s1):
        if s not in Dv:
            Dv[s] = ch.D_iv(iv.mpf(s))
    h = s1 - s0
    M2 = absmax_iv(ch.enclose(mpi(s0, s1), 2, DEPTH)) + T2
    slack = Ca - (max(hi(Dv[s0]), hi(Dv[s1])) + T0 + M2 * h**2 / 8)
    if slack > 0 or level >= max_level:
        return slack, 1
    m = (s0 + s1) / 2
    a1, n1 = chord_cell(ch, s0, m, Dv, Ca, T0, T2, level + 1, max_level)
    a2, n2 = chord_cell(ch, m, s1, Dv, Ca, T0, T2, level + 1, max_level)
    return min(a1, a2), n1 + n2


def analyse(A, x, w, nsamp=200, dps=40, quiet=False):
    t0 = time.time()
    ch = Channel(A, x, w, dps)
    say = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, **k, flush=True))
    xs = ch.xp
    K = ch.K
    Ci = ch.C_iv()
    Ca, Cb = lo(Ci), hi(Ci)
    say(f"A = {A:.4f}, K = {K}, C in [{nstr(Ca, 25)}, {nstr(Cb, 25)}]  (ny = {ch.ny}, depth {DEPTH})")
    T = {o: ch.tail_bound(mpf(10) ** -8, o) for o in range(4)}
    say(f"tail bounds (x_lo = 1e-8): " + " ".join(f"T{o} = {nstr(T[o], 2)}" for o in range(4)))
    eta = [mpf(0)]

    # (i) monotone runs, with the left sliver lemma and the right grid starting at A (q32_piece_i.py)
    from q32_piece_i import piece_i

    left_reach, right_reach, ok1, info1 = piece_i(ch, xs[-1], T1=T[1], T2=T[2], say=say)
    D0 = ch.D_iv(iv.mpf(0))
    DA = ch.D_iv(iv.mpf(xs[-1]))
    eta.append(hi(D0) - Ca)
    eta.append(hi(DA) + T[0] - Ca)
    say(f"    D(0) - C <= {nstr(eta[1], 3)}, D(A) - C <= {nstr(eta[2], 3)}   [{time.time()-t0:.0f}s]")

    # (ii) radii with proved M3
    rad = {}
    ok2 = True
    say("(ii) cubic radii, M3 proved by depth-3 enclosure with adaptive pieces; largest certified radius")
    for i in range(1, K - 1):
        r, d2, M3, pieces = radius(ch, xs[i], xs[i - 1], xs[i + 1], T[2], T[3])
        rad[i] = r
        Da = ch.D_iv(iv.mpf(xs[i]))
        Dpa = ch.Dp_iv(iv.mpf(xs[i]))
        e = hi(Da) + T[0] - Ca + (absmax_iv(Dpa) + T[1]) * r
        eta.append(e)
        good = hi(d2) + T[2] < 0 and r > 0
        ok2 = ok2 and good
        say(
            f"    atom {i:2d} x = {float(xs[i]):9.4f}  D'' <= {nstr(hi(d2), 5):>12}  M3 = {nstr(M3, 5) if M3 else '-':>11} ({pieces:4d} pieces)  r = {float(r):7.4f}  "
            f"eta_atom {nstr(e, 3):>9}  {'ok' if good else 'NO RADIUS'}   [{time.time()-t0:.0f}s]"
        )

    # (iii) chord bound on every gap
    say(
        "(iii) chord bound on every gap, M2 proved by depth-3 enclosure, cells split where the slack is negative"
    )
    allok = ok2 and ok1
    worst_all = None
    for i in range(K - 1):
        a, b = xs[i], xs[i + 1]
        lo_edge = left_reach if i == 0 else a + rad.get(i, mpf(0))
        hi_edge = right_reach if i == K - 2 else b - rad.get(i + 1, mpf(0))
        ta, tb = sqrt(a), sqrt(b)
        pts = [(ta + (tb - ta) * mpf(j) / nsamp) ** 2 for j in range(nsamp + 1)]
        Dv = {}
        worst = None
        cnt = 0
        splits = 0
        for j in range(nsamp):
            s0, s1 = pts[j], pts[j + 1]
            if (s1 <= lo_edge) or (s0 >= hi_edge):
                continue
            if s0 <= 0:
                allok = False
                say("    first cell touches 0 and is not covered by the monotone run: FAIL")
                break
            cnt += 1
            slack, ncell = chord_cell(ch, s0, s1, Dv, Ca, T[0], T[2])
            splits += ncell - 1
            if worst is None or slack < worst:
                worst = slack
        good = (cnt == 0) or (worst is not None and worst > 0)
        allok = allok and good
        if worst is not None:
            worst_all = worst if worst_all is None else min(worst_all, worst)
        say(
            f"    [{float(a):8.4f},{float(b):9.4f}]  checked {cnt:3d} (+{splits:3d} split)  worst C - bound {nstr(worst, 5) if worst is not None else 'all inside radii':>18}  {'ok' if good else 'FAILS'}   [{time.time()-t0:.0f}s]"
        )
    eta_max = max(eta)
    say(
        f"all three pieces hold with proved constants: {allok};  max_x D(x) <= C + eta with eta = {nstr(eta_max, 3)};  worst chord slack {nstr(worst_all, 4) if worst_all is not None else '-'}   [{time.time()-t0:.0f}s]"
    )
    return {
        "ok": bool(allok),
        "eta": float(eta_max),
        "worst_slack": float(worst_all) if worst_all is not None else None,
        "C_lo": float(Ca),
        "C_hi": float(Cb),
        "K": K,
        "A": float(A),
        "radii": {i: float(r) for i, r in rad.items()},
        "seconds": time.time() - t0,
    }


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "100"
    if which == "100":
        d = json.load(open(os.path.join(HERE, "results_q32_A100_mp2.json")))["11"]
        res = analyse(100.0, d["x"], d["w"])
    elif which == "150":
        d = [
            v
            for k, v in json.load(open(os.path.join(HERE, "results_q32_A150.json"))).items()
            if "insert" in k
        ][0]
        res = analyse(150.0, d["x"], d["w"])
    json.dump(res, open(os.path.join(HERE, f"results_q32_certificate_v3_A{which}.json"), "w"), indent=1)
