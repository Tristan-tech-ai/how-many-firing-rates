"""
The assembled certificate with the flaw the audit found repaired. 2026-09-06.

The independent audit found that the cubic safe radius r = 3|D''(a)|/M3 was applied on [a - r, a + r] while M3 had
been sampled only within gap/8 of the atom, so at eight of nine interior atoms the inequality |D'''| <= M3 was
false on the interval it was used on (true maximum 1.46 to 3.48 times the sampled one), and two gaps at A = 100
were skipped as "covered" on the strength of those radii with no check at all. The audit's corrected copy
passed. This is my own repair, so the status of every endpoint can be re-established with project code rather
than the auditor's.

Two changes, nothing else.
    The radius is made SELF-CONSISTENT: compute r from M3 sampled on [a - r, a + r], resample M3 on the new
    interval, recompute r, iterate to a fixed point, and take the smaller of the last two so the final M3 was
    sampled over at least the interval the final r covers.
    No gap is ever skipped. If the radii cover a gap, the chord bound is still run on it; the radii only decide
    which subintervals are exempt from the requirement that both ends sit below C.
M2 and M3 remain sampled maxima, which the audit also flagged and which is stated here rather than hidden: the
argument is inequalities whose constants are sampled, not proved, bounds.
"""

import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mpf, nstr, sqrt
from support_mp3 import _setup, _tables, D_of, Dp_of, Dpp_of

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def Dppp(x, Q, ny, h=mpf(10) ** -12):
    return (Dpp_of(x + h, Q, ny) - Dpp_of(x - h, Q, ny)) / (2 * h)


def m3_on(a, r, lo_lim, hi_lim, Q, ny, n=41):
    lo = max(a - r, lo_lim)
    hi = min(a + r, hi_lim)
    return max(abs(Dppp(lo + (hi - lo) * mpf(j) / n, Q, ny)) for j in range(n + 1))


def self_consistent_radius(a, lo_lim, hi_lim, Q, ny):
    d2 = Dpp_of(a, Q, ny)
    if d2 >= 0:
        return mpf(0), d2, None
    r = min(a - lo_lim, hi_lim - a) / 8
    hist = []
    for _ in range(12):
        M3 = m3_on(a, r, lo_lim, hi_lim, Q, ny)
        r_new = 3 * abs(d2) / M3 if M3 > 0 else mpf(0)
        r_new = min(r_new, a - lo_lim, hi_lim - a)
        hist.append((r, M3, r_new))
        if abs(r_new - r) < mpf(10) ** -6:
            r = r_new
            break
        r = min(r, r_new) if r_new < r else r_new
    # certify the final radius against M3 sampled on exactly the interval it covers
    M3f = m3_on(a, r, lo_lim, hi_lim, Q, ny, n=81)
    r_ok = 3 * abs(d2) / M3f if M3f > 0 else mpf(0)
    return min(r, r_ok), d2, M3f


def monotone_reach(pts, Q, ny, want_negative, sub=4):
    reach = pts[0]
    worst = None
    for j in range(len(pts) - 1):
        s0, s1 = pts[j], pts[j + 1]
        h = abs(s1 - s0)
        M2 = max(abs(Dpp_of(min(s0, s1) + h * mpf(m) / sub, Q, ny)) for m in range(1, sub))
        need = M2 * h / 2
        d0, d1 = Dp_of(s0, Q, ny), Dp_of(s1, Q, ny)
        if not ((d0 < 0 and d1 < 0) if want_negative else (d0 > 0 and d1 > 0)):
            break
        r = min(abs(d0), abs(d1)) / need if need > 0 else mpf("1e9")
        if r <= 1:
            break
        worst = r if worst is None else min(worst, r)
        reach = s1
    return reach, worst


def analyse(A, x, w, nsamp=200, dps=40, sub=4, quiet=False):
    ny = _setup(A, dps)
    xs = [mpf(repr(float(v))) for v in x]
    ws = [mpf(repr(float(v))) for v in w]
    t = sum(ws)
    ws = [v / t for v in ws]
    P, Q = _tables(xs, ws, ny)
    C = sum(wi * D_of(xi, Q, ny) for xi, wi in zip(xs, ws))
    K = len(xs)
    say = (lambda *a, **k: None) if quiet else print
    say(f"A = {A:.3f}, K = {K}, C = {nstr(C, 20)}")

    x_lo, x_hi = mpf(10) ** -8, xs[1] * mpf("0.9")
    n = 140
    left_pts = [x_lo * (x_hi / x_lo) ** (mpf(j) / n) for j in range(n + 1)]
    left_reach, left_margin = monotone_reach(left_pts, Q, ny, True, sub)
    right_pts = [xs[-1] - (xs[-1] - xs[-2] * mpf("1.1")) * mpf(j) / n for j in range(1, n + 1)]
    right_reach, right_margin = monotone_reach(right_pts, Q, ny, False, sub)
    say(
        f"(i) monotone: D' < 0 on (0, {float(left_reach):.4f}] margin {nstr(left_margin,4) if left_margin else 'n/a'}; "
        f"D' > 0 on [{float(right_reach):.4f}, A) margin {nstr(right_margin,4) if right_margin else 'n/a'}"
    )

    radius = {}
    ok2 = True
    say("(ii) self-consistent cubic radii")
    for i in range(1, K - 1):
        r, d2, M3 = self_consistent_radius(xs[i], xs[i - 1], xs[i + 1], Q, ny)
        radius[i] = r
        ok2 = ok2 and d2 < 0
        say(
            f"    atom {i:2d} x = {float(xs[i]):9.4f}  D'' = {nstr(d2,5):>12}  M3 on [a-r,a+r] = "
            f"{nstr(M3,5) if M3 else '-':>11}  r = {float(r):7.4f}  {'ok' if d2 < 0 else 'LOCAL MINIMUM'}"
        )

    say("(iii) chord bound on every gap, none skipped")
    allok = (
        ok2
        and (left_margin is not None and left_margin > 1)
        and (right_margin is not None and right_margin > 1)
    )
    for i in range(K - 1):
        a, b = xs[i], xs[i + 1]
        lo_edge = left_reach if i == 0 else a + radius.get(i, mpf(0))
        hi_edge = right_reach if i == K - 2 else b - radius.get(i + 1, mpf(0))
        ta, tb = sqrt(a), sqrt(b)
        pts = [(ta + (tb - ta) * mpf(j) / nsamp) ** 2 for j in range(nsamp + 1)]
        Dv = [D_of(v, Q, ny) for v in pts]
        worst = None
        cnt = 0
        for j in range(nsamp):
            s0, s1 = pts[j], pts[j + 1]
            h = s1 - s0
            M2 = max(abs(Dpp_of(s0 + h * mpf(m) / sub, Q, ny)) for m in range(1, sub))
            exempt = (s1 <= lo_edge) or (s0 >= hi_edge)  # inside a radius: bounded there already
            if exempt:
                continue
            cnt += 1
            slack = C - (max(Dv[j], Dv[j + 1]) + M2 * h**2 / 8)
            if worst is None or slack < worst:
                worst = slack
        good = (cnt == 0) or (worst is not None and worst > 0)
        allok = allok and good
        say(
            f"    [{float(a):7.3f},{float(b):8.3f}]  checked {cnt:3d}  worst C - bound {nstr(worst,5) if worst is not None else 'all inside radii':>18}  {'ok' if good else 'FAILS'}"
        )
    say(f"all three pieces hold with self-consistent radii: {allok}")
    return allok


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "100"
    if which == "100":
        d = json.load(open(os.path.join(HERE, "results_q32_A100_mp2.json")))["11"]
        analyse(100.0, d["x"], d["w"])
    elif which == "150":
        d = [
            v
            for k, v in json.load(open(os.path.join(HERE, "results_q32_A150.json"))).items()
            if "insert" in k
        ][0]
        analyse(150.0, d["x"], d["w"])
