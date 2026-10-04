"""Q481: the TAIL LEMMA that turns q470's truncated-alphabet certificates (617, 624, and the A = 2000 run) into
statements about the true Poisson channel, whose output alphabet is all of N_0.

q470 sums every output series over y = 0 .. Ymax, Ymax = floor(A + 16 sqrt(A) + 60). The true functions carry a
tail over y > Ymax. This script bounds every tail rigorously and re-runs the Krawczyk test with the bounds added.

THE BOUNDS. For y > Ymax >= A and 0 <= x <= A the Poisson pmf P(y|x) = x^y e^-x / y! increases in x (its
x-derivative is P (y - x)/x > 0), so P(y|x) <= P(y|A); the output law Q = sum_k w_k P(.|x_k) satisfies
w_A P(y|A) <= Q(y) <= P(y|A). Hence, with q = log P(y|x) - log Q(y),
     q <= log P(y|A) - log(w_A P(y|A)) = -log w_A,
     q >= log P(y|x) - log P(y|A) = y log(x/A) + (A - x) >= -y log(A/x_lo),
so |q| <= |log w_min| + y log(A/x_lo) on x >= x_lo. With u = (y - x)/x, |u| <= U := y/x_lo + 1 and
|u'|, |u''|/2, |u'''|/6 <= U (x_lo >= 1), every integrand of D, D', D'', D''', D'''' is bounded by
50 U^4 (|q| + 1) (worked out for D'''' in 624(2); the lower orders are smaller), and every Jacobian entry of the
stationary system by 2 x that divided by w_min (the weight derivatives carry P_l/Q <= 1/w_l). So one number
     delta(x_lo) = 100/w_min * sum_{y > Ymax} P(y|A) U^4 (|log w_min| + y log(A/x_lo) + 1)
bounds the tail of every row of F, every entry of J, and D, D', D'', D''' anywhere on [x_lo, A]. On the stretch
(h0, x_1) below the first interior atom the same holds with P(y|x_1) in place of P(y|A) and x_lo = h0.
THE SUM is evaluated in ball arithmetic up to Y2 = Ymax + 4000 and the rest bounded geometrically by the term
ratio at Y2, which is below A/(Y2 + 1) (1 + 1/(Y2 + x_lo))^4 (1 + 1/Y2) < 1.

PRE-REGISTERED: PASS for a state if (a) Krawczyk with F and J widened by delta still contains the sweep box,
(b) 2 delta is below the worst free-stretch margin of its q470 log, (c) delta is below the smallest |D''| upper
bound over the atom neighbourhoods and below the wall's D' lower bound (both read from the log), and (d) the
stretch below atom 1 has a tail bound below the same margin. FAIL on any of them names which.
Usage: RBOX=... NEWTON_TARGET=... py q481_tail_lemma.py STATE.json PREC LOG1 [LOG2 ...]"""

import os, sys, re, time
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STATE = sys.argv[1]
PREC = int(sys.argv[2])
LOGS = sys.argv[3:]
sys.argv = [sys.argv[0], str(PREC), "8", STATE]
import q470_support_certificate as Q
from flint import arb, arb_mat, ctx

ctx.prec = PREC
K, n, Aa, fl = Q.K, Q.n, Q.Aa, Q.fl
RBOX = Q.RBOX
t0 = time.time()


def tail_sum(Pbase, x_lo, wmin, ymax, extra=4000):
    """rigorous upper bound of sum_{y > ymax} P(y|Pbase) U^4 (|log wmin| + y log(A/x_lo) + 1), U = y/x_lo + 1"""
    L = (Aa / x_lo).log()
    c = abs(wmin.log()) + 1
    lp = lambda y: y * Pbase.log() - Pbase - (arb(y) + 1).lgamma()
    term = lambda y: (lp(y)).exp() * (arb(y) / x_lo + 1) ** 4 * (c + arb(y) * L)
    s = arb(0)
    Y2 = ymax + extra
    for y in range(ymax + 1, Y2 + 1):
        s += term(y)
    ratio = Pbase / (Y2 + 1) * (1 + 1 / (arb(Y2) + x_lo)) ** 4 * (1 + arb(1) / Y2)
    assert ratio.upper() < 1, "geometric remainder does not converge"
    s += term(Y2 + 1) / (1 - ratio)
    return s.upper()


def parse_logs(paths):
    worst = None
    d2 = None
    d1 = None
    atoms = set()
    npieces = 0
    fails = 0
    for p in paths:
        for line in open(os.path.join(HERE, p), encoding="utf-8", errors="replace"):
            m = re.search(
                r"atom\s+(\d+) x =.*D'' on I_k upper\s+(\S+)\s+(PASS|FAIL).*worst margin ([0-9.e+-]+)", line
            )
            if m:
                atoms.add(int(m.group(1)))
                v = float(m.group(2))
                wm = float(m.group(4))
                d2 = v if d2 is None else max(d2, v)
                worst = wm if worst is None else min(worst, wm)
                fails += m.group(3) == "FAIL"
            m = re.search(r"wall A = .*lower (\S+)\s+(PASS|FAIL).*worst margin ([0-9.e+-]+)", line)
            if m:
                d1 = float(m.group(1))
                atoms.add(K - 1)
                worst = min(worst, float(m.group(3))) if worst is not None else float(m.group(3))
            m = re.search(
                r"pieces evaluated: (\d+); worst free-stretch margin \(inf C - sup D\): ([0-9.e+-]+); failures: (\d+)",
                line,
            )
            if m:
                npieces += int(m.group(1))
                fails += int(m.group(3))
                worst = min(worst, float(m.group(2))) if worst is not None else float(m.group(2))
    return worst, d2, d1, sorted(atoms), npieces, fails


if __name__ == "__main__":
    say(f"tail lemma for {STATE}: K = {K}, A = {fl(Aa)}, Ymax = {Q.YMAX}, prec {PREC}, sweep box {RBOX:.0e}")
    NT = float(os.environ.get("NEWTON_TARGET", "1e-28"))
    v = list(Q.xs0[1 : K - 1]) + list(Q.ws0)
    for it in range(30):
        S = Q.State(*Q.unpack(v))
        F, J = S.FJ()
        Fm = np.array([fl(f) for f in F])
        Jm = np.array([[fl(J[i][j]) for j in range(n)] for i in range(n)])
        if np.abs(Fm).max() < NT:
            break
        dv = np.linalg.solve(Jm, Fm)
        v = [v[i] - arb(float(dv[i])) for i in range(n)]
    say(f"   point: max |F| = {np.abs(Fm).max():.3e}")
    Dr, Dc, _ = Q.ruiz(Jm)
    Js = (Dr[:, None] * Jm) * Dc[None, :]
    Y = np.linalg.inv(Js)
    Yarb = arb_mat([[arb(float(Y[i, j])) for j in range(n)] for i in range(n)])
    DrM, DcM = Q.diag(Dr), Q.diag(Dc)
    I = Q.diag(np.ones(n))
    Xv = [arb(v[i].mid(), float(RBOX * Dc[i])) for i in range(n)]
    SX = Q.State(*Q.unpack(Xv))
    xb, wb = SX.x, SX.w
    wmin = min(w.lower() for w in wb)
    x1lo = xb[1].lower()
    h0 = min(arb("1e-3"), xb[1].mid() / 5)
    dlt = tail_sum(Aa, x1lo, wmin, Q.YMAX) * 100 / wmin
    dlt_low = tail_sum(xb[1].upper(), h0, wmin, Q.YMAX) * 100 / wmin
    say(f"   w_min {wmin.str(6, radius=False)}, x_1 >= {x1lo.str(6, radius=False)}")
    say(
        f"   delta on [x_1, A]  = {dlt.str(4, radius=False)}      (Poisson tail of A beyond Ymax, all orders, all rows and entries)"
    )
    say(f"   delta on (h0, x_1) = {dlt_low.str(4, radius=False)}")
    # (a) Krawczyk with F and J widened by delta
    Fp = [f + arb(0, dlt) for f in F]
    Fs = arb_mat([[arb(float(Dr[i])) * Fp[i]] for i in range(n)])
    YF = Yarb * Fs
    yf = [YF[i, 0].abs_upper() for i in range(n)]
    _, JX = SX.FJ()
    JXp = [[JX[i][j] + arb(0, dlt) for j in range(n)] for i in range(n)]
    C = I - Yarb * DrM * arb_mat(JXp) * DcM
    rows = [sum(C[i, j].abs_upper() for j in range(n)) for i in range(n)]
    ra = arb(RBOX)
    oka = all((yf[i] + rows[i] * ra) < ra for i in range(n)) and max(rows) < arb(1)
    say(
        f"(a) Krawczyk with the tail: ||Y F~||_inf {max(yf).str(4, radius=False)}, ||I - Y J~(X)||_inf {max(rows).str(6, radius=False)}, "
        f"max_i (|YF|_i + r row_i)/r = {max((yf[i] + rows[i]*ra)/ra for i in range(n)).str(6, radius=False)} -> {'CONTAINED' if oka else 'NOT CONTAINED'}"
    )
    # (b)-(d) against the sweep logs
    worst, d2, d1, atoms, npieces, fails = parse_logs(LOGS)
    missing = sorted(set(range(1, K)) - set(atoms))
    say(
        f"   logs: {len(LOGS)} file(s), atoms covered {len(atoms)}/{K - 1}{' MISSING ' + str(missing) if missing else ''}, "
        f"failures {fails}, pieces {npieces}"
    )
    say(
        f"   worst free-stretch margin {worst:.3e}; largest D'' upper bound over neighbourhoods {d2:.3e}; wall D' lower {d1}"
    )
    okb = worst is not None and 2 * fl(dlt) < worst and 2 * fl(dlt_low) < worst
    okc = d2 is not None and fl(dlt) < -d2 and (d1 is None or fl(dlt) < d1)
    say(
        f"(b) 2 delta = {2*fl(dlt):.3e} (and {2*fl(dlt_low):.3e} below x_1) against the worst margin {worst:.3e} -> {'PASS' if okb else 'FAIL'}"
    )
    say(f"(c) delta against |D''| >= {-d2:.3e} and D'(wall) >= {d1} -> {'PASS' if okc else 'FAIL'}")
    ok = oka and okb and okc and not missing and fails == 0
    say(
        f"VERDICT: {'PASS' if ok else 'FAIL'}"
        + (
            f" - the untruncated Poisson channel at A = {fl(Aa)} has a unique stationary {K}-atom state in the "
            f"sweep box, a probability distribution satisfying the KKT inequality on [0, A]: |supp P*| = {K}."
            if ok
            else ""
        )
    )
    say(f"[{time.time() - t0:.0f}s]")
