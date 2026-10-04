"""
The whole certificate chain replicated in a SECOND INSTRUMENT: arb ball arithmetic (python-flint). 2026-09-06.

Everything Amendments 69 to 71a did with mpmath.iv is re-done here with arb (Johansson), an independent
implementation of rigorous arithmetic with its own rounding and error propagation, for the exact-point
statement of Amendment 70 at every endpoint:
    1. Krawczyk inclusion for the KKT system on the box of radius rho around the 40-digit point;
    2. piece (i): the left sliver lemma on (0, 1e-8], the left grid, the right grid starting at A;
    3. piece (ii): cubic radii with M3 enclosed by the depth-3 centered form and adaptive pieces;
    4. piece (iii): chord bounds on every non-exempt cell, slack = inf C - sup bound;
all with the parameters as the box, so every enclosure covers the exact KKT point F*. The closed forms and
coefficient tables are shared with support_iv.py (pure Python; they were checked symbolically there). Every
comparison below is an arb comparison, which is True only when the balls prove it.
Agreement with the mpmath run at every endpoint (verdicts, radii and slacks to the printed digits) reduces
"the correctness of the interval library" to "two independent libraries would have to be wrong in the same
way". arb is also about fifty times faster, so this runs all twelve cases in minutes.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from flint import arb, arb_mat, ctx
from mpmath import mp, mpf, sqrt as msqrt
from support_iv import H, CC, MAXORDER, _poly, _const

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEPTH = 3
mp.dps = 50


def pt(v):
    return v if isinstance(v, arb) else arb(str(v))


def box(a, b):
    return pt(a).union(pt(b))


def up(b):
    return b.upper()


def dn(b):
    return b.lower()


def amax(b):
    return abs(b).upper()


def fl(b):
    return float(b.str(20, radius=False))


class ArbChannel:
    def __init__(self, A_dec, xs, ws, rho=None):
        """A_dec decimal string; xs, ws lists of decimal strings (40 digits) or arb balls; rho a decimal string
        or None. With rho the interior atoms and all weights become balls x~ +- rho."""
        self.A = pt(A_dec)
        Af = float(A_dec)
        self.ny = int(Af + 18 * Af**0.5 + 50)
        self.LF = [arb(y + 1).lgamma() for y in range(self.ny)]
        K = len(xs)
        self.K = K
        r = pt(rho) if rho is not None else None
        self.x = []
        for i, v in enumerate(xs):
            b = v if isinstance(v, arb) else pt(v)
            if r is not None and 0 < i < K - 1:
                b = b + box("-" + str(rho), rho)
            self.x.append(b)
        w = []
        for v in ws:
            b = v if isinstance(v, arb) else pt(v)
            if r is not None:
                b = b + box("-" + str(rho), rho)
            w.append(b)
        t = arb(0)
        for v in w:
            t += v
        self.w = [v / t for v in w]
        self.xmid = [fl(b.mid()) for b in self.x]
        self.zero = [i for i in range(K) if not (self.x[i] > 0)]  # atoms that are (or touch) 0
        self.Q = []
        for y in range(self.ny):
            q = arb(0)
            for k in range(K):
                if k in self.zero:
                    if y == 0:
                        q += self.w[k]
                else:
                    q += self.w[k] * (y * self.x[k].log() - self.x[k] - self.LF[y]).exp()
            self.Q.append(q)
        self.logQ = [q.log() for q in self.Q]

    def _sum(self, X, order):
        if not (X > 0):
            raise ValueError("ball touches 0")
        xinv = 1 / X
        lX = X.log()
        s = arb(0)
        for y in range(self.ny):
            Lv = y * lX - X - self.LF[y]
            s += Lv.exp() * _poly(order, y * xinv, xinv) * (Lv - self.logQ[y])
        return s + _const(order, xinv)

    def D0(self):
        return -self.logQ[0]

    def enclose(self, X, order, depth=DEPTH):
        """X a ball; mean-value form to the given depth"""
        if depth <= 0 or order + 1 > MAXORDER or X.rad() == 0:
            return self._sum(X, order)
        m = X.mid()
        half = X.rad()
        return self._sum(m, order) + self.enclose(X, order + 1, depth - 1) * box(-half, half)

    def absmax(self, lo, hi, order, pieces, depth=DEPTH):
        best = arb(0)
        for j in range(pieces):
            a = lo + (hi - lo) * j / pieces
            b = lo + (hi - lo) * (j + 1) / pieces
            v = amax(self.enclose(box(a, b), order, depth))
            if v > best:
                best = v
        return best

    def C(self):
        c = arb(0)
        for k in range(self.K):
            c += self.w[k] * (self.D0() if k in self.zero else self._sum(self.x[k], 0))
        return c

    def tail(self, x_lo, order):
        A = self.A
        ny = self.ny
        wA = dn(self.w[-1])
        rho = A / ny
        rho2 = rho * (1 + arb(1) / ny) ** (order + 2)
        assert rho2 < 1
        PA = (ny * A.log() - A - arb(ny + 1).lgamma()).exp()
        g1 = A + (1 / wA).log()
        g2 = ny * (A / x_lo).log()
        gb = up(g1) if g1 > g2 else up(g2)
        hb = arb(0)
        for coef, i, j in H[order]:
            hb += abs(coef) * (ny / x_lo) ** i / x_lo**j
        return up(PA * hb * gb / (1 - rho2))

    def sliver(self, x1):
        """upper bound on D'(x) over (0, x1], the lemma of Amendment 71a"""
        X1 = pt(x1)
        Q0 = up(self.Q[0])
        Q1 = dn(self.Q[1])
        assert Q0 < 1 and X1 < -Q0.log() and X1 < Q1 and X1 < arb(-1).exp()
        t1 = (1 - X1) * (-X1).exp() * (X1.log() - Q1.log())
        s = arb(0)
        absl = -X1.log()
        for y in range(2, self.ny):
            lq = abs(self.logQ[y]).upper()
            s += ((y - 1) * X1.log() - arb(y).lgamma()).exp() * (y * absl + X1 + arb(y + 1).lgamma() + lq)
        y = self.ny
        Tny = ((y - 1) * X1.log() - arb(y).lgamma()).exp() * (
            y * absl + X1 + 2 * arb(y + 1).lgamma() + self.A + (1 / dn(self.w[-1])).log()
        )
        ratio = 2 * X1 / y
        assert ratio < 1
        return up(t1 + s + Tny / (1 - ratio))


def monotone(ch, pts, negative, T1, T2):
    reach = pts[0]
    worst = None
    dprev = None
    for j in range(len(pts) - 1):
        s0, s1 = pts[j], pts[j + 1]
        c0, c1 = (s0, s1) if float(s0) < float(s1) else (s1, s0)
        h = pt(c1) - pt(c0)
        d0 = dprev if dprev is not None else ch._sum(pt(s0), 1)
        d1 = ch._sum(pt(s1), 1)
        dprev = d1
        if negative:
            u0, u1 = up(d0) + T1, up(d1) + T1
            if not (u0 < 0 and u1 < 0):
                break
            dmin = -u0 if -u0 < -u1 else -u1
        else:
            l0, l1 = dn(d0) - T1, dn(d1) - T1
            if not (l0 > 0 and l1 > 0):
                break
            dmin = l0 if l0 < l1 else l1
        M2 = amax(ch.enclose(box(c0, c1), 2)) + T2
        need = M2 * h / 2
        r = dmin / need
        if not (r > 1):
            break
        worst = r if worst is None or r < worst else worst
        reach = s1
    return reach, worst


def piece_i(ch, A_dec, x1="1e-8", n=140):
    T1 = ch.tail(pt(x1), 1)
    T2 = ch.tail(pt(x1), 2)
    sl = ch.sliver(x1)
    x1m = mpf(x1)
    x_hi = mpf(ch.xmid[1]) * mpf("0.9")
    left = [str(x1m * (x_hi / x1m) ** (mpf(j) / n)) for j in range(n + 1)]
    lr, lm = monotone(ch, left, True, T1, T2)
    Am = mpf(A_dec)
    xk2 = mpf(ch.xmid[-2]) * mpf("1.1")
    right = [A_dec] + [str(Am - (Am - xk2) * mpf(j) / n) for j in range(1, n + 1)]
    rr, rm = monotone(ch, right, False, T1, T2)
    ok = (sl < 0) and lm is not None and rm is not None and (lm > 1) and (rm > 1) and float(rr) < float(A_dec)
    return (
        ok,
        {
            "sliver": fl(sl),
            "left_reach": float(lr),
            "left_margin": fl(lm) if lm is not None else None,
            "right_reach": float(rr),
            "right_margin": fl(rm) if rm is not None else None,
        },
        lr,
        rr,
    )


def m3_box(ch, L, R, r, d, T3, pieces=32, max_pieces=2048):
    while True:
        Mi = ch.absmax(L, R, 3, pieces) + T3
        Ms = arb(0)
        for j in range(pieces + 1):
            v = amax(ch._sum(pt(L + (R - L) * j / pieces), 3))
            if v > Ms:
                Ms = v
        cert = pt(r) <= 3 * d / Mi
        genuine = Mi <= arb("1.5") * Ms
        if cert or genuine or pieces >= max_pieces:
            return Mi, pieces, cert
        f = fl(((Mi - Ms) / (arb("0.5") * Ms)) ** arb("0.25")) if Ms > 0 else 4.0
        newp = pieces * 2
        while newp < pieces * f:
            newp *= 2
        pieces = min(max(newp, pieces * 2), max_pieces)


def radius_box(ch, a_lo, a_hi, lo_lim, hi_lim, T2, T3):
    d2 = ch._sum(box(a_lo, a_hi), 2)
    d2u = up(d2) + T2
    if not (d2u < 0):
        return 0.0, d2, None, 0
    d = -d2u
    rmax = min(a_lo - lo_lim, hi_lim - a_hi)
    best = None
    Mb = None
    pb = 0
    r = rmax / 4
    for _ in range(16):
        Mi, pieces, cert = m3_box(ch, a_lo - r, a_hi + r, r, d, T3)
        if cert:
            best, Mb, pb = r, Mi, pieces
            break
        r /= 2
    if best is None:
        return 0.0, d2, None, 0
    lo_r, hi_r = best, min(2 * best, rmax)
    for _ in range(3):
        mid = (lo_r + hi_r) / 2
        Mi, pieces, cert = m3_box(ch, a_lo - mid, a_hi + mid, mid, d, T3)
        if cert:
            lo_r, best, Mb, pb = mid, mid, Mi, pieces
        else:
            hi_r = mid
    return best, d2, Mb, pb


def chord_cell(ch, s0, s1, Dv, Ca, T0, T2, level=0):
    for s in (s0, s1):
        if s not in Dv:
            Dv[s] = up(ch._sum(pt(s), 0))
    h = pt(s1) - pt(s0)
    M2 = amax(ch.enclose(box(s0, s1), 2)) + T2
    top = Dv[s0] if Dv[s0] > Dv[s1] else Dv[s1]
    slack = Ca - (top + T0 + M2 * h * h / 8)
    if slack > 0 or level >= 4:
        return slack, 1
    m = str((mpf(s0) + mpf(s1)) / 2)
    a1, n1 = chord_cell(ch, s0, m, Dv, Ca, T0, T2, level + 1)
    a2, n2 = chord_cell(ch, m, s1, Dv, Ca, T0, T2, level + 1)
    return (a1 if a1 < a2 else a2), n1 + n2


class KKT:
    def __init__(self, ch, interior):
        self.ch = ch
        self.interior = interior
        self.n = ch.K + len(interior)
        ny = ch.ny
        K = ch.K
        self.P = [
            [
                (
                    (arb(1) if y == 0 else arb(0))
                    if k in ch.zero
                    else (y * ch.x[k].log() - ch.x[k] - ch.LF[y]).exp()
                )
                for y in range(ny)
            ]
            for k in range(K)
        ]
        self.R = [[self.P[k][y] / ch.Q[y] for y in range(ny)] for k in range(K)]
        self.U = [[None if k in ch.zero else (y / ch.x[k] - 1) for y in range(ny)] for k in range(K)]
        self.L = [
            [None if k in ch.zero else (y * ch.x[k].log() - ch.x[k] - ch.LF[y]) for y in range(ny)]
            for k in range(K)
        ]

    def D(self, k):
        if k in self.ch.zero:
            return self.ch.D0()
        return sum((self.P[k][y] * (self.L[k][y] - self.ch.logQ[y]) for y in range(self.ch.ny)), arb(0))

    def Dp(self, k):
        return sum(
            (self.P[k][y] * self.U[k][y] * (self.L[k][y] - self.ch.logQ[y]) for y in range(self.ch.ny)),
            arb(0),
        )

    def Dpp(self, k):
        x = self.ch.x[k]
        return (
            sum(
                (
                    self.P[k][y]
                    * ((self.U[k][y]) ** 2 - (self.U[k][y] + 1) / x)
                    * (self.L[k][y] - self.ch.logQ[y])
                    for y in range(self.ch.ny)
                ),
                arb(0),
            )
            + 1 / x
        )

    def G(self):
        K = self.ch.K
        Dv = [self.D(k) for k in range(K)]
        g = [Dv[i] - Dv[0] for i in range(1, K)]
        g.append(sum(self.ch.w, arb(0)) - 1)
        g += [self.Dp(i) for i in self.interior]
        return g

    def J(self):
        K = self.ch.K
        n = self.n
        inter = self.interior
        ny = self.ch.ny
        w = self.ch.w
        Dp_at = {i: self.Dp(i) for i in inter}
        Dpp_at = {i: self.Dpp(i) for i in inter}
        dD_dw = lambda i, k: -sum((self.P[i][y] * self.R[k][y] for y in range(ny)), arb(0))
        dD_dx = lambda i, m: -sum(
            (self.P[i][y] * w[m] * self.R[m][y] * self.U[m][y] for y in range(ny)), arb(0)
        )
        dDp_dw = lambda i, k: -sum((self.P[i][y] * self.U[i][y] * self.R[k][y] for y in range(ny)), arb(0))
        dDp_dx = lambda i, m: -sum(
            (self.P[i][y] * self.U[i][y] * w[m] * self.R[m][y] * self.U[m][y] for y in range(ny)), arb(0)
        )
        Jm = [[arb(0)] * n for _ in range(n)]
        for row, i in enumerate(range(1, K)):
            for k in range(K):
                Jm[row][k] = dD_dw(i, k) - dD_dw(0, k)
            for j, m in enumerate(inter):
                v = dD_dx(i, m) - dD_dx(0, m)
                if m == i:
                    v = v + Dp_at[i]
                Jm[row][K + j] = v
        for k in range(K):
            Jm[K - 1][k] = arb(1)
        for a, i in enumerate(inter):
            row = K + a
            for k in range(K):
                Jm[row][k] = dDp_dw(i, k)
            for j, m in enumerate(inter):
                v = dDp_dx(i, m)
                if m == i:
                    v = v + Dpp_at[i]
                Jm[row][K + j] = v
        return Jm


def krawczyk(A_dec, x40, w40, rho):
    K = len(x40)
    interior = list(range(1, K - 1))
    ch0 = ArbChannel(A_dec, x40, w40, None)
    S0 = KKT(ch0, interior)
    G0 = S0.G()
    J0 = S0.J()
    n = S0.n
    Y = arb_mat([[J0[i][j].mid() for j in range(n)] for i in range(n)]).inv()
    chb = ArbChannel(A_dec, x40, w40, rho)
    JX = KKT(chb, interior).J()
    YG = [sum((Y[i, l] * G0[l] for l in range(n)), arb(0)) for i in range(n)]
    worst = arb(0)
    for i in range(n):
        contraction = arb(0)
        for j in range(n):
            s = arb(1 if i == j else 0)
            for l in range(n):
                s = s - Y[i, l] * JX[l][j]
            contraction += amax(s)
        ratio = (amax(YG[i]) + contraction * pt(rho)) / pt(rho)
        if up(ratio) > worst:
            worst = up(ratio)
    return (worst < 1), fl(worst)


def certify(label, A_dec, x40, w40, rho, say=print):
    t0 = time.time()
    kok, kratio = krawczyk(A_dec, x40, w40, rho)
    say(
        f"{label}: Krawczyk at rho = {rho}: ratio {kratio:.3e} {'inside' if kok else 'NOT inside'}   [{time.time()-t0:.0f}s]"
    )
    ch = ArbChannel(A_dec, x40, w40, rho)
    K = ch.K
    Cb = ch.C()
    Ca = dn(Cb)
    T = {o: ch.tail(pt("1e-8"), o) for o in range(4)}
    ok1, info1, lr, rr = piece_i(ch, A_dec)
    say(
        f"   (i) sliver {info1['sliver']:.3f}, left margin {info1['left_margin']}, right margin {info1['right_margin']} {'ok' if ok1 else 'FAIL'}   [{time.time()-t0:.0f}s]"
    )
    rad = {}
    ok2 = True
    rhof = float(rho)
    xt = [float(mpf(v)) for v in x40]
    for i in range(1, K - 1):
        a_lo, a_hi = mpf(x40[i]) - mpf(rho), mpf(x40[i]) + mpf(rho)
        lo_lim = (mpf(x40[i - 1]) + mpf(rho)) if i - 1 > 0 else mpf(0)
        hi_lim = (mpf(x40[i + 1]) - mpf(rho)) if i + 1 < K - 1 else mpf(A_dec)
        r, d2, M3, pieces = radius_box(ch, a_lo, a_hi, lo_lim, hi_lim, T[2], T[3])
        rad[i] = mpf(r) - mpf(rho)
        good = (up(d2) + T[2] < 0) and r > rhof
        ok2 = ok2 and good
        say(
            f"   atom {i:2d} x~ = {xt[i]:9.4f}  sup D'' = {fl(up(d2)):.5e}  M3 = {fl(M3) if M3 is not None else float('nan'):.5e} ({pieces:4d})  r = {float(r):7.4f}  {'ok' if good else 'NO RADIUS'}   [{time.time()-t0:.0f}s]"
        )
    allok = ok1 and ok2
    worst_all = None
    xs = [mpf(v) for v in x40]
    for i in range(K - 1):
        a, b = xs[i], xs[i + 1]
        lo_edge = mpf(lr) if i == 0 else a + rad.get(i, mpf(0))
        hi_edge = mpf(rr) if i == K - 2 else b - rad.get(i + 1, mpf(0))
        ta, tb = msqrt(a), msqrt(b)
        pts = [str((ta + (tb - ta) * mpf(j) / 200) ** 2) for j in range(201)]
        Dv = {}
        worst = None
        cnt = 0
        for j in range(200):
            s0, s1 = pts[j], pts[j + 1]
            if mpf(s1) <= lo_edge or mpf(s0) >= hi_edge:
                continue
            if mpf(s0) <= 0:
                allok = False
                break
            cnt += 1
            slack, _ = chord_cell(ch, s0, s1, Dv, Ca, T[0], T[2])
            worst = slack if worst is None or slack < worst else worst
        good = cnt == 0 or (worst is not None and worst > 0)
        allok = allok and good
        if worst is not None:
            worst_all = worst if worst_all is None or worst < worst_all else worst_all
        say(
            f"   [{float(a):8.4f},{float(b):9.4f}] checked {cnt:3d}  worst slack {fl(dn(worst)) if worst is not None else float('nan'):.5e}  {'ok' if good else 'FAILS'}   [{time.time()-t0:.0f}s]"
        )
    verdict = allok and kok
    say(
        f"   => D(x; F*) <= C(F*) on [0, A] for the exact point: {'PROVED (arb)' if verdict else 'NOT PROVED'}; worst slack {fl(dn(worst_all)) if worst_all is not None else float('nan'):.4e}   [{time.time()-t0:.0f}s]"
    )
    return {
        "ok": bool(verdict),
        "krawczyk_ok": bool(kok),
        "krawczyk_ratio": kratio,
        "worst_slack": fl(dn(worst_all)) if worst_all is not None else None,
        "radii": {i: float(r) for i, r in rad.items()},
        "piece_i": info1,
        "K": K,
        "A": float(A_dec),
        "rho": rho,
        "seconds": time.time() - t0,
    }


if __name__ == "__main__":
    ctx.dps = 40
    out_p = os.path.join(HERE, "results_q32_certificate_arb.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    items = []
    e = json.load(open(os.path.join(HERE, "results_q32_A100_exact40.json")))
    items.append(("A = 100 (control)", "100", e["x"], e["w"], "1e-30"))
    for label, rec in json.load(open(os.path.join(HERE, "results_q32_exact40_all.json"))).items():
        items.append((label, repr(float(rec["A"])), rec["x"], rec["w"], rec["rho"] or "1e-30"))
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for label, A_dec, x40, w40, rho in items:
        if only and only not in label:
            continue
        if label in out:
            print(f"{label}: already done ({'PROVED' if out[label]['ok'] else 'NOT PROVED'})", flush=True)
            continue
        assert mpf(x40[-1]) == mpf(A_dec) and mpf(x40[0]) == 0
        res = certify(label, A_dec, x40, w40, rho, say=lambda *a: print(*a, flush=True))
        out[label] = res
        json.dump(out, open(out_p, "w"), indent=1)
    bad = [k for k, v in out.items() if not v["ok"]]
    print("arb replication: " + ("PROVED at every case" if not bad else f"NOT PROVED at {bad}"), flush=True)
