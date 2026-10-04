"""
The certificate for the EXACT KKT point: closes the audit's caveat 1. 2026-09-06.

q32_certificate_v3.py proves max_x D(x; F) <= I(F) + eta for the decimal input F. q32_krawczyk.py proves that
the KKT equality system has exactly one zero F* in the box B = {|w_k - w~_k| <= rho, |x_i - x~_i| <= rho}
around the 40-digit numerical point (x_0 = 0 and x_{K-1} = A exact). This file runs the same three-piece
argument with the PARAMETERS AS THE BOX, so every enclosure covers F*, and reads the pieces for F*:
    (i)   D'(x; F) < 0 on (0, left_reach] for every F in B, hence for F*; so D(x; F*) <= D(0; F*) = C(F*)
          there, the last equality being a KKT row (D(x_k) = D(x_0) for all atoms, sum w = 1). Same on the right.
    (ii)  at each interior atom a* in [a~ - rho, a~ + rho]: D'(a*; F*) = 0 and D(a*; F*) = C(F*) exactly (KKT),
          D''(a*; F*) <= sup_B D'' over the atom box < 0, and |D'''| <= M3 on [a~ - rho - r, a~ + rho + r]
          for every F in B; the cubic bound then gives D(x; F*) <= C(F*) for |x - a*| <= r, which covers
          x in [a~ - (r - rho), a~ + (r - rho)] whatever a* is in its box.
    (iii) on every cell of every gap not inside a radius: D(x; F*) <= sup_B [max(D(s0), D(s1)) + M2 h^2/8]
          <= inf_B C(F) <= C(F*) whenever the box slack is positive.
Together: D(x; F*) <= C(F*) on all of [0, A]. By the KKT sufficient condition (Smith 1971; for the Poisson
channel Shamai 1990), F* is capacity-achieving. The capacity-achieving input of this channel is unique
(Shamai 1990, IEE Proc. I 137(6):424-430, as cited by Barletta-Dytso-Shamai 2024, arXiv 2401.05045: "unique
and discrete with finitely many mass points"), so F* IS the optimum and N*(A) = K exactly, with no eta.

What remains outside the computation: the two cited theorems (KKT sufficiency; uniqueness), the Krawczyk
theorem (Krawczyk 1969, Moore 1977), and the correctness of mpmath's directed rounding.

Control: A = 100, K = 11, rho = 1e-30. Must pass with slacks within 1e-25 of v3's (the box is 1e-30 wide).
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mpf, mpi, iv, nstr, sqrt
from support_iv import Channel
from q32_certificate_v3 import lo, hi, absmax_iv, monotone_reach, decades_negative, chord_cell, DEPTH

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def m3_box(ch, L, R, r, d, T3, pieces0=32, max_pieces=2048):
    pieces = pieces0
    while True:
        Mi = ch.absmax(L, R, 3, pieces, DEPTH) + T3
        Ms = max(abs(ch.Dppp(L + (R - L) * mpf(j) / pieces)) for j in range(pieces + 1))
        cert = r <= 3 * d / Mi
        genuine = Mi <= mpf("1.5") * Ms
        if cert or genuine or pieces >= max_pieces:
            return Mi, Ms, pieces, cert
        f = ((Mi - Ms) / (mpf("0.5") * Ms)) ** mpf("0.25") if Ms > 0 else mpf(4)
        newp = pieces * 2
        while newp < pieces * f:
            newp *= 2
        pieces = min(max(newp, pieces * 2), max_pieces)


def radius_box(ch, a_lo, a_hi, lo_lim, hi_lim, T2, T3):
    d2 = ch._sum(mpi(a_lo, a_hi), 2)  # naive over the atom box (width 2 rho)
    d2u = hi(d2) + T2
    if d2u >= 0:
        return mpf(0), d2, None, 0
    d = -d2u
    rmax = min(a_lo - lo_lim, hi_lim - a_hi)
    best = None
    Mbest = None
    pbest = 0
    r = rmax / 4
    for _ in range(16):
        Mi, Ms, pieces, cert = m3_box(ch, a_lo - r, a_hi + r, r, d, T3)
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
            Mi, Ms, pieces, cert = m3_box(ch, a_lo - mid, a_hi + mid, mid, d, T3)
            if cert:
                lo_r, best, Mbest, pbest = mid, mid, Mi, pieces
            else:
                hi_r = mid
    return best, d2, Mbest, pbest


def analyse_box(A, x40, w40, rho, nsamp=200, dps=40, quiet=False):
    t0 = time.time()
    from mpmath import mp

    mp.dps = dps
    iv.dps = dps  # BEFORE parsing the 40-digit strings, or the box is centred on a 15-digit point
    rho = mpf(rho)
    xt = [mpf(v) for v in x40]
    wt = [mpf(v) for v in w40]
    K = len(xt)
    A_dec = mpf(repr(float(A)))  # the decimal amplitude, identical to the stored last atom
    assert xt[0] == 0 and xt[-1] == A_dec, "boundary atoms must be exactly 0 and the decimal A"
    Xb = [iv.mpf(0)] + [mpi(xt[i] - rho, xt[i] + rho) for i in range(1, K - 1)] + [iv.mpf(A_dec)]
    Wb = [mpi(w - rho, w + rho) for w in wt]
    ch = Channel(A, Xb, Wb, dps)
    say = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, **k, flush=True))
    xs = ch.xp
    Ci = ch.C_iv()
    Ca, Cb = lo(Ci), hi(Ci)
    say(
        f"A = {A:.4f}, K = {K}, rho = {nstr(rho, 2)}, C(F) over the box in [{nstr(Ca, 25)}, {nstr(Cb, 25)}]  (ny = {ch.ny}, depth {DEPTH})"
    )
    T = {o: ch.tail_bound(mpf(10) ** -8, o) for o in range(4)}
    say(f"tail bounds: " + " ".join(f"T{o} = {nstr(T[o], 2)}" for o in range(4)))

    # (i) with the left sliver lemma and the right grid starting at A (q32_piece_i.py)
    from q32_piece_i import piece_i

    left_reach, right_reach, ok1, info1 = piece_i(ch, A_dec, T1=T[1], T2=T[2], say=say)
    D0 = ch.D_iv(iv.mpf(0))
    DA = ch.D_iv(iv.mpf(A_dec))
    say(
        f"    consistency: D(0) - C over box within {nstr(max(abs(hi(D0) - Ca), abs(lo(D0) - Cb)), 2)}, D(A) - C within {nstr(max(abs(hi(DA) - Ca), abs(lo(DA) - Cb)), 2)}   [{time.time()-t0:.0f}s]"
    )

    # (ii)
    rad = {}
    ok2 = True
    say(
        "(ii) cubic radii about the atom boxes; M3 proved on [a~ - rho - r, a~ + rho + r] for every F in the box"
    )
    for i in range(1, K - 1):
        a_lo, a_hi = xt[i] - rho, xt[i] + rho
        lo_lim = (xt[i - 1] + rho) if i - 1 > 0 else mpf(0)
        hi_lim = (xt[i + 1] - rho) if i + 1 < K - 1 else A_dec
        r, d2, M3, pieces = radius_box(ch, a_lo, a_hi, lo_lim, hi_lim, T[2], T[3])
        rad[i] = r - rho
        Dpa = ch.Dp_iv(mpi(a_lo, a_hi))
        Da = ch.D_iv(mpi(a_lo, a_hi))
        good = hi(d2) + T[2] < 0 and r > rho
        ok2 = ok2 and good
        say(
            f"    atom {i:2d} x~ = {float(xt[i]):9.4f}  sup D'' = {nstr(hi(d2), 5):>12}  M3 = {nstr(M3, 5) if M3 else '-':>11} ({pieces:4d} pieces)  r = {float(r):7.4f}  "
            f"consistency |D'| <= {nstr(absmax_iv(Dpa), 2)}, D - C within {nstr(max(abs(hi(Da) - Ca), abs(lo(Da) - Cb)), 2)}  {'ok' if good else 'NO RADIUS'}   [{time.time()-t0:.0f}s]"
        )

    # (iii)
    say("(iii) chord bound on every gap for every F in the box, slack = inf C - sup bound")
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
            f"    [{float(a):8.4f},{float(b):9.4f}]  checked {cnt:3d} (+{splits:3d} split)  worst inf C - sup bound {nstr(worst, 5) if worst is not None else 'all inside radii':>18}  {'ok' if good else 'FAILS'}   [{time.time()-t0:.0f}s]"
        )
    say(
        f"D(x; F*) <= C(F*) on [0, A] for the exact KKT point F* in the Krawczyk box: {'PROVED' if allok else 'NOT PROVED'} "
        f"(all three pieces {allok}); worst chord slack {nstr(worst_all, 4) if worst_all is not None else '-'}   [{time.time()-t0:.0f}s]"
    )
    return {
        "ok": bool(allok),
        "rho": float(rho),
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
    rho = sys.argv[2] if len(sys.argv) > 2 else "1e-30"
    d = json.load(open(os.path.join(HERE, f"results_q32_A{which}_exact40.json")))
    kr = json.load(open(os.path.join(HERE, f"results_q32_krawczyk_A{which}.json")))
    assert kr[rho]["ok"], f"Krawczyk test not passed at rho = {rho}"
    res = analyse_box(d["A"], d["x"], d["w"], rho)
    res["krawczyk"] = kr[rho]
    json.dump(res, open(os.path.join(HERE, f"results_q32_certificate_v4_A{which}.json"), "w"), indent=1)
