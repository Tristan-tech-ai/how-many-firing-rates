"""
Q35: CERTIFIED support sizes with dark current, by the shift identity. 2026-09-06.

The Poisson channel with dark current lambda on the input interval [0, A] is the Poisson channel without dark
current on the input interval [lambda, A + lambda]: Y ~ Poisson(x + lambda) with x in [0, A] is Y ~ Poisson(u)
with u in [lambda, A + lambda]. So the exact-point certificate of Amendment 70 (Krawczyk inclusion on the KKT
system, then the three-piece argument with the parameters as the box, here on arb) applies with the left
boundary atom at lambda instead of 0. Two things change and nothing else: the left end is a regular point
(no sliver lemma, no decade sweep; the left monotone run is a linear grid from lambda to the first interior
atom), and the 40-digit Newton pins both end atoms (support_mp3.solve with pin_ends).

Pipeline per (lambda, A, K): float64 continuation (q34_poisson_dark.step_solve) from the certified five-atom
solution at A = 27.9452 to the target A, shift by lambda, 40-digit Newton with pinned ends, Krawczyk ladder,
certificate. CONTROL: lambda = 0 at A = 27.9452, K = 5 through this code path must PROVE with worst slack
5.5174e-5 (Amendment 71c). TARGETS: the float64 transitions 5 -> 6 at lambda = 0.5 (A = 33.0386) and
lambda = 8 (A = 54.2724): certify K = 5 just below and K = 6 just above each, which turns two rows of the Q34
table into rigorous brackets. Output: results_q35_dark_certify.json.
"""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mp, mpf, nstr, sqrt as msqrt
from flint import arb, ctx
from support_mp3 import solve as newton40
from q34_poisson_dark import step_solve
from q32_certificate_arb import (
    ArbChannel,
    KKT,
    pt,
    box,
    up,
    dn,
    amax,
    fl,
    monotone,
    radius_box,
    chord_cell,
    krawczyk,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ctx.dps = 40


def piece_i_shift(ch, lo_dec, hi_dec, n=140):
    """left run from the left boundary atom (regular point), right run from the right boundary atom"""
    x_lo = mpf(lo_dec)
    T1 = ch.tail(pt(lo_dec), 1)
    T2 = ch.tail(pt(lo_dec), 2)
    x_hi = x_lo + (mpf(ch.xmid[1]) - x_lo) * mpf("0.9")
    left = [str(x_lo + (x_hi - x_lo) * mpf(j) / n) for j in range(n + 1)]
    lr, lm = monotone(ch, left, True, T1, T2)
    Am = mpf(hi_dec)
    xk2 = mpf(ch.xmid[-2]) + (Am - mpf(ch.xmid[-2])) * mpf("0.1")
    right = [hi_dec] + [str(Am - (Am - xk2) * mpf(j) / n) for j in range(1, n + 1)]
    rr, rm = monotone(ch, right, False, T1, T2)
    ok = (
        lm is not None
        and rm is not None
        and (lm > 1)
        and (rm > 1)
        and float(lr) > float(lo_dec)
        and float(rr) < float(hi_dec)
    )
    return (
        ok,
        {
            "left_reach": float(lr),
            "left_margin": fl(lm) if lm is not None else None,
            "right_reach": float(rr),
            "right_margin": fl(rm) if rm is not None else None,
        },
        lr,
        rr,
    )


def certify_shift(label, lo_dec, hi_dec, x40, w40, rho, say=print):
    """x40 are the shifted atoms (x40[0] = lo_dec = lambda, x40[-1] = hi_dec = A + lambda)"""
    t0 = time.time()
    kok, kratio = krawczyk(hi_dec, x40, w40, rho)
    say(
        f"{label}: Krawczyk at rho = {rho}: ratio {kratio:.3e} {'inside' if kok else 'NOT inside'}   [{time.time()-t0:.0f}s]"
    )
    ch = ArbChannel(hi_dec, x40, w40, rho)
    K = ch.K
    Cb = ch.C()
    Ca = dn(Cb)
    if float(lo_dec) == 0:  # the lambda = 0 control: the original piece (i) with the sliver lemma
        from q32_certificate_arb import piece_i as piece_i0

        T = {o: ch.tail(pt("1e-8"), o) for o in range(4)}
        ok1, info1, lr, rr = piece_i0(ch, hi_dec)
    else:
        T = {o: ch.tail(pt(lo_dec), o) for o in range(4)}
        ok1, info1, lr, rr = piece_i_shift(ch, lo_dec, hi_dec)
    say(
        f"   (i) left margin {info1.get('left_margin')}, right margin {info1.get('right_margin')} {'ok' if ok1 else 'FAIL'}   [{time.time()-t0:.0f}s]"
    )
    rad = {}
    ok2 = True
    rhof = float(rho)
    xt = [float(mpf(v)) for v in x40]
    for i in range(1, K - 1):
        a_lo, a_hi = mpf(x40[i]) - mpf(rho), mpf(x40[i]) + mpf(rho)
        lo_lim = (mpf(x40[i - 1]) + mpf(rho)) if i - 1 > 0 else mpf(lo_dec)
        hi_lim = (mpf(x40[i + 1]) - mpf(rho)) if i + 1 < K - 1 else mpf(hi_dec)
        r, d2, M3, pieces = radius_box(ch, a_lo, a_hi, lo_lim, hi_lim, T[2], T[3])
        rad[i] = mpf(r) - mpf(rho)
        good = (up(d2) + T[2] < 0) and r > rhof
        ok2 = ok2 and good
        say(
            f"   atom {i:2d} u~ = {xt[i]:9.4f}  sup D'' = {fl(up(d2)):.5e}  M3 = {fl(M3) if M3 is not None else float('nan'):.5e} ({pieces:4d})  r = {float(r):7.4f}  {'ok' if good else 'NO RADIUS'}   [{time.time()-t0:.0f}s]"
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
        f"   => {label}: D(u; F*) <= C(F*) on [lambda, A + lambda] for the exact point: {'PROVED (arb)' if verdict else 'NOT PROVED'}; worst slack {fl(dn(worst_all)) if worst_all is not None else float('nan'):.4e}   [{time.time()-t0:.0f}s]"
    )
    return {
        "ok": bool(verdict),
        "krawczyk_ok": bool(kok),
        "krawczyk_ratio": kratio,
        "worst_slack": fl(dn(worst_all)) if worst_all is not None else None,
        "radii": {i: float(r) for i, r in rad.items()},
        "piece_i": info1,
        "K": K,
        "rho": rho,
        "seconds": time.time() - t0,
    }


def float_solution(lam, A_target, xs0, ws0, A0, step=1.0, say=print):
    """float64 continuation from (xs0, ws0) at A0 to A_target at dark current lam; returns the clean solution"""
    xs, ws = np.array(xs0, float), np.array(ws0, float)
    A = A0
    while A < A_target - 1e-9:
        A_new = min(A + step, A_target)
        xs = xs * (A_new / A)
        xs[0] = 0.0
        A = A_new
        xs, ws, C, viol, res, clean = step_solve(A, lam, xs, ws)
    xs, ws, C, viol, res, clean = step_solve(A_target, lam, xs, ws)
    say(
        f"   float64 at lambda = {lam}, A = {A_target}: K = {len(xs)}, viol {viol:+.1e}, resid {res:.1e}, {'clean' if clean else 'UNCLEAN'}"
    )
    return xs, ws, clean


def run_case(label, lam, A, K_expect, xs0, ws0, A0, say=print):
    xs, ws, clean = float_solution(lam, A, xs0, ws0, A0, say=say)
    if len(xs) != K_expect or not clean:
        say(f"   {label}: float64 solution has K = {len(xs)} (expected {K_expect}) or is unclean; skipped")
        return {"ok": False, "reason": "float64 stage", "K": len(xs), "clean": clean}
    # shift to [lambda, A + lambda] and refine at 40 digits with pinned ends
    mp.dps = 40
    lo_dec = repr(float(lam))
    hi_dec = repr(float(A + lam))
    pts = [lo_dec] + [repr(float(v + lam)) for v in xs[1:-1]] + [hi_dec]
    r = newton40(A + lam, pts, list(ws), dps=40, newton_steps=30, tol_pow=8, pin_ends=True)
    x40 = [nstr(v, 40) for v in r["x"]]
    w40 = [nstr(v, 40) for v in r["w"]]
    x40[0] = lo_dec
    x40[-1] = hi_dec
    say(f"   40-digit Newton: residual {nstr(r['newton_resid'], 3)}  history {r['hist']}")
    res = certify_shift(label, lo_dec, hi_dec, x40, w40, "1e-30", say=say)
    res.update({"lambda": lam, "A": A, "x40": x40, "w40": w40, "newton_residual": nstr(r["newton_resid"], 3)})
    return res


if __name__ == "__main__":
    e = json.load(open(os.path.join(HERE, "results_q32_exact40_all.json")))["5 to 6 lower"]
    xs0 = [float(v) for v in e["x"]]
    ws0 = [float(v) for v in e["w"]]
    A0 = e["A"]
    say = lambda *a: print(*a, flush=True)
    out = {}
    cases = [
        ("control lambda=0 A=27.9452 K=5", 0.0, 27.9452, 5),
        ("lambda=0.5 below 5->6 (A=32.9)", 0.5, 32.9, 5),
        ("lambda=0.5 above 5->6 (A=33.2)", 0.5, 33.2, 6),
        ("lambda=8 below 5->6 (A=54.0)", 8.0, 54.0, 5),
        ("lambda=8 above 5->6 (A=54.6)", 8.0, 54.6, 6),
    ]
    for label, lam, A, K in cases:
        say(f"\n== {label}")
        out[label] = run_case(label, lam, A, K, xs0, ws0, A0, say=say)
        json.dump(out, open(os.path.join(HERE, "results_q35_dark_certify.json"), "w"), indent=1)
    print("\nsummary:")
    for k, v in out.items():
        print(f"   {k:36s} {'PROVED' if v.get('ok') else 'not proved'}   slack {v.get('worst_slack')}")
