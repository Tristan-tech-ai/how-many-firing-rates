"""
Q80: the transition amplitude A_{K+1} defined by the newborn weight -> 0, from the 40-digit unfolding.
From the K-atom KKT point at A_try (just above the transition) insert the newborn at the violation peak and
unfold (feasible-step) at eps = 1e-3, 5e-4, 2.5e-4; A(eps) is linear in eps near birth, so A_{K+1} =
A(0) by extrapolation, and dA/d(eps) is the inverse birth slope. This replaces the float64 bisection with a
1e-9 violation threshold, which sits about 0.03 above the transition because the violation grows slowly.
Usage: py q80_AK_by_unfolding.py <src.json> <A_try> <tag>
"""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from mpmath import mp, mpf, nstr
from support_unfold import solve_unfold
from support_mp3 import solve as newton40
import q34_poisson_dark as qp

say = lambda *a: print(*a, flush=True)
t0 = time.time()
src = sys.argv[1]
A_try = float(sys.argv[2])
A_dec = repr(A_try)
tag = sys.argv[3]
e = json.load(open(src))
x0 = np.array([float(mpf(v)) for v in e["x"]])
w0 = [repr(float(mpf(v))) for v in e["w"]]
xs = x0 * (A_try / x0[-1])
pts0 = [repr(float(v)) for v in xs]
pts0[0] = "0.0"
pts0[-1] = A_dec
SEED = os.environ.get("SEED")
if SEED:
    _s = json.load(open(SEED))
    pts = list(_s["x"])
    wts = list(_s["w"])
    fixed = int(_s["fixed"])
    say(
        f"seed state {os.path.basename(SEED)}: K = {len(pts)}, A = {pts[-1][:16]}, eps = {_s['eps']}, fixed = {fixed}"
    )
else:
    rK = newton40(A_dec, pts0, w0, dps=40, newton_steps=40, tol_pow=8, pin_ends=True, feasible_step=True)
    xs = np.array([float(v) for v in rK["x"]])
    ws = np.array([float(v) for v in rK["w"]])
    xs[0] = 0.0
    say(
        f"K = {len(xs)} KKT point at A = {A_try}: residual {nstr(rK['newton_resid'], 3)}   [{time.time()-t0:.0f}s]"
    )
    y = np.arange(qp.ny_of(A_try, 0.0))
    xq = np.linspace(0, A_try, 20001)
    Dat, _ = qp.D_and_Dp(xs, xs, ws, 0.0, y)
    C = float(ws @ Dat)
    Dq, _ = qp.D_and_Dp(xq, xs, ws, 0.0, y)
    viol = Dq - C
    x_ins = float(xq[viol.argmax()])
    say(f"   max(D - C) = {viol.max():+.3e} at t = {2*np.sqrt(x_ins):.3f}")
    pairs = sorted([(float(u), float(w)) for u, w in zip(xs, ws)] + [(x_ins, 1e-3)])
    pts = [repr(u) for u, _ in pairs]
    wts = [repr(w) for _, w in pairs]
    fixed = [u for u, _ in pairs].index(x_ins)
pts[0] = "0.0"
pts[-1] = A_dec
out = {}
for eps in os.environ.get("RUNGS", "1e-3,5e-4,2.5e-4").split(","):
    r = solve_unfold(
        pts[-1],
        pts,
        wts,
        fixed,
        eps,
        dps=40,
        newton_steps=int(os.environ.get("NEWTON_STEPS", "60")),
        tol_pow=8,
        feasible_step=True,
        max_dx=(float(os.environ["MAX_DX"]) if os.environ.get("MAX_DX") else None),
    )
    xa = np.array([float(v) for v in r["x"]])
    say(
        f"   eps = {eps}: residual {nstr(r['newton_resid'], 3)} in {len(r['hist'])} steps, A = {nstr(r['A'], 14)}, newborn t = {2*np.sqrt(xa[fixed]):.4f}   [{time.time()-t0:.0f}s]"
    )
    say("   resid hist (last 8): " + " ".join(str(h) for h in r["hist"][-8:]))
    conv = r["newton_resid"] < mpf("1e-25")
    pts = [nstr(v, 40) for v in r["x"]]
    wts = [nstr(v, 40) for v in r["w"]]
    pts[0] = "0.0"
    _st = {
        "A": nstr(r["A"], 40),
        "K": len(pts),
        "x": list(pts),
        "w": list(wts),
        "fixed": fixed,
        "eps": eps,
        "resid": nstr(r["newton_resid"], 3),
    }
    _st["x"][-1] = _st["A"]
    json.dump(
        _st,
        open(
            os.path.join(
                HERE, f"results_q80_state_K{len(pts)}_eps{eps}_{tag}{'' if conv else '_unconv'}.json"
            ),
            "w",
        ),
        indent=1,
    )
    if conv:
        out[eps] = float(r["A"])
if len(out) >= 2:
    es = np.array([float(k) for k in out])
    As = np.array([out[k] for k in out])
    coef = np.polyfit(es, As, 1)
    A0 = coef[1]
    say(
        f"   A(eps) = {A0:.6f} + {coef[0]:.3f} eps (linear fit over {len(out)} points); A_K+1 = {A0:.6f}; dw/dA at birth = {1/coef[0]:.5f} per unit A"
    )
    if len(out) == 3:
        coef2 = np.polyfit(es, As, 2)
        say(f"   quadratic fit: A(0) = {coef2[2]:.6f} (difference {coef2[2]-A0:+.6f})")
    json.dump(
        {"tag": tag, "A_eps": out, "A_transition": A0, "dA_deps": coef[0]},
        open(os.path.join(HERE, f"results_q80_A_transition_{tag}.json"), "w"),
        indent=1,
    )
say(f"done [{time.time()-t0:.0f}s]")
