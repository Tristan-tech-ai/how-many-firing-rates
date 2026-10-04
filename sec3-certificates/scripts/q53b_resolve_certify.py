"""
Q53b: re-solve a birth point CONSISTENTLY (A as a decimal string, dps 50) and certify it on arb. 2026-09-06.
Why: the q52/q54 save path wrote the atoms from a Newton run at the BINARY value of a float A but stored A as
its decimal repr; parsed at 40 digits the strings carry a 1.6e-16 residual, which the Krawczyk step, with
condition number 4.6e9 at the light newborn, magnifies past the 1e-30 box (ratio 1.35e16). Fix: parse A_dec
at 50 digits, Newton (pinned ends) at dps 50, save 50-digit strings; Krawczyk with rho = 1e-28.
Usage: py q53b_resolve_certify.py <json> <rho> <label> [<A_g next transition, for c(K+1)>]
"""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from mpmath import mp, mpf, nstr

say = lambda *a: print(*a, flush=True)
t0 = time.time()
src, rho, label = sys.argv[1], sys.argv[2], sys.argv[3]
Ag_next = float(sys.argv[4]) if len(sys.argv) > 4 else None
mp.dps = 50
from support_mp3 import solve as newton

d = json.load(open(src))
A_dec = repr(float(d["A"])) if not isinstance(d["A"], str) else d["A"]
r = newton(A_dec, d["x"], d["w"], dps=50, newton_steps=30, tol_pow=8, pin_ends=True)
x50 = [nstr(v, 50) for v in r["x"]]
w50 = [nstr(v, 50) for v in r["w"]]
x50[0] = "0.0"
x50[-1] = A_dec
say(
    f"{label}: re-solved at dps 50 with A = {A_dec}: residual {nstr(r['newton_resid'], 3)} in {len(r['hist'])} steps (hist[0] = {r['hist'][0]})   [{time.time()-t0:.0f}s]"
)
out_fn = src.replace("_exact40.json", "_exact50.json")
json.dump({"A": A_dec, "K": len(x50), "x": x50, "w": w50, "C": nstr(r["C"], 50)}, open(out_fn, "w"), indent=1)
# residual of the SAVED strings, parsed at dps 50 (the number the certificate sees)
from support_mp3 import _tables, D_of, Dp_of

x = [mpf(v) for v in x50]
w = [mpf(v) for v in w50]
K = len(x)
Af = float(A_dec)
ny = int(Af + 18 * Af**0.5 + 60)
P, Q = _tables(x, w, ny)
Dv = [D_of(xi, Q, ny) for xi in x]
G = [Dv[i] - Dv[0] for i in range(1, K)] + [sum(w) - 1] + [Dp_of(x[i], Q, ny) for i in range(1, K - 1)]
say(f"   residual of the saved strings at dps 50: {nstr(max(abs(g) for g in G), 3)}")
from flint import ctx

ctx.dps = 50
from q32_certificate_arb import certify
import q34_poisson_dark as qp

res = certify(label, A_dec, x50, w50, rho, say=say)
out = {
    "A": Af,
    "K": K,
    "source": os.path.basename(out_fn),
    "rho": rho,
    "certificate": {k: v for k, v in res.items() if k != "radii"},
}
say(
    f"certificate: {'PROVED' if res['ok'] else 'NOT PROVED'}, Krawczyk ratio {res['krawczyk_ratio']:.3e}, worst slack {res['worst_slack']}   [{time.time()-t0:.0f}s]"
)
xs = np.array([float(mpf(v)) for v in x50])
ws = np.array([float(mpf(v)) for v in w50])
lo, hi = Af, Af + 45.0
vlo, vhi = qp.viol_of(lo, 0.0, xs, ws), qp.viol_of(hi, 0.0, xs, ws)
say(f"violation of the {K}-atom family at A = {lo:.3f}: {vlo:.2e}; at {hi:.3f}: {vhi:.2e}")
if vlo <= 1e-9 < vhi:
    for _ in range(18):
        mid = (lo + hi) / 2
        if qp.viol_of(mid, 0.0, xs, ws) <= 1e-9:
            lo = mid
        else:
            hi = mid
    out[f"A_{K+1}"] = (lo + hi) / 2
    msg = f"A_{K+1} = {out[f'A_{K+1}']:.4f} (sqrt = {np.sqrt(out[f'A_{K+1}']):.4f})"
    if Ag_next:
        out[f"c_{K+1}"] = float(np.sqrt(out[f"A_{K+1}"]) - Ag_next)
        msg += f"; Gaussian transition A_g = {Ag_next}; c({K+1}) = {out[f'c_{K+1}']:.4f} (law 0.346 - 0.531/{K+1} = {0.346-0.531/(K+1):.4f})"
    say(msg + f"   [{time.time()-t0:.0f}s]")
json.dump(out, open(os.path.join(HERE, f"results_q53b_certify{K}_A{Af:.3f}.json"), "w"), indent=1)
say(f"done [{time.time()-t0:.0f}s]")
