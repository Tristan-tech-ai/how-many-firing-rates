"""Certify the 16-atom point (results_q48_K16_exact40.json) on arb, rho = 1e-30; A_17 by bisection on its violation."""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from mpmath import mp, mpf
from flint import ctx

ctx.dps = 40
from q32_certificate_arb import certify
import q34_poisson_dark as qp

say = lambda *a: print(*a, flush=True)
e = json.load(open(os.path.join(HERE, "results_q48_K16_exact40.json")))
res = certify("K = 16 at A = 204.776", e["x"][-1], e["x"], e["w"], "1e-30", say=say)
out = {"A": float(mpf(e["x"][-1])), "certificate": {k: v for k, v in res.items() if k != "radii"}}
say(f"certificate: {'PROVED' if res['ok'] else 'NOT PROVED'}, worst slack {res['worst_slack']}")
mp.dps = 40
xs = np.array([float(mpf(v)) for v in e["x"]])
ws = np.array([float(mpf(v)) for v in e["w"]])
A = out["A"]
lo, hi = A, A + 45.0
vlo, vhi = qp.viol_of(lo, 0.0, xs, ws), qp.viol_of(hi, 0.0, xs, ws)
if vlo <= 1e-9 < vhi:
    for _ in range(16):
        mid = (lo + hi) / 2
        if qp.viol_of(mid, 0.0, xs, ws) <= 1e-9:
            lo = mid
        else:
            hi = mid
    out["A_17"] = (lo + hi) / 2
    say(
        f"A_17 = {out['A_17']:.4f} (sqrt = {np.sqrt(out['A_17']):.4f}); the Gaussian 16 -> 17 transition is not yet computed"
    )
json.dump(out, open(os.path.join(HERE, "results_q49_certify16.json"), "w"), indent=1)
