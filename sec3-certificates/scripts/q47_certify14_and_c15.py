"""
Q47: certify the 14-atom point (arb, exact point) and extend c(K) to K = 15, 16. 2026-09-06, night.
Inputs: results_q46_A164_exact40.json (14 atoms at A = 166.668, Newton residual 1.6e-38). Steps:
  (1) Krawczyk + arb exact-point certificate at rho = 1e-30 -> N*(166.668) = 14 PROVED or not;
  (2) A_15 by bisection on the 14-atom solution's KKT violation (float64), c(15) = sqrt(A_15) - 12.8187;
  (3) birth of K = 15 from the Gaussian K = 15 optimum near A_g = 13.4 (same shifted map), Newton 40 digits;
      if it converges, A_16 by bisection on it, c(16) = sqrt(A_16) - 13.5834.
Readings (CONJECTURE_LANE3, test 3): c(15), c(16) within 0.02 of 0.309, 0.311 keep the 1/K law.
"""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from mpmath import mp, mpf, nstr
from flint import ctx

ctx.dps = 40
import q33_gauss_support as qg
import q34_poisson_dark as qp
from support_mp3 import solve as newton40
from q32_certificate_arb import certify

say = lambda *a: print(*a, flush=True)
t0 = time.time()
e = json.load(open(os.path.join(HERE, "results_q46_A164_exact40.json")))
A_dec = e["x"][-1]
x40 = e["x"]
w40 = e["w"]
out = {"A": float(mpf(A_dec))}
# (1) certificate
res = certify("K = 14 at A = 166.668", A_dec, x40, w40, "1e-30", say=say)
out["certificate"] = {k: v for k, v in res.items() if k != "radii"}
say(
    f"(1) certificate: {'PROVED' if res['ok'] else 'NOT PROVED'}, worst slack {res['worst_slack']}, Krawczyk ratio {res['krawczyk_ratio']:.2e}   [{time.time()-t0:.0f}s]"
)
json.dump(out, open(os.path.join(HERE, "results_q47_certify14_c15.json"), "w"), indent=1)

# (2) A_15 by bisection on the 14-atom violation
mp.dps = 40
xs14 = np.array([float(mpf(v)) for v in x40])
ws14 = np.array([float(mpf(v)) for v in w40])
A14 = float(mpf(A_dec))


def bisect(xs, ws, lo, hi):
    vlo, vhi = qp.viol_of(lo, 0.0, xs, ws), qp.viol_of(hi, 0.0, xs, ws)
    if not (vlo <= 1e-9 < vhi):
        return None, (vlo, vhi)
    for _ in range(16):
        mid = (lo + hi) / 2
        if qp.viol_of(mid, 0.0, xs, ws) <= 1e-9:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, None


A15, info = bisect(xs14, ws14, A14, A14 + 40.0)
if A15 is None:
    say(f"(2) bisection bracket failed: violations {info}")
else:
    c15 = float(np.sqrt(A15) - 12.8187)
    out["A_15"] = A15
    out["c_15"] = c15
    say(f"(2) A_15 = {A15:.4f}, c(15) = {c15:.3f} (predicted 0.309)   [{time.time()-t0:.0f}s]")
json.dump(out, open(os.path.join(HERE, "results_q47_certify14_c15.json"), "w"), indent=1)

# (3) birth of K = 15 from the Gaussian K = 15 optimum
G = json.load(open(os.path.join(HERE, "results_q33_gauss_support.json")))["history"]
row = next(r for r in G if abs(r["Ag"] - 12.0) < 1e-9 and r["clean"])
hist, _ = qg.continuation([], 12.0, 13.5, row["us"], row["ws"], step=0.05, say=lambda *a: None)
g15 = [r for r in hist if r["clean"] and r["K"] == 15]
if not g15:
    say("(3) no clean Gaussian K = 15 step up to 13.5")
else:
    g = max(g15, key=lambda r: r["Ag"])
    Ag = g["Ag"]
    tG = np.array(g["us"]) + Ag
    wG = np.array(g["ws"])
    shift = np.where(tG < Ag, 0.5, np.where(tG < Ag + 2, 0.25, 0.0))
    shift[0] = 0.0
    tP = tG + shift
    A = (Ag + 0.31) ** 2
    tP = tP * ((2 * np.sqrt(A)) / tP[-1])
    tP[0] = 0.0
    xP = (tP / 2) ** 2
    r = newton40(
        A,
        [0.0] + [float(v) for v in xP[1:-1]] + [float(A)],
        list(wG),
        dps=40,
        newton_steps=40,
        tol_pow=8,
        pin_ends=True,
    )
    xa = np.array([float(v) for v in r["x"]])
    wa = np.array([float(v) for v in r["w"]])
    say(
        f"(3) K = 15 at A = {A:.3f} from Gaussian A_g = {Ag:.2f}: Newton residual {nstr(r['newton_resid'], 3)}, min gap {np.diff(xa).min():.3f}, min w {wa.min():.3e}   [{time.time()-t0:.0f}s]"
    )
    if r["newton_resid"] < mpf("1e-30") and np.diff(xa).min() > 0.5 and wa.min() > 1e-3:
        x15 = [nstr(v, 40) for v in r["x"]]
        w15 = [nstr(v, 40) for v in r["w"]]
        x15[0] = "0.0"
        x15[-1] = repr(float(A))
        json.dump(
            {"A": A, "K": 15, "x": x15, "w": w15, "C": nstr(r["C"], 40)},
            open(os.path.join(HERE, "results_q46_A%d_exact40_K15.json" % int(A)), "w"),
            indent=1,
        )
        A16, info = bisect(xa, wa, A, A + 45.0)
        if A16 is not None:
            c16 = float(np.sqrt(A16) - 13.5834)
            out["A_16"] = A16
            out["c_16"] = c16
            say(f"    A_16 = {A16:.4f}, c(16) = {c16:.3f} (predicted 0.311)   [{time.time()-t0:.0f}s]")
        else:
            say(f"    bisection bracket for A_16 failed: {info}")
json.dump(out, open(os.path.join(HERE, "results_q47_certify14_c15.json"), "w"), indent=1)
say(f"done [{time.time()-t0:.0f}s]")
