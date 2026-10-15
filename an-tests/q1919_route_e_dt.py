"""q1919 (15 Oct 2026): route E of paper 2 (task P2FBL), registered in Amendment 2268 before any on-off versus graded value was
computed. Instrument: the binned DT bound of q1918 (Reading 482; grid H = 2e-5, certified bracket), imported unchanged.

Question. Paper 2 reports that the normal approximation, maximized over inputs, picks an on-off code at short windows. Does the
exact DT achievability bound (Polyanskiy, Poor, Verdu 2010, Theorem 17) for i.i.d. inputs pick on-off as well?
Channel: Poisson, peak count A, no dark count. Cells: A in {4, 5, 6}, n in {5, 10, 20}, eps in {0.1, 0.01} (18 cells).
Input families (fixed here):
  on-off: {0, A} with weight w on A, w in {0.05, 0.10, ..., 0.95} (19 laws);
  graded: {0, x, A} with x in {0.25, 0.40, 0.55} A, middle weight in {0.1, 0.2, 0.3}, top weight in {0.2, 0.3, 0.4, 0.5, 0.6}
          (45 laws, all with total weight below 1), plus the capacity-achieving law from Blahut-Arimoto on 401 points (atoms
          below 1e-6 dropped).
For each law and cell: log M_DT = largest log M with DT <= eps, computed from both sides of the bracket: log M_lo from the DT upper
bracket (conservative) and log M_hi from the DT lower bracket. Per family the best over its laws, separately for lo and hi.
Cell scored ON-OFF if best on-off log M_lo >= best graded log M_hi; GRADED if best graded log M_lo > best on-off log M_hi;
UNRESOLVED otherwise.
VERDICT (H_E, registered): PASS if at least 12 of the 18 cells are ON-OFF, FAIL otherwise; VOID if any bracket is wider than 1e-2
nats in log M, or a control fails; INCOMPLETE if fewer than 18 cells were computed (an interrupted run).
Readings: per cell the two best log M values (nats), the difference, the best on-off weight and the best graded law; the
normal-approximation rate n C - sqrt(n V) Q^-1(eps) of each family's best law is not computed here.
Usage: py q1919_route_e_dt.py [--registration-run]
"""
import json
import math
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1918_route_e_controls as R

OUT = os.path.join(HERE, "paper_neuron", "q1919_cells.json")
AS, NS, EPSS = (4.0, 5.0, 6.0), (5, 10, 20), (0.1, 0.01)
WIDTH_BAR = 1e-2


def cap_law(A, n=401, it=200000):
    x = A * np.linspace(0, 1, n); ymax = int(A + 40 * math.sqrt(A + 1) + 60); y = np.arange(ymax + 1)
    lx = np.log(np.maximum(x[:, None], 1e-300)); L = -x[:, None] + y[None, :] * lx - gammaln(y + 1)[None, :]
    L[:, 0] = -x; L[0, 1:] = -np.inf; P = np.exp(L)
    p = np.full(n, 1.0 / n)
    for _ in range(it):
        q = p @ P
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(P > 0, P * (np.log(P) - np.log(q[None, :])), 0.0).sum(1)
        if D.max() - p @ D < 1e-9:
            break
        p = p * np.exp(D); p /= p.sum()
    keep = p > 1e-6
    return list(x[keep]), list(p[keep] / p[keep].sum())


def families(A):
    on = [("onoff w=%.2f" % w, [0.0, A], [1 - w, w]) for w in np.round(np.arange(0.05, 0.951, 0.05), 2)]
    gr = []
    for fx in (0.25, 0.40, 0.55):
        for wm in (0.1, 0.2, 0.3):
            for wt in (0.2, 0.3, 0.4, 0.5, 0.6):
                if wm + wt < 1 - 1e-12:
                    gr.append(("3pt x=%.2fA wm=%.1f wt=%.1f" % (fx, wm, wt), [0.0, fx * A, A], [1 - wm - wt, wm, wt]))
    xs, ps = cap_law(A)
    gr.append(("capacity law (%d atoms)" % len(xs), xs, ps))
    return on, gr


def laws(xs, ps, n):
    v, p, cut = R.letter_law(xs, ps, R.poisson_W)
    s_dn, pr_dn = R.binned_sum_law(v, p, n, up=False)
    s_up, pr_up = R.binned_sum_law(v, p, n, up=True)
    return (s_up, pr_up, 0.0), (s_dn, pr_dn, 1.0 - (1.0 - cut) ** n)


def best_logM(law, eps):
    s, pr, extra = law
    dt = lambda c: R.dt_from_law(s, pr, c) + extra
    a, b = -20.0, 200.0
    if dt(a) > eps:
        return -math.inf
    for _ in range(70):
        m = 0.5 * (a + b)
        (a, b) = (m, b) if dt(m) <= eps else (a, m)
    return math.log(2 * math.exp(a) + 1)


def score(on_lo, on_hi, gr_lo, gr_hi):
    if on_lo >= gr_hi:
        return "ON-OFF"
    if gr_lo > on_hi:
        return "GRADED"
    return "UNRESOLVED"


def decide(cells, widths_ok=True, controls_ok=True):
    if not controls_ok or not widths_ok:
        return "VOID", "controls %s, widths %s, cells %d" % (controls_ok, widths_ok, len(cells))
    if len(cells) != 18:
        return "INCOMPLETE", "cells %d of 18" % len(cells)
    k = sum(c["score"] == "ON-OFF" for c in cells)
    return ("PASS" if k >= 12 else "FAIL"), "ON-OFF in %d of 18 cells (bar >= 12)" % k


def registration_run():
    ok = R.main() == 0
    hi = [{"score": "ON-OFF"}] * 12 + [{"score": "GRADED"}] * 6
    lo = [{"score": "ON-OFF"}] * 11 + [{"score": "UNRESOLVED"}] * 7
    d = (decide(hi)[0], decide(lo)[0], decide(hi[:17])[0], decide(hi, widths_ok=False)[0])
    print("control decide(): %s %s %s %s (expected PASS FAIL INCOMPLETE VOID)" % d)
    ok &= d == ("PASS", "FAIL", "INCOMPLETE", "VOID")
    s = (score(1.0, 1.1, 0.8, 0.9), score(0.8, 0.9, 1.0, 1.1), score(1.0, 1.2, 1.1, 1.3))
    print("control score(): %s %s %s (expected ON-OFF GRADED UNRESOLVED)" % s)
    ok &= s == ("ON-OFF", "GRADED", "UNRESOLVED")
    xs, ps = cap_law(5.0)
    print("control capacity law at A = 5: atoms %s, weights %s" % (np.round(xs, 3).tolist(), np.round(ps, 4).tolist()))
    ok &= abs(sum(ps) - 1) < 1e-12 and xs[0] == 0.0 and abs(xs[-1] - 5.0) < 1e-12
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1919: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1919 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    cells = []; widths_ok = True
    for A in AS:
        on, gr = families(A)
        for n in NS:
            res = {}
            for fam, L in (("on", on), ("gr", gr)):
                vals = []
                for name, xs, ps in L:
                    up, dn = laws(xs, ps, n)
                    vals.append((name, {e: (best_logM(dn, e), best_logM(up, e)) for e in EPSS}))
                res[fam] = vals
            for e in EPSS:
                on_lo = max(v[e][0] for _, v in res["on"]); on_hi = max(v[e][1] for _, v in res["on"])
                gr_lo = max(v[e][0] for _, v in res["gr"]); gr_hi = max(v[e][1] for _, v in res["gr"])
                wmax = max(v[e][1] - v[e][0] for f in ("on", "gr") for _, v in res[f])
                widths_ok &= wmax <= WIDTH_BAR
                bon = max(res["on"], key=lambda t: t[1][e][0])[0]; bgr = max(res["gr"], key=lambda t: t[1][e][0])[0]
                c = {"A": A, "n": n, "eps": e, "on_lo": on_lo, "on_hi": on_hi, "gr_lo": gr_lo, "gr_hi": gr_hi, "width_max": wmax,
                     "best_on": bon, "best_gr": bgr, "score": score(on_lo, on_hi, gr_lo, gr_hi)}
                cells.append(c)
                print("Reading: A %g n %d eps %g: on-off %.5f..%.5f (%s), graded %.5f..%.5f (%s), on-off minus graded %.5f -> %s"
                      % (A, n, e, on_lo, on_hi, bon, gr_lo, gr_hi, bgr, on_lo - gr_lo, c["score"]), flush=True)
            json.dump(cells, open(OUT, "w"), indent=1)
    v, why = decide(cells, widths_ok=widths_ok)
    print(why)
    print("VERDICT q1919 (Amendment 2268, H_E): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2, "INCOMPLETE": 3}[v]


if __name__ == "__main__":
    sys.exit(main())
