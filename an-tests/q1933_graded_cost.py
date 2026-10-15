"""q1933 (15 Oct 2026): task P2GRADEDCOST, registered in Amendment 2281 before any per-fibre information was computed.

Question (Barlow frame). Under a natural-like level law, how much single-window information does a fibre's measured graded rate map
give up against the infomax staircase, which attains the capacity C(A, lam) (Proposition L1)?
Data and selection as q1926 (PH recordings, 0 <= t < 50 ms, >= 4 levels, >= 20 repeats per level, A >= 1).
Per fibre: level means m_l at levels L_l (dB); lam = min m_l, A = max m_l - lam. Rate map r(L): linear interpolation of m_l in L,
constant beyond the measured range. Level law: Gaussian with SD 8.5 dB (the median implied SD of Amendment 2274) and mean mu at the
level where the interpolated map crosses lam + A/2 (the fibre's mid level; the adapted case, which favours the fibre).
I_map = I(X; Y), X = r(L), on 801 levels from mu - 4.5 SD to mu + 4.5 SD with Gaussian weights renormalised; C = capacity on [lam,
lam + A] (q1913.capacity, gap 1e-7). loss = 1 - I_map/C.
VERDICT (H_cost, registered): PASS if the median loss over fibres is below 0.10 (the prediction written in the queue); FAIL otherwise;
VOID if fewer than 20 fibres or a capacity run misses its gap; INCOMPLETE if the census is missing.
Readings: quartiles of loss; median loss for A below and above 5.
Usage: py q1933_graded_cost.py [--registration-run]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1913_capacity_loss_a1_v2 as Q
import q1926_ph_steps as S

SD = 8.5


def info_of_map(levels, means, lam, A):
    L = np.asarray(levels, float); m = np.asarray(means, float)
    order = np.argsort(L); L, m = L[order], m[order]
    target = lam + A / 2.0
    grid = np.linspace(L.min(), L.max(), 2001); rg = np.interp(grid, L, m)
    idx = np.where(rg >= target)[0]
    mu = float(grid[idx[0]]) if len(idx) else float(np.median(L))
    Ls = np.linspace(mu - 4.5 * SD, mu + 4.5 * SD, 801)
    w = norm.pdf((Ls - mu) / SD); w /= w.sum()
    x = np.interp(Ls, L, m)
    return Q.mutual_info(x, w), mu


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    if not all(r["ok"] for r in rows):
        return "VOID", "a capacity run missed its gap"
    med = float(np.median([r["loss"] for r in rows]))
    return ("PASS" if med < 0.10 else "FAIL"), "median loss %.4f over %d fibres (bar < 0.10)" % (med, len(rows))


def registration_run():
    ok = True
    lam, A = 0.5, 8.0
    C, _, conv = Q.capacity(lam, A)
    # (a) a map that realizes the capacity-achieving law under the Gaussian level law: build pi by fine Blahut-Arimoto, then quantile map
    xg = lam + A * np.linspace(0, 1, 401); ymax = int(lam + A + 12 * np.sqrt(lam + A + 1) + 30)
    P = Q.pois_matrix(xg, ymax); p = np.full(401, 1 / 401)
    for _ in range(20000):
        D, I, U = Q._gap_stats(P, p)
        if U - I < 1e-8:
            break
        p = p * np.exp(D); p /= p.sum()
    Ls = np.linspace(-4.5 * SD, 4.5 * SD, 801); w = norm.pdf(Ls / SD); w /= w.sum()
    Fs = np.cumsum(w); Gc = np.cumsum(p); xs = xg[np.minimum(np.searchsorted(Gc, Fs), 400)]
    la = 1 - Q.mutual_info(xs, w) / C
    # (b) a linear map spanning only the middle third of the level range
    lv = np.array([-30.0, -15.0, 0.0, 15.0, 30.0]); mb = np.interp(lv, [-5.0, 5.0], [lam, lam + A])
    Ib, _ = info_of_map(lv, mb, lam, A); lb = 1 - Ib / C
    print("control (a) quantile staircase of pi under the level law: loss %.4f; (b) steep linear map: loss %.4f" % (la, lb))
    ok &= conv and la < 0.01 and lb > 0.0
    d = (decide([{"loss": 0.05, "ok": True}] * 30)[0], decide([{"loss": 0.2, "ok": True}] * 30)[0], decide([{"loss": 0.05, "ok": True}] * 5)[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d)
    ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1933: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1933 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(S.CENSUS):
        print("VERDICT q1933: INCOMPLETE (census missing)"); return 3
    rows = []
    for rec in (x for x in json.load(open(S.CENSUS)) if "error" not in x):
        f = S.fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        means = [c.mean() for c in counts]
        I, mu = info_of_map(levels, means, lam, A)
        C, _, ok = Q.capacity(lam, A)
        rows.append({"path": rec["path"], "A": A, "lam": lam, "I": I, "C": C, "ok": bool(ok), "loss": 1 - I / C, "mu": mu})
    json.dump(rows, open(os.path.join(HERE, "paper_neuron", "q1933_rows.json"), "w"))
    if rows:
        L = np.array([r["loss"] for r in rows]); As = np.array([r["A"] for r in rows])
        print("Reading: fibres %d; loss quartiles %.4f %.4f %.4f" % (len(rows), *np.percentile(L, [25, 50, 75])))
        for name, m in (("A < 5", As < 5), ("A >= 5", As >= 5)):
            if m.any():
                print("Reading: %s: %d fibres, median loss %.4f" % (name, int(m.sum()), float(np.median(L[m]))))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1933 (Amendment 2281, H_cost): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
