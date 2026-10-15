"""q1956 (15 Oct 2026): task P2WEAKAUDIT, H_stepsLS, the q1953 likelihood statistic with K computed on the fibre's hull range.
Registered in Amendment 2299 only after the controls below pass.

Why. In the theory lam is the spontaneous count. q1926, and through it q1950 and q1953, took lam as the smallest PH level mean. Reading 553
(q1951) found, before the q1953 result, that this mean exceeds the mean count of the RLF silent trials by more than one spike in 22.2
percent of the 239 fibres with at least 5 silent trials. The range is then too short and K too small. The statistic, its variance model
and its bootstrap are those of q1953 (Amendment 2297), unchanged; only K changes.
Range. lam_S = min(smallest PH level mean, mean 0-50 ms count of the RLF silent trials if there are at least 5, else none);
top = largest PH level mean (no RLF level enters); K = rate-dependent K of q1950 with the fibre's Fano line on [lam_S, top].
VERDICT (H_stepsLS): PASS if at least 80 percent of the fibres have p >= 0.05; FAIL otherwise; VOID below 20 fibres.
Readings: the number of fibres whose range changed and how many of them gain a step; the fraction for A >= 8 and A < 8 and by thirds of
the median Fano; the fraction among fibres whose range changed and among the others; the fraction consistent with K + 1 steps.
Usage: py q1956_ph_steps_lik_hull.py [--registration-run]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1950_ph_steps_ratefano as R  # noqa: E402
import q1953_ph_steps_lik as L  # noqa: E402
from q1952_ph_steps_hull import hull, silent_mean  # noqa: E402

OUT = os.path.join(HERE, "paper_neuron", "q1956_rows.json")
ALPHA = 0.05


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean([r["p"] >= ALPHA for r in rows]))
    return ("PASS" if f >= 0.8 else "FAIL"), "consistent with at most K steps (hull range) in %.3f of %d fibres (bar 0.80)" % (f, len(rows))


def registration_run(nfib=100, B=100):
    ok = True; rng = np.random.default_rng(2299); a, b = 0.7034, -0.0355
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        Kfull, _ = R.k_rate(A, lam, a, b); vals = np.linspace(lam, lam + A, Kfull)
        tested = np.linspace(lam + 1.0, lam + A, 7)                 # levels start one spike above the spontaneous count
        stair = vals[np.argmin(np.abs(tested[:, None] - vals[None, :]), axis=1)]
        res = {}
        for name, rates in (("staircase", stair), ("graded", tested)):
            P, y = L.sampler(rates, a, b); Ps, ys = L.sampler([lam], a, b); rej = []
            for _ in range(nfib):
                cs = [rng.choice(y, size=40, p=P[i]).astype(float) for i in range(7)]
                sil = float(rng.choice(ys, size=10, p=Ps[0]).mean())          # 10 silent trials, as in the data
                means = [c.mean() for c in cs]
                l2, A2 = hull(min(means), max(means) - min(means), sil)
                K, _ = R.k_rate(A2, l2, a, b)
                rej.append(L.pvalue(cs, K, a, b, B, rng)[0] < ALPHA)
            res[name] = float(np.mean(rej))
        print("control: A %.0f lam %.1f, levels from lam + 1, silent count measured, K from the hull: rejected %.3f of staircase and "
              "%.3f of graded fibres (K full range %d)" % (A, lam, res["staircase"], res["graded"], Kfull))
        ok &= res["staircase"] <= 0.10 and res["graded"] >= 0.70
    h = (hull(2.0, 6.0, None), hull(2.0, 6.0, 3.0), hull(2.0, 6.0, 0.5))
    print("control hull(): %s (expected (2, 6), (2, 6), (0.5, 7.5))" % (h,)); ok &= h == ((2.0, 6.0), (2.0, 6.0), (0.5, 7.5))
    rows = [{"p": 0.5}] * 25; bad = [{"p": 0.01}] * 25
    d = (decide(rows)[0], decide(bad)[0], decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1956: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1956 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (R.CENSUS, R.R26, R.R36, L.R50)):
        print("VERDICT q1956: INCOMPLETE (input missing)"); return 3
    lines, r26, r36 = R.fibre_lines(); r50 = {r["path"]: r for r in json.load(open(L.R50))}
    recs = {rec["path"]: rec for rec in json.load(open(R.CENSUS)) if rec.get("path") in r26}
    rng = np.random.default_rng(2299); rows = []
    for path, levels, counts, lam, A, a, b, flat in lines:
        l2, A2 = hull(lam, A, silent_mean(recs[path]))
        K = r50[path]["K"] if l2 == lam else R.k_rate(A2, l2, a, b)[0]
        p, T = L.pvalue(counts, K, a, b, 200, rng)
        p1, _ = L.pvalue(counts, K + 1, a, b, 200, rng)
        rows.append({"path": path, "K": K, "K_q1950": r50[path]["K"], "changed": l2 < lam, "lam_S": l2, "A_S": A2, "T": T, "p": p,
                     "p_Kplus1": p1, "A": A, "F": r36[path]["median_fano"]})
        json.dump(rows, open(OUT, "w"))
    if rows:
        f = lambda S, k="p": float(np.mean([r[k] >= ALPHA for r in S])) if S else float("nan")
        ch = [r for r in rows if r["changed"]]; un = [r for r in rows if not r["changed"]]
        print("Reading: range extended in %d fibres, K raised in %d of them" % (len(ch), sum(r["K"] > r["K_q1950"] for r in ch)))
        print("Reading: consistent among changed %.3f (%d), among unchanged %.3f (%d)" % (f(ch), len(ch), f(un), len(un)))
        for name, sub in (("A >= 8", [r for r in rows if r["A"] >= 8]), ("A < 8", [r for r in rows if r["A"] < 8])):
            print("Reading: %s, fibres %d, consistent %.3f" % (name, len(sub), f(sub)))
        F = np.array([r["F"] for r in rows]); q = np.percentile(F, [33.3, 66.7])
        for name, m in (("most regular", F < q[0]), ("middle", (F >= q[0]) & (F < q[1])), ("least regular", F >= q[1])):
            print("Reading: %s third, consistent %.3f" % (name, f([r for r, mm in zip(rows, m) if mm])))
        print("Reading: consistent with K + 1 steps %.3f" % f(rows, "p_Kplus1"))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1956 (Amendment 2299, H_stepsLS): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
