"""q1952 (15 Oct 2026): task P2WEAKAUDIT, H_stepsS, registered in Amendment 2297 before any K of this range was computed on data.

Why. q1926 took lam as the smallest PH level mean and A as the range of the PH level means, excluding the silent trials. In the theory
lam is the spontaneous count, the count without a stimulus. The PH levels of a fibre are presented at its best frequency, as are the
RLF levels and their silent trials (Heeringa et al. 2024). Reading 553 (q1951_out.txt) found that the smallest PH level mean exceeds
the mean count of the RLF silent trials by more than one spike in 22.2 percent of the 239 fibres with at least 5 silent trials: the
lowest PH level already drives these fibres, so the range read by q1926 is too short and its K too small. This repairs the instrument
that reads the range; it can only raise K, and the control below shows that the test still separates staircases from graded curves.
Range. lam_S = min(smallest PH level mean, mean 0-50 ms count of the RLF silent trials if there are at least 5, else none);
top = largest PH level mean (unchanged; no RLF level enters, to keep the bias of a maximum over noisy means out); A_S = top - lam_S.
Channel (primary). The rate-dependent count of q1950 (Amendment 2294) with each fibre's Fano line, on [lam_S, lam_S + A_S].
K, D, merge, tolerance: as in q1950; D from q1926, unchanged.
VERDICT (H_stepsS, registered): PASS if D <= K in at least 80 percent of the fibres (bar unchanged); FAIL otherwise; VOID below 20 fibres.
Readings: the same fraction with the Poisson count and with the matched count of q1939 (one zeta per fibre) on the same range; the
number of fibres whose range changed; by thirds of the median Fano; A >= 8 and A < 8.
Usage: py q1952_ph_steps_hull.py [--registration-run]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1950_ph_steps_ratefano as R  # noqa: E402
from q1939_ph_steps_compoisson import k_cmp, k_poisson_merged  # noqa: E402

R39 = R.R39
OUT = os.path.join(HERE, "paper_neuron", "q1952_rows.json")
W = 0.050


def silent_mean(rec):
    s = rec["spikes"].get("RLF_silent", [])
    if len(s) < 5:
        return None
    return float(np.mean([sum(1 for t in tr if 0.0 <= t < W) for tr in s]))


def hull(lam, A, sil):
    top = lam + A
    lo = lam if sil is None else min(lam, sil)
    return lo, top - lo


def registration_run():
    ok = True
    # 1. hull(): no change when the silent count is missing or above lam; extension when below
    h = (hull(2.0, 6.0, None), hull(2.0, 6.0, 3.0), hull(2.0, 6.0, 0.5))
    print("control hull(): %s (expected (2, 6), (2, 6), (0.5, 7.5))" % (h,)); ok &= h == ((2.0, 6.0), (2.0, 6.0), (0.5, 7.5))
    # 2. staircase and graded fibres whose tested levels cover only the upper part of the range; K from the full range
    from q1926_ph_steps import count_distinct
    rng = np.random.default_rng(2297); a_med, b_med = 0.7034, -0.0355; LEV = list(np.arange(0, 70, 10.0))
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        res = {"staircase": [], "graded": []}
        K, _ = R.k_rate(A, lam, a_med, b_med)
        vals = np.linspace(lam, lam + A, K)
        for _ in range(200):
            sil = rng.poisson(lam, 10).mean()                       # 10 silent trials, as in the data
            lo_t = lam + 1.0                                        # lowest tested level one spike above spontaneous
            tested = np.linspace(lo_t, lam + A, 7)
            stair = vals[np.argmin(np.abs(tested[:, None] - vals[None, :]), axis=1)]
            for name, rates in (("staircase", stair), ("graded", tested)):
                zz, _ = R.zeta_for(np.maximum(rates, 1e-3), np.clip(a_med + b_med * rates, R.FLO, R.FHI))
                P = R.cmp_rows(rates, zz); y = np.arange(P.shape[1])
                cs = [rng.choice(y, size=40, p=row / row.sum()).astype(float) for row in P]
                means = np.array([c.mean() for c in cs]); l0, a0 = hull(float(means.min()), float(means.max() - means.min()), sil)
                Kh, _ = R.k_rate(a0, l0, a_med, b_med)
                res[name].append(count_distinct(LEV, cs) <= Kh)
        fs, fg = float(np.mean(res["staircase"])), float(np.mean(res["graded"]))
        print("control: A %.0f lam %.1f, tested levels from lam + 1, K from the hull: D <= K in %.3f of staircase and %.3f of graded fibres"
              % (A, lam, fs, fg))
        ok &= fs >= 0.80 and fg <= 0.30
    rows = [{"D": 2, "K": 3}] * 25; bad = [{"D": 4, "K": 3}] * 25
    d = (R.decide(rows)[0], R.decide(bad)[0], R.decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1952: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1952 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (R.CENSUS, R.R26, R.R36, R39)):
        print("VERDICT q1952: INCOMPLETE (input missing)"); return 3
    lines, r26, r36 = R.fibre_lines(); r39 = {r["path"]: r for r in json.load(open(R39))}
    recs = {rec["path"]: rec for rec in json.load(open(R.CENSUS)) if rec.get("path") in r26}
    rows = []
    for path, levels, counts, lam, A, a, b, flat in lines:
        sil = silent_mean(recs[path]); l2, A2 = hull(lam, A, sil)
        K, _ = R.k_rate(A2, l2, a, b)
        nu = r39[path]["nu"]
        Kp = k_poisson_merged(A2, l2); Km = k_cmp(A2, l2, nu) if nu > 1.0 else Kp
        rows.append({"path": path, "D": r26[path]["D"], "K": K, "K_poisson_S": Kp, "K_matched_S": Km, "A": A, "lam": lam,
                     "lam_S": l2, "A_S": A2, "changed": l2 < lam, "F": r36[path]["median_fano"]})
        json.dump(rows, open(OUT, "w"))
    if rows:
        f = lambda S, k: float(np.mean([r["D"] <= r[k] for r in S]))
        print("Reading: range extended in %d of %d fibres" % (sum(r["changed"] for r in rows), len(rows)))
        print("Reading: same range, Poisson K: D <= K in %.3f; matched K (one zeta): %.3f" % (f(rows, "K_poisson_S"), f(rows, "K_matched_S")))
        F = np.array([r["F"] for r in rows]); q = np.quantile(F, [1 / 3, 2 / 3])
        for name, m in (("most regular", F <= q[0]), ("middle", (F > q[0]) & (F <= q[1])), ("least regular", F > q[1])):
            sub = [r for r, mm in zip(rows, m) if mm]
            print("Reading: %s third, fibres %d, D <= K in %.3f" % (name, len(sub), f(sub, "K")))
        for name, sub in (("A >= 8", [r for r in rows if r["A"] >= 8]), ("A < 8", [r for r in rows if r["A"] < 8])):
            print("Reading: %s (q1926 A), fibres %d, D <= K in %.3f" % (name, len(sub), f(sub, "K")))
    v, why = R.decide(rows)
    print(why)
    print("VERDICT q1952 (Amendment 2297, H_stepsS): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
