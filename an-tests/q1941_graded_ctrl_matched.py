"""q1941 (15 Oct 2026): task P2GRADEDCTRL, registered in Amendment 2292 before the simulation is run.

Question. The matched-count staircase test (Amendments 2290-2291) compares D, from q1926's tests between neighbouring sound levels, with
K computed for a Conway-Maxwell-Poisson count. Its synthetic graded control was run only for Poisson counts (q1926 registration run).
Can the test still fail for graded fibres whose counts are as regular as the recorded ones?
Design, mirroring the Poisson control of q1926: 7 sound levels, 40 presentations per level, counts Conway-Maxwell-Poisson with
zeta = 3 (q1939's cmp_pmf; zeta near the median 2.99 of the fibres). K = k_cmp(A, lam, 3) from q1939. Staircase fibres: K values evenly
spaced over [lam, lam + A], assigned monotonically to the 7 levels as in q1926. Graded fibres: 7 rates evenly spaced over [lam, lam + A].
D = count_distinct of q1926 (exact conditional binomial test, Holm at 0.05). 200 synthetic fibres per kind and point, seed 2292.
Points (A, lam) = (8, 0.5) and (15, 1).
VERDICT (H_gradedM, registered): PASS if, at both points, graded fibres meet D <= K in at most 0.30 of cases AND staircase fibres in at
least 0.80; FAIL otherwise. VOID if the sampler control fails; INCOMPLETE if q1926 or q1939 is missing.
Readings: the four fractions; K at each point.
Usage: py q1941_graded_ctrl_matched.py [--registration-run]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q1926_ph_steps import count_distinct  # noqa: E402
from q1939_ph_steps_compoisson import cmp_pmf, fano_at, k_cmp  # noqa: E402

ZETA = 3.0
N_FIB = 200
LEVELS = list(np.arange(0, 70, 10.0))
POINTS = ((8.0, 0.5), (15.0, 1.0))


def sampler(rates, zeta, n, rng):
    P = cmp_pmf(np.asarray(rates, float), zeta); y = np.arange(P.shape[1])
    return [rng.choice(y, size=n, p=row / row.sum()).astype(float) for row in P]


def fractions(A, lam, rng):
    K = k_cmp(A, lam, ZETA)
    vals = np.linspace(lam, lam + A, K); idx = np.minimum((np.arange(7) * K) // 7, K - 1); stair = vals[idx]
    graded = np.linspace(lam, lam + A, 7)
    res = {}
    for name, rates in (("staircase", stair), ("graded", graded)):
        res[name] = float(np.mean([count_distinct(LEVELS, sampler(rates, ZETA, 40, rng)) <= K for _ in range(N_FIB)]))
    return K, res


def decide(res):
    ok = all(r["graded"] <= 0.30 and r["staircase"] >= 0.80 for r in res)
    return "PASS" if ok else "FAIL"


def registration_run():
    rng = np.random.default_rng(1); ok = True
    for r in (0.7, 3.0, 9.0):
        x = sampler([r], ZETA, 40000, rng)[0]; f = x.var() / x.mean(); ft = fano_at(r, ZETA)
        print("control sampler: rate %.1f, sample mean %.3f, sample Fano %.3f, exact Fano %.3f" % (r, x.mean(), f, ft))
        ok &= abs(x.mean() - r) < 0.03 * max(r, 1) and abs(f - ft) < 0.03
    d = (decide([{"graded": 0.1, "staircase": 0.9}] * 2), decide([{"graded": 0.5, "staircase": 0.9}, {"graded": 0.1, "staircase": 0.9}]),
         decide([{"graded": 0.1, "staircase": 0.7}] * 2))
    print("control decide(): %s %s %s (expected PASS FAIL FAIL)" % d); ok &= d == ("PASS", "FAIL", "FAIL")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1941: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1941 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(os.path.join(HERE, f)) for f in ("q1926_ph_steps.py", "q1939_ph_steps_compoisson.py")):
        print("VERDICT q1941: INCOMPLETE (an instrument script is missing)"); return 3
    if not registration_run():
        print("VERDICT q1941: VOID (sampler control failed)"); return 2
    rng = np.random.default_rng(2292); res = []
    for A, lam in POINTS:
        K, r = fractions(A, lam, rng); res.append(r)
        print("Reading: A %.0f lam %.1f, K (zeta 3) %d: D <= K in %.3f of staircase fibres and %.3f of graded fibres" % (
            A, lam, K, r["staircase"], r["graded"]))
    print("VERDICT q1941 (Amendment 2292, H_gradedM): %s" % decide(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
