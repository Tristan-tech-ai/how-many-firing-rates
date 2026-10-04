"""
PART A2 — stop arguing from the hull: RUN the Lagrangian on this objective and watch which energies it skips.

Part A showed F(E) sits strictly below its concave majorant across a band. That is an argument that a
Lagrangian cannot reach those energies. This script is the direct demonstration: sweep lam over a fine grid,
maximise  obj(q) - lam*rate(q)  with the SAME machinery and the SAME objective used to build F(E), and record
the mean rate actually achieved at each lam. If a band of energies is never achieved for any lam, the
Lagrangian demonstrably skips it — no hull reasoning required.

Also adds the SECOND SIDE of the instrument control:
  C2  (in partA_hull.py) proved the hull code does not INVENT a gap on a provably concave function.
  C2b (here)             proves the hull code FINDS a gap that is really there, on a hand-computable case.
  Together they are two-sided. C2 alone is not.

Deterministic. Run: py partA2_lagrangian.py
"""

import numpy as np, json, os, sys
from core import asym_front, comp_P, front_point, Qinv
from partA_hull import concave_majorant, GAP_FLOOR

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))

T = 0.05
K = 12
RATES = np.linspace(0, 60, K)
EPS = 0.01
E_TARGETS = np.concatenate([np.linspace(0.4, 10, 13), np.linspace(11.5, 29, 11)])


def c2b_two_sided_control():
    """KNOWN POSITIVE, computable by hand: points (0,0), (1,0), (2,2).
    The upper concave envelope at x=1 is the chord from (0,0) to (2,2), whose value there is exactly 1.0.
    So the true gap at x=1 is 1.0 - 0.0 = 1.0 EXACTLY. The instrument must report that."""
    E = np.array([0.0, 1.0, 2.0])
    F = np.array([0.0, 0.0, 2.0])
    env, _ = concave_majorant(E, F)
    gap = env - F
    expect = np.array([0.0, 1.0, 0.0])
    err = float(np.nanmax(np.abs(gap - expect)))
    ok = err < 1e-12
    print("C2b two-sided control — KNOWN NON-CONCAVE case, gap computable by hand:")
    print(f"   points (0,0),(1,0),(2,2) -> envelope {np.round(env,12)}, gap {np.round(gap,12)}")
    print(f"   expected gap [0, 1, 0]; max abs error {err:.2e} -> {'PASS' if ok else '*** FAIL ***'}")
    return ok, err


def run():
    print("PART A2 — direct Lagrangian sweep on the SAME objective\n")
    ok_c2b, err_c2b = c2b_two_sided_control()

    P = comp_P(RATES, 1.0, T)
    rows = asym_front(P, RATES, E_TARGETS)
    E_grid = np.array([r["rate"] for r in rows])
    Qi = Qinv(EPS)
    seeds = tuple(r["q"] for r in rows)
    E_max = float(RATES.max())

    res = {"C2b": {"pass": bool(ok_c2b), "max_abs_err": err_c2b}, "byN": {}}

    for n in (10, 50):
        # ---- the direct-constraint front, for reference (same as Part A) ----
        F = []
        for r in rows:
            f = front_point(
                P,
                RATES,
                r["rate"],
                lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi,
                seed_qs=(r["q"],),
                allow_silence=False,
            )
            F.append(float(f["obj"]) if f is not None else np.nan)
        F = np.array(F)

        # ---- the LAGRANGIAN sweep: unconstrained max of [obj(q) - lam*rate(q)] ----
        lams = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 120)])
        achieved = []
        for lam in lams:
            f = front_point(
                P,
                RATES,
                E_max,  # E_max => the rate constraint is inactive; lam does the work
                lambda C, V, rr, _l=lam: n * C - np.sqrt(max(n * V, 0)) * Qi - _l * rr,
                seed_qs=seeds,
                allow_silence=False,
            )
            achieved.append(float(f["rate"]) if f is not None else np.nan)
        achieved = np.array(achieved)
        got = np.sort(achieved[np.isfinite(achieved)])

        # ---- which grid energies does the Lagrangian never land near? ----
        tol = 0.25  # Hz; a grid energy counts as "reached" if some lam lands within this
        reached = np.array([bool(np.any(np.abs(got - e) <= tol)) for e in E_grid])
        skipped = E_grid[~reached]

        print(f"\n--- n = {n} ---")
        print(f"   lam grid: {len(lams)} values, 0 to {lams[-1]:g}")
        print(f"   distinct achieved rates (Hz): {np.round(np.unique(np.round(got,3)),2)}")
        print(f"   grid energies NEVER reached by any lam (tol {tol} Hz): {np.round(skipped,2)}")
        if len(skipped):
            print(
                f"   -> contiguous skipped span: {skipped.min():.2f} - {skipped.max():.2f} Hz "
                f"({len(skipped)} of {len(E_grid)} grid points)"
            )
        ov = skipped[(skipped >= 4.0) & (skipped <= 20.0)]
        print(f"   skipped energies inside 4-20 Hz: {np.round(ov,2) if len(ov) else 'NONE'}")

        res["byN"][str(n)] = {
            "lams": lams.tolist(),
            "achieved_rate_Hz": achieved.tolist(),
            "E_grid": E_grid.tolist(),
            "reached": reached.tolist(),
            "skipped_Hz": skipped.tolist(),
            "skipped_in_4_20": ov.tolist(),
            "F_direct": F.tolist(),
        }

    json.dump(res, open(os.path.join(DIR, "resultsA2.json"), "w"), indent=1)
    print("\n[saved] resultsA2.json")


if __name__ == "__main__":
    run()
