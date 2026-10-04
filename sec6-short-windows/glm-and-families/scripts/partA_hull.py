"""
PART A — is the "no optimum at 4-20 Hz" a LAGRANGIAN ARTIFACT? Demonstrated, not argued.

A Lagrangian sweep  max_q [ obj(q) - lam*rate(q) ]  over any grid of lam >= 0 can only ever return points
lying on the UPPER CONCAVE ENVELOPE of the front F(E). Points strictly below that envelope are unreachable
for every lam. So: build F(E) with the direct rate constraint, compute its concave majorant on the same grid,
and report where F sits strictly below it.

Controls (both can fail):
  C2  the ASYMPTOTIC front is a capacity-cost function and is PROVABLY CONCAVE. The identical instrument run
      on it must report NO gap. A gap there means the instrument is broken and Part A is void.
  C3  the saved configuration must reproduce pareto_front/results.json (C=0.5944, V=0.2224, supp 3-4 vs 2).

Deterministic. Run: py partA_hull.py
"""

import numpy as np, json, os, sys
from core import asym_front, comp_P, front_point, Qinv, CV_from_P, fano_of_P

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))

# ---- configuration IDENTICAL to pareto_front/run.py so C3 is a real reproduction check ----
T = 0.05
K = 12
RATES = np.linspace(0, 60, K)
EPS = 0.01
E_TARGETS = np.concatenate([np.linspace(0.4, 10, 13), np.linspace(11.5, 29, 11)])

# thresholds FIXED IN PREREG.md before computing
GAP_FLOOR = 1e-6  # nats; below this is numerical noise
MIN_BAND = 2  # consecutive grid points required to call it a band, not optimiser wobble


def concave_majorant(E, F):
    """Upper concave envelope of the points (E_i, F_i), evaluated on the same grid.
    This is EXACTLY the set of values a Lagrangian sweep can reach: for slope lam the Lagrangian selects the
    point where the supporting line of slope lam touches the front, so it traces the concave majorant and
    never a point strictly beneath it. Computed as the upper convex hull via a monotone chain on slopes."""
    E = np.asarray(E, float)
    F = np.asarray(F, float)
    ok = np.isfinite(E) & np.isfinite(F)
    idx = np.argsort(E[ok])
    xs = E[ok][idx]
    ys = F[ok][idx]
    hull = []  # build upper hull: keep only right turns (concave)
    for x, y in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # cross product; if the middle point is on/below the chord, it is not on the upper hull
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    hx = np.array([p[0] for p in hull])
    hy = np.array([p[1] for p in hull])
    return np.interp(E, hx, hy, left=np.nan, right=np.nan), (hx, hy)


def bands(E, gap):
    """Contiguous runs of >= MIN_BAND grid points where gap > GAP_FLOOR."""
    hit = np.isfinite(gap) & (gap > GAP_FLOOR)
    out, i = [], 0
    while i < len(hit):
        if hit[i]:
            j = i
            while j + 1 < len(hit) and hit[j + 1]:
                j += 1
            if (j - i + 1) >= MIN_BAND:
                out.append((float(E[i]), float(E[j]), int(j - i + 1), float(np.nanmax(gap[i : j + 1]))))
            i = j + 1
        else:
            i += 1
    return out


def run():
    print("PART A — Lagrangian reachability of the direct-constraint front\n")
    print(f"channel: COM-Poisson Fano=1.0 (Poisson), T={T*1000:.0f} ms, K={K} levels 0-60 Hz, eps={EPS}")
    print(f"thresholds fixed in PREREG: gap > {GAP_FLOOR:g} nats, band >= {MIN_BAND} consecutive points\n")

    P = comp_P(RATES, 1.0, T)
    rows = asym_front(P, RATES, E_TARGETS)
    E = np.array([r["rate"] for r in rows])
    C_as = np.array([r["C"] for r in rows])
    V_as = np.array([r["V"] for r in rows])
    supp_as = np.array([int((r["q"] > 1e-3).sum()) for r in rows])

    # ---------------- C3: reproduction of the saved run ----------------
    iC = int(np.argmax(C_as))
    print("C3 reproduction check against pareto_front/results.json:")
    print(
        f"   capacity point: C={C_as[iC]:.4f} (paper 0.5944)  V={V_as[iC]:.4f} (paper 0.2224)  "
        f"at rate {E[iC]:.2f} Hz (paper 28.30)"
    )
    c3_C = abs(C_as[iC] - 0.5944) < 5e-4
    c3_V = abs(V_as[iC] - 0.2224) < 5e-4
    print(f"   -> C {'PASS' if c3_C else 'FAIL'} | V {'PASS' if c3_V else 'FAIL'}")

    res = {
        "config": {"T": T, "K": K, "eps": EPS, "gap_floor": GAP_FLOOR, "min_band": MIN_BAND},
        "E": E.tolist(),
        "C_asym": C_as.tolist(),
        "V_asym": V_as.tolist(),
        "supp_asym": supp_as.tolist(),
        "C3": {
            "C": float(C_as[iC]),
            "V": float(V_as[iC]),
            "rate": float(E[iC]),
            "pass_C": bool(c3_C),
            "pass_V": bool(c3_V),
        },
        "byN": {},
    }

    # ---------------- C2: instrument validation on a PROVABLY CONCAVE object ----------------
    # the asymptotic front n*C(E) is a capacity-cost function -> concave by theorem (not by my construction)
    print("\nC2 instrument control — hull run on the ASYMPTOTIC front (provably concave, must show NO gap):")
    for n in (10, 50):
        env_a, _ = concave_majorant(E, n * C_as)
        gap_a = env_a - n * C_as
        bad = bands(E, gap_a)
        worst = float(np.nanmax(gap_a))
        print(
            f"   n={n:<3} worst gap = {worst:.3e} nats   bands found = {len(bad)}   "
            f"-> {'PASS' if (worst <= GAP_FLOOR and not bad) else '*** FAIL — INSTRUMENT BROKEN ***'}"
        )
        res.setdefault("C2", {})[str(n)] = {"worst_gap": worst, "n_bands": len(bad)}

    # ---------------- the actual test: finite-N front vs its concave majorant ----------------
    Qi = Qinv(EPS)
    for n in (10, 50):
        F, supp_f = [], []
        for r in rows:
            f = front_point(
                P,
                RATES,
                r["rate"],
                lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi,
                seed_qs=(r["q"],),
                allow_silence=False,
            )
            if f is None:
                F.append(np.nan)
                supp_f.append(-1)
            else:
                F.append(float(f["obj"]))
                supp_f.append(int((f["q"] > 1e-3).sum()))
        F = np.array(F)
        supp_f = np.array(supp_f)
        env, (hx, hy) = concave_majorant(E, F)
        gap = env - F
        bd = bands(E, gap)

        print(f"\n--- n = {n} ---")
        print("    E(Hz)   F(E) direct    concave env      gap      supp_asym  supp_finN   Lagrangian?")
        for e, f, v, g, sa, sf in zip(E, F, env, gap, supp_as, supp_f):
            reach = "reachable" if not (np.isfinite(g) and g > GAP_FLOOR) else "UNREACHABLE"
            print(f"   {e:6.2f}  {f:11.4f}  {v:12.4f}  {g:9.5f}      {sa:^7d}   {sf:^7d}    {reach}")
        print(f"   bands (>= {MIN_BAND} consecutive pts, gap > {GAP_FLOOR:g}): {bd if bd else 'NONE'}")
        overlap = [b for b in bd if not (b[1] < 4.0 or b[0] > 20.0)]
        print(f"   bands overlapping 4-20 Hz: {overlap if overlap else 'NONE'}")

        res["byN"][str(n)] = {
            "F_direct": F.tolist(),
            "concave_env": env.tolist(),
            "gap": gap.tolist(),
            "supp_finN": supp_f.tolist(),
            "bands": bd,
            "bands_overlapping_4_20": overlap,
            "hull_vertices_E": hx.tolist(),
        }

    # ---------------- pre-registered verdict ----------------
    print("\n" + "=" * 78)
    verdicts = {}
    for n in ("10", "50"):
        b = res["byN"][n]
        ov = b["bands_overlapping_4_20"]
        if ov:
            v = "A-CONFIRMED"
        elif b["bands"]:
            v = "A-AMBIGUOUS (non-concave, but not overlapping 4-20 Hz)"
        else:
            v = "A-REFUTED (front is concave on this grid — diagnosis was wrong)"
        verdicts[n] = v
        print(f"n={n}: {v}")
    res["verdict"] = verdicts
    json.dump(res, open(os.path.join(DIR, "resultsA.json"), "w"), indent=1)
    print(f"\n[saved] resultsA.json")


if __name__ == "__main__":
    run()
