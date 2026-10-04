"""
The peak-amplitude constraint (Kostal & Shinomoto Eq 1) as a free parameter L.

Q1: at fixed T = 50 ms, sweep L. Where does the capacity code turn graded?
    Pre-registered identity: L = 30/40/50 BINARY; L = 60/80/100/150/300 GRADED. Boundary between 50 and 60.
Q2: the (L, T) boundary curve. Pre-registered arithmetic T* = 2.75/L: 55 ms @50, 45.8 @60, 27.5 @100, 9.2 @300.

Protocol is window_sweep/collapse.py's, ported and control-validated in glm_contrib1/partB3: K = 21 levels,
a SINGLE energy at rho = 0.30 of peak, support tolerance 1e-3, T_total = 500 ms so n = T_total/T.
Deterministic. Run: py run_peak.py
"""

import numpy as np, json, os, sys
from core import comp_P, CV_from_P, asym_front, front_point, Qinv, fano_of_P

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
DIR = os.path.dirname(os.path.abspath(__file__))

T_TOTAL, EPS, K = 0.5, 0.01, 21
RHO = 0.30  # their average-to-peak ratio (Eq 2), our energy budget as a fraction of peak
SUPP_TOL = 1e-3
A_STAR = 2.75  # measured like-for-like in glm_contrib1/partB3, control-validated

PRED = {
    30: "BINARY",
    40: "BINARY",
    50: "BINARY",
    60: "GRADED",
    80: "GRADED",
    100: "GRADED",
    150: "GRADED",
    300: "GRADED",
}


def point(L, T, n, rho=RHO, fano=1.0):
    rates = np.linspace(0.0, float(L), K)
    P = comp_P(rates, fano, T)
    Qi = Qinv(EPS)
    E = rho * float(L)  # Eq (2): average power as a fraction of peak L
    r = asym_front(P, rates, np.array([E]))[0]
    qc = np.asarray(r["q"])
    Cc, Vc = CV_from_P(P, qc)
    if Cc <= 0:
        return None
    obj = lambda C, V: n * C - np.sqrt(max(n * V, 0.0)) * Qi
    o_cap = obj(Cc, Vc)
    f = front_point(
        P, rates, r["rate"], lambda C, V, rr: obj(C, V), p_grid=61, seed_qs=(qc,), allow_silence=False
    )
    if f is None:
        return None
    return {
        "L": float(L),
        "T_ms": T * 1000,
        "A": float(L * T),
        "n": n,
        "supp_cap": int((qc > SUPP_TOL).sum()),
        "supp_fin": int((f["q"] > SUPP_TOL).sum()),
        "gain_pct": float(100.0 * (f["obj"] - o_cap) / (n * Cc)),
        "C": float(Cc),
        "V": float(Vc),
        "rate": float(r["rate"]),
        "transmits": bool(o_cap > 0),
    }


out = {
    "config": {"T_TOTAL": T_TOTAL, "EPS": EPS, "K": K, "RHO": RHO, "A_STAR": A_STAR},
    "prediction": {str(k): v for k, v in PRED.items()},
}

# ---------------- C3 falsifier: duplicate symbols must give C = 0 ----------------
dup = np.array([30.0] * K)
Pd = comp_P(dup, 1.0, 0.05)
qd, Cd, _ = __import__("core").ba_capacity_cost(Pd, dup, 0.0, iters=800, tol=1e-11)
print(f"C3 falsifier (duplicate symbols): C = {Cd:.3e} -> {'PASS' if abs(Cd) < 1e-9 else '*** FAIL ***'}")
out["C3_falsifier_C"] = float(Cd)

# ---------------- Q1: sweep L at fixed T = 50 ms ----------------
T = 0.05
n = int(round(T_TOTAL / T))
print(f"\nQ1 — peak-amplitude sweep at T = {T*1000:.0f} ms, n = {n}, rho = {RHO}")
print(f"   pre-registered boundary: L* = A*/T = {A_STAR}/{T} = {A_STAR/T:.1f} Hz\n")
print("      L(Hz)     A=L*T   supp_cap/fin   gain%    observed   predicted   match")
rows, miss = [], []
for L in sorted(PRED):
    p = point(L, T, n)
    if p is None:
        print(f"   {L:6.0f}   ---- no transmission ----")
        continue
    obs = "GRADED" if p["supp_cap"] >= 3 else "BINARY"
    ok = obs == PRED[L]
    if not ok:
        miss.append(L)
    p["observed"] = obs
    p["predicted"] = PRED[L]
    p["match"] = bool(ok)
    rows.append(p)
    print(
        f"   {L:6.0f}   {p['A']:7.2f}     {p['supp_cap']}/{p['supp_fin']}      "
        f"{p['gain_pct']:7.2f}   {obs:<8}   {PRED[L]:<8}   {'OK' if ok else '*** MISS ***'}"
    )
out["Q1_rows"] = rows
out["Q1_misses"] = miss

# locate the boundary finely between the last binary and first graded
fine = []
print("\n   fine scan for the boundary:")
for L in np.arange(44.0, 70.1, 2.0):
    p = point(float(L), T, n)
    if p is None:
        continue
    fine.append(p)
    print(f"   {L:6.1f}   {p['A']:7.2f}     {p['supp_cap']}/{p['supp_fin']}      {p['gain_pct']:7.2f}")
out["Q1_fine"] = fine
Lb = None
for i in range(1, len(fine)):
    if fine[i - 1]["supp_cap"] == 2 and fine[i]["supp_cap"] >= 3:
        Lb = (fine[i - 1]["L"], fine[i]["L"])
        break
out["L_boundary"] = Lb
print(f"   -> boundary bracketed at L in {Lb} Hz" if Lb else "   -> no boundary found in 44-70 Hz")

# ---------------- Q2: the (L, T) boundary curve ----------------
print(f"\nQ2 — minimum counting window T* per peak rate L (pre-registered T* = {A_STAR}/L)")
print("      L(Hz)   T* predicted(ms)   T* measured(ms)   |diff|")
curve = []
for L in (50.0, 60.0, 100.0, 150.0, 300.0):
    Tpred = A_STAR / L
    found = None
    for Tms in np.arange(4.0, 121.0, 1.0):
        Tt = Tms / 1000.0
        nn = max(1, int(round(T_TOTAL / Tt)))
        p = point(L, Tt, nn)
        if p is not None and p["supp_cap"] >= 3:
            found = Tms
            break
    curve.append({"L": L, "T_star_pred_ms": Tpred * 1000, "T_star_meas_ms": found})
    d = abs(found - Tpred * 1000) if found else float("nan")
    print(
        f"   {L:6.0f}       {Tpred*1000:8.1f}          {found if found else float('nan'):8.1f}      {d:6.1f}"
    )
out["Q2_curve"] = curve

json.dump(out, open(os.path.join(DIR, "results_peak.json"), "w"), indent=1)
print("\n[saved] results_peak.json")
