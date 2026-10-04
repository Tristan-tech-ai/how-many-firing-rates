"""
Corrected window sweep: the gap measured ONLY where transmission is real.

The first pass reported 90-160% "gains" at E=3 Hz. That is meaningless: with T_total=500 ms and E=3 Hz the whole
decision carries N_sp = 1.5 spikes, phi ~ 2, and BOTH objectives are deeply negative -- the ratio of two
negative numbers is not a gain. Restrict to energies where the CAPACITY code itself achieves log M* > 0, i.e.
where a neuron actually transmits something. (Same failure mode as the clipped-ties artifact in pareto_front.)

Because T_total is held fixed, fixing E also fixes the total spike budget N_sp = T_total*E, so comparing across
T at fixed E is comparing at matched total spikes -- the correct control.

Run: py sweep_valid.py
"""

import numpy as np, json, os, sys
from core import comp_P, CV_from_P, asym_front, front_point, Qinv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DIR, "results_valid.json")
PEAK, T_TOTAL, EPS, K = 60.0, 0.5, 0.01, 21


def point(T, E, n, K=K, eps=EPS):
    rates = np.linspace(0.0, PEAK, K)
    P = comp_P(rates, 1.0, T)
    Qi = Qinv(eps)
    r = asym_front(P, rates, np.array([E]))[0]
    qc = r["q"]
    Cc, Vc = CV_from_P(P, qc)
    obj = lambda C, V: n * C - np.sqrt(max(n * V, 0.0)) * Qi
    o_cap = obj(Cc, Vc)
    f = front_point(
        P, rates, r["rate"], lambda C, V, rr: obj(C, V), p_grid=61, seed_qs=(qc,), allow_silence=False
    )
    if f is None:
        return None
    return {
        "T_ms": T * 1000,
        "n": n,
        "E_Hz": float(r["rate"]),
        "N_sp": float(n * r["rate"] * T),
        "obj_cap": float(o_cap),
        "obj_dag": float(f["obj"]),
        "transmits": bool(o_cap > 0),
        "gain_pct": float(100.0 * (f["obj"] - o_cap) / (n * Cc)) if Cc > 0 else float("nan"),
        "supp_cap": int((qc > 1e-3).sum()),
        "supp_dag": int((f["q"] > 1e-3).sum()),
        "C_cap": float(Cc),
        "V_cap": float(Vc),
        "phi": float(np.sqrt(n * Vc) * Qi / (n * Cc)) if Cc > 0 else float("nan"),
    }


Ts = [0.002, 0.005, 0.010, 0.015, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200]
Es = [16.0, 20.0, 24.0, 28.0]
rows = []
print(f"CORRECTED SWEEP — gap only where the capacity code TRANSMITS (log M* > 0).")
print(f"T_total={T_TOTAL*1000:.0f} ms fixed, K={K}, eps={EPS}, peak={PEAK:.0f} Hz\n")
print(f"{'T(ms)':>7}{'n':>5}{'A=cnt':>7} |" + "".join(f"{'E=%.0f' % e:>22}" for e in Es))
print(f"{'':>7}{'':>5}{'':>7} |" + "".join(f"{'gain%':>8}{'sc/sd':>8}{'tx':>6}" for _ in Es))
for T in Ts:
    n = max(1, int(round(T_TOTAL / T)))
    line = f"{T*1000:>7.0f}{n:>5}{PEAK*T:>7.2f} |"
    for E in Es:
        p = point(T, E, n)
        if p is None:
            line += f"{'--':>22}"
            continue
        rows.append(p)
        tx = "yes" if p["transmits"] else "NO"
        supp = "{}/{}".format(p["supp_cap"], p["supp_dag"])
        line += f"{p['gain_pct']:>8.2f}{supp:>8}{tx:>6}"
    print(line)

valid = [r for r in rows if r["transmits"]]
print("\n--- onset of the GRADED capacity code (support >= 3), among transmitting points ---")
for T in Ts:
    sub = [r for r in valid if abs(r["T_ms"] - T * 1000) < 1e-6]
    if not sub:
        continue
    sc = max(r["supp_cap"] for r in sub)
    g = max(r["gain_pct"] for r in sub)
    print(
        f"  T={T*1000:>6.0f} ms  A={PEAK*T:>5.2f} counts   max supp_cap={sc}   max gain={g:>7.2f}%   "
        f"{'GRADED' if sc >= 3 else 'binary (phenomenon absent)'}"
    )
json.dump({"rows": rows, "T_total_s": T_TOTAL, "K": K, "eps": EPS}, open(OUT, "w"), indent=1)
print(f"\n[saved] {OUT}")
