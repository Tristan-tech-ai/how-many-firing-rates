"""
Is the controlling variable the WINDOW T, or the maximum spike count per window A = peak_rate * T?

If the gain collapses onto a single curve in A across different peak rates, then "we chose 50 ms" is the wrong
framing: the phenomenon is governed by the spike-count dynamic range within one integration window, and any
(peak rate, window) pair with A above the onset shows it. That is a much stronger and more falsifiable claim.

Energy budget is held at a FIXED FRACTION of the peak rate so the comparison is fair across peak rates.
Run: py collapse.py
"""

import numpy as np, json, os, sys
from core import comp_P, CV_from_P, asym_front, front_point, Qinv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_collapse.json")
T_TOTAL, EPS, K = 0.5, 0.01, 21
FRAC = 0.30  # energy budget as a fraction of peak rate (E = FRAC * peak)


def point(peak, T, n, eps=EPS):
    rates = np.linspace(0.0, peak, K)
    P = comp_P(rates, 1.0, T)
    Qi = Qinv(eps)
    E = FRAC * peak
    r = asym_front(P, rates, np.array([E]))[0]
    qc = r["q"]
    Cc, Vc = CV_from_P(P, qc)
    obj = lambda C, V: n * C - np.sqrt(max(n * V, 0.0)) * Qi
    o_cap = obj(Cc, Vc)
    f = front_point(
        P, rates, r["rate"], lambda C, V, rr: obj(C, V), p_grid=61, seed_qs=(qc,), allow_silence=False
    )
    if f is None or Cc <= 0:
        return None
    return {
        "peak_Hz": peak,
        "T_ms": T * 1000,
        "A": peak * T,
        "n": n,
        "gain_pct": float(100.0 * (f["obj"] - o_cap) / (n * Cc)),
        "supp_cap": int((qc > 1e-3).sum()),
        "supp_dag": int((f["q"] > 1e-3).sum()),
        "transmits": bool(o_cap > 0),
        "N_sp": float(n * r["rate"] * T),
    }


rows = []
print(f"COLLAPSE TEST: does the gap depend on A = peak*T rather than on T?")
print(f"(T_total={T_TOTAL*1000:.0f} ms, E = {FRAC:.2f} x peak, eps={EPS}, K={K})\n")
print(f"{'peak(Hz)':>9}{'T(ms)':>8}{'A':>7}{'n':>6}{'gain%':>9}{'sc/sd':>8}{'tx':>5}")
for peak in (30.0, 60.0, 120.0, 240.0):
    for A_target in (0.6, 1.2, 2.0, 3.0, 4.5, 6.0, 9.0):
        T = A_target / peak
        if T > 0.25:
            continue  # keep the window physically sane
        n = max(1, int(round(T_TOTAL / T)))
        p = point(peak, T, n)
        if p is None:
            continue
        rows.append(p)
        supp = "{}/{}".format(p["supp_cap"], p["supp_dag"])
        print(
            f"{peak:>9.0f}{T*1000:>8.1f}{p['A']:>7.2f}{n:>6}{p['gain_pct']:>9.2f}{supp:>8}"
            f"{'yes' if p['transmits'] else 'NO':>5}"
        )

print("\n--- collapse check: gain grouped by A, across peak rates ---")
print(f"{'A':>6} |" + "".join(f"{'peak=%.0f' % p:>12}" for p in (30, 60, 120, 240)))
for A_target in (0.6, 1.2, 2.0, 3.0, 4.5, 6.0, 9.0):
    line = f"{A_target:>6.1f} |"
    for peak in (30.0, 60.0, 120.0, 240.0):
        m = [r for r in rows if abs(r["A"] - A_target) < 1e-6 and abs(r["peak_Hz"] - peak) < 1e-9]
        line += f"{m[0]['gain_pct']:>12.2f}" if m else f"{'--':>12}"
    print(line)
json.dump({"rows": rows, "frac": FRAC, "T_total_s": T_TOTAL, "eps": EPS, "K": K}, open(OUT, "w"), indent=1)
print(f"\n[saved] {OUT}")
