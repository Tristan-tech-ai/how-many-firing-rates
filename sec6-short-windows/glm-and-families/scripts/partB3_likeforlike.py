"""
PART B3 — like-for-like onset, under window_sweep's OWN protocol.

Part B2 measured A*_realised ~ 1.45 for GLMs. A mandatory control then showed my instrument returns
A* ~ 1.5 on the COM-Poisson Fano=1.0 channel, where window_sweep measured A* ~ 2.7 on the SAME channel.
So "2.7 -> 1.45" was apples-to-oranges and is withdrawn.

The instruments differ in three ways, read off window_sweep/collapse.py:
    window_sweep : K = 21 levels, a SINGLE energy E = 0.30 * peak, support tolerance 1e-3
    mine (B2)    : K = 12 levels, MAX support over an energy grid 0.10-0.80 of peak, tolerance 1e-2
Taking a max over a range of energies asks "does a graded code exist ANYWHERE on the front"; a single
energy asks "is the code graded AT this budget". Mine necessarily fires earlier. Neither is wrong.

This script ports window_sweep's protocol verbatim and re-measures, so the Poisson and GLM numbers are
comparable. The Poisson arm is a CONTROL: it must return ~2.7, a number I did not produce. If it does not,
my port is broken and no GLM comparison may be drawn from it.

Method note: for the GLM the bin WIDTH is held fixed at 2 ms (so Kbins = round(T/0.002)), matching Part B's
25 bins over 50 ms, so the refractory kernel is discretised consistently as T varies. The Poisson arm uses
comp_P, an exact unbinned count channel, exactly as collapse.py does -- so a residual binned-vs-unbinned
difference remains between the two arms and is stated rather than hidden.

Deterministic. Run: py partB3_likeforlike.py
"""

import numpy as np, json, os, sys
from core import comp_P, CV_from_P, asym_front, front_point, Qinv, mean_rate_cost, fano_of_P
from glm import glm_count_P

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
DIR = os.path.dirname(os.path.abspath(__file__))

# ---- window_sweep/collapse.py protocol, verbatim ----
T_TOTAL, EPS, K = 0.5, 0.01, 21
FRAC = 0.30  # energy budget as a fraction of peak rate
SUPP_TOL = 1e-3  # collapse.py uses (q > 1e-3).sum()
BIN_W = 0.002  # GLM bin width held fixed at 2 ms

KERNELS = [
    (None, "h0"),
    ({"amp": 2.0, "tau_ms": 4.0}, "amp2_tau4"),
    ({"amp": 4.0, "tau_ms": 4.0}, "amp4_tau4"),
    ({"amp": 8.0, "tau_ms": 4.0}, "amp8_tau4"),
]
A_GRID = [0.6, 1.2, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.5, 6.0]
PEAK = 60.0


def point(kernel, peak, T, n):
    """One (channel, window) point under collapse.py's protocol. kernel=None -> exact COM-Poisson channel."""
    rates = np.linspace(0.0, peak, K)
    if kernel == "poisson":
        P = comp_P(rates, 1.0, T)
        cost = rates.copy()
    else:
        Kb = max(4, int(round(T / BIN_W)))
        P = glm_count_P(rates, kernel, T, Kb)
        cost = mean_rate_cost(P, T)
    Qi = Qinv(EPS)
    E = FRAC * float(cost.max())  # budget as a fraction of the REALISED peak
    r = asym_front(P, cost, np.array([E]))[0]
    qc = np.asarray(r["q"])
    Cc, Vc = CV_from_P(P, qc)
    if Cc <= 0:
        return None
    obj = lambda C, V: n * C - np.sqrt(max(n * V, 0.0)) * Qi
    o_cap = obj(Cc, Vc)
    f = front_point(
        P, cost, r["rate"], lambda C, V, rr: obj(C, V), p_grid=61, seed_qs=(qc,), allow_silence=False
    )
    if f is None:
        return None
    return {
        "A_nominal": float(peak * T),
        "A_realised": float(cost.max() * T),
        "T_ms": T * 1000,
        "n": n,
        "supp_cap": int((qc > SUPP_TOL).sum()),
        "supp_dag": int((f["q"] > SUPP_TOL).sum()),
        "gain_pct": float(100.0 * (f["obj"] - o_cap) / (n * Cc)),
        "transmits": bool(o_cap > 0),
        "fano_top": float(fano_of_P(P, np.eye(K)[-1])),
    }


def onset_of(rows):
    """First A (ascending) where supp_cap >= 3 and it stays >= 3 for the rest of the grid."""
    for i, r in enumerate(rows):
        if r["supp_cap"] >= 3 and all(x["supp_cap"] >= 3 for x in rows[i:]):
            return r
    return None


def run():
    print("PART B3 — like-for-like onset under window_sweep's own protocol")
    print(f"(K={K}, E={FRAC:.2f} x peak, tol={SUPP_TOL:g}, T_total={T_TOTAL*1000:.0f} ms, eps={EPS})\n")
    out = {
        "protocol": {
            "K": K,
            "FRAC": FRAC,
            "SUPP_TOL": SUPP_TOL,
            "T_TOTAL": T_TOTAL,
            "EPS": EPS,
            "BIN_W": BIN_W,
            "peak_Hz": PEAK,
        },
        "arms": {},
    }

    arms = [("poisson", "COM-Poisson Fano=1.0 [CONTROL, must give ~2.7]")] + [
        (k, f"GLM {t}") for k, t in KERNELS
    ]
    tags = ["poisson"] + [t for _, t in KERNELS]

    for (kern, label), tag in zip(arms, tags):
        rows = []
        for A in A_GRID:
            T = A / PEAK
            n = max(1, int(round(T_TOTAL / T)))
            p = point(kern, PEAK, T, n)
            if p is not None:
                rows.append(p)
        on = onset_of(rows)
        out["arms"][tag] = {"label": label, "rows": rows, "onset": on}
        print(f"=== {label} ===")
        print("   A_nom   A_real   T(ms)    n   supp_cap/dag   gain%    tx")
        for r in rows:
            print(
                f"  {r['A_nominal']:6.2f}  {r['A_realised']:6.2f}  {r['T_ms']:6.1f} {r['n']:4d}"
                f"      {r['supp_cap']}/{r['supp_dag']}      {r['gain_pct']:7.2f}  "
                f"{'yes' if r['transmits'] else 'NO'}"
            )
        if on:
            print(
                f"  -> ONSET at A_nominal = {on['A_nominal']:.2f}, A_realised = {on['A_realised']:.2f}, "
                f"induced Fano(top) = {on['fano_top']:.3f}\n"
            )
        else:
            print("  -> NO ONSET in the swept range\n")

    # ---------------- the control verdict ----------------
    print("=" * 78)
    pon = out["arms"]["poisson"]["onset"]
    if pon is None:
        print("CONTROL FAILED — no onset found on the Poisson channel. Port is broken; no GLM comparison.")
        out["control_pass"] = False
    else:
        A_p = pon["A_nominal"]
        ok = 2.4 <= A_p <= 3.0  # window_sweep: flips between A=2.50 and 2.75
        out["control_pass"] = bool(ok)
        print(
            f"CONTROL: Poisson onset A = {A_p:.2f}  vs window_sweep's 2.50-2.75  "
            f"-> {'PASS — port reproduces a number I did not produce' if ok else '*** FAIL — port differs ***'}"
        )
        if ok:
            print("\nLike-for-like GLM onsets under the SAME protocol:")
            nom, rea, fan = [], [], []
            for _, t in KERNELS:
                o = out["arms"][t]["onset"]
                if o:
                    nom.append(o["A_nominal"])
                    rea.append(o["A_realised"])
                    fan.append(o["fano_top"])
                    print(
                        f"   {t:<12} A_nominal = {o['A_nominal']:.2f}   A_realised = {o['A_realised']:.2f}"
                        f"   Fano(top) = {o['fano_top']:.3f}"
                    )
                else:
                    print(f"   {t:<12} NO ONSET in range")
            if nom:
                nom = np.array(nom)
                rea = np.array(rea)
                print(
                    f"\n   Poisson A_nom = {A_p:.2f}   |   GLM A_nom mean = {nom.mean():.2f} "
                    f"(spread {100*(nom.max()-nom.min())/nom.mean():.1f}%)"
                )
                print(
                    f"   {'':>18}   |   GLM A_real mean = {rea.mean():.2f} "
                    f"(spread {100*(rea.max()-rea.min())/rea.mean():.1f}%)"
                )
                out["poisson_onset_A_nom"] = float(A_p)
                out["glm_onset_A_nom"] = nom.tolist()
                out["glm_onset_A_real"] = rea.tolist()
    json.dump(out, open(os.path.join(DIR, "resultsB3.json"), "w"), indent=1)
    print("\n[saved] resultsB3.json")


if __name__ == "__main__":
    run()
