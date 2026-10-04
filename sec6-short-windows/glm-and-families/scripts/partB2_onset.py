"""
PART B2 — the Part B prediction FAILED. Measure the replacement constant.

Part B verdict stands unrevised: B-NO-EFFECT, the named identity missed on 3 of 4 kernels, and the
realised-A mechanism is wrong in the direction OPPOSITE to the one predicted. This script does not repair
that prediction. It measures the quantity the prediction was built on, for the channel it actually failed on.

A* ~= 2.7 was measured in window_sweep/ on a POISSON count channel. Question: what is the onset for a GLM,
and in which currency is it stable -- nominal (input alphabet) or realised (what the neuron emits)?

Sweep the window T (which moves A continuously) per kernel; locate the smallest A at which the CAPACITY
support flips 2 -> >=3. Report in both currencies against induced Fano. Pre-registered readings and the
"roughly constant = spread <= 15% of mean" rule are in PREREG.md (amendment).

Also runs the support-robustness control on the Part B verdict itself: GRADED must survive a 1% and 2%
input-mass tolerance, else the Part B verdict is withdrawn as inflated support.

Deterministic. Run: py partB2_onset.py
"""

import numpy as np, json, os, sys
from core import asym_front, front_point, Qinv, fano_of_P, mean_rate_cost
from glm import glm_count_P

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
DIR = os.path.dirname(os.path.abspath(__file__))

K = 12
RATES = np.linspace(0, 60, K)
EPS = 0.01
KB_COUNT = 25
KERNELS = [
    (None, "h0"),
    ({"amp": 2.0, "tau_ms": 4.0}, "amp2_tau4"),
    ({"amp": 4.0, "tau_ms": 4.0}, "amp4_tau4"),
    ({"amp": 8.0, "tau_ms": 4.0}, "amp8_tau4"),
]
TOLS = [1e-3, 1e-2, 2e-2]  # support tolerances for the robustness control


def max_supp_on_front(Pc, cost, T, tol):
    """Largest capacity-optimal support over an energy grid spanning this channel's front."""
    top = float(cost.max())
    E_grid = top * np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65, 0.80])
    rows = asym_front(Pc, cost, E_grid)
    return max(int((np.asarray(r["q"]) > tol).sum()) for r in rows)


def run():
    print("PART B2 — measuring the GLM onset after the Part B prediction failed\n")
    out = {"tols": TOLS, "kernels": {}, "robustness": {}}

    # ---------------- support-robustness control on the Part B verdict ----------------
    print("ROBUSTNESS CONTROL on the Part B GRADED verdict (T = 50 ms, the Part B configuration):")
    print("  a level must hold this share of the INPUT mass to count toward the support\n")
    print("  kernel        tol=1e-3   tol=1e-2   tol=2e-2   verdict survives 1%?")
    for h, tag in KERNELS:
        Pc = glm_count_P(RATES, h, 0.05, KB_COUNT)
        cost = mean_rate_cost(Pc, 0.05)
        s = [max_supp_on_front(Pc, cost, 0.05, t) for t in TOLS]
        surv = s[1] >= 3
        print(
            f"  {tag:<12}  {s[0]:^8d}   {s[1]:^8d}   {s[2]:^8d}   {'YES' if surv else '*** NO — WITHDRAW ***'}"
        )
        out["robustness"][tag] = {"supp_by_tol": s, "survives_1pct": bool(surv)}

    # ---------------- the onset sweep ----------------
    Ts = np.concatenate([np.arange(0.008, 0.030, 0.002), np.arange(0.030, 0.085, 0.005)])
    print(
        f"\nONSET SWEEP — {len(Ts)} windows from {Ts[0]*1000:.0f} to {Ts[-1]*1000:.0f} ms, "
        f"capacity support at tol=1e-2 (the 1% level, not the loose one)\n"
    )

    for h, tag in KERNELS:
        recs = []
        for T in Ts:
            Pc = glm_count_P(RATES, h, float(T), KB_COUNT)
            cost = mean_rate_cost(Pc, float(T))
            A_nom = float(RATES.max() * T)
            A_real = float(cost.max() * T)
            supp = max_supp_on_front(Pc, cost, float(T), 1e-2)
            fano = float(fano_of_P(Pc, np.eye(K)[-1]))
            recs.append(
                {"T": float(T), "A_nominal": A_nom, "A_realised": A_real, "supp": supp, "fano_top": fano}
            )
        # onset = first T (ascending) at which supp >= 3 and it never drops back below on the rest of the grid
        onset = None
        for i, r in enumerate(recs):
            if r["supp"] >= 3 and all(x["supp"] >= 3 for x in recs[i:]):
                onset = r
                break
        out["kernels"][tag] = {"sweep": recs, "onset": onset}
        if onset:
            print(
                f"  {tag:<12} onset at T = {onset['T']*1000:5.1f} ms   "
                f"A_nominal = {onset['A_nominal']:.3f}   A_realised = {onset['A_realised']:.3f}   "
                f"induced Fano(top) = {onset['fano_top']:.3f}"
            )
        else:
            print(
                f"  {tag:<12} NO ONSET in the swept range "
                f"(supp at smallest T = {recs[0]['supp']}, at largest = {recs[-1]['supp']})"
            )

    # ---------------- which currency is stable? ----------------
    ons = [out["kernels"][t]["onset"] for _, t in KERNELS]
    print("\n" + "=" * 78)
    if all(o is not None for o in ons):
        nom = np.array([o["A_nominal"] for o in ons])
        rea = np.array([o["A_realised"] for o in ons])
        sp_n = float((nom.max() - nom.min()) / nom.mean())
        sp_r = float((rea.max() - rea.min()) / rea.mean())
        print(f"A*_nominal  across kernels: {np.round(nom,3)}  mean {nom.mean():.3f}  spread {sp_n*100:.1f}%")
        print(f"A*_realised across kernels: {np.round(rea,3)}  mean {rea.mean():.3f}  spread {sp_r*100:.1f}%")
        if sp_n <= 0.15 and sp_r > 0.15:
            v = (
                f"NOMINAL-STABLE — the onset is a property of the INPUT ALPHABET, at "
                f"A*_nominal = {nom.mean():.2f} (spread {sp_n*100:.1f}%). Refractoriness is irrelevant to it; "
                f"charging the realised rate is right for ENERGY but wrong for the onset."
            )
        elif sp_r <= 0.15 and sp_n > 0.15:
            v = (
                f"REALISED-STABLE — A*_realised = {rea.mean():.2f} (spread {sp_r*100:.1f}%). Part B used the "
                f"right currency and the wrong constant; this is the replacement."
            )
        elif sp_n <= 0.15 and sp_r <= 0.15:
            v = (
                f"BOTH STABLE — A*_nominal = {nom.mean():.2f} ({sp_n*100:.1f}%), "
                f"A*_realised = {rea.mean():.2f} ({sp_r*100:.1f}%); the kernels do not separate them here."
            )
        else:
            v = (
                f"FANO-DEPENDENT — neither currency is constant (nominal {sp_n*100:.1f}%, "
                f"realised {sp_r*100:.1f}%); the onset is a two-parameter object A*(Fano)."
            )
        out["currency_verdict"] = v
        out["A_star_nominal"] = nom.tolist()
        out["A_star_realised"] = rea.tolist()
        out["spread_nominal"] = sp_n
        out["spread_realised"] = sp_r
    else:
        v = "NO-ONSET for at least one kernel in the swept range."
        out["currency_verdict"] = v
    print("\nVERDICT: " + v)
    json.dump(out, open(os.path.join(DIR, "resultsB2.json"), "w"), indent=1)
    print("\n[saved] resultsB2.json")


if __name__ == "__main__":
    run()
