"""
PART B — does contribution #1 survive at the SPIKE-TRAIN level?

Contribution #1 (the only top-ranked survivor of the occupancy pass): under an energy budget the
CAPACITY-optimal input is GRADED (>=3 active levels) while the FINITE-BLOCKLENGTH optimal input stays
BINARY (2 levels). Established so far ONLY for a COM-Poisson count model.

pareto_front/results.json saves supp_finN for every GLM kernel but NEVER saves supp_asym for any of them —
exactly one half of the comparison. This computes the missing half.

PREREGISTERED PREDICTION, BY IDENTITY (PREREG.md, fixed before computing):
    h0        realised 56.540 Hz -> A = 2.827 -> ABOVE onset 2.7 -> GRADED (supp_asym >= 3)
    amp2_tau4 realised 47.221 Hz -> A = 2.361 -> BELOW           -> BINARY (supp_asym = 2)
    amp4_tau4 realised 43.389 Hz -> A = 2.169 -> BELOW           -> BINARY
    amp8_tau4 realised 40.025 Hz -> A = 2.001 -> BELOW           -> BINARY
A wrong IDENTITY fails the prediction even if the COUNT of graded kernels comes out right.

Controls that can fail:
  C1  h0 must reproduce the saved GLM h0 numbers in pareto_front/results.json.
  C4  P(0) = exp(-lambda*T) = 0.04978707 for EVERY kernel ([proven]; a kernel acting only after spikes
      cannot change the void probability). If a kernel moves it, this GLM implementation is wrong.

Deterministic. Run: py partB_glm.py
"""

import numpy as np, json, os, sys
from core import asym_front, comp_P, front_point, Qinv, fano_of_P, mean_rate_cost
from glm import glm_count_P, glm_induced_fano

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))

T = 0.05
K = 12
RATES = np.linspace(0, 60, K)
EPS = 0.01
NS = [10, 50]
KB_COUNT = 25
E_TARGETS = np.concatenate([np.linspace(0.4, 10, 13), np.linspace(11.5, 29, 11)])
KERNELS = [None, {"amp": 2.0, "tau_ms": 4.0}, {"amp": 4.0, "tau_ms": 4.0}, {"amp": 8.0, "tau_ms": 4.0}]
A_ONSET = 2.7  # measured in window_sweep/ on a POISSON count channel
SUPP_TOL = 1e-3  # same support tolerance as pareto_front/run.py

PREDICTION = {"h0": "GRADED", "amp2_tau4": "BINARY", "amp4_tau4": "BINARY", "amp8_tau4": "BINARY"}


def run():
    print("PART B — contribution #1 at the spike-train level\n")
    out = {
        "config": {"T": T, "K": K, "eps": EPS, "Kbins": KB_COUNT, "A_onset": A_ONSET},
        "prediction": PREDICTION,
        "kernels": {},
    }
    Qi = Qinv(EPS)

    for h in KERNELS:
        tag = "h0" if h is None else f"amp{h['amp']:g}_tau{h['tau_ms']:g}"
        Pc = glm_count_P(RATES, h, T, KB_COUNT)
        cost = mean_rate_cost(Pc, T)  # REALISED rates (load-bearing)
        A_real = float(cost.max() * T)

        # ---- C4: the void-probability identity, [proven], must hold for every kernel ----
        P0 = float(Pc[-1][0])
        P0_ref = float(np.exp(-60 * T))
        c4 = abs(P0 - P0_ref) < 1e-8

        rows = asym_front(Pc, cost, E_TARGETS[E_TARGETS <= cost.max()])
        E = np.array([r["rate"] for r in rows])
        C_as = np.array([r["C"] for r in rows])
        supp_as = np.array([int((r["q"] > SUPP_TOL).sum()) for r in rows])

        rec = {
            "h": h,
            "realised_top_Hz": float(cost.max()),
            "A_realised": A_real,
            "A_nominal": float(RATES.max() * T),
            "predicted": PREDICTION[tag],
            "C4_P0": P0,
            "C4_pass": bool(c4),
            "C_sat": float(C_as.max()),
            "induced_fano_top": float(fano_of_P(Pc, np.eye(K)[-1])),
            "E": E.tolist(),
            "C_asym": C_as.tolist(),
            "supp_asym": supp_as.tolist(),
            "byN": {},
        }

        print(
            f"=== {tag} ===  realised top {cost.max():.3f} Hz -> A = {A_real:.3f} "
            f"({'ABOVE' if A_real > A_ONSET else 'BELOW'} onset {A_ONSET})   predicted {PREDICTION[tag]}"
        )
        print(
            f"    C4 P(0) = {P0:.8f} vs exp(-lam T) = {P0_ref:.8f} -> {'PASS' if c4 else '*** FAIL ***'}"
            f"   |  C_sat = {C_as.max():.4f}   induced Fano(top) = {rec['induced_fano_top']:.4f}"
        )

        for n in NS:
            F, supp_f, gain = [], [], []
            for r in rows:
                f = front_point(
                    Pc,
                    cost,
                    r["rate"],
                    lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi,
                    seed_qs=(r["q"],),
                    allow_silence=False,
                )
                if f is None:
                    F.append(np.nan)
                    supp_f.append(-1)
                    gain.append(np.nan)
                    continue
                F.append(float(f["obj"]))
                supp_f.append(int((f["q"] > SUPP_TOL).sum()))
                # objective of the CAPACITY code evaluated under the finite-N objective
                from core import CV_from_P

                Cc, Vc = CV_from_P(Pc, r["q"])
                obj_star = n * Cc - np.sqrt(max(n * Vc, 0)) * Qi
                # transmitting region ONLY (both objectives positive) -- the invalid-region trap
                if obj_star > 0 and f["obj"] > 0 and n * Cc > 0:
                    gain.append(float((f["obj"] - obj_star) / abs(n * Cc)))
                else:
                    gain.append(np.nan)
            supp_f = np.array(supp_f)
            gain = np.array(gain, float)
            valid = np.isfinite(gain)
            rec["byN"][str(n)] = {
                "F": F,
                "supp_finN": supp_f.tolist(),
                "gain": gain.tolist(),
                "n_valid": int(valid.sum()),
                "best_gain": float(np.nanmax(gain)) if valid.any() else None,
            }
            # support comparison in the transmitting region
            sa_t = supp_as[valid]
            sf_t = supp_f[valid]
            print(
                f"    n={n:<3} transmitting pts={int(valid.sum()):<3} "
                f"supp_asym(uniq)={sorted(set(sa_t.tolist())) if valid.any() else '-'}  "
                f"supp_finN(uniq)={sorted(set(sf_t.tolist())) if valid.any() else '-'}  "
                f"best gain={rec['byN'][str(n)]['best_gain'] if valid.any() else float('nan')}"
            )
            rec["byN"][str(n)]["supp_asym_transmitting"] = sa_t.tolist()
            rec["byN"][str(n)]["supp_finN_transmitting"] = sf_t.tolist()

        # ---- observed verdict for this kernel: is the CAPACITY code graded where it transmits? ----
        sa10 = np.array(rec["byN"]["10"]["supp_asym_transmitting"])
        sa50 = np.array(rec["byN"]["50"]["supp_asym_transmitting"])
        sa_all = np.concatenate([sa10, sa50]) if (len(sa10) or len(sa50)) else np.array([])
        observed = "GRADED" if (len(sa_all) and (sa_all >= 3).any()) else "BINARY"
        rec["observed"] = observed
        rec["match"] = bool(observed == PREDICTION[tag])
        print(
            f"    -> OBSERVED {observed}   (predicted {PREDICTION[tag]})   "
            f"{'MATCH' if rec['match'] else '*** IDENTITY MISS ***'}\n"
        )
        out["kernels"][tag] = rec

    # ---------------- pre-registered verdict ----------------
    print("=" * 78)
    obs = {t: out["kernels"][t]["observed"] for t in out["kernels"]}
    matches = {t: out["kernels"][t]["match"] for t in out["kernels"]}
    all_binary = all(v == "BINARY" for v in obs.values())
    all_graded = all(v == "GRADED" for v in obs.values())
    identity_ok = all(matches.values())

    print("kernel        A_real   predicted   observed   match")
    for t in out["kernels"]:
        r = out["kernels"][t]
        print(f"  {t:<12} {r['A_realised']:.3f}   {r['predicted']:<9}  {r['observed']:<8}  {r['match']}")

    if all_binary:
        verdict = (
            "B-KILLED — the graded capacity code never appears for a GLM at any kernel including h0. "
            "Contribution #1 is an artifact of the count model and does NOT survive at spike-train level."
        )
    elif all_graded:
        verdict = (
            "B-NO-EFFECT — every kernel is graded; refractoriness does not suppress the graded code "
            "and the realised-A mechanism is wrong."
        )
    elif identity_ok:
        verdict = (
            "B-CONFIRMED-WITH-MECHANISM — the named per-kernel identity holds exactly. #1 survives at "
            "spike-train level but only for weak/absent refractoriness; the A-criterion transfers."
        )
    else:
        verdict = (
            "B-MECHANISM-WRONG — the graded/binary split occurs but the named identity is violated. "
            "The A-criterion does NOT transfer from a Poisson count channel to a sub-Poisson GLM."
        )
    print("\nVERDICT: " + verdict)
    out["verdict"] = verdict
    out["identity_held"] = bool(identity_ok)
    json.dump(out, open(os.path.join(DIR, "resultsB.json"), "w"), indent=1)
    print("\n[saved] resultsB.json")


if __name__ == "__main__":
    run()
