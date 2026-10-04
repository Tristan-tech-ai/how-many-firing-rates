"""PREREG_Q26 verdict from results_q26.json: per-orientation contrast maps and the 36-cell map."""

import json, os, sys, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results_q26.json")))
out = {}
for W in ("0.25", "0.5"):
    print(f"\n===== window {W} s =====")
    # (a) per-orientation maps
    ents = [e for u in R["units"] for e in u["W"][W]["per_orientation"] if "Ncv_by_m" in e]
    tuned = [e for e in ents if e["Ncv_by_m"][-1] >= 2 and e.get("K1_shuffle_Ncv", 1) == 1]
    print(
        f"(a) per-orientation maps: {len(ents)} with A>=0.05, tuned (N_cv(all)>=2, K1 pass) {len(tuned)}; N* distribution among tuned: {dict(zip(*np.unique([e['Nstar'] for e in tuned], return_counts=True)))}"
        if tuned
        else "(a) no tuned maps"
    )
    if tuned:
        nall = np.array([e["Ncv_by_m"][-1] for e in tuned])
        n8 = np.array([e["Ncv_by_m"][0] for e in tuned])
        ns = np.array([e["Nstar"] for e in tuned])
        stair = np.mean(nall <= ns + 1)
        graded = np.mean(nall > ns + 1)
        growth = np.mean(nall > n8)
        print(
            f"    N_cv(all) mean {nall.mean():.2f} (max possible 9); P-stair fraction (N_cv <= N*+1) {100*stair:.0f}%; P-graded fraction {100*graded:.0f}%; growth 8->15 {100*growth:.0f}%"
        )
        for tag in ("shrunk", "plugin"):
            cont = np.array([e["comp"][tag]["cont"][-1] for e in tuned])
            disc = np.array([e["comp"][tag]["disc"][-1] for e in tuned])
            near = np.mean(np.abs(cont - nall) < np.abs(disc - nall)) + 0.5 * np.mean(
                np.abs(cont - nall) == np.abs(disc - nall)
            )
            print(
                f"    {tag:6s} seeds: continuous-own mean {cont.mean():.2f}, discrete-own mean {disc.mean():.2f} vs data {nall.mean():.2f}; units nearer continuous {100*near:.0f}%"
            )
        verdict_a = (
            "P-graded" if (graded >= 0.6 and growth >= 0.6) else ("P-stair" if stair >= 0.6 else "P-mixed")
        )
        print(f"    VERDICT (a): {verdict_a}")
        out[f"{W}|a"] = {
            "n_tuned": len(tuned),
            "Ncv_all_mean": float(nall.mean()),
            "stair": float(stair),
            "graded": float(graded),
            "growth": float(growth),
            "verdict": verdict_a,
        }
    # (b) 36-cell maps
    m36 = [u["W"][W]["map36"] for u in R["units"] if u["W"][W]["map36"]]
    t36 = [e for e in m36 if e["Ncv_by_m"][-1] >= 2]
    if t36:
        nall = np.array([e["Ncv_by_m"][-1] for e in t36])
        n8 = np.array([e["Ncv_by_m"][0] for e in t36])
        ns = np.array([e["Nstar"] for e in t36])
        print(
            f"(b) 36-cell maps: {len(m36)} with A>=0.05, tuned {len(t36)}; N_cv(all) mean {nall.mean():.2f}; plateau at N*+1 or below {100*np.mean(nall <= ns+1):.0f}%; growth 8->15 {100*np.mean(nall > n8):.0f}%"
        )
        for tag in ("shrunk", "plugin"):
            cont = np.array([e["comp"][tag]["cont"][-1] for e in t36])
            disc = np.array([e["comp"][tag]["disc"][-1] for e in t36])
            near = np.mean(np.abs(cont - nall) < np.abs(disc - nall)) + 0.5 * np.mean(
                np.abs(cont - nall) == np.abs(disc - nall)
            )
            print(
                f"    {tag:6s} seeds: continuous-own mean {cont.mean():.2f}, discrete-own mean {disc.mean():.2f} vs data {nall.mean():.2f}; units nearer continuous {100*near:.0f}%"
            )
        out[f"{W}|b"] = {
            "n_tuned": len(t36),
            "Ncv_all_mean": float(nall.mean()),
            "near_cont_shrunk": float(
                np.mean(
                    np.abs(np.array([e["comp"]["shrunk"]["cont"][-1] for e in t36]) - nall)
                    < np.abs(np.array([e["comp"]["shrunk"]["disc"][-1] for e in t36]) - nall)
                )
            ),
        }
json.dump(out, open(os.path.join(HERE, "results_q26_verdict.json"), "w"), indent=1)
print("\n[saved] results_q26_verdict.json")
