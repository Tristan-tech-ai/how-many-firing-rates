"""Within-condition Fano factor of the six datasets of paper 2, Section V (Table III), from the q8 results files.

The same rows q8_verdict.py analyses: every (unit, window) pair with an N_cv curve; "powered" as stored by the q8 run
(K2 separation >= 0.8). Prints and writes the median Fano factor over powered pairs and over all pairs, per dataset.
Writes fano_medians.json next to this file. Reads only; never writes into temporal_within/.
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
SETS = [("M1", "results_q8_rep.json"), ("area 2", "results_q8_area2.json"), ("DMFC", "results_q8_dmfc.json"),
        ("retina", "results_q8_retina.json"), ("V1", "results_q8_v1.json"), ("A1", "results_q8_a1.json")]


def main():
    out = {}
    for name, fn in SETS:
        R = json.load(open(os.path.join(TW, fn), encoding="utf-8"))
        WS = [str(w) for w in R["config"]["W"]]
        rows = [u["W"][W] for u in R["units"] for W in WS if "Ncv_by_m" in u["W"].get(W, {})]
        pw = [r for r in rows if r["powered"]]
        f_all = np.array([r["fano"] for r in rows], float)
        f_pw = np.array([r["fano"] for r in pw], float)
        out[name] = {"source": fn, "pairs": len(rows), "powered_pairs": len(pw),
                     "median_fano_powered": float(np.nanmedian(f_pw)), "median_fano_all": float(np.nanmedian(f_all)),
                     "q25_fano_powered": float(np.nanpercentile(f_pw, 25)), "q75_fano_powered": float(np.nanpercentile(f_pw, 75))}
        print(f"{name:8s} pairs {len(rows):5d} powered {len(pw):5d}  median Fano powered {out[name]['median_fano_powered']:.4f}"
              f"  all {out[name]['median_fano_all']:.4f}  IQR powered {out[name]['q25_fano_powered']:.3f}-{out[name]['q75_fano_powered']:.3f}")
    # the COM-Poisson count at those Fano factors: results_noiseNstar.json (support_general.py, grid FS below, A <= 12)
    FS = (0.7, 0.9, 1.0, 1.13, 1.3, 1.5)
    N = json.load(open(os.path.join(TW, "results_noiseNstar.json"), encoding="utf-8"))
    i1 = FS.index(1.0)
    used = [f for f in FS if 0.9 <= f <= 1.5]
    dmax = max(abs(v[FS.index(f)] - v[i1]) for v in N.values() for f in used)
    never_raised = all(v[FS.index(f)] <= v[i1] for v in N.values() for f in FS if f > 1.0)
    lowered = sum(1 for v in N.values() for f in FS if f > 1.0 and v[FS.index(f)] < v[i1])
    out["com_poisson_count"] = {"fano_grid_used": used, "cells": len(N), "max_abs_diff_from_poisson": dmax,
                                "above_poisson_never_raises": never_raised, "above_poisson_cells_lowered": lowered}
    print("COM-Poisson count, Fano 0.9..1.5 on the grid:", out["com_poisson_count"])
    json.dump(out, open(os.path.join(HERE, "fano_medians.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
