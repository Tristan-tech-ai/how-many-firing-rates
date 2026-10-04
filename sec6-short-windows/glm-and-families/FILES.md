# Files in sec6-short-windows/glm-and-families

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/collapse.py` | Is the controlling variable the WINDOW T, or the maximum spike count per window A = peak_rate * T? If the gain collapses onto a single curve in A across different peak rates, then "we chose 50 ms" is the wrong framing: the phenomenon is gov | `window_sweep/collapse.py` |
| `scripts/core.py` | Core machinery for the energy-vs-capacity Pareto front. | `glm_contrib1/core.py` |
| `scripts/family.py` | The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory, Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1. | `glm_contrib1/family.py` |
| `scripts/glm.py` | GLM spike-train channel, EXACT (no Monte Carlo, so no error bars are needed). | `glm_contrib1/glm.py` |
| `scripts/partA2_lagrangian.py` | PART A2 — stop arguing from the hull: RUN the Lagrangian on this objective and watch which energies it skips. | `glm_contrib1/partA2_lagrangian.py` |
| `scripts/partA_hull.py` | PART A — is the "no optimum at 4-20 Hz" a LAGRANGIAN ARTIFACT? Demonstrated, not argued. | `glm_contrib1/partA_hull.py` |
| `scripts/partB2_onset.py` | PART B2 — the Part B prediction FAILED. | `glm_contrib1/partB2_onset.py` |
| `scripts/partB3_likeforlike.py` | PART B3 — like-for-like onset, under window_sweep's OWN protocol. | `glm_contrib1/partB3_likeforlike.py` |
| `scripts/partB_glm.py` | PART B — does contribution #1 survive at the SPIKE-TRAIN level? Contribution #1 (the only top-ranked survivor of the occupancy pass): under an energy budget the CAPACITY-optimal input is GRADED (>=3 active levels) while the FINITE-BLOCKLENG | `glm_contrib1/partB_glm.py` |
| `scripts/run.py` | Main computation: energy-vs-capacity Pareto fronts. | `pareto_front/run.py` |

`logs/` holds 5 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
