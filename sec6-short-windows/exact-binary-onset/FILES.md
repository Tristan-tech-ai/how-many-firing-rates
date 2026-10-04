# Files in sec6-short-windows/exact-binary-onset

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/binary_cert2.py` |  | `binary_cert2.py` |
| `scripts/core.py` | Core machinery for the energy-vs-capacity Pareto front. | `core.py` |
| `scripts/family.py` | The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory, Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1. | `family.py` |
| `scripts/glm.py` | GLM spike-train channel, EXACT (no Monte Carlo, so no error bars are needed). | `glm.py` |
| `scripts/glm_recert2.py` | Recertify Part B capacity-code classification on a FINE grid (0.5 Hz) with a continuum-sense certificate: BINARY iff a 2-level code {0} U {j, j+1} (time-sharing between ADJACENT grid levels allowed = one interior level) reaches C_BA within  | `glm_recert2.py` |
| `scripts/oneshot_refine.py` | Arm 3 refinement: exact one-shot MAP codebooks on a finer rate grid. | `oneshot_refine.py` |

`logs/` holds 3 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
