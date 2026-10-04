# Files in sec6-short-windows/energy-capacity-front

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/core.py` | Core machinery for the energy-vs-capacity Pareto front. | `pareto_front/core.py` |
| `scripts/family.py` | The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory, Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1. | `pareto_front/family.py` |
| `scripts/figure.py` | Builds the figure from results.json. | `pareto_front/figure.py` |
| `scripts/glm.py` | GLM spike-train channel, EXACT (no Monte Carlo, so no error bars are needed). | `pareto_front/glm.py` |
| `scripts/pins.py` | Pins that could fail. | `pareto_front/pins.py` |
| `scripts/run.py` | Main computation: energy-vs-capacity Pareto fronts. | `pareto_front/run.py` |

`logs/` holds 2 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
