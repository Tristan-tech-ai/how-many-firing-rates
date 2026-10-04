# Files in sec6-short-windows/window-sweep

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/collapse.py` | Is the controlling variable the WINDOW T, or the maximum spike count per window A = peak_rate * T? If the gain collapses onto a single curve in A across different peak rates, then "we chose 50 ms" is the wrong framing: the phenomenon is gov | `window_sweep/collapse.py` |
| `scripts/core.py` | Core machinery for the energy-vs-capacity Pareto front. | `window_sweep/core.py` |
| `scripts/family.py` | The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory, Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1. | `window_sweep/family.py` |
| `scripts/sweep.py` | Window sweep: is contribution #1 an artifact of the 50 ms window? Control: hold TOTAL decision time T_total = n*T fixed, vary the window T (so n = T_total/T). | `window_sweep/sweep.py` |
| `scripts/sweep_valid.py` | Corrected window sweep: the gap measured ONLY where transmission is real. | `window_sweep/sweep_valid.py` |

`logs/` holds 3 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
