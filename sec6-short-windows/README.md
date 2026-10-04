# Section VI: short windows

Without an energy budget the long-window code is on-off up to the first count change, A = 3.3679, and graded beyond it. Under an
energy budget at A = 3 it is graded at some budgets and gains at most 0.8 % there. For short windows the normal approximation to the
finite-blocklength rate was maximized by search; its maximizer was binary in all 185 cases computed, many of them outside the
approximation's accuracy range. Each subfolder is one line of work, with its own copies of the shared modules (`core.py`,
`family.py`, `glm.py`) so that every script runs against the modules it was written with.

| subfolder | paper item | main scripts | results |
|---|---|---|---|
| [`exact-binary-onset/`](exact-binary-onset/) | the exact onset A = 3.3679 and the recertified graded-vs-binary classification | `binary_cert2.py`, `glm_recert2.py`, `oneshot_refine.py` | `results_binary_cert.json`, `results_glm_recert2.json`, `CYCLE2_READOUT.md` |
| [`glm-and-families/`](glm-and-families/) | the 185 short-window cases (the Poisson neuron and the refractory GLM) and the Lagrangian sweep of Section VI-D | `partA_hull.py`, `partA2_lagrangian.py`, `partB_glm.py`, `partB2_onset.py`, `partB3_likeforlike.py` | `glm_contrib1__results*.json` |
| [`energy-capacity-front/`](energy-capacity-front/) | the energy-capacity fronts (Fig. 3) | `run.py`, `pins.py`, `figure.py` | `pareto_front__results.json`, `pareto_front__pins.json` |
| [`window-sweep/`](window-sweep/) | the window length at fixed decision time, and the collapse onto the peak count A | `sweep.py`, `sweep_valid.py`, `collapse.py` | `window_sweep__results*.json` |
| [`peak-amplitude/`](peak-amplitude/) | the peak rate as a free parameter (the comparison with Kostal and Shinomoto) | `run_peak.py` | `peak_amplitude__results_*.json` |

`CYCLE2_READOUT.md` records the 2 September correction: the graded onset near A = 2.7 read from Blahut-Arimoto mass was an artefact,
and the exact binary-optimality check puts it at A = 3.3679. The short-window fronts come from a search and carry no certificate,
as the paper states.

See the `FILES.md` of each subfolder for every script.
