# Section VII and Table III: do recorded neurons use discrete levels?

The test was written down before it was run and is the same for every dataset. For each tuned unit, the number of firing levels
that best predicts held-out trials is counted as the number of trials grows. A neuron that uses a few discrete rates should stay
flat at that number; a continuous rate map should keep growing. Each unit is simulated both ways with its own noise and trial
count, and only units whose two simulations separate are scored. A shuffle control must give one level.

| step | scripts | results |
|---|---|---|
| fetch the public data | `cache_dandi.py`, `cache_allen.py` | |
| level counts and their growth with trials | `q8_levels.py`, `q8_levels_rep.py`, `q8_levels_rep_fixed.py`, `q8_popsum.py`, `q8b_mixture.py` | `results_q8*.json` |
| the verdict | `q8_verdict.py` | `results_q8_verdict.json`, `results_q8_rep_verdict.json` |
| support sizes of the optimal codes used as the discrete hypothesis | `support_exact_v3.py`, `support_gauss_v3.py`, `support_general.py` | `results_q30_*.json` |
| robustness and mechanism checks (noise, drift, heavy tails, dependence, matched comparators, own-map comparator) | `q10_*.py` to `q29_universal.py` | `results_q1*.json`, `results_q2*.json` |

`READOUT_Q8.md` is the full readout of these runs, with every verdict and its pre-registration.

## Data

MC_Maze, Area2_Bump and DMFC_RSG (Neural Latents Benchmark; DANDI 000128, 000127, 000130), larval salamander retina (Dryad
10.5061/dryad.4qrfj6qm8), mouse V1 (Allen Visual Coding Neuropixels, DANDI 000021) and mouse A1 (DANDI 000986). The data are not
copied here.

See [`FILES.md`](FILES.md) for every script.
