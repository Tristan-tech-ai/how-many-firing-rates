# Section III, Table II and Fig. 1: the count law

The paper conjectures, and tests for five to twenty-four rates, that in long windows the number of rates of the Poisson neuron equals
the number of levels of the unit-noise Gaussian channel, read at the half-length sqrt(A) shortened by a wall constant c(K).

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| Gaussian count changes A_g(K) (Table II) | `q143_gaussian_transitions.py`, `q84_gauss_symmetric_chain.py`, `q89_gauss_ba.py` | `results_q143_Ag_all.json`, `results_q143_Ag.log`, `results_q89_gauss_ba.json` |
| A_g(15) to A_g(26) relocated at 128 bits | `c60_cap_fullchain.py` | `c60_K14.json` to `c60_K25.json` with their logs, `c61_out.txt` |
| Table II, every value, built only from the output files below | `build_table2.py` | `table2_primary.json` |
| A_K by the newborn-weight unfolding (K = 11, 13 and 14 rerun on 4 Oct 2026) | `q80_AK_by_unfolding.py` with `support_unfold.py`, `support_mp3.py`, `q34_poisson_dark.py` | `results_q80_A_transition_*.json`, `results_q80_gate_K*.log` |
| A_g(4) to A_g(7) and A_g(14) relocated at 128 bits (4 Oct 2026) | `c60_cap_fullchain.py` | `c60_gate_K*.json` with their logs |
| the Gaussian side of the correspondence | `q33_gauss_support.py` | |
| earlier table of wall constants, superseded by `table2_primary.json` (it mixed older A_g values with back-computed A_K) | (producing script not kept) | `results_q89_cK_table.json` |
| growth of the count, predictions made in advance | `q173_conjecture_table.py`, `q181_growth_verdict.py` | |

`c60_cap_fullchain.py` is run once per event as `python c60_cap_fullchain.py K A0 DA PREC TAG`; the file `c60_K<K>.json` holds the
event for the count K before it (K = 14 gives A_g(15)). `c61_out.txt` lists the twelve relocated values next to the earlier stored ones;
the two differ by at most 7e-5, below the precision of the wall constants in Table II.

`q33_gauss_support_split_FAILED.py` is the failed first version of `q33_gauss_support.py`, kept because the record refers to it.
The certified counts that bracket the Poisson count changes A_K of Table II are in [`../sec3-certificates/`](../sec3-certificates/).

See [`FILES.md`](FILES.md) for every script.
