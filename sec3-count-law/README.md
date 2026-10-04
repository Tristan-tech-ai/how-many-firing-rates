# Section III, Table II and Fig. 1: the count law

The paper conjectures, and tests for five to twenty-four rates, that in long windows the number of rates of the Poisson neuron equals
the number of levels of the unit-noise Gaussian channel, read at the half-length sqrt(A) shortened by a wall constant c(K).

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| Gaussian count changes A_g(K) (Table II) | `q143_gaussian_transitions.py`, `q84_gauss_symmetric_chain.py`, `q89_gauss_ba.py` | `results_q143_Ag_all.json`, `results_q143_Ag.log`, `results_q89_gauss_ba.json` |
| A_g(15) to A_g(26) relocated at 128 bits (the values printed in Table II) | `c60_cap_fullchain.py` | `c60_K14.json` to `c60_K25.json` with their logs, `c61_out.txt` |
| the Gaussian side of the correspondence | `q33_gauss_support.py` | |
| wall constant c(K) (Table II) | (producing script not kept) | `results_q89_cK_table.json` |
| growth of the count, predictions made in advance | `q173_conjecture_table.py`, `q181_growth_verdict.py` | |

`c60_cap_fullchain.py` is run once per event as `python c60_cap_fullchain.py K A0 DA PREC TAG`; the file `c60_K<K>.json` holds the
event for the count K before it (K = 14 gives A_g(15)). `c61_out.txt` lists the twelve relocated values next to the earlier stored ones;
the two differ by at most 7e-5, below the precision of the wall constants in Table II.

`q33_gauss_support_split_FAILED.py` is the failed first version of `q33_gauss_support.py`, kept because the record refers to it.
The certified counts that bracket the Poisson count changes A_K of Table II are in [`../sec3-certificates/`](../sec3-certificates/).

See [`FILES.md`](FILES.md) for every script.
