# Section IV: neurons more regular or more variable than Poisson

A neuron whose spike count is more regular (binomial) or more variable (negative binomial, generalized Poisson) than Poisson keeps
the Poisson wall at silence and adds one finite-size term: the count is read at h_eff = h_F - kappa (h_F - q_0 sqrt(A)) / K, with
kappa = 0.42 +- 0.02 (equation (6) of the paper).

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| wall chains of the three families | `q98_binomial_wall_chain.py`, `q99_negbin_wall_chain.py`, `q101_genpoisson_wall_chain.py` | |
| symmetric binomial channel, count changes by continuation in n (the two registered predictions) | `q134_symmetric_binomial_mirror.py`, `q135_symmetric_binomial_mp.py` | `results_q135_sym_bin_c.log`, `results_q135_sym_bin_from190.json` |
| the finite-size coefficient kappa | (producing script not kept) | `results_q138_kappa_points.json` |

See [`FILES.md`](FILES.md) for every script.
