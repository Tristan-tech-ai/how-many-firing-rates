# Files in sec4-non-poisson

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/q101_genpoisson_wall_chain.py` | the sixth family, generalized Poisson (Consul), dispersion index CONSTANT in mu. | `q101_genpoisson_wall_chain.py` |
| `scripts/q134_symmetric_binomial_mirror.py` | Symmetric binomial channel Bin(n, p): continuation in n with the MIRROR-REDUCED KKT system (symmetry imposed, so the antisymmetric soft mode of a central pitchfork is absent). | `q134_symmetric_binomial_mirror.py` |
| `scripts/q135_symmetric_binomial_mp.py` | symmetric binomial channel Bin(n, p) on p in [0, 1], mirror-reduced KKT system in mpmath (dps 30), continuation in n. | `q135_symmetric_binomial_mp.py` |
| `scripts/q98_binomial_wall_chain.py` | the binomial WALL by the half-line chain (the setting of Q74/Q75 with the binomial kernel). | `q98_binomial_wall_chain.py` |
| `scripts/q99_negbin_wall_chain.py` | the fifth family, over-dispersed: negative binomial NB(r, mean mu), variance mu + mu^2/r. | `q99_negbin_wall_chain.py` |

`logs/` holds 3 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
