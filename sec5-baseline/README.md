# Section V: a baseline rate

With a baseline (dark-current) rate lambda the neuron fires even in its silent state. By the shift identity the channel becomes
the Poisson channel on [lambda, A + lambda], and the wall constant moves from its Poisson value to the Gaussian one.

| paper item | scripts | outputs in `logs/` |
|---|---|---|
| the offset c as a function of lambda | `q34_poisson_dark.py` | |
| certified counts with a baseline | `q35_dark_certify.py` (with the certificate modules it imports) | `results_q35_dark_certify.json` |
| the 25 published count changes with a baseline, predicted with no free parameter | (producing script not kept) | `results_q111_bd_comparison.json` |

The published values are those of Barletta and Dytso (ICC 2022), whose data files are public at
[github.com/ucando83/PoissonCapacity](https://github.com/ucando83/PoissonCapacity). `results_q111_bd_comparison.json` has 25 rows,
one per published count change, each starting with lambda and K. The six changes to three and four rates lie below K = 5, outside
the range on which the chain law was fitted.

See [`FILES.md`](FILES.md) for every script.
