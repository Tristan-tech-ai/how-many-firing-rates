# Files in sec3-count-law

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/build_table2.py` | Build Table II of paper 2 (count changes and wall constant) from primary output files only. | `gate_truth/build_table2.py` |
| `scripts/c60_cap_fullchain.py` | full-chain solver for the amplitude-constrained Gaussian CAPACITY on [-A, A] (symmetric), and bisection-quality relocation of its transitions. | `centre_scripts/c60_cap_fullchain.py` |
| `scripts/q143_gaussian_transitions.py` | transitions A_g(K+1) of the unit-noise Gaussian channel Y = X + Z on [-A, A] for large K, by a mirror-reduced KKT solver (float, scipy hybr) with a structured guess (wall profile of the stored K = 25 Gaussian point of q84, interior uniform) | `q143_gaussian_transitions.py` |
| `scripts/q173_conjecture_table.py` | the two growth conjectures and the measured Gaussian staircase, written as predictions for the symmetric binomial transitions n*(K) at K = 55..72, so the thirty-digit transitions can be compared the moment they land (Amendments 251, 253b, 2 | `q173_conjecture_table.py` |
| `scripts/q181_growth_verdict.py` | the growth-conjecture verdict from the first DIRECT transition, and the pre-registered predictions for the next one. | `q181_growth_verdict.py` |
| `scripts/q33_gauss_support.py` | the Gaussian side of the correspondence: N_Gauss(sqrt A) against N_Poisson(A). | `q33_gauss_support.py` |
| `scripts/q33_gauss_support_split_FAILED.py` | the Gaussian side of the correspondence: N_Gauss(sqrt A) against N_Poisson(A). | `q33_gauss_support_split_FAILED.py` |
| `scripts/q34_poisson_dark.py` | the offset c as a function of dark current lambda. | `q34_poisson_dark.py` |
| `scripts/q80_AK_by_unfolding.py` | the transition amplitude A_{K+1} defined by the newborn weight -> 0, from the 40-digit unfolding. | `q80_AK_by_unfolding.py` |
| `scripts/q84_gauss_symmetric_chain.py` | the Gaussian birth chain in the SYMMETRIC parametrisation (right half only), which removes the antisymmetric zero mode that stalls the full-system Newton at a centre birth (pitchfork). | `q84_gauss_symmetric_chain.py` |
| `scripts/q89_gauss_ba.py` | Gaussian optimal input from scratch: Blahut-Arimoto on a fine symmetric input grid, then clustering, symmetric Newton refinement (q84 functions) and a violation check. | `q89_gauss_ba.py` |
| `scripts/support_mp3.py` | KKT Newton with an ANALYTIC JACOBIAN (2026-09-05). | `support_mp3.py` |
| `scripts/support_unfold.py` | Unfolding the birth of an atom: KKT Newton with the NEW atom's weight fixed at eps and the amplitude A free. | `support_unfold.py` |

`logs/` holds 82 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
