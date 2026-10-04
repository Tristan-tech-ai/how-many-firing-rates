# Files in sec7-data-test

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/cache_allen.py` |  | `v1/cache_allen.py` |
| `scripts/cache_dandi.py` | Build Q8-style caches for two more NLB datasets by streaming NWB from DANDI (remfile + h5py, no browser). | `cache_dandi.py` |
| `scripts/core.py` | Core machinery for the energy-vs-capacity Pareto front. | `core.py` |
| `scripts/family.py` | The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory, Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1. | `family.py` |
| `scripts/q10_retina_full.py` | PREREG_Q10.md Amendment 1: shape test on the retina's full PSTH (all bins). | `q10_retina_full.py` |
| `scripts/q10_shape.py` | shape of the continuous map. | `q10_shape.py` |
| `scripts/q11_mech.py` | threshold-exponential transduction of Gaussian input; | `q11_mech.py` |
| `scripts/q11b_mech.py` | threshold-exponential transduction of Gaussian input; | `q11b_mech.py` |
| `scripts/q11d_norm.py` | normalised map shape vs dynamic range, data vs two models. | `q11d_norm.py` |
| `scripts/q12_noise.py` | modulated-Poisson (gain-fluctuation) noise. | `q12_noise.py` |
| `scripts/q13_drift.py` | contiguous vs random trial subsampling. | `q13_drift.py` |
| `scripts/q14_heavytail.py` | heavy-tail diagnostic. | `q14_heavytail.py` |
| `scripts/q15_matched.py` | per-condition moment-matched noise vs unit-wide Fano vs data. | `q15_matched.py` |
| `scripts/q16_dependence.py` | residual ACF in session order + copula AR(1) dependence-matched simulation. | `q16_dependence.py` |
| `scripts/q16_m1fix.py` | residual ACF in session order + copula AR(1) dependence-matched simulation. | `q16_m1fix.py` |
| `scripts/q17_copula_check.py` | is the Q16 copula sampler (a = rho = 0) distributionally the same as the Q15/Q17 numpy sampler? Area 2, 3 seeds each, mean N_cv curve. | `q17_copula_check.py` |
| `scripts/q17_m1fix.py` | Q17 instrument check: seed spread of the mean N_cv curves (data and per-condition matched independent sim). | `q17_m1fix.py` |
| `scripts/q17_seeds.py` | Q17 instrument check: seed spread of the mean N_cv curves (data and per-condition matched independent sim). | `q17_seeds.py` |
| `scripts/q18_plugin.py` | plug-in artefact check. | `q18_plugin.py` |
| `scripts/q19_shrink.py` | shrinkage-seeded comparator vs data vs plug-in comparator. | `q19_shrink.py` |
| `scripts/q20_analytic.py` | one-constant analytic resolution rule. | `q20_analytic.py` |
| `scripts/q21_discrete_own.py` | discrete-own (N* levels on the unit's shrunken map) vs continuous-own (shrunken map), matched noise. | `q21_discrete_own.py` |
| `scripts/q21_retina.py` | PREREG_Q21.md Amendment 1: own-map discrete vs continuous comparators for the retina data. | `q21_retina.py` |
| `scripts/q21_retina_plugin.py` | PREREG_Q21.md Amendment 1: own-map discrete vs continuous comparators for the retina data. | `q21_retina_plugin.py` |
| `scripts/q23_pilot.py` | predict N_cv at m=32 and m=all from a 16-trial pilot (shrunk and plug-in seeds). | `q23_pilot.py` |
| `scripts/q25_deweese.py` | binary spiking vs low-rate Poisson in CRCNS ac-2 (anaesthetised rat A1, whole-cell). | `q25_deweese.py` |
| `scripts/q26_aggregate.py` | Aggregate the Q26 contrast test across all completed Allen functional-connectivity sessions. | `q26_aggregate.py` |
| `scripts/q26_contrast.py` | grating contrast as the graded axis, Allen FC session 779839471, VISp good units. | `q26_contrast.py` |
| `scripts/q26_verdict.py` | PREREG_Q26 verdict from results_q26.json: per-orientation contrast maps and the 36-cell map. | `q26_verdict.py` |
| `scripts/q27_class.py` | plateau units = cell class or SNR floor? Logistic regression, CV AUC, bootstrap. | `q27_class.py` |
| `scripts/q29_universal.py` | cross-dataset universality of the normalised within-neuron rate map. | `q29_universal.py` |
| `scripts/q8_levels.py` | Q8 test per PREREG_Q8.md: K1 shuffle control -> K2 power simulation -> data N_cv(u, W, m) -> verdict. | `q8_levels.py` |
| `scripts/q8_levels_rep.py` | Q8 test per PREREG_Q8.md: K1 shuffle control -> K2 power simulation -> data N_cv(u, W, m) -> verdict. | `q8_levels_rep.py` |
| `scripts/q8_levels_rep_fixed.py` | Q8 test per PREREG_Q8.md: K1 shuffle control -> K2 power simulation -> data N_cv(u, W, m) -> verdict. | `q8_levels_rep_fixed.py` |
| `scripts/q8_popsum.py` | level count of the SUMMED population vs N*(A_pop). | `q8_popsum.py` |
| `scripts/q8_retina.py` | Q8 on salamander retina (PREREG_Q8_retina.md + Amendment 1). | `q8_retina.py` |
| `scripts/q8_verdict.py` | Mechanical verdict for Q8 per PREREG_Q8.md. | `q8_verdict.py` |
| `scripts/q8b_mixture.py` | Q8b per PREREG_Q8b.md: within-condition mixture test on powered (unit, W) pairs from results_q8.json. | `q8b_mixture.py` |
| `scripts/support_exact.py` | Certified support size N*(A, lam) of the capacity-achieving input of Y ~ Poisson(lam + x), 0 <= x <= A. | `support_exact.py` |
| `scripts/support_exact_v3.py` | certified support of the capacity-achieving input of Y ~ Poisson(lam + x), 0 <= x <= A, for large A. | `support_exact_v3.py` |
| `scripts/support_gauss_v3.py` | certified support of the capacity-achieving input of the amplitude-constrained AWGN channel Y = x + Z, Z ~ N(0, 1), 0 <= x <= a. | `support_gauss_v3.py` |
| `scripts/support_general.py` | N*(A, lam, Fano) by cutting-plane + KKT on the COM-Poisson family. | `support_general.py` |

`logs/` holds 33 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
