# READOUT — Q8: do motor-cortex neurons use the discrete firing levels optimal-code theory predicts?
Date 2026-09-02. Files: `PREREG_Q8.md`, `q8_levels.py`, `q8_verdict.py`, `results_q8.json`, `results_q8_verdict.json`.

## Verdict
- **By the pre-registered rule: Q8-MIXED.** Plateau 46%, growth 54%, neither reached the 70% bar.
- **By the pre-registered comparator (K2 simulations, per pair): the data are indistinguishable from a
  continuous tuning curve carrying the unit's own noise, at every window, and nowhere near the discrete
  prediction.** This reading is post hoc in the sense that the PREREG did not name it as an outcome rule;
  it is not post hoc in the sense that every number in it was fixed and computed before the verdict.

## Controls (all pass)
| control | result |
|---|---|
| K1 shuffle: condition labels permuted | N_cv = 1 in **529 / 529** pairs. The statistic finds no levels in noise |
| K2 power: discrete vs smooth simulation separate at m = all in >= 80% | **469 / 529** pairs powered, 178 / 182 units |
| K3 instrument | `support_exact.py` reproduces A* = 3.3679 |
| comparator behaviour | discrete sims: growth 17%, mean N_cv 2.63 (they plateau at N*); smooth sims: growth 89%, mean N_cv 5.58 |

## The numbers (powered pairs)
| quantity | data | discrete sim (same pairs) | smooth sim (same pairs) |
|---|---|---|---|
| mean N_cv at all trials | **5.48** | 2.61 | 5.79 |
| W = 100 ms, tuned pairs (n = 121) | 5.93 | 2.66 | 5.91 |
| W = 200 ms, tuned pairs (n = 130) | 6.32 | 2.30 | 6.40 |
| W = 400 ms, tuned pairs (n = 150) | 6.43 | 2.21 | 6.58 |
| A + lambda >= 3.4, tuned (n = 66): data vs N* | **7.15** vs N* = 2.14 | 2.16 | 6.93 |
| plateau at 2 among low-A pairs (n = 402) | **2%** | — | — |
| nearer which simulation, tuned pairs (n = 401) | smooth **308 (77%)**, discrete 89 (22%), tie 4 | | |

N_cv(all) histogram, powered: 1: 68, 2: 31, 3: 31, 4: 42, 5: 18, 6: 43, 7: 55, **8 (ceiling): 181**.
N* histogram, powered: 2: 460, 3: 9.

## Why the pre-registered rule returned MIXED, and why that is my design flaw
1. **Ceiling.** NMAX = 8 was too low: 181 pairs sit at N = 8 with all trials, and 62 of them were already
   at 8 with 32 trials, so "growth := N_cv(all) > N_cv(32)" cannot register growth that has saturated. Those
   count as "plateau", but a plateau at the ceiling is the opposite of a plateau at N* = 2.
2. **Untuned pairs.** 68 powered pairs have N_cv = 1 (no resolvable tuning). The +-1 match admits them as
   "plateau at N* = 2". Among tuned pairs, plateau-and-match is **5%**.
Both flaws push toward "plateau", i.e. toward the theory. Correcting either would move the verdict toward
SMOOTH, not away from it. A replication with NMAX = 16 is pre-registered as Amendment 1 and running.

## What this says
[confirmed numerically, 178 units, 3 windows, one dataset] In macaque M1 during delayed reaching, the
per-condition tuning of single neurons across 36 conditions is continuous at the resolution 55-75 trials
afford, at 100, 200 and 400 ms windows. The single-neuron optimal-code prediction of discrete levels
(Bethge 2002/2003: binary below a peak count of ~3; Nikitin 2009 / Dytso 2024: N*(A) levels) is **not
observed**: the predicted 2-level plateau appears in 2% of low-count pairs, and the pairs above threshold
use about seven resolvable levels where the theory predicts two.

## What it does not say (caveats, in the theory's own words)
- **Context smoothing (Bethge 2003, lines 1673-1679):** if the neuron encodes x = s + c with c an
  uncontrolled trial-to-trial variable, a discrete f(x) is measured as a smooth f(s). MC_Maze conditions
  are maze layouts; kinematics vary within a condition. This is exactly the escape hatch, and it is being
  tested now (`PREREG_Q8b.md`, `q8b_mixture.py`): a discrete code smoothed by context predicts a
  within-condition MIXTURE at the certified levels; a smooth code predicts a single NB.
- **Population, not single neuron.** Nikitin 2009 says the level count grows with population size; the
  single-neuron prediction may simply be the wrong object for M1. This test cannot separate "the theory is
  wrong" from "the theory's object is not a single M1 neuron".
- One dataset, one area, one task. Retina under repeated natural stimuli would be the theory's home turf.

## Labels
K1-K3 [controls, pass]; the comparison table [confirmed numerically]; the interpretation "continuous at
this resolution" [confirmed numerically within this dataset]; "the theory's prediction fails" [argued, subject
to Q8b and to the population caveat].

## Addendum — the population caveat is weaker than I wrote (grepped after the readout)
Nikitin et al. 2009, `papers/nikitin2009.txt` lines 213-216: "suppose an overall population consists of K
neurons and M−1 sub-populations, within which each neuron is identical, and **binary with rates 0 and γ_i**
... the only way of achieving f(x) would be a single sub-population, where each neuron is identical, and able
to fire at two rates". In their population reading, the M-ary code is a population property built from
subpopulations of BINARY neurons. So both theories predict individual neurons with two levels; the M1 data
show ~7 at the N = 8 ceiling and are reaching 16 in the ceiling-raised replication (first unit: N_cv = 16 at
100 and 200 ms). The caveat that "the object is the population" does not rescue the per-neuron prediction.
[found-with-citation]

## Replication (Amendment 1: NMAX = 16, seed 2) — all three pre-registered predictions held
`q8_levels_rep.py`, `results_q8_rep.json`, `results_q8_rep_verdict.json`. Shuffle 529/529 = 1; powered 473/529.
| quantity | data | discrete sim | smooth sim |
|---|---|---|---|
| mean N_cv(all), powered | **8.86** | 2.71 | 8.73 |
| tuned pairs, W = 100 / 200 / 400 ms | 9.43 / 10.14 / 11.25 | 2.72 / 2.33 / 2.23 | 8.82 / 9.71 / 10.38 |
| A + lambda >= 3.4, tuned (n = 66) | **12.29** vs N* = 2.14 | 2.14 | 11.57 |
| mean N_cv vs m = 8 / 16 / 32 / all, tuned | **1.68 / 3.44 / 6.38 / 10.34** | 3.02 / 3.15 / 2.80 / 2.41 | 1.81 / 3.66 / 6.92 / 9.69 |
| growth among tuned pairs | **79%** (prediction: > 64%) | | |
| plateau-and-match among powered | **4%** (prediction: < 10%) | | |
| nearer smooth simulation, tuned pairs | **74%** (prediction: >= 70%) | 25% nearer discrete | |
| plateau at 2 among low-count pairs (n = 407) | **2%** | | |
| pairs at the new ceiling N = 16 | 82 | | |

By the letter of the original rule the replication is again **Q8-MIXED** (growth 67% overall), and again the
whole shortfall is the 75 untuned pairs (N_cv = 1, 16%) that the +-1 match counts as plateaus. Among pairs with
any resolvable tuning the pre-registered growth criterion is met at 79%. The data's N_cv-versus-trials curve
lies on the smooth simulation at every m and never on the discrete one. [confirmed numerically, replicated
with a fresh seed and a doubled ceiling]

## Q8b — the escape hatch, tested (`PREREG_Q8b.md`, `q8b_mixture.py`, `results_q8b.json`)
**Verdict: HATCH CLOSED** (pre-registered rule: Delta <= 0 in >= 70% of powered units; observed 97%).
| | |
|---|---|
| pairs / powered_b (K4 both sides >= 80%) | 469 / **376** over 154 units; unpowered pairs are the low-count ones (median A 0.21 vs 1.33) |
| K4 power | discrete-smoothed simulation detected (Delta > 0) in **100%**; smooth simulation rejected (Delta <= 0) in **85%** |
| data: Delta = LL(mixture at certified levels) − LL(single NB), per trial, held out | Delta > 0 in **3%** of powered pairs; median **−0.0167** nats/trial (p10 −0.076, p90 −0.003) |
| by window 100 / 200 / 400 ms | Delta > 0 in 4% / 1% / 4%; median −0.009 / −0.016 / −0.029 |
| A + lambda >= 3.4 (n = 66) | Delta > 0 in 3%; median −0.039 |

Reading: if the neuron used N* discrete levels and an uncontrolled context variable moved it between them
from trial to trial (Bethge 2003, lines 1673-1679), the within-condition count distribution would be a
mixture at those levels. It is not. A single overdispersed distribution at the condition's own rate wins in
97% of powered pairs. [confirmed numerically]

## Final statement for Q8 (2026-09-02)
The per-neuron prediction of discrete firing levels — binary below a peak count of 3.37 (Bethge 2002/2003;
Cao et al. 2014), N*(A, lambda) levels in general (Nikitin 2009; Dytso 2024), each neuron binary even in
Nikitin's population model — is **not observed in macaque M1 (MC_Maze) at 100, 200 or 400 ms**:
1. level count grows with trials exactly as a continuous tuning curve with the unit's own noise would
   (data 8.86 vs smooth 8.73 vs discrete 2.71; replicated, fresh seed, doubled ceiling);
2. the predicted 2-level plateau appears in 2% of low-count pairs, both runs;
3. the context-smoothing escape is closed: no within-condition mixture at the certified levels (97%);
4. the population escape is closed by the theory's own text (each neuron binary).
By the letter of the pre-registered rule both level-count runs are MIXED, entirely because untuned pairs
(N_cv = 1) are counted as plateaus; among tuned pairs the growth criterion is met at 64% and 79%. I keep the
label and report both readings. Scope: one dataset, one area, one task, 36 conditions, 55-75 repeats. The
theory's easy regime (retina, 16.7 ms bins) is pre-registered in `PREREG_Q8_retina.md` and awaits the data.

## Bookkeeping correction (2026-09-02, found while building the second-area caches)
`mc_maze_cache.npz` stores NWB END offsets under `sidx`; `q8_levels*.py` read them as START offsets. So
"unit k" in `results_q8*.json` and `results_q8b.json` is NWB unit k+1, NWB unit 0 was never analysed, and
"unit 181" is empty (skipped as A < 0.05, visible in the results). 181 real units were analysed under shifted
labels. No number or conclusion changes. New caches (`cache_dandi.py`) store start offsets.

## Second and third areas (PREREG_Q8_area2.md; DANDI 000127 and 000130, streamed; same scripts, m in {8, 16, all})
| | Area2_Bump (somatosensory area 2) | DMFC_RSG (dorsomedial frontal) |
|---|---|---|
| units / conditions / repeats | 65 / 16 / ~22 | 54 / 40 / ~23 |
| K1 shuffle | 193/195 = 1 (99%) PASS | 162/162 PASS |
| powered pairs (K2) | 167/195 | 130/162 |
| **rule verdict** | Q8-MIXED | Q8-MIXED |
| mean N_cv(all), powered: data / discrete sim / smooth sim | **8.10** / 3.02 / 6.63 | **5.39** / 2.96 / 7.40 |
| tuned pairs nearer smooth / discrete | **72%** / 28% | 49% / 48% |
| mean curve m = 8 / 16 / all, tuned: data | 4.82 / 8.01 / 9.59 | 3.36 / 5.37 / 7.07 |
| same, smooth sim / discrete sim | 4.42 / 6.51 / 7.60 ; 3.06 / 2.98 / 2.90 | 4.35 / 7.51 / 9.16 ; 3.05 / 2.81 / 2.73 |
| flat at 2 (N_cv(16) = N_cv(all) = 2), tuned | 5 / 138 (4%) | 12 / 94 (13%) |
| pairs at the condition ceiling | 34 (of 16 conditions) | 9 (of 40) |
| **Q8b hatch** | CLOSED: Delta > 0 in 3% of 121 powered pairs, median −0.077; K4 0.98 / 0.80 | CLOSED: Delta > 0 in 1% of 103, median −0.058; K4 1.00 / 0.83 |

**Area 2** replicates M1: the data resolve MORE levels than the uniform-continuous simulation (direction
tuning across 8 directions is U-shaped in rate, which separates better than uniform), 34 pairs resolve every
condition, and the hatch is closed. [confirmed numerically]

**DMFC** is the theory's best showing and still a minority result. Unrestricted, the tuned pairs split
evenly between the two simulations and 13% are flat at two levels. Amendment 2 (written before computing)
restricts to pairs tuned to the graded variable ts (71 of 130, permutation p < 0.01), where the theory's
claim is non-trivial: **data 7.51 vs discrete 2.76 vs smooth 9.00; curve 3.75 / 5.69 / 7.51 (growing) vs
discrete 3.08 / 2.90 / 2.76 (flat); flat-at-2 in 6%; growth in 59%.** Of the 24 two-level pairs in the main
run, 14 are ts-tuned and only 4 split on a binary task factor at >= 0.85 agreement, so the two-level pairs
are neither mostly design artefacts nor a majority. Why DMFC sits below the uniform-continuous simulation:
tuning to ts is monotone and skewed, and a skewed continuous map resolves fewer levels than a uniform one at
the same SNR; the uniform null is conservative in the direction of the theory. [confirmed numerically for the
restriction; the skewness explanation is argued, not tested]

**Across three cortical areas and three task structures** the per-neuron discrete-levels prediction is not
observed as the rule: the two-level plateau appears in 2% (M1), 4% (area 2) and 6-13% (DMFC) of tuned pairs,
the context-smoothing mixture loses in 97% / 97% / 99%, and level counts grow with trials everywhere the
theory says they should not. The DMFC minority is the one place a defender of the theory could stand, and it
is reported at full size.

## Retina — the theory's easy regime (PREREG_Q8_retina.md + Amendment 1; Dryad 10.5061/dryad.4qrfj6qm8)
Larval salamander RGCs, 93 cells x 5 natural movies = 465 units, conditions = 36 random 16.7 ms bins (seed 3),
repeats 64-91, Bernoulli data at 16.7 ms (simulations Bernoulli there), summed to 50 and 100 ms.
- K1 shuffle 1000/1003 = 1 (99.7%). Powered 932/1003 pairs over 381 units.
- **Rule verdict: Q8-SMOOTH** (growth 73% >= 70%; this is the first dataset where untuned pairs, 3%, do not
  push the letter of the rule to MIXED). Data 6.55 vs discrete sim 3.10 vs smooth sim 8.43.
| window | tuned pairs | data N_cv (m = 8/16/32/all) | discrete sim | smooth sim | flat at 2 | nearer smooth |
|---|---|---|---|---|---|---|
| **16.7 ms** (A median 0.17, max 0.90) | 226 | 1.99 / 2.84 / 3.63 / **5.10** | 1.72 / 2.62 / 3.26 / 3.39 | 1.02 / 1.17 / 2.21 / 6.57 | **6%** | 45% |
| 50 ms (A median 0.33) | 323 | 2.84 / 3.76 / 4.92 / **6.63** | 2.57 / 3.40 / 3.55 / 3.14 | 1.14 / 1.88 / 4.06 / 8.56 | 5% | 54% |
| 100 ms (A median 0.56, max 3.52) | 359 | 3.56 / 4.71 / 6.17 / **7.77** | 3.04 / 3.48 / 3.37 / 2.85 | 1.55 / 2.98 / 5.71 / 9.88 | 4% | 63% |

Reading. At 16.7 ms, where Bethge's binary prediction is stated to hold, the data keep resolving more
levels from 32 repeats to all (3.63 -> 5.10) while the discrete simulation has flattened (3.26 -> 3.39); the
uniform-continuous simulation only resolves at the last step because a uniform spread of tiny rates is
noise-limited. The endpoint sits between the two simulations (45/55), the SHAPE is the continuous one, and
the predicted two-level plateau appears in 6% of tuned pairs. At 50 and 100 ms the endpoint moves toward the
continuous simulation (54%, 63%) and flat-at-2 falls to 5% and 4%. As in DMFC, the data lie below the
uniform-continuous simulation: a PSTH sampled at random bins is skewed (mostly near baseline, few transients),
and a skewed continuous map resolves fewer levels than a uniform one; the uniform null is conservative in the
theory's favour. [confirmed numerically; the skewness explanation argued]
A better null for the paper: continuous tuning with a skewed (e.g. lognormal) rate distribution matched to
the unit's own range — to be pre-registered before use, not fitted after.

## Skewed continuous null (PREREG_skewnull.md; lognormal-quantile rates, same noise, same trial counts)
| dataset | tuned pairs | data curve | discrete | uniform | **skewed** | nearest-of-three at m = all (disc / unif / skew) | skewed-null growth | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | 398 | 1.68 / 3.44 / 6.38 / 10.34 | 3.02 / 3.15 / 2.80 / 2.41 | 1.81 / 3.66 / 6.92 / 9.69 | 1.55 / 2.77 / 5.74 / 9.13 | 20% / 40% / 40% | 95% | ✓ | ✓ | ✓ |
| Area 2 | 138 | 4.82 / 8.01 / 9.59 | 3.06 / 2.98 / 2.90 | 4.42 / 6.51 / 7.60 | 3.56 / 5.65 / 7.17 | 24% / 44% / 32% | 80% | ✓ | ✓ | ✓ |
| DMFC | 94 | 3.36 / 5.37 / 7.07 | 3.05 / 2.81 / 2.73 | 4.35 / 7.51 / 9.16 | 3.33 / 6.23 / 7.96 | **39%** / 28% / 33% | 82% | ✓ | ✓ | **✗** |
| Retina | 908 | 2.91 / 3.91 / 5.09 / 6.70 | 2.54 / 3.24 / 3.41 / 3.09 | 1.27 / 2.14 / 4.25 / 8.58 | 1.17 / 1.81 / 4.03 / 8.70 | **34%** / 32% / 34% | 100% | ✗ (marginal: skewed 8.70 vs uniform 8.58) | ✓ | **✗** |

P1 (skewed between discrete and uniform) and P2 (skewed still grows, >= 60%) hold in all three cortical
datasets: a realistic skewed continuous map is still separated from the discrete prediction by SHAPE. P3
holds in M1 and area 2 and **fails in DMFC**: on the endpoint metric the discrete simulation is nearest for
39% of DMFC tuned pairs (46% at 100 ms), above the 30% bar. The DMFC data curve itself grows (3.36 -> 5.37 ->
7.07) like the skewed null and unlike the flat discrete simulation, and the pre-registered ts-tuned restriction
gave flat-at-2 in 6%; but the endpoint sits where a mixed population would put it. DMFC is reported as the
dataset where the theory is not excluded on the endpoint metric. [confirmed numerically]

**Retina under the skewed null.** P2 holds (growth 100%); P1 fails marginally (the skewed null ends 0.12
above the uniform one, i.e. at these tiny rates the two continuous nulls are the same object); P3 fails
(discrete nearest 34% at m = all, 40% at 16.7 ms). The endpoint is a three-way tie. The CURVES are not: at 8
repeats the data resolve 2.91 levels, matching the discrete simulation (2.54) and not the continuous ones
(1.17-1.27); at all repeats they reach 6.70 while the discrete simulation sits at 3.09. Reading: retinal
responses to natural movies are sparse, mostly silent bins with occasional firing events. At coarse
resolution that IS a two-level code, events versus silence, which is what Bethge's binary prediction
describes and why the discrete simulation matches at 8 repeats. With enough repeats the event amplitudes
resolve into graded levels, which is what the theory denies. So in the theory's own regime the binary
prediction is a low-resolution appearance of a graded rate map, not a property of the map. [confirmed
numerically for the curves; the sparse-code reading is argued and should be checked against the per-bin rate
histograms before it goes into a paper]

## Final cross-dataset statement (2026-09-02, pending only the retina hatch test)
Four datasets, three cortical areas and retina, one pre-registered pipeline. The per-neuron discrete-levels
prediction of optimal Poisson rate coding (2 levels below a peak count of 3.37; N*(A, λ) in general):
- by the pre-registered rule: MIXED in M1, area 2 and DMFC (all three for the same untuned-pair reason),
  SMOOTH in retina;
- by curve shape: level counts grow with repeats in every dataset and window while every discrete simulation
  flattens; no dataset shows the predicted plateau in more than 6% of tuned pairs (13% in DMFC before the
  graded-variable restriction);
- by endpoint against three nulls: the discrete simulation is nearest in 20% (M1), 24% (area 2), 39% (DMFC)
  and 34% (retina) of tuned pairs, so DMFC and retina do not exclude it on that metric;
- the context-smoothing escape is closed in M1 (97%), area 2 (97%) and DMFC (99%); retina pending;
- the population escape is closed by Nikitin 2009's own text.
What the theory gets right: at low resolution (few repeats, tiny rates) sparse responses look two-level, and
the discrete simulation matches the data there. What it gets wrong: that this is a property of the rate map.

## Secondary result — what does set the resolvable level count? (all four datasets, 1538 tuned powered pairs)
Not an optimal code: a resolution law. N_cv(all) rises with the unit's signal-to-noise in every dataset
(correlation of log N with log SNR, SNR = A / sqrt(Fano (lambda + A/2)): M1 0.45, area 2 0.62, DMFC 0.64,
retina 0.55; with log A alone 0.39-0.60). Pooled power law N ~ A^0.26 (per-dataset exponents 0.30-0.58); the
PREREG's naive Poisson guess was 0.5. The certified staircase N*(A) would predict a STEP at A = 3.37, not a
power law. Measured within-condition Fano medians: M1 1.13, area 2 1.03, DMFC 1.07, retina 0.90 — all near
Poisson, which is why the noise-corrected N* (PREREG_noiseNstar.md, running) is expected to move little.
[confirmed numerically; secondary, not pre-registered as a prediction]
Fig 6 `figs/fig6_retina_rate_hist.png`: 97% of retinal 16.7 ms bins are silent (p < 0.05); the non-silent bins
form a graded continuum (median p 0.11, 90th pct 0.30, CV 0.53) — the "sparse, graded" reading is substantiated.
**Quantitative test of the resolution law (from `results_skewnull.json`).** Log-log slope of N_cv(all) against A,
per dataset, data vs the three simulations run on the same pairs with the same noise and trial counts:
| | M1 | area 2 | DMFC | retina | all |
|---|---|---|---|---|---|
| data | 0.30 | 0.40 | 0.58 | 0.33 | 0.26 |
| skewed continuous sim | 0.27 | 0.48 | 0.57 | 0.40 | 0.20 |
| uniform continuous sim | 0.25 | 0.49 | 0.42 | 0.54 | 0.29 |
| discrete N*(A, λ) sim | **−0.20** | **−0.06** | **−0.14** | **−0.26** | **−0.16** |
Per-pair correlation of data N_cv with the skewed simulation's N_cv: 0.54 / 0.56 / 0.61 / 0.59. A continuous
map under the unit's own noise reproduces how the level count scales with dynamic range in every dataset; the
discrete prediction scales with the wrong sign. [confirmed numerically; secondary]

## Straw-man check — N* under each dataset's measured noise (PREREG_noiseNstar.md, `support_general.py`)
Control: the COM-Poisson instrument at Fano = 1 reproduces the Poisson staircase at every tested (A, λ).
At Fano 0.9-1.13 (all four datasets' medians) N*(A, λ, F) differs from the Poisson value by at most ONE level
anywhere in A <= 12, and equals 2 throughout the retina regime (A <= 0.9). Super-Poisson noise (Fano 1.13-1.5,
cortex) LOWERS N* (2 up to A = 3.5-5 at λ = 0); only strongly sub-Poisson units (Fano 0.7) would gain a level
at A = 3. Predictions (1)-(3) of the PREREG hold: the discrete prediction tested above is the theory's own
under realistic noise, not a Poisson caricature. [confirmed numerically]

## Growth with trials (secondary): log N_cv vs log m over the m grid, tuned powered pairs
| | M1 | area 2 | DMFC | retina |
|---|---|---|---|---|
| data | +0.87 | +0.69 | +0.70 | +0.37 |
| continuous sim | +0.82 | +0.54 | +0.72 | +0.87 |
| discrete sim | −0.11 | −0.05 | −0.11 | +0.08 |
Cortex grows with trials exactly like the continuous simulation; retina grows more slowly than the uniform
continuous simulation (it resolves 2-3 levels already at 8 repeats, the sparse-code signature) but far from
the flat discrete prediction. [confirmed numerically]

## Retina hatch test (Q8b at 50 / 100 ms; Bernoulli mixtures unidentifiable at 16.7 ms)
**HATCH CLOSED**, with the largest minority of the four datasets: 539 powered pairs over 317 units (K4 power
0.99 / 0.84); Delta > 0 in **13%** at both windows (median −0.0035 and −0.0069 nats/trial). Unpowered pairs are
the low-count ones (median A 0.20 vs 0.56). Reading: a minority of retinal bins are genuinely event-vs-no-event
mixtures across repeats (Berry-Warland-Meister's firing events with variable occurrence); for 87% a single
overdispersed distribution at the bin's own rate is the better description. [confirmed numerically]

## Final cross-dataset statement — updated 2026-09-02 (all runs complete)
Hatch closed in all four: M1 97%, area 2 97%, DMFC 99%, retina 87% of powered pairs favour a single NB.
Everything else as stated above. The theory's binary prediction is the low-resolution appearance of a
graded sparse code; the level count is set by signal-to-noise, with exponents a continuous map reproduces
and the discrete code contradicts in sign; the prediction survives only as a minority (2-13%) that is
largest exactly where responses are sparsest.

## Population version tested directly (PREREG_Q8_population.md, `q8_popsum.py`) — M1 complete, retina in progress
Summed population response over K random M1 units (10 subsets each), W = 200 ms, 36 conditions:
| K | A_pop | N*(A_pop, λ_pop) | data N_cv (m = 8/16/32/all) | discrete sim | continuous sim | within ±1 of N* | growth |
|---|---|---|---|---|---|---|---|
| 1 | 0.98 | 2 | 1.3 / 2.4 / 5.1 / 7.4 | 3.0 / 3.3 / 2.8 / 3.3 | 1.2 / 2.8 / 4.8 / 7.2 | 40% | 60% |
| 4 | 3.27 | 2 | 1.6 / 2.6 / 7.9 / **12.5** | 2.3 / 2.6 / 2.0 / 2.0 | 3.6 / 4.7 / 8.6 / 10.1 | 0% | 70% |
| 16 | 7.00 | 2 | 1.8 / 6.0 / 8.2 / **13.0** | 2.1 / 2.1 / 2.0 / 2.0 | 3.2 / 5.4 / 7.4 / 10.4 | 0% | 70% |
| 64 | 16.05 | 2.1 | 1.7 / 6.0 / 11.0 / **13.1** | 2.5 / 2.1 / 2.1 / 2.1 | 3.6 / 5.8 / 9.2 / 10.8 | 0% | 70% |
| 128 | 22.29 | 2 | 2.4 / 6.3 / 9.1 / **12.9** | 2.2 / 2.0 / 2.0 / 2.0 | 2.4 / 4.1 / 6.8 / 8.9 | 0% | 70% |
The population sum's high baseline acts as dark current, so the certified population staircase predicts 2
levels at every K; the summed data resolve 10-13 and keep growing with trials. **P-pop-S in M1.** Retina
(movies 0-3, K = 93): 10-16 levels resolved vs N* = 2-3, within ±1 of N* in 0% of subsets. The population
escape is closed by direct test, not only by Nikitin's wording. [confirmed numerically]
Retina population sums complete (`results_q8_pop.json`): at 16.7 ms, K = 4 / 16 / 93 cells resolve 6.4 / 11.3 /
13.9 levels vs N* = 2 / 2 / 2.6 (within ±1 in 22% / 12% / 0%); at 100 ms, 11.7 / 14.4 / 15.3 vs 2.2 / 2.9 / 5.8
(2% / 2% / 0%). **P-pop-S in both tissues; the population escape is closed by direct test.** [confirmed numerically]
Population growth with size (`results_q8_pop.json`, subsets below the 16-level ceiling): N_cv(K) ~ K^γ with data
γ = +0.21 (M1), +0.35 (retina 16.7 ms), +0.27 (retina 100 ms); continuous simulation +0.08 / +0.35 / +0.19;
discrete −0.06 / −0.04 / +0.12. Growth with population size is real and sub-√K (shared noise correlations
make the sum's noise grow faster than √K); the discrete prediction is flat. [confirmed numerically; secondary]

## Q10 — shape of the continuous map (PREREG_Q10.md, `q10_shape.py`, `results_q10.json`): UNDERPOWERED, with a hint
Four shape hypotheses for the per-condition means (sqrt-uniform = noise-adjusted equalisation, Han 2015 /
Wang-Stocker-Lee 2012; rate-uniform = Laughlin 1981; log-uniform; two-level), KS classification, per-pair
recovery control. Powered pairs (all four shapes recovered >= 60%): M1 32/398, area 2 7/138, DMFC 11/94,
retina **0/908**; pooled **50/1538 (3%)**. Among powered pairs: **log-uniform 58%, sqrt 34%, rate 8%,
two-level 0%**. Mean recovery over all pairs: sqrt 0.75, rate 0.49, log 0.47, two 0.14, so the classifier
leans to sqrt when it cannot see; the unpowered tally (sqrt 69%) is therefore not a result and is not one.
Verdict: neither equalisation prediction holds where the test has power; the map is more right-skewed than
noise-adjusted equalisation (log-uniform wins), and the two-level code never wins. K1 was vacuous by
construction (noted in the PREREG). With 36 conditions the shape question is beyond this instrument's
resolution for 97% of pairs. Next: the retina's 1,203 bins per cell-movie (Amendment 1). [confirmed numerically]
**Q10 Amendment 1 — retina, full PSTH (`q10_retina_full.py`, `results_q10_retina_full.json`).** Non-silent bins
per cell-movie 139-231 (silent floor excluded: 81% / 57% / 31% of bins at 16.7 / 50 / 100 ms). Powered pairs
31 / 27 / 19 (16-49%). Among powered pairs **log-uniform wins 100% / 100% / 95%**; sqrt 0 / 0 / 5%; rate 0;
two-level 0. Mean KS among powered: log 0.27, sqrt 0.38, two-level 0.49, rate 0.55. Verdict: **both
equalisation predictions fail (P-sqrt, P-rate)**; the best of four is log-uniform, and its residual KS of 0.27
says the map is more skewed still: sparse, heavy-tailed, most bins silent. [confirmed numerically]

## Q11 — a mechanistic model (PREREG_Q11.md, `q11_mech.py`, `results_q11.json`): FAILS by the letter, right in direction
Threshold-exponential transduction (gain k = 2, fixed) of Gaussian input across conditions, Poisson/NB noise at
the unit's Fano and trial counts, three cortical datasets.
| | M1 | area 2 | DMFC | bar |
|---|---|---|---|---|
| rate skew > 0 | 100% | 98% | 100% | >= 80% ✓ |
| log-skew median | +0.40 | +0.30 | +0.21 | within ±0.3: ✗ / borderline / ✓ |
| conditions in lowest quarter (data 44-48%) | 64% | 62% | 60% | ±10 pts: ✗ (too sparse) |
| N_cv curve max |model − data| | 1.48 | 2.97 | 1.35 | <= 1: ✗ |
| A-exponent model vs data | +0.05 vs +0.30 | +0.25 vs +0.40 | +0.19 vs +0.58 | ±0.15: ✗ / ✓ / ✗ |
| growth exponent model vs data | +0.61 vs +0.81 | +0.35 vs +0.48 | +0.53 vs +0.47 | — |
| Q10 class of model maps (log / sqrt / rate / two) | 71 / 29 / 0 / 0 | 65 / 31 / 4 / 0 | 70 / 30 / 0 / 0 | same ordering as data ✓ |
Verdict: P1 and P2 fail; P3 holds. A gain-2 exponential on unit-variance Gaussian input is too compressive: it
produces the right kind of map (skewed, log-like, never two-level) but too sparse and with too few resolvable
levels that depend too weakly on dynamic range. The gain was not retuned after the result (pre-registered).
Next, if pursued: calibrate the gain on one dataset by the lowest-quarter statistic and test the level-count law
on the others (cross-validated, pre-registered). [confirmed numerically]

## Fifth dataset — mouse V1, static gratings (PREREG_Q8_v1.md; Allen Visual Coding Neuropixels, DANDI 000021, ses-715093703)
135 VISp 'good' units, 36 of 120 orientation x spatial-frequency x phase conditions (seed 12), 45-50 repeats,
W = 250 ms. K1 shuffle 124/124. Powered 100/124 units. **Rule verdict: Q8-MIXED** (growth 62%; 27 units, 27%,
untuned to the random grating subset). Data mean N_cv 6.46 vs discrete sim 2.67 vs continuous sim 7.92. Among
tuned units (n = 73): **growth 85%, plateau-and-match 8%**; plateau at 2 among low-count units 8%; high-count
units (A + λ >= 3.4, n = 37) growth 81%. N_cv distribution bimodal: 27 at 1 (untuned), 27 at >= 14 (near the
ceiling). Same picture as the four other datasets, now on the canonical graded-variable stimulus. Hatch test
running. [confirmed numerically]

## Q11b — cross-validated gain (PREREG_Q11b.md, `q11b_mech.py`)
Calibration on M1 by the lowest-quarter statistic over the pre-registered grid: k = 0.5 -> 39%, **0.75 -> 44%**,
1.0 -> 50%, 1.25 -> 53%, 1.5 -> 56%, 2.0 -> 64%, 3.0 -> 69% (data 44%). **k* = 0.75**, frozen. On M1 at k*: P1
passes (skew > 0 99%, log-skew −0.18, lowest-quarter 44%); P2: N_cv max |diff| 1.07 (bar 1.0, marginal),
A-exponent 0.11 vs 0.30 (fail), growth exponent 0.68 vs 0.81 (pass). At EVERY gain the model's A-exponent is
0.05-0.14 against the data's 0.30: the level count's dependence on dynamic range is not a gain effect. Test
sets (area 2, DMFC, V1) running at k* = 0.75; retina not run (the model has no Bernoulli-bin form; deviation
from the PREREG's test-set list, stated).
**V1 hatch test (Q8b):** CLOSED. 75 powered units; Delta > 0 in **24%** (the largest minority of the five datasets;
bar 30%), median −0.013 nats/trial; K4 1.00 / 0.80. [confirmed numerically]

**Q11b test sets at k\* = 0.75** (`q11b_test_log.txt`):
| | area 2 | DMFC | bar |
|---|---|---|---|
| rate skew > 0 / log-skew / lowest-quarter (data 44-48%) | 93% / −0.22 / 44% ✓ | 98% / −0.33 / 45% ✓ | P1 passes both |
| N_cv max |model − data| | 1.54 ✗ | 2.74 ✗ | <= 1 |
| A-exponent model vs data | 0.28 vs 0.40 ✓ | 0.31 vs 0.58 ✗ | ±0.15 |
| growth exponent model vs data | 0.36 vs 0.48 ✓ | 0.40 vs 0.47 ✓ | ±0.2 |
| Q10 class (sqrt / log) | 54 / 30 ✗ | 67 / 28 ✗ | log > sqrt as in data |
**Verdict: Q11b FAILS** (0 of 2 test sets pass P1 and P2; V1 run pending a script fix). The cross-validated
gain fixes sparseness and nothing else: the calibrated model has too few resolvable levels where the data
have many (area 2) and too many where they have few (DMFC), and its maps are sqrt-like rather than log-like.
Sparseness and resolution are set by different things; the Gaussian-input transduction model captures the
first. Per the PREREG's kill clause, the next candidate (heavy-tailed input, or input spread scaling with
dynamic range) is a separate pre-registration, not a retune. [confirmed numerically]
V1 test set at k\* = 0.75: P1 passes (skew > 0 100%, log-skew −0.16, lowest-quarter 44%); P2 fails badly (N_cv
curve [3.26, 5.75, 9.34] vs data [1.41, 2.29, 8.48], max |diff| 3.47; A-exponent 0.18 vs 0.70; growth 0.56 vs
1.00); P3 sqrt 55% vs log 41% ✗. **Q11b final: 0 of 3 test sets pass P1 and P2 — FAILS.** V1 tuning is weak per
trial and resolves only at many repeats, a signal-to-noise structure the Gaussian-input model does not have.

## Q11c — input spread scaling with dynamic range (PREREG_Q11c.md): FAILS AT CALIBRATION; branch stopped
Grid c in {0, 0.25, 0.5, 1, 2} on M1: model A-exponent 0.12 / 0.12 / −0.01 / −0.09 / −0.23 against the data's
0.30. A wider input saturates the nonlinearity and compresses the map; the best value is c = 0, i.e. Q11b. Per
the stop rule the mechanistic branch stops here. Standing statement: a one-parameter
threshold-exponential transduction of Gaussian input reproduces the SHAPE of single-neuron rate maps on
held-out datasets (sparseness, skew, log-symmetry) and does not reproduce their RESOLUTION law (level count vs
trials and vs dynamic range). Open diagnostic for the next cycle: the random continuous nulls (uniform or
lognormal-quantile rates over [λ, λ + A]) give A-exponents of 0.27-0.57, above the transduction model's 0.1,
so the model's normalised map is too invariant across units; whether the real normalised map changes shape
with A is a cheap pre-registerable test (Q11d). [confirmed numerically]

## Q11d — does the normalised map change shape with dynamic range? (PREREG_Q11d.md, `q11d_norm.py`): INVARIANT
Tuned powered units split into A-terciles; y = (m − min)/(max − min). KS between the pooled normalised maps of the
lowest and highest A-terciles, data vs the transduction model (k = 0.75) and the lognormal-quantile null simulated
per unit (sampling-noise dependence only):
| | n | data | transduction | lognull | data − max(models) | verdict |
|---|---|---|---|---|---|---|
| M1 | 129 | 0.077 | 0.108 | 0.032 | −0.03 | invariant |
| area 2 | 48 | 0.172 | 0.152 | 0.109 | +0.02 | invariant |
| DMFC | 30 | 0.150 | 0.070 | 0.040 | +0.08 | invariant (bar 0.10) |
| V1 | 73 | 0.107 | 0.071 | 0.044 | +0.04 | invariant |
| retina | 359 | 0.160 | 0.096 | 0.140 | +0.02 | invariant |
Sparseness (fraction of conditions with y < 0.25) drifts from ~50% in the low-A tercile to ~40% in the high-A
tercile in M1 and DMFC, within what the models' sampling noise produces. **Verdict: Q11d-INVARIANT in 5/5.** The
models' failure on the resolution law is not a shape-versus-range effect; by elimination it lies in the
trial-to-trial variability structure that the negative-binomial-at-Fano description does not carry (V1's growth
exponent of 1.0 is the pointer). Per the Q11c stop rule, no further model is tried. [confirmed numerically]

## Sixth dataset — mouse auditory cortex (PREREG_Q8_a1.md; DANDI 000986, sub-LA11 ses-1, 235 units, 5 tone frequencies x ~1,490 repeats, 20 ms tones, 50 ms window)
**DeWeese-Wehr-Zador 2003 "binary spiking" re-tested directly (descriptive, pre-registered).** 214 units with
>= 50 evoked spikes at their best frequency. Best-frequency mean count: median 0.25 (p90 1.34). P(count >= 2):
median **3.6%** (p75 13%, p90 40%); 29% of units below 1% ("binary" by count). Fano factor at best frequency:
median **1.03**. Observed P(>= 2) divided by the Poisson expectation at the same mean: median **1.07**. Reading: in
awake mouse A1 the two-or-more fraction is what a Poisson response of 0.25 spikes per tone produces; the
sub-Poisson binary regime DeWeese et al. reported (Fano well below 1, anaesthetised rat, cell-attached) is not
present in this preparation. "Binary" here is the low-rate appearance of Poisson-like spiking, the same
low-resolution appearance the retina showed for the level count. [confirmed numerically; different
preparation, so not a refutation of their measurement, a non-replication of the regime]
Level-count test (5 conditions, up to ~1,490 repeats) running.
**A1 level-count test** (`results_q8_a1.json`, `a1/verdict.txt`): 5 conditions, so the ceiling is 5. K1 154/161.
Powered 123/161 units; N* = 2 for all (A + λ median 0.37). Mean N_cv over m = 8 / 16 / 32 / 64 / 128 / 256 / all
(tuned, n = 117): **data 1.46 / 1.89 / 2.43 / 3.06 / 3.68 / 4.15 / 4.43**; discrete sim 1.65 / 1.99 / 2.14 / 2.23 /
2.28 / 2.26 / 2.25; continuous sim 1.15 / 1.44 / 1.91 / 2.47 / 2.85 / 3.44 / 4.20. N_cv(all) histogram: 2: 10,
3: 12, 4: 13, **5: 82** (70% of tuned units resolve all five frequencies). Plateau at 2 among low-count units:
**8%**. Rule verdict MIXED (a plateau at the 5-level ceiling counts as a plateau: 76%). The theory's binary
prediction, at its most favourable (A = 0.37, 20 ms tones), is contradicted by five resolved levels in most
units once repeats are sufficient; the same low-resolution appearance as retina at few repeats (1.5 levels
at 8 repeats). [confirmed numerically] Hatch test running.
**A1 hatch test (Q8b):** CLOSED. 79 powered units; Delta > 0 in **16%**, median −0.002 nats/trial; K4 0.89 / 0.81.
Six datasets: hatch minorities 3 / 3 / 1 / 13 / 24 / 16%. Fig 5 now carries all six (eight panels). [confirmed numerically]

## Within-dataset replication — V1 drifting gratings (PREREG_Q8_v1_drifting.md): the weakest run, reported at full size
Same 135 VISp units, 40 direction x temporal-frequency conditions, **15 repeats**, windows 250-2000 ms. K1 538/539.
Powered 423/539 pairs, but **205 (48%) untuned** (N_cv = 1). Among tuned pairs (n = 218): growth 8 -> 15 trials in
**73%**, plateau-and-match 17%; flat at 2 in 11 / 6 / 16 / 3% by window. Endpoint: data 3.57 vs discrete sim
4.34 vs continuous sim 5.59; per window the data (5.2 / 6.0 / 5.5 / 6.8) sit BELOW both simulations (discrete
3.6-5.7, continuous 7.4-8.8; A median 4-14 so N* = 3-12 for many units), nearer the continuous one in only
34-46%. Hatch: CLOSED at **28%** (bar 30%), the highest of seven runs. Rule verdict MIXED.
Reading: at 15 repeats and large A, real drifting-grating responses resolve fewer levels than any simulation
with negative-binomial-at-Fano noise, the same per-trial-variability signature seen in V1 static at low m and
in the Q11 growth-exponent mismatch. The growth signature survives; the endpoint comparison does not
discriminate here. This is the run a referee would point to, and it points to the noise model (Q12).
[confirmed numerically]

## Q12 — modulated-Poisson (gain-fluctuation) noise (PREREG_Q12.md, `q12_noise.py`, `results_q12.json`): FAILS
Run 2026-09-02 ~23:20, later on the same day as the pre-registration, unmodified.
| | area 2 | DMFC | V1 | M1 |
|---|---|---|---|---|
| sigma_G^2 median (> 0 in) | 0.000 (33%) | 0.001 (50%) | 0.240 (96%) | 0.137 (84%) |
| P1 growth exponent: data / continuous-NB / continuous-MoP | 0.73 / 0.55 / 0.53 ✗ | 0.53 / 0.72 / 0.69 ✗ | 1.01 / 0.69 / 0.75 ✗ | 0.88 / 0.85 / 0.93 ✓ |
| P2 transduction+MoP curve max |model − data| | 1.54 ✗ | 1.03 ✗ (bar 1.0) | 2.37 ✗ | 1.40 (calibration set) |
Verdict: P1 fails on all three test sets, P2 fails 0/3. Why, stated after the fact but not used to retune: a
trial-wise multiplicative gain adds variance that averages at the same 1/n rate as Poisson noise, so it cannot
change the rate at which levels resolve with trials, which is the quantity the data disagree on. A growth
exponent near 1 (V1) needs variability that averages more slowly than 1/sqrt(n): heavy-tailed trial noise or
slow drift across trials. Per the PREREG kill clause the resolution law stays an empirical regularity with no
model in hand. Diagnostic Q13 (contiguous vs random trial draws, data only) is pre-registered next. [confirmed numerically]

## Q13 — session drift diagnostic (PREREG_Q13.md, `q13_drift.py`, `results_q13.json`): no drift in M1/V1, drift in A1
Run 2026-09-02 ~23:30, minutes after pre-registration, unmodified. Contiguous minus random N_cv at m = 8 / 16 / 32:
| | data | NB-at-Fano control | units contiguous > random at m=8 (vs <) |
|---|---|---|---|
| M1 (n=129) | −0.01 / 0.00 / −0.38 | +0.20 / −0.08 / −0.35 | 9% vs 9% |
| V1 (n=73) | +0.21 / +0.23 / −0.19 | +0.08 / −0.45 / +0.16 | 14% vs 3% |
| A1 (n=117) | +0.32 / +0.16 / +0.49 | −0.08 / −0.03 / −0.15 | 25% vs 8% |
Instrument note (stated after the fact): the control itself swings by up to 0.45 levels at R = 20, so the
pre-registered control bar (|mean| < 0.15) is not met by the instrument; the resolution of this diagnostic is
about 0.3-0.4 levels. Verdict by the letter: P-drift fails (1 of 3 datasets), P-null fails (A1). Reading:
M1 and V1 data track their controls (no drift signal); A1, with ~1,490 repeats per tone over a long session,
shows a consistent positive signature at every m with a negative control, i.e. slow drift is real there.
Drift is therefore not the missing half where the growth-exponent mismatch lives (V1: 1.0 vs 0.7). The
remaining candidate on the noise side is heavy-tailed trial noise (Q14, pre-registered next). [confirmed numerically]

## Q14 — heavy-tail diagnostic (PREREG_Q14.md, `q14_heavytail.py`, `results_q14.json`): P-NULL
Run 2026-09-02 ~23:40, unmodified. (a) Per-unit excess kurtosis above the paired NB-at-Fano simulation in
40% (M1), 21% (V1), 59% (A1) of units; medians M1 3.58 vs 3.98, V1 2.24 vs 2.72, A1 11.3 vs 8.3. Trial noise
in M1/V1 is LIGHTER-tailed than the simulation. (b) Trimmed-minus-mean estimator swap, paired, R = 40:
D = M1 [+0.13, +0.53, +0.74], V1 [+0.03, +1.16, +1.64], A1 [−0.03, −0.04, −0.05] at m = 8/16/32; the positive
D in M1/V1 means trimming hurts the simulation more than the data, i.e. the simulation is the heavier-tailed
one, consistent with (a). P-heavy fails on (a); P-null holds for M1/V1. Heavy tails are not the missing half.
[confirmed numerically]

## Q15 — per-condition moment-matched noise (PREREG_Q15.md, `q15_matched.py`, `results_q15.json`): OUTSIDE BOTH PREDICTIONS, and it relocates the wall
Run 2026-09-02 ~23:45, unmodified. N_cv at m = 8 / 16 / 32 / all (data | matched independent noise | unit-Fano):
| | data | matched | unit-Fano | growth exp data / matched |
|---|---|---|---|---|
| M1 (n=129, 64 trials) | 1.46 / 2.76 / 5.50 / 7.98 | 1.46 / 2.83 / 6.04 / **9.71** | 1.68 / 3.15 / 6.43 / 9.87 | +0.84 / +0.93 |
| V1 (n=73, 48) | 1.40 / 2.30 / 6.30 / 8.52 | 1.49 / 2.84 / **8.45** / **10.62** | 1.47 / 3.40 / 8.78 / 10.92 | +1.06 / +1.16 |
| A1 (n=117, 1489) | 1.48 / 1.91 / 2.34 / 4.44 | 1.37 / 1.76 / 2.33 / 4.47 | 1.46 / 1.76 / 2.31 / 4.41 | +0.20 / +0.21 |
| area 2 (n=48, 23) | 4.60 / 7.60 / 9.46 / 9.10 | 4.08 / 7.44 / 8.90 / 8.92 | 5.17 / 7.46 / 9.52 / 10.31 | +0.70 / +0.76 |
| DMFC (n=30, 24) | 3.73 / 6.73 / 7.93 / 7.70 | 3.77 / 6.97 / **9.67** / **9.97** | 3.17 / 7.20 / 9.73 / 10.03 | +0.71 / +0.87 |
P-match fails (only A1 within 0.5 at every m; area 2 misses by 0.06). P-null fails too: at m = 8 the matched
simulation is within 0.11 level everywhere except area 2 (0.52), and every growth exponent is within 0.16.
Two things follow. (1) The small-m "starvation" seen against the uniform-map comparator (Fig 5, V1 at 8
trials) was the sparse MAP SHAPE, not the noise: a simulation on the data's own condition means reproduces
small-m counts with independent noise. (2) The real mismatch is at LARGE m: with all trials, M1, V1 and DMFC
resolve 1.7-2.3 fewer levels than independent moment-matched noise predicts; area 2 shows none; A1 sits at its
5-condition ceiling. Fewer levels than independent noise allows, at the largest trial counts only, is the
signature of positive dependence between trials (a slow shared component), which lowers the effective trial
count when all trials are used but not when 32 are drawn from 1,489. Q13's contiguous-block test does not
exclude this: under a shared slow state its sign is ambiguous. Direct test = residual autocorrelation in
session order (Q16). [confirmed numerically]

## Q16 — trial-to-trial dependence (PREREG_Q16.md, `q16_dependence.py`, `results_q16.json`): P-NULL, and the instrument is exposed
Run 2026-09-03 ~00:20, unmodified. (a) Lag-1 autocorrelation of standardised residuals in session order:
| | M1 | V1 | DMFC | area 2 | A1 |
|---|---|---|---|---|---|
| lag-1 ACF mean (shuffle sd) | +0.002 (0.021) | **+0.234** (0.023) | **+0.129** (0.029) | +0.007 (0.049) | +0.065 (0.011) |
| lags 2-10 mean | 0.000 | +0.165 | +0.100 | +0.002 | +0.053 |
| units above shuffle 97.5% | 9% | 97% | 80% | 4% | 79% |
Dependence is real and slow in V1 (rho ~0.8 per trial, 27% of variance shared), moderate in DMFC and A1
(consistent with Q13's drift there), absent in M1 and area 2. Prediction (a) fails on M1, which has the large-m
deficit without any dependence. Prior art on the phenomenon: slow drift of population activity over tens of
minutes in V4/PFC (Cowley et al. 2020, Neuron 108:551-567); the V1 number here is that literature, not new.
(b) Dependence-matched copula AR(1) simulation at m = all: M1 10.26, V1 10.12, DMFC 10.10, area 2 10.46 vs
data 7.98 / 8.52 / 7.70 / 9.10; residual +1.6 to +2.4; it closes nothing. P-null by (b).
Instrument exposure: for area 2 and M1 the dependence parameters are zero, so this simulation IS the Q15
independent model re-drawn with another seed, yet it gives 10.46 vs 8.92 (area 2) and 10.26 vs 9.71 (M1) at
m = all. And the M1 data value at m = all is 7.98 here against 10.3 in the Q8 pipeline (RESULTS_FIG5 draft).
The full-trial level count therefore carries a realisation spread of 0.5-1.5 levels and an implementation
sensitivity of ~2 levels, and the Q15 "deficit of 1.7-2.3" has no error bar yet. Per PREREG_Q16 the next
candidate is the instrument: Q17 = seed spread of the mean curves (5 seeds, data and matched) and the
reconciliation of the Q8 vs Q15 code paths. No model is touched until that is known. [confirmed numerically]

## Q17 reconciliation, part 1 (code paths): the Q8 and Q15 level-count functions are identical; the M1 gap is bookkeeping
`q8_levels_rep.py` and `q15_matched.py` share the same split (half train / half test of the m subsample), the
same quantiser, NMAX = 16, R = 20, FLOOR = 1e-2. The M1 difference (10.14 vs 7.98 at m = all, 1.63 vs 1.46 at
m = 8, same 129 selected units) comes from the cache: `mc_maze_cache.npz` stores END offsets, the Q8 scripts
read them as starts (the documented one-unit shift, internally consistent), and the Q13-Q17 loaders use the
corrected reading while taking the powered/tuned flags from the Q8 results, so for M1 the flags and the spikes
describe different neurons. The other four caches store START offsets and are unaffected. Consequence: every
M1 number in Q13-Q16 is on a mis-selected set (still real neurons, still the same statistic, but the
selection did not apply to them); the V1/DMFC/area 2/A1 numbers stand. Fix: `q17_m1fix.py` reads M1 the Q8 way.
The Q8 M1 result itself is unchanged (consistent within its own reading). [confirmed by code inspection]

## Q17 part 2 (`q17_m1fix.py`, 5 seeds, M1 with consistent indexing): the M1 deficit survives its error bar
Data 1.55 / 3.24 / 6.48 / 10.54 (seed sd 0.04-0.24) vs matched independent 1.66 / 3.60 / 7.72 / 11.57 (sd 0.06-0.30)
at m = 8 / 16 / 32 / all; matched minus data 0.11 / 0.36 / **1.24 (sd 0.27)** / **1.03 (sd 0.20)**. The mean
curves are stable across seeds, so the Q16 "0.5-1.5 realisation spread" inference is withdrawn for this
estimator; the Q15-vs-Q16 area 2 discrepancy is examined when Q17 part 1 reports. The M1 deficit at large m
is real relative to the plug-in matched simulation. Whether the plug-in construction itself inflates the
simulation's level count is Q18. [confirmed numerically]
Q16 rerun for M1 with consistent indexing (`q16_m1fix.py`): lag-1 ACF +0.000 (median −0.001, shuffle sd 0.020),
lags 2-10 0.000, 8% of units above the shuffle band. M1 has no trial dependence on the right units either, and
its 1.0 +/- 0.2 deficit stands next to that zero. Q17 part 1 (Q8-flag set, for the record): V1 deficit
2.28 +/- 0.61 at m = all (1.62 +/- 0.77 at 32), DMFC 1.34 +/- 1.55 at m = all (n = 30, too noisy) and
1.43 +/- 0.70 at 32. [confirmed numerically]

## Q17 part 1 complete (`q17_seeds.py`, 5 seeds; `results_q17.json`): the large-m deficit with error bars
| matched − data (seed sd) | m = 8 | 16 | 32 | all |
|---|---|---|---|---|
| M1 (consistent indexing, part 2) | 0.11 (0.08) | 0.36 (0.20) | **1.24 (0.27)** | **1.03 (0.20)** |
| V1 | 0.05 (0.06) | 0.91 (0.40) | **1.62 (0.77)** | **2.28 (0.61)** |
| DMFC (n = 30) | −0.08 (0.48) | 0.92 (0.38) | **1.43 (0.70)** | 1.34 (1.55) |
| area 2 | −0.12 (0.65) | 0.12 (0.40) | 0.50 (0.41) | −0.15 (0.65) |
Seed spread of the mean curves is 0.03-0.7 levels (1.1 for DMFC's matched curve at m = all, n = 30). The
deficit is real in M1 and V1 at 32 and all trials, real in DMFC at 32, absent in area 2. The Q16 area 2 control
value (10.46) sits 3.5 seed-sd above the 5-seed matched mean (9.14) and is checked separately
(`q17_copula_check.py`). M1 shows the deficit with zero residual autocorrelation, so first-order dependence
is not its cause; the plug-in construction of the comparator is (Q18, running). [confirmed numerically]

## Q17 sampler control (`q17_copula_check.py`): numpy and copula samplers agree (area 2, 3 seeds: 9.37 +/- 0.21 vs
9.38 +/- 0.07 at m = all). The Q16 area 2 value 10.46 was a single-draw outlier, not a code difference. [confirmed numerically]

## Q18 — plug-in artefact (PREREG_Q18.md, `q18_plugin.py`, `results_q18.json`): P-ARTEFACT in M1, mostly in V1 (partial, DMFC/area 2 pending)
Run 2026-09-03 ~00:55, unmodified. stage 2 − stage 1 (2 seeds) next to the Q17 deficit, at m = 8 / 16 / 32 / all:
| | stage 2 − stage 1 (sd) | Q17 matched − data (sd) | share of deficit |
|---|---|---|---|
| M1 | 0.00 / 0.10 / 0.58 (0.50) / **0.92 (0.18)** | 0.11 / 0.36 / 1.24 (0.27) / 1.03 (0.20) | 89% at m = all |
| V1 | 0.10 / 0.95 / 1.23 (0.54) / **1.27 (0.71)** | 0.05 / 0.91 / 1.62 (0.77) / 2.28 (0.61) | 56% at all, 76% at 32, 100% at 16 |
A simulation seeded from estimated condition means resolves about one level more than the neurons it was
built from, for no biological reason: the estimation error of the seed adds between-condition spread of
variance v_c / n_c, which is exactly the near-neighbour separation the level count reads at large m.
Bias-corrected deficit: M1 0.11 +/- 0.27 at m = all, 0.66 +/- 0.57 at 32; V1 1.0 +/- 0.9 at all, 0.4 +/- 0.9
at 32. None is distinguishable from zero. The Q15 "large-m deficit" is withdrawn as a property of neurons;
it was the comparator. Every simulation in this project that seeds true rates from estimated ones (the
uniform-map comparator uses only A and lambda, so it is less exposed) carries this bias at the largest m; the
Q8 verdicts do not depend on m = all comparisons against plug-in maps (they compare growth vs plateau and use
the uniform-map comparator), but the RESULTS paragraph numbers at m = all should be read with a ~1-level
comparator bias in mind. Fix = Q19 (shrinkage-seeded comparator). [confirmed numerically]
Q18 complete. DMFC: stage 2 − stage 1 = 0.15 / 0.15 / 0.43 (0.09) / **0.77 (0.24)** against the Q17 deficit
−0.08 / 0.92 / 1.43 / 1.34 (57% at m = all, 30% at 32). Area 2: 0.57 / 0.12 / −0.08 / −0.09, no deficit to
explain and no bias at large m (its count sits near the condition ceiling). Verdict by the letter: IN BETWEEN
(P-artefact met in M1 only at the 70% bar; P-real met nowhere). Bias-corrected residual deficits at m = all:
M1 0.11 +/- 0.27, V1 1.0 +/- 0.9, DMFC 0.6 +/- 1.6, area 2 −0.06 +/- 0.8; all consistent with zero, V1 and DMFC
not precisely so. Q19 (shrinkage-seeded comparator, 3 seeds) is the clean version of this comparison.

## Q19 — shrinkage-seeded comparator (PREREG_Q19.md, `q19_shrink.py`, `results_q19.json`, 3 seeds): P-FIXED for M1/V1, mild P-OVER in area 2
Run 2026-09-03 ~01:20, unmodified. shrunk − data (seed sd) | plug-in − data, at m = 8 / 16 / 32 / all:
| | shrunk − data | plug-in − data |
|---|---|---|
| M1 (consistent idx) | −0.17 (0.04) / +0.07 (0.21) / −0.01 (0.15) / −0.10 (0.64) | +0.03 / +0.46 / +1.10 / +1.43 |
| V1 | −0.04 / −0.21 / −0.37 (0.34) / −0.37 (0.21) | +0.06 / +0.39 / +1.48 / +1.85 |
| DMFC (n = 30) | −0.51 (0.66) / −0.14 / +0.07 / +0.20 (1.01) | +0.02 / +0.22 / +2.72 / +1.13 |
| area 2 | −0.56 / −0.51 / −0.61 (0.59) / −0.32 (0.09) | −0.35 / +0.41 / +0.50 / +0.24 |
Reading: the two datasets with the deficit (M1, V1) are reproduced within 0.4 level at every m once the seed
is de-noised; DMFC within 0.2 at 32 and all trials; area 2 is over-corrected by 0.3-0.6 (shrinkage removes
real fine structure in a well-resolved 23-trial map). Plug-in and shrunk comparators bracket the data
everywhere. Conclusion for the paper: the level count as a function of trials is reproduced, in every
dataset, by the unit's own rate map plus independent noise matched to each condition's measured variance.
No noise structure beyond the first two per-condition moments is needed. Fig 5 primary comparator becomes
the shrinkage-seeded one, with the plug-in as the upper bracket; the uniform-map comparator stays for the
discrete-vs-continuous contrast (it does not seed from estimated means). [confirmed numerically]

## Q20 — one-constant analytic resolution rule (PREREG_Q20.md, `q20_analytic.py`, `results_q20.json`): P-NULL
Fitted on M1: c* = 0.75, yet the rule gives 3.37 / 4.71 / 6.38 / 8.21 against data 1.64 / 3.12 / 6.37 / 10.29;
held out: V1 max |diff| 2.52, DMFC 1.74, area 2 2.53, A1 0.47. The cross-validated count grows faster with
trials than the number of groups separated by c standard errors does, in every dataset; a greedy grouping is
not the statistic. Dropped per the pre-registration; the matched-noise simulation stays the comparator.
[confirmed numerically]

## Q21 — discrete vs continuous, both from the unit's own de-noised map (PREREG_Q21.md, `q21_discrete_own.py`, `results_q21.json`, 3 seeds): P-CONT
Run 2026-09-03 ~01:45, unmodified. N* median 2 in every dataset. m = 8 / 16 / 32 / all:
| | data | continuous-own | discrete-own | |cont−data| 32/all | |disc−data| 32/all | units nearer continuous (all / 32) |
|---|---|---|---|---|---|---|
| M1 (129) | 1.62 / 3.05 / 6.30 / 10.40 | 1.49 / 2.92 / 6.02 / 9.67 | 1.21 / 1.52 / 1.99 / 2.37 | 0.29 / 0.73 | 4.32 / 8.02 | 76% / 65% |
| V1 (73) | 1.37 / 2.27 / 6.66 / 8.64 | 1.34 / 2.18 / 5.90 / 8.51 | 1.26 / 1.47 / 1.92 / 2.14 | 0.76 / 0.13 | 4.74 / 6.50 | 65% / 62% |
| DMFC (30) | 3.54 / 6.24 / 8.20 / 8.36 | 3.30 / 6.52 / 8.23 / 8.39 | 1.88 / 2.48 / 2.39 / 2.54 | 0.03 / 0.03 | 5.81 / 5.81 | 59% / 62% |
| area 2 (48) | 4.75 / 7.55 / 9.03 / 9.67 | 4.24 / 7.30 / 8.89 / 9.03 | 2.30 / 2.74 / 2.95 / 3.25 | 0.15 / 0.63 | 6.08 / 6.42 | 71% / 70% |
| A1 (117) | 1.48 / 1.85 / 2.42 / 4.42 | 1.44 / 1.82 / 2.32 / 4.39 | 1.26 / 1.50 / 1.85 / 2.20 | 0.10 / 0.03 | 0.57 / 2.22 | 86% / 55% |
Verdict: P-cont (dataset means 5/5; per-unit >= 60% in 4/5, DMFC 59%). With the discrete hypothesis allowed to
place its certified N* levels where the unit's own map puts the mass, and with the seeding bias removed from
both sides, the discrete comparator plateaus at 2-3 levels at every trial count and the data climb to 8-10;
the continuous own-map comparator tracks the data within 0.8 level everywhere. This is also the direct
control that the statistic does not grow on a truly discrete map (discrete-own is flat in m). This is the
version of the Q8 test the paper should lead with. [confirmed numerically]

## Q8 M1 rerun with corrected unit offsets (`q8_levels_rep_fixed.py` -> `results_q8_rep_fixed.json`, 476 s): no conclusion changes
| W = 200 ms | original (shifted index) | corrected |
|---|---|---|
| powered / tuned units | 153 / 129 | 158 / 136 |
| mean N_cv at 8 / 16 / 32 / all | 1.63 / 3.37 / 6.19 / 10.14 | 1.48 / 3.21 / 6.19 / 10.00 |
| continuous sim | 1.73 / 3.47 / 6.89 / 9.71 | 1.69 / 3.51 / 6.93 / 9.71 |
| discrete sim | 3.18 / 3.19 / 2.79 / 2.33 | 3.13 / 3.18 / 2.77 / 2.39 |
| nearest continuous at all trials | 71% | 74% |
| plateau at 2 (flat 32 -> all) | 0.8% | 3.7% |
| growth 32 -> all | 80% | 77% |
| K1 shuffle N_cv = 1 | 529 / 529 | 530 / 532 |
The corrected file is the one the paper uses; drafts quoting 1.7 -> 10.3 (M1) become 1.5 -> 10.0, and the
plateau range across datasets stays 1-13%. [confirmed numerically]

## Q21 Amendment 1 — retina, own-map comparators (`q21_retina.py`, `results_q21_retina.json`, 2 seeds, fresh 36-bin draws): P-CONT at 50/100 ms, P-MIXED at 16.7 ms
| window | n | data (8/16/32/all) | continuous-own | discrete-own | |cont−data| 32/all | |disc−data| 32/all | nearer continuous (all / 32) |
|---|---|---|---|---|---|---|---|
| 16.7 ms (Bernoulli) | 226 | 1.88 / 2.49 / 3.26 / 4.42 | 1.54 / 1.89 / 2.43 / 3.41 | 1.43 / 1.75 / 2.10 / 2.36 | 0.83 / 1.02 | 1.15 / 2.06 | 59% / 55% |
| 50 ms | 323 | 2.87 / 3.72 / 4.75 / 6.20 | 2.45 / 3.15 / 3.99 / 5.30 | 2.10 / 2.48 / 2.72 / 2.67 | 0.76 / 0.90 | 2.03 / 3.53 | 69% / 61% |
| 100 ms | 359 | 3.43 / 4.61 / 5.90 / 7.53 | 3.02 / 4.07 / 5.23 / 6.80 | 2.29 / 2.58 / 2.75 / 2.78 | 0.67 / 0.73 | 3.16 / 4.74 | 74% / 68% |
The discrete own-map comparator is flat at 2.1-2.8 levels at every window; the data reach 4.4-7.5 and grow at
every step. At 16.7 ms the per-unit fraction (59%) sits just under the 60% bar, so P-mixed by the letter with
the mean favouring continuous; this is the window Q8 flagged as nearest-discrete at low repeats, and even
here the own-map discrete comparator does not reach the data. Caveat: the shrunken continuous comparator
undershoots the retina data by 0.7-1.0 at the full repeat count (over-correction, as in area 2), so the data
are bracketed from below here and the plug-in comparator would bracket from above. Six datasets, one answer,
with the fairest comparators. [confirmed numerically]

## Naming note (02:50)
The Q18 two-stage check is the classical parametric-bootstrap estimate of estimator bias (Efron and
Tibshirani 1993, ch. 10): fit, simulate from the fit, re-fit, and read the bias as the difference. The paper
should name it that way and cite it; nothing about the check is new, only its application to the level-count
comparator. Shrinkage seeding (Q19) is empirical-Bayes de-noising of the seed, also standard. [named]
Retina, plug-in-seeded comparators (`q21_retina_plugin.py`, `results_q21_retina_plugin.json`, 02:40): the
continuous own-map comparator seeded from the raw condition means sits within 0.02-0.13 level of the data at
every window and every m (16.7 ms: 1.92 / 2.50 / 3.20 / 4.40 vs data 1.88 / 2.49 / 3.26 / 4.42; 50 ms: 2.75 /
3.61 / 4.63 / 6.08 vs 2.78 / 3.67 / 4.72 / 6.15; 100 ms: 3.47 / 4.59 / 5.84 / 7.59 vs 3.42 / 4.64 / 5.97 /
7.52); discrete-own stays at 2.3-2.9. Units nearer continuous 65% / 74% / 78% at m = all, so with plug-in seeds
retina is P-cont at all three windows including 16.7 ms. The plug-in bias that reached ~1 level in cortex is
absent in retina, and the shrunk comparator's undershoot there is over-correction. The anomaly candidate
("data more separable than independent matched noise") is CLOSED: the data are bracketed in every dataset.
[confirmed numerically]

## Q23 — pilot-to-full prediction (PREREG_Q23.md, `q23_pilot.py`, `results_q23.json`, 2 seeds): IN BETWEEN, a bracket not a calculator
From 16 trials per condition, predicted N_cv at m = 32 / all vs data:
| | data | shrunk-pilot (err) | plug-in-pilot (err) |
|---|---|---|---|
| M1 | 6.54 / 10.57 | 6.14 / 9.73 (−0.40 / −0.83) | 9.63 / 13.20 (+3.09 / +2.64) |
| V1 | 6.03 / 8.44 | 6.69 / 8.84 (+0.66 / +0.40) | 10.49 / 12.84 (+4.45 / +4.40) |
| DMFC | 7.60 / 7.88 | 8.78 / 8.52 (+1.18 / +0.63) | 10.75 / 11.07 (+3.15 / +3.18) |
| area 2 | 9.01 / 9.50 | 8.47 / 8.25 (−0.54 / −1.25) | 9.98 / 10.23 (+0.97 / +0.73) |
| A1 | 2.52 / 4.39 | 2.48 / 3.31 (−0.04 / −1.08) | 3.06 / 4.19 (+0.54 / −0.20) |
P-tool condition 1 (shrunk within 1.0 at m = all in >= 4/5) fails at 3/5; condition 2 (data bracketed or
within 0.5 of a prediction in >= 4/5) passes at 4/5; P-fail not met. The plug-in seed from 16 trials
overshoots by 2.6-4.4 levels in cortex (seed error four times that of the full data), the shrunk seed
undershoots by up to 1.25. Usable statement: a 16-trial pilot brackets the full-trial level count between
its shrunk and plug-in simulations in 4 of 5 datasets, with a bracket 3-4 levels wide in cortex. A bracket
that wide is a caveat for the supplement, not a tool to advertise. [confirmed numerically]

## Q24/Q25 — CRCNS route (2026-09-03 morning): DATA WALL, then DATASET KILL
Q24 (graded-intensity test): no CRCNS dataset passes the pre-registered bar (PREREG_Q24 Amendment 1): ac-2's
intensity axis is 4 amplitudes in 7 cells, 6 of them under TTX; no other auditory or visual set states an
intensity or contrast series; the site search finds none. Wall material: DATA.
Q25 (DeWeese "binary spiking" in its own preparation, ac-2 anaesthetised set, 33 whole-cell cells, 32 tones at
65 dB, 11-98 repeats per tone; `q25_deweese.py`, `results_q25.json`): the stimulus-locked traces contain NO
spikes in 31 of 33 cells at any threshold (+25 to +55 mV, including the authors' own per-cell `spikethresh`),
1 and 4 spikes in the other two; raw 8-s sweeps confirm 0-6 spike-like events per whole recording. The
release is a membrane-potential dataset (QX-314 in the pipette for most cells, consistent with the 2004/2006
papers it comes from); the 2003 binary-spiking cells are not in it. Pre-registered kill condition ("no way to
recover spike times") fires: no verdict on P-binary / P-Poisson. The awake-mouse A1 result (DANDI 000986)
stays the only test of the DeWeese regime in hand. [confirmed numerically; 253 MB fetched from CRCNS,
md5 verified]
Wall anatomy: material = data (the relevant cells were never released); method through = ask the DeWeese lab
for the 2003 cell-attached recordings, or any A1 lab with an intensity series; behind it =
a real answer to "is binary spiking a preparation effect", small but referee-relevant, not world-changing.

## Q26 — grating contrast as the graded axis (PREREG_Q26.md + Amendments 1-2; Allen FC session 779839471, 241 VISp good units; `q26_contrast.py`, `q26_verdict.py`, `results_q26*.json`): session 1
Design (from the table): 9 contrasts x 4 orientations x 15 presentations, 0.5 s. N* mostly 2 (A per map
0.05-9 spikes). Per-orientation contrast maps (9 conditions x 15 repeats), tuned = N_cv(15) >= 2 and K1 pass:
| window | tuned maps | N_cv(15) mean (max 9) | <= N*+1 | > N*+1 | growth 8->15 | cont-own (shrunk / plug-in) | disc-own | nearer continuous (shrunk / plug-in) | verdict (a) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 s | 164 | 3.90 | 51% | 49% | 87% | 3.63 / 4.31 | 2.26 / 2.52 | 64% / 60% | P-mixed by the stair/graded split; P-cont by the Q21 rule |
| 0.5 s | 219 | 4.19 | 48% | 52% | 82% | 3.86 / 4.69 | 2.62 / 2.89 | 57% / 57% | P-mixed |
36-cell contrast x orientation map: 0.5 s, 84 tuned units, N_cv(15) 6.31 vs continuous-own 6.36 (shrunk) /
7.71 (plug-in) vs discrete-own 2.82 / 3.10; nearer continuous 62% / 56%; growth 83% -> P-cont (shrunk). 0.25 s:
57 tuned, 4.91 vs 5.09 / 6.82 vs 2.53 / 2.75; nearer continuous 58% / 37% (plug-in overshoots at 15 repeats,
the Q18 bias).
Reading: along a single graded axis with 15 repeats the per-unit test is at its power limit (half of the
tuned maps resolve <= 3 levels, which a continuous map also does at this repeat count), so the pre-registered
stair/graded split lands at 50/50 and the verdict is P-mixed by the letter. The comparator reading is not
mixed: the data sit within 0.3 level of the continuous own-map comparator and 1.3-1.6 above the discrete one,
82-87% of tuned maps still gain levels from 8 to 15 repeats, and the 36-cell map reproduces the earlier V1
result (6.3 levels vs 2.8 discrete). Same direction as the six datasets, lower power per unit. Replication on
session 794812542 running with the same scripts. [confirmed numerically]
### Q26 replication, session 794812542 (226 VISp good units, identical design), same scripts
| window | tuned maps | N_cv(15) mean | <= N*+1 | > N*+1 | growth | cont-own (shrunk / plug-in) | disc-own | nearer continuous | verdict (a) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 s | 150 | 3.54 | 59% | 41% | 83% | 3.54 / 4.03 | 2.23 / 2.55 | 57% / 51% | P-mixed |
| 0.5 s | 203 | 3.77 | 53% | 47% | 77% | 3.88 / 4.29 | 2.60 / 2.81 | 59% / 55% | P-mixed |
36-cell map, 0.5 s: 89 tuned, 4.51 vs continuous-own 5.28 / 6.41 vs discrete-own 2.64 / 2.96; nearer continuous
54% / 38%; growth 80% -> P-mixed (session 1 was P-cont at 62%).
Two sessions agree to a tenth of a level on the contrast axis: data 4.19 and 3.77, continuous-own 3.86 and
3.88, discrete-own 2.62 and 2.60, nearer-continuous 57% and 59%, growth 82% and 77%. Consolidated Q26 verdict:
P-MIXED by the pre-registered per-unit bars in both sessions; the dataset-level comparison favours the
continuous own-map comparator in both (within 0.3 level, against 1.2-1.6 above the discrete one), and 77-87%
of tuned maps still gain levels between 8 and 15 repeats. In the theory's best regime on a public graded axis
the discrete hypothesis does not win; with 15 repeats the per-unit test cannot reject it at the 60% bar
either. The limit is the Allen design (15 presentations per contrast), stated as a power limit, not as
support. Figure: figs/fig_q26_contrast.png (both sessions). [confirmed numerically]

## Q27 — who are the plateau units? (PREREG_Q27.md, `q27_class.py`, `results_q27.json`): P-SNR
256 tuned VISp units with public waveform and depth metrics (static 83, contrast sessions 84 and 89; plateau
rate 0.26). Cross-validated AUC for plateau status: SNR baseline (log A, log Fano, log peak) 0.771 +/- 0.010;
with waveform duration, peak-trough ratio, depth and firing rate added, 0.766 +/- 0.007 (delta −0.005; per
session −0.017, −0.072, −0.079: the class variables only overfit). Pooled coefficients: log Fano +0.80 [+0.44,
+1.30] (the SNR direction), peak-trough ratio −1.05 [−3.06, −0.03] but −5.9 / +0.35 / −0.63 across sessions
(sign not stable), firing rate +0.66 [+0.01, +1.58] pooled but never within a session. P-SNR by the letter:
the two-level minority is the resolution floor. Closed. [confirmed numerically]

## Q29 — is the normalised within-neuron map shape universal? (PREREG_Q29.md, `q29_universal.py`, `results_q29.json`): split by the letter, one cortical shape and one retinal shape in structure
Eight datasets with >= 16 conditions, tuned units, condition means from 13 repeats (equalised) or all, shrunk,
normalised to [0, 1]; pooled CDF per dataset; KS between datasets against a split-half null.
| dataset (units) | median x, eq / all | frac < 0.25, eq / all | skew, eq / all |
|---|---|---|---|
| M1 (128) | 0.42 / 0.31 | 0.34 / 0.44 | +0.14 / +0.57 |
| area 2 (48) | 0.38 / 0.35 | 0.37 / 0.39 | +0.30 / +0.43 |
| DMFC (30) | 0.40 / 0.36 | 0.32 / 0.37 | +0.35 / +0.43 |
| V1 static (72) | 0.39 / 0.28 | 0.32 / 0.46 | +0.34 / +0.83 |
| V1 drifting (47) | 0.32 / 0.32 | 0.42 / 0.43 | +0.49 / +0.50 |
| V1 contrast s1 (84) | 0.30 / 0.29 | 0.44 / 0.45 | +0.55 / +0.61 |
| V1 contrast s2 (89) | 0.25 / 0.24 | 0.50 / 0.52 | +0.87 / +0.93 |
| retina 50 ms (250 / 311) | 0.00 / 0.00 | 0.86 / 0.86 | +2.52 / +2.54 |
Pairs "same shape": equalised 19/28 (68%, P-mixed by the bar); all repeats 21/28 (75%, P-universal by the
bar). Structure: among the seven CORTICAL datasets 19/21 pairs agree at equalised repeats (the two
exceptions involve contrast session 2, the sparsest) and 21/21 with all repeats, KS 0.03-0.16; the retina
differs from every cortical set by KS 0.60-0.78. The sparseness criterion (within 0.10 of the pooled value in
every dataset) fails only because the pooled value includes the retina. Reading: one normalised map shape
across three macaque areas and mouse V1 under three stimulus classes, and a distinct, far sparser retinal
shape at 50 ms bins under natural movies. Occupancy against the lifetime-sparseness literature to be checked
before any claim; the Treves-Rolls index per dataset is computed next for that comparison. [confirmed numerically]
Q29 occupancy (`results_q29b_sparseness.json`): Treves-Rolls lifetime sparseness, median per dataset: M1 0.72,
area 2 0.70, DMFC 0.76, V1 static 0.80, V1 drifting 0.67, V1 contrast 0.68 / 0.62, retina 0.11. The cortical
band (0.6-0.8) is the published range for visual cortex and IT, and a constant lifetime sparseness across
V1, V2 and V4 is already a stated result (Willmore, Mazer & Gallant 2011, J Neurophysiol, "Sparse coding in
striate and extrastriate visual cortex"); extreme retinal sparseness to natural movies is Berry & Meister
territory. Q29 therefore EXTENDS a known regularity to motor, somatosensory and frontal cortex with a
distribution-level (KS) instrument; it is a supplement figure and a Discussion sentence, not a frontier.
Recorded as OCCUPIED IN KIND. [confirmed numerically; citation to verify]

## Q30 — open-problem lane: support size of the capacity-achieving input of the amplitude-constrained Poisson channel (PREREG_Q30.md + Amendments 1-3; `support_exact_v3.py`, `support_gauss_v3.py`, `results_q30_v3.json`, `results_q30_gauss_compare.json`)
Problem as stated in print: Dytso-Barletta-Shamai 2021: Omega(sqrt A) <= N*(A) <= O(A log^2 A); Dytso et al. 2024: O(A).
Certified staircase (KKT slack <= 1e-6; v3 solver validated on A* = 3.3679): N* = 4, 5, 5, 6, 7, 7, 8, 9, 11, 12, 14,
20, 23 at A = 10, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250; within +/-1 of 1 + sqrt(A) up to 150, then
above it; low mass points A-independent (0, 2.44, 7.25, 13.6, 21.1, 29.7, 39); spacing 2.0-2.4 sqrt(x) up to 150,
1.45 sqrt(x) by 250.
Correspondence (new, tested 14:10): the amplitude-constrained AWGN channel (unit noise, input [0, a]) solved with
the same certified method at a = 2 sqrt(A) gives N_Gauss = 4, 5, 5, 6, 7, 8, 8, 9, 11, 12, 14 at the same A list up
to 150: identical to the Poisson count at 10 of 11 amplitudes (off by one at A = 50); Gaussian spacing 1.8-2.5
noise units matching the Poisson coefficient. The variance-stabilising map t = sqrt(x) (sqrt(Y) ~ N(sqrt x, 1/4))
therefore carries the support size across, at least at finite A. Consequence: the AWGN lower bound of order
a sqrt(log a) (Wang, Barletta & Dytso, arXiv 2512.22691, Dec 2025; super-linear growth, Wang arXiv 2510.20723)
transfers heuristically to N_Poisson(A) >= c sqrt(A log A), above the 2021 Omega(sqrt A); the certified counts
beyond A = 150 already bend upward. Limit stated: the KKT certificate proves epsilon-optimality, so at large A
the count is an epsilon-support size; the exact order is a theorem question (Q31, the proof attempt redirected to a
transfer theorem). [confirmed numerically; theorem pending]
Q30/Q31 status (17:00): Poisson epsilon-support 44/58/76 at A = 500/700/1000; Gaussian correspondence +/-1 at every
A from 10 to 400 (figs/fig_q30_support.png). Exact ingredients of the AWGN super-linear bound read from
WBD 2512.22691 (wrapping chi^2 bound + O(1/A) stability) and the stated open problem quoted; Poisson transfer plan
(I')/(II') written in PREREG_Q31 Amendment 2; the proof attempt resumed on it after a pause.
Q31 numerics (17:35): approximation barrier min chi^2 ~ exp(-8 K^2/A) (Poisson vs Jeffreys reference) and
exp(-7.3 K^2/A) (Gaussian at a = 2 sqrt A vs uniform reference); stability D(Q_J || Q*) ~ A^{-0.3}. The published
Gaussian constant (39.5) is loose by ~5x. Theorem pending. [confirmed numerically]
Q31 honesty check (2026-09-04): the certified staircase cannot discriminate sqrt(A) from sqrt(A log A) (both flat
over A = 50-150); large-A counts are epsilon-inflated. The super-sqrt claim rests on the correspondence plus the
published AWGN theorem. Recorded in Q31_PROOF_SKELETON.md Step 7.

## Q26 aggregate over 17 Allen sessions (`q26_aggregate.py`, `results_q26_aggregate.json`): the contrast-axis test at full sample
3308 tuned contrast maps (9 contrasts x 15 repeats, 0.5 s) and 1223 tuned 36-cell maps, from 17 functional-
connectivity sessions (24 of the 26 have VISp units; the batch was stopped after 17 because the numbers had
stabilised).
| quantity | mean | range across sessions |
|---|---|---|
| N_cv at 15 repeats, per-orientation | 4.32 | 3.77-4.77 |
| continuous-own comparator | 4.08 | 3.74-4.56 |
| discrete-own comparator | 2.64 | 2.35-2.88 |
| fraction of maps at or below N*+1 | 45.5% | 33.7-53.2% |
| fraction above N*+1 | 54.5% | 46.8-66.3% |
| growth 8 -> 15 repeats | 77.5% | 72.4-82.6% |
| units nearer the continuous comparator | 61.0% | 55.1-65.1% |
| 36-cell map: data / continuous-own / discrete-own | 6.03 / 6.42 / 2.92 | 4.51-8.79 |
| 36-cell nearer continuous | 59.5% | 45.9-80.0% |
Verdicts. By the pre-registered stair/graded split the result stays P-MIXED at every session (the split is
45/55 pooled, and no session reaches the 60% bar on either side). By the PREREG_Q21 comparator rule the pooled
fraction nearer the continuous own-map comparator is 61.0%, above the 60% bar, and 10 of 17 sessions clear it
individually (7 of 17 on the 36-cell map); growth from 8 to 15 repeats holds in 72-83% of maps in every session. The discrete comparator sits
at 2.6 to 2.9 levels everywhere while the data reach 4.3 (per orientation) and 6.0 (36 cells).
Reading: with 3308 maps instead of 219, the contrast axis gives the same answer as the other six datasets by the
comparator reading, and remains power-limited by the 15-repeat design on the per-unit staircase test.
[confirmed numerically]

---

## Correction to the Q8b hatch verdicts (2026-09-05). See NEURAL_LANE_AUDIT_2026-09-05.md
The hatch was declared CLOSED in all seven runs against a pre-registered bar of 30%, chosen to match the 70/30
convention of Q8. That bar was never compared with the test's own false-positive rate, and it sits above every
observed value, so it could not discriminate. Three things follow.
1. The specificity quoted per dataset (0.80 to 0.85) is the mean over ALL tested pairs, while the reported
   "mixture wins" fraction is over POWERED pairs. Different populations. The specificity over powered rows is
   0.94 to 0.99, but that figure is selected on its own noise: it comes from five simulations with rows kept at
   0.8 or above. A fresh measurement at fourteen simulations on the same M1 powered rows gives 0.862, not 0.981,
   so the whole gap is selection. On the unselected population fresh and stored agree (0.824 against 0.846), so
   the estimator is fine and only the filter is at fault.
2. Seeding the smooth simulation from empirical condition means inflates it, exactly as Q18 found for the
   level-count comparator. Shrinkage seeds raise M1 specificity from 0.862 to 0.887 and V1's from 0.787 to 0.858.
3. Even after both corrections M1 sits five standard errors BELOW its own null, so the simulation is still not a
   faithful null and no "exceeds chance" claim from it is trustworthy.
WHAT SURVIVES WITHOUT A NULL. The same statistic and pipeline across datasets: motor and somatosensory pooled
16/600 = 2.7%, sensory pooled 152/880 = 17.3%. The gap holds under stratification on dynamic range, baseline
rate and Fano (Mantel-Haenszel odds ratio 5.5 to 6.1), under window matching where the windows overlap (retina
12.7% against motor 3.0% at 100 ms, z = 3.43), and at cluster level every sensory rate exceeds every motor rate
(probability 1/35 = 0.029).
STATUS OF THE VERDICTS. Closed stands at M1, area 2 and DMFC. WITHDRAWN at retina, A1, V1 static and V1
drifting, where the correct label is that the hatch test shows a large excess and cannot be called closed.
Whether that excess is about sensory coding or about how those recordings were made needs matching on trials per
condition, which the stored rows do not carry; the hatch test has to be re-run keeping those covariates.
