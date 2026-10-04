# CYCLE 2 READOUT (2026-09-02) — Arms 2 and 3 closed; one artefact found that corrects the record

## Arm 3 — the two "anomalies" dissolve on refinement (owned computation)
- **One-shot codebooks** (`oneshot_refine.py`, exact MAP, 0.5-1 Hz grid): M=3 -> {0, 20, 60} Hz, M=4 ->
  {0, 20, 40, 60} Hz, M=5 -> {0, 1, 20, 40, 60} Hz. In a 50 ms window these are mean counts {0, 1, 2, 3}:
  **one codeword per integer count up to the peak**, extremes always present, and the fifth codeword is a
  near-duplicate of silence. The earlier "[5,21,38,54]" was a misremembering of the 3 Hz-grid result
  {0,21,39,60}. Not an anomaly; a count-resolution packing. [confirmed numerically]
- **Duty cycle 0.404 vs 1/e**: `frontier/results_law.json` shows 0.404 is `p_mean`, the mean over all rows of
  the already-RETRACTED 1/3-law test, including degenerate rows with p = 1.0. Not a quantity. [on disk]

## Arm 2 — occupied at the identity level (downloaded and grepped)
`papers/sagawa_1206.2479.txt` line 237: **"⟨e^{−σ+∆I}⟩ = 1"** (their Eq. 3), with I the stochastic mutual
information (lines 206-212). The mapping "information density as a thermodynamic work-like variable with an
integral fluctuation theorem" is Sagawa-Ueda PRL 109, 180602 (2012). The neural specialisation adds nothing
the identity does not already say. [found-with-citation] Closed.

## The find of the cycle — the graded-code onset was a Blahut-Arimoto artefact
Triggered by an occupancy search that returned "binary optimal iff A <= 3.3679" while our onset was 2.6.
- `papers/dytso_2401.05045.txt` line 366, **Proposition 1: "For A + λ < e, |supp(P_X⋆)| = 2"**; the exact
  threshold is attributed to Cao, Hranilovic, Chen, IEEE Trans. Commun. 62(1) 2014, "Part II: Binary inputs".
- `peak_amplitude/results_peak.json` `supp_cap` = 5,5,5,4,5,5,4,4,5,5 across L = 52..70: non-monotone,
  i.e. BA mass-spreading near the transition, counted as symbols by `SUPP_TOL = 1e-3`.
- **Own certificate** `binary_cert2.py`: interior KKT slack max_{0<x<A}[D(P_x||Q) − C] is negative through
  A = 3.36 and positive from A = 3.37 (x ≈ 1.29). Bisection **A* = 3.3679** — the literature value to four
  decimals, from an independent method. [proven condition, confirmed numerically]
- Consequences: L* = 67.4 Hz at 50 ms (not 50-52); Kostal-Shinomoto's L = 50 Hz cap is **26% below** the
  onset (not 4%): clearance, not knife-edge; T*(L) = 3.3679/L; retina shortfall ~40% (not 20-30%); the
  rho-vs-A numbers withdrawn (qualitative point exact by construction: the channel depends on L, T only
  through A = LT). `glm_contrib1/` contrast rows (36-46%) need recertification with the same certificate.
- Record corrected in: `peak_amplitude/READOUT.md` (retraction section).

## Lesson
Never read a support size off BA mass near a transition. The KKT slack is the certificate, it is cheap, and
it is exact. This is the same failure class as the residual-mass atom count in P2 (12 atoms that were 3).

## Addendum (cycle 3) — GLM Part B recertified twice
- `glm_recert.py` (K = 12 grid, exact KKT on the grid): graded rows 16/15/14/10 of 24 per kernel vs the
  old mass-count 16/14/13/11. One-row differences only. But the grid alternated GRADED/BINARY/GRADED with
  budget, which exposed the second artefact: on a coarse grid, one interior level between two grid rates is
  written as two adjacent support points and counted as "graded".
- `glm_recert2.py` (0.5 Hz grid, continuum sense): graded 14/11/9/8 of 24. Contrast recount **84/192 =
  44%** of transmitting (E, n) rows (old 60/130 = 46%; the old denominator used a stricter mask). The
  qualifier: the graded advantage over the best binary code is <= 3.8e-3 nats in every row, under 1% of C.
  [confirmed numerically] Email draft updated with the certified figure and the qualifier.
