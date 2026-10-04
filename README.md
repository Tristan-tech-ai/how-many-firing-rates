# How Many Firing Rates Should a Neuron Use? Supplementary material

Code, certificates and result files for the paper

> I Made Tristan Hope Firdaus, *How Many Firing Rates Should a Neuron Use? Poisson and Non-Poisson Spike Counts in Long and
> Short Windows*, working paper, 2026. The PDF is in [`paper/`](paper/).

A neuron whose firing rate a stimulus can set anywhere between silence and a peak, read through its spike count in a window, is a
noisy channel from rates to counts. The paper asks how many distinct rates the information-maximizing code uses, in long and in
short windows, for Poisson neurons and for neurons more regular or more variable than Poisson, and tests the prediction on six
public datasets.

## Where to find what

| folder | paper section | what it holds |
|---|---|---|
| [`sec3-certificates/`](sec3-certificates/) | III, Appendix A, Table IV | the ball-arithmetic certificates that fix the exact count at 35 peak counts (runs on its own) |
| [`sec3-count-law/`](sec3-count-law/) | III, Table II, Fig. 1 | Gaussian count changes (with the relocated values of Table II), the wall constants and the growth predictions |
| [`sec4-non-poisson/`](sec4-non-poisson/) | IV | binomial, negative-binomial and generalized-Poisson wall chains; the finite-size term |
| [`sec5-baseline/`](sec5-baseline/) | V | the baseline (dark-current) rate and the comparison with published count changes |
| [`sec6-short-windows/`](sec6-short-windows/) | VI, Figs. 2-3 | short-window optima, the exact binary onset A = 3.3679, energy-capacity fronts, refractoriness |
| [`sec7-data-test/`](sec7-data-test/) | VII, Table III | the pre-registered level-count test on six public datasets |
| [`checks/`](checks/) | all | the code that checked every number and figure of the paper against primary files, with what it reads |
| [`paper/`](paper/) | | the paper |

Each folder has a `README.md` (what is there and how it maps to the paper) and a `FILES.md` (one line per script: what it does,
and the name it had when it ran). The project's registration log, in which predictions were written before they were tested, is
public in the companion repository: [how-many-atoms/registration-log](https://github.com/Tristan-tech-ai/how-many-atoms/tree/main/registration-log).

## Two kinds of folder

**`sec3-certificates/` can be run.** The scripts read their input states from their own folder. We reran the control certificate
(peak count A = 100, eleven rates) from this folder before publishing; `logs/rerun_q32_certificate_v4_A100.log` and
`logs/rerun_results_q32_certificate_v4_A100.json` are that run, and `logs/results_q32_certificate_v4_A100.log` and `.json` the
original. The two agree in every printed number. One line differs in form: the script gained a lemma for the interval (0, 1e-8]
after the original run, and the rerun prints it; this is the form that Appendix A of the paper describes.

**The other folders are records.** They hold the scripts as they ran and the result files they wrote, with the original file
names. Some scripts expect the folder layout of the original project, so rerunning one can mean adjusting a path at the top of
the script. The Python files were reformatted with [black](https://github.com/psf/black), which changes layout only.

Three result files are kept although the short scripts that wrote them were not. `sec3-count-law/logs/results_q89_cK_table.json`
is an earlier table of wall constants, now superseded; Table II is rebuilt by `sec3-count-law/scripts/build_table2.py` from
output files only. `sec4-non-poisson/logs/results_q138_kappa_points.json` holds the points behind kappa = 0.42, which
`checks/scripts/kappa_counts.py` recounts from the run logs. `sec5-baseline/logs/results_q111_bd_comparison.json` is the
comparison with the published dark-current count changes. Their numbers can be checked against the certificates and against the
published data they compare with.

## How the numbers were checked

Every number printed in the paper was checked by code against a primary file: a run log, a result file, or an entry of the
registration log. The figures are drawn by scripts that read their data from such files. [`checks/`](checks/) holds that code,
the formula behind each derived number, and the input files that the other folders do not already hold.

## Requirements

Python 3.10 or later and the packages in [`requirements.txt`](requirements.txt). `mpmath` and `python-flint` carry the rigorous
steps. The data test also needs `h5py` and `remfile` to stream the public recordings.

## Quick start: one certificate

```
cd sec3-certificates/scripts
python q32_certificate_v4.py 100 1e-30
```

About 30 minutes on one core. The last line should read `D(x; F*) <= C(F*) on [0, A] for the exact KKT point F* in the Krawczyk
box: PROVED (all three pieces True)`. See [`sec3-certificates/README.md`](sec3-certificates/README.md).

## Data

The six datasets of Table III are public: the Neural Latents Benchmark sets MC_Maze, Area2_Bump and DMFC_RSG (DANDI 000128,
000127, 000130), larval salamander retina (Dryad 10.5061/dryad.4qrfj6qm8), mouse V1 (Allen Visual Coding Neuropixels, DANDI 000021)
and mouse A1 (DANDI 000986). They are not copied here; `sec7-data-test/scripts/cache_dandi.py` streams the DANDI sets.

## License and citation

Code: MIT License (see [`LICENSE`](LICENSE)). If you use this material, please cite the paper above.

Contact: I Made Tristan Hope Firdaus, Independent Researcher, Bali, Indonesia (madetristanfirdaus@gmail.com, ORCID
0009-0002-3048-3624).
