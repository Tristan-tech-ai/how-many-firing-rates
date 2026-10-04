# Section III, Appendix A and Table V: certificates of the exact number of rates

Each certificate proves, in directed-rounding or ball arithmetic, that the capacity-achieving input of the Poisson channel with
peak count A has exactly K mass points. The argument has three parts: a Krawczyk test shows that the optimality equations have
exactly one solution in a small box around a 40-digit numerical point; the optimality function is then shown to stay below the
capacity on all of [0, A] (a lemma near silence, cubic bounds around each level, chord bounds on every other cell); KKT sufficiency
and the uniqueness of the optimum (Shamai 1990) then give N(A) = K.

## Run it

```
cd scripts
python q32_certificate_v4.py 100 1e-30        # control: A = 100, K = 11, about 30 minutes
```

| script | role |
|---|---|
| `q32_krawczyk.py` | Krawczyk existence and uniqueness test around the 40-digit point (writes `results_q32_krawczyk_A100.json`) |
| `q32_certificate_v4.py` | the certificate for the exact optimality point in that box (mpmath, directed rounding) |
| `q32_recertify_v4.py` | the same certificate at every bracket endpoint of Table V (inputs in `results_q32_exact40_all.json`) |
| `q32_certificate_arb.py` | the whole chain repeated in a second instrument, Arb ball arithmetic (python-flint) |
| `q470_support_certificate.py` | the certificate for large peak counts: K = 21 at A = 311.37, K = 46 at A = 1144.25, K = 66 at A = 2000 |
| `q481_tail_lemma.py` | the tail lemma that turns q470's truncated output alphabet into a statement about the true channel |

Earlier versions (`q32_certificate_v2.py`, `_v3.py`, `q32_recertify_v2.py`, `_v3.py`, `q32_piece_i*.py`) are imported by the
current ones and are kept for that reason. Input states: `results_q32_A100_exact40.json`, `results_q32_exact40_all.json`,
`results_q66_K21_A311.369_exact40.json`, `results_q351_A1144.248_K46_arb40.json`, `results_q473_seed_A2000_K66_v7_ascent_arb40.json`.

## Results

| file in `logs/` | content |
|---|---|
| `results_q32_certificate_v4_A100.log`, `.json` | the original control run |
| `rerun_q32_certificate_v4_A100.log`, `rerun_results_q32_certificate_v4_A100.json` | the control run repeated from this folder before publishing; every printed number agrees with the original |
| `results_q32_recertify_v4.log`, `.json` | the certificates at the Table V amplitudes |
| `results_q32_certificate_arb.json` | the second-instrument replication |
| `q470_*.log`, `q481_*.log` | the large-A certificates and their tail lemma |

See [`FILES.md`](FILES.md) for every script.
