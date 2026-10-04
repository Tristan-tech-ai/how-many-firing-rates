# Files in sec3-certificates

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/q32_certificate_arb.py` | The whole certificate chain replicated in a SECOND INSTRUMENT: arb ball arithmetic (python-flint). | `q32_certificate_arb.py` |
| `scripts/q32_certificate_v2.py` | The assembled certificate with the flaw the audit found repaired. | `q32_certificate_v2.py` |
| `scripts/q32_certificate_v3.py` | The certificate with PROVED constants. | `q32_certificate_v3.py` |
| `scripts/q32_certificate_v4.py` | The certificate for the EXACT KKT point: closes the audit's caveat 1. | `q32_certificate_v4.py` |
| `scripts/q32_krawczyk.py` | Krawczyk existence test: an EXACT KKT point lies within rho of the numerical one. | `q32_krawczyk.py` |
| `scripts/q32_piece_i.py` | Piece (i) of the certificate, with two gaps closed. | `q32_piece_i.py` |
| `scripts/q32_piece_i_fix.py` | run piece (i) alone, with both gaps closed, at every endpoint already certified. | `q32_piece_i_fix.py` |
| `scripts/q32_recertify_v2.py` | run every bracket endpoint certificate with the repaired argument. | `q32_recertify_v2.py` |
| `scripts/q32_recertify_v3.py` | run every bracket endpoint certificate with PROVED constants. | `q32_recertify_v3.py` |
| `scripts/q32_recertify_v4.py` | point certificate (q32_certificate_v4.py) at every bracket endpoint. | `q32_recertify_v4.py` |
| `scripts/q470_support_certificate.py` | T2 of 613(7)/614(4) - a rigorous SUPPORT-SIZE certificate for one capacity-achieving state of the amplitude-constrained Poisson channel, in ball arithmetic (python-flint arb). | `q470_support_certificate.py` |
| `scripts/q481_tail_lemma.py` | the TAIL LEMMA that turns q470's truncated-alphabet certificates (617, 624, and the A = 2000 run) into statements about the true Poisson channel, whose output alphabet is all of N_0. | `q481_tail_lemma.py` |
| `scripts/support_iv.py` | arithmetic evaluation of D and its derivatives to order six, for PROVED bounds. | `support_iv.py` |
| `scripts/support_mp3.py` | KKT Newton with an ANALYTIC JACOBIAN (2026-09-05). | `support_mp3.py` |

`logs/` holds 24 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
