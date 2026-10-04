# Files in sec5-baseline

Scripts are formatted with black (layout only). The original path is the name used in the registration log.

| script | what it does (from its own docstring) | original path |
|---|---|---|
| `scripts/q32_certificate_arb.py` | The whole certificate chain replicated in a SECOND INSTRUMENT: arb ball arithmetic (python-flint). | `q32_certificate_arb.py` |
| `scripts/q32_certificate_v2.py` | The assembled certificate with the flaw the audit found repaired. | `q32_certificate_v2.py` |
| `scripts/q34_poisson_dark.py` | the offset c as a function of dark current lambda. | `q34_poisson_dark.py` |
| `scripts/q35_dark_certify.py` | CERTIFIED support sizes with dark current, by the shift identity. | `q35_dark_certify.py` |
| `scripts/support_iv.py` | arithmetic evaluation of D and its derivatives to order six, for PROVED bounds. | `support_iv.py` |
| `scripts/support_mp3.py` | KKT Newton with an ANALYTIC JACOBIAN (2026-09-05). | `support_mp3.py` |

`logs/` holds 2 output files of the runs reported in the paper; a `__` in a name stands for a folder separator of the original tree.
