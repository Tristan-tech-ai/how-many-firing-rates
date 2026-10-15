# How many firing rates should a neuron use? Code, outputs and registration log

Code, outputs and the dated registration log for the paper *How many firing rates should a neuron use? Infomax staircases for a
given stimulus distribution and a test in the auditory nerve* by I Made Tristan Hope Firdaus (2026). The paper is in `paper/`.

The paper studies a sensory neuron whose spike count in a window is Poisson, with a mean between its spontaneous count and that count
plus a peak count. It proves that the most informative tuning curve for a given stimulus distribution is a staircase whose steps sit at
quantiles of the stimulus distribution (Proposition 1). It also proves that the number of steps grows at least like 0.056 times the
two-thirds power of the peak count (Theorem 3). It states a conjectured law for the exact number of steps, with predictions written
before they were computed. It then reports registered tests of the shape of rate-level functions and of the number of distinct rates
in gerbil auditory-nerve fibres.

## Where to find what

| folder | what it holds |
|---|---|
| `paper/` | the paper (`paper2_v2_draft.pdf`; the submission layout `paper2_plos.pdf` with its source `paper2_plos.tex`), its sources, figures, figure scripts and the supporting text `supplement_p2.pdf` |
| `theory-checks/` | interval-arithmetic and numerical checks of the constants of Theorem 3 (`q1906`, `q1907`, `q1908`, `q1921`), of the finiteness proof in the supporting text (`q1945`), of the extension to a spontaneous count (`q1958`), and of the direction of asymmetry of the fitted curves (`q1960`) |
| `count-law/` | the walk that locates count changes (`q1899`) and the states on both sides of them (`q1910`) |
| `proved-counts/` | the computer-assisted proof of a count (`q470_support_certificate.py`, `q481_tail_lemma.py`), the logs of every run, and the states they read |
| `an-tests/` | the registered tests on auditory-nerve fibres and their outputs: shape (`q1925`, `q1932`, `q1935`, `q1957`), distinct rates (`q1926`, `q1930`, `q1937`, `q1939`, `q1950`, `q1953`, `q1956`), drift (`q1929`), variability of counts (`q1936`, `q1949`), synthetic controls (`q1941`), tables (`q1942`), and the earlier analyses the paper cites |
| `registration-log/` | the public copy of the registration log, and `timestamps/` with the files that date it |

Each test script prints its decision as its last line, starting with `VERDICT`. Its output file (`*_out.txt`, and `*_rows.json` with
one row per fibre) is the one the paper and the log quote. A script run with `--registration-run` runs only the synthetic controls
that were run before the recorded fibres were analysed.

## Data

The recordings are the public dataset 001262 of the DANDI archive (https://dandiarchive.org/dandiset/001262), described by Heeringa
(Sci. Data 11:411, 2024). They are not copied here. To rebuild the files the tests read, run from `an-tests/`:

    python q1895_dandi001262_census.py     # spike counts per trial for every recording, written to data_ext/dandi001262/census.json
    python q1931_subject_ages.py           # ages of the animals, written to data_ext/dandi001262/subject_ages.json

The second script reads `data_ext/dandi001262/assets.json`, which the first one writes. Several scripts import helpers from
`an-tests/` (for example `support_exact.py`); run them from that folder or put it on `PYTHONPATH`.

## Proved count

At peak count A = 412.42711264916346 the optimal input has exactly 24 mass points. To check it, run from `proved-counts/`:

    RBOX=1e-26 python q470_support_certificate.py 192 8 state_412.427112649163.json > A25.log
    RBOX=1e-26 python q481_tail_lemma.py state_412.427112649163.json 192 A25.log

Both last lines must read `VERDICT: PASS`. The logs of the runs reported in the log are `lawcert_A25_lo.log` and
`lawcert_A25_lo_tail.log`. The other logs record the runs at the lower ends of the 47th and 67th steps. They did not prove those
counts for the channel with an unbounded count, as the registration log describes.

## Registration log and time stamps

`registration-log/registration_log.md` is the public copy of the log in which every test and prediction was written before its data
were analysed or its value was computed. Entries from 14 October 2026 on are dated by an independent authority. The manifests in
`registration-log/timestamps/` give the length and hash of the log at each stamp, and the README there explains how to check a stamp
with OpenSSL and with the OpenTimestamps client.

## Changes in this copy

`REDACTIONS.md` lists the files from which absolute local folder names were removed, with the hash of each file as written by the
computation and of the copy here. `MANIFEST.json` gives the source and hash of every file.

## License

MIT, as in `LICENSE` at the top of the repository.
