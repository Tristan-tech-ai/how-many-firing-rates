"""q1929 (15 Oct 2026): task P2DRIFT, follow-up of the H_steps FAIL (Amendment 2273), registered in Amendment 2275 before any drift
statistic of the data was computed.

Question. Does slow drift across repeats make adjacent levels look different in the PH recordings, inflating D?
Data and selection exactly as q1926 (PH recordings, 0 <= t < 50 ms, >= 4 levels, >= 20 repeats per level, A >= 1). Trial order is the
order of the repeats as stored in the census (the order of the NWB units table).
Per fibre and level: means of the first and the second half of the repeats, z = (m1 - m2)/sqrt(m1/n1 + m2/n2) (Poisson standard error,
with 0.5/n added to each mean to avoid a zero). A fibre DRIFTS if max |z| over its levels exceeds the two-sided Bonferroni threshold
for its number of levels at 0.05 (under stationarity about 5 percent of fibres drift).
VERDICT (H_drift, registered): PASS if at least 30 percent of the fibres drift; FAIL otherwise. VOID if fewer than 20 fibres;
INCOMPLETE if the census is missing.
Readings: fraction drifting; fraction drifting among fibres with D > K and with D <= K (from paper_neuron/q1926_rows.json, matched by
path); median of the signed mean change (second half minus first half) over levels.
Usage: py q1929_ph_drift.py [--registration-run]
"""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1926_ph_steps as S

CENSUS = S.CENSUS
ROWS26 = os.path.join(HERE, "paper_neuron", "q1926_rows.json")


def drift_stats(counts):
    zs, ch = [], []
    for c in counts:
        h = len(c) // 2; a, b = c[:h], c[h:]
        m1, m2 = a.mean() + 0.5 / len(a), b.mean() + 0.5 / len(b)
        zs.append((m1 - m2) / np.sqrt(m1 / len(a) + m2 / len(b))); ch.append(b.mean() - a.mean())
    thr = norm.ppf(1 - 0.025 / len(counts))
    return bool(np.max(np.abs(zs)) > thr), float(np.mean(ch))


def decide(flags):
    if len(flags) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean(flags))
    return ("PASS" if f >= 0.3 else "FAIL"), "fraction drifting %.3f over %d fibres (bar >= 0.3)" % (f, len(flags))


def registration_run():
    rng = np.random.default_rng(2275); ok = True
    rates = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    stat = [drift_stats([rng.poisson(r, 40).astype(float) for r in rates])[0] for _ in range(400)]
    drift = [drift_stats([np.concatenate([rng.poisson(r, 20), rng.poisson(r * 1.25, 20)]).astype(float) for r in rates])[0]
             for _ in range(400)]
    print("control: stationary synthetic fibres (6 levels, 40 repeats): fraction flagged %.3f (expected about 0.05)" % np.mean(stat))
    print("control: fibres whose rates rise 25 percent in the second half: fraction flagged %.3f" % np.mean(drift))
    ok &= np.mean(stat) <= 0.08 and np.mean(drift) >= 0.8
    d = (decide([True] * 10 + [False] * 20)[0], decide([True] * 5 + [False] * 25)[0], decide([True] * 5)[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d)
    ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1929: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1929 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS):
        print("VERDICT q1929: INCOMPLETE (census missing)"); return 3
    r26 = {r["path"]: r for r in json.load(open(ROWS26))} if os.path.exists(ROWS26) else {}
    flags, changes, over, under = [], [], [], []
    for rec in (x for x in json.load(open(CENSUS)) if "error" not in x):
        f = S.fibre(rec)
        if f is None:
            continue
        flag, ch = drift_stats(f[1]); flags.append(flag); changes.append(ch)
        if rec["path"] in r26:
            (over if r26[rec["path"]]["D"] > r26[rec["path"]]["K"] else under).append(flag)
    if flags:
        print("Reading: fibres %d; fraction drifting among D > K fibres %.3f (%d), among D <= K fibres %.3f (%d); median mean change %.3f spikes"
              % (len(flags), np.mean(over) if over else float("nan"), len(over), np.mean(under) if under else float("nan"), len(under),
                 float(np.median(changes))))
    v, why = decide(flags)
    print(why)
    print("VERDICT q1929 (Amendment 2275, H_drift): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
