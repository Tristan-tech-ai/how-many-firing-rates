"""q1932 (15 Oct 2026): task P2REPL, registered in Amendment 2279 before any per-group statistic was computed. Replication of H_L1
(Amendment 2274) within animal groups. Per-fibre log-likelihood differences d = loglik(M_sq) - loglik(M_log) are read from
paper_neuron/q1925_rows.json (written by the registered q1925 run of Amendment 2271; not refitted). Subject ages from
data_ext/dandi001262/subject_ages.json (q1931, DANDI asset metadata). Groups as in Table 1 of Heeringa et al. (Sci. Data 2024, local
text): young-adult below 12 months (age < 365.25 days), quiet-aged above 36 months (age > 1095.75 days); middle-aged fibres enter a
reading only.
VERDICT (H_L1rep): PASS if the bootstrap 95 percent lower bound of mean d (2000 resamples of fibres, seed 2279, the q1925 rule) is
above 0 in BOTH the young-adult and the quiet-aged group; FAIL otherwise; VOID if either group has fewer than 20 fibres; INCOMPLETE if
an input file is missing.
Usage: py q1932_hl1_replication.py [--registration-run]"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1925_rlf_shape as Q

ROWS = os.path.join(HERE, "paper_neuron", "q1925_rows.json")
AGES = os.path.join(HERE, "data_ext", "dandi001262", "subject_ages.json")


def group(age_days):
    return "young" if age_days < 365.25 else ("old" if age_days > 1095.75 else "middle")


def decide(dy, do):
    if len(dy) < 20 or len(do) < 20:
        return "VOID", "a group has fewer than 20 fibres (young %d, old %d)" % (len(dy), len(do))
    ly, lo = Q.boot_lower(dy, seed=2279), Q.boot_lower(do, seed=2279)
    ok = ly > 0 and lo > 0
    return ("PASS" if ok else "FAIL"), "lower bound young %.4f (%d fibres), old %.4f (%d fibres) (bar > 0 in both)" % (ly, len(dy), lo, len(do))


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1932: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        rng = np.random.default_rng(1)
        pos = list(rng.normal(0.3, 0.5, 200)); neg = list(rng.normal(-0.2, 0.5, 200))
        d = (decide(pos, pos)[0], decide(pos, neg)[0], decide(pos, pos[:10])[0])
        g = (group(150), group(700), group(1200))
        print("control decide(): %s %s %s (expected PASS FAIL VOID); group(): %s (expected young middle old)" % (d + (g,)))
        ok = d == ("PASS", "FAIL", "VOID") and g == ("young", "middle", "old")
        print("VERDICT q1932 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not (os.path.exists(ROWS) and os.path.exists(AGES)):
        print("VERDICT q1932: INCOMPLETE (input missing)"); return 3
    ages = json.load(open(AGES)); rows = json.load(open(ROWS)); by = {"young": [], "middle": [], "old": []}
    for r in rows:
        a = ages.get(r["path"].split("/")[0])
        if a:
            by[group(int(re.sub(r"\D", "", a)))].append(r["d"])
    for k in ("young", "middle", "old"):
        if by[k]:
            print("Reading: %s fibres %d, mean d %.4f, lower bound %.4f" % (k, len(by[k]), float(np.mean(by[k])), Q.boot_lower(by[k], seed=2279)))
    v, why = decide(by["young"], by["old"])
    print(why)
    print("VERDICT q1932 (Amendment 2279, H_L1rep): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
