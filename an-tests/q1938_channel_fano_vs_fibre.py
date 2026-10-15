"""q1938 (15 Oct 2026): reading only, no verdict. For the channel that q1937 (repaired) used in each fibre, Bin(n, r/n) with
n = ceil(lam + A), compare its Fano factor at the fibre's mean count, 1 - mbar/n, with the fibre's measured median Fano factor F.
Counts how many fibres had a channel at least as regular as the fibre (1 - mbar/n <= F) and how many a less regular one.
Usage: py q1938_channel_fano_vs_fibre.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q1926_ph_steps import fibre  # noqa: E402

CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
R37 = os.path.join(HERE, "paper_neuron", "q1937_rows.json")


def main():
    r37 = {r["path"]: r for r in json.load(open(R37))}
    out = []
    for rec in json.load(open(CENSUS)):
        if "error" in rec or rec.get("path") not in r37:
            continue
        f = fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        row = r37[rec["path"]]
        mbar = float(np.mean([c.mean() for c in counts]))
        if row["n"] is None:
            out.append(("poisson", row))
            continue
        n = int(row["n"]); assert n == max(int(np.ceil(lam + A)), 1)
        Fch = 1 - mbar / n
        out.append(("regular" if Fch <= row["F"] else "noisier", row, Fch))
    print("fibres matched to q1937 rows: %d of %d" % (len(out), len(r37)))
    for cls in ("regular", "noisier", "poisson"):
        sub = [o for o in out if o[0] == cls]
        if sub:
            fr = np.mean([o[1]["D"] <= o[1]["K"] for o in sub])
            extra = ""
            if cls != "poisson":
                extra = ", median channel Fano %.3f, median fibre Fano %.3f" % (np.median([o[2] for o in sub]), np.median([o[1]["F"] for o in sub]))
            print("Reading: channel %s than or as the fibre: %d fibres, D <= K in %.3f%s" % (cls, len(sub), fr, extra))
    if len(out) != len(r37):
        print("Reading: INCOMPLETE match")


if __name__ == "__main__":
    main()
