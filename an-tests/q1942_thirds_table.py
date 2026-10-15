"""q1942 (15 Oct 2026): reading only, no verdict. Source of the table of paper 2 that splits the 265 phase-locking fibres into three
groups of equal size by their median Fano factor (q1936) and gives the fraction with D <= K for the K of each count model: Poisson
(q1926), binomial (q1937, Amendments 2286-2288) and matched Conway-Maxwell-Poisson (q1939, Amendments 2290-2291). The group boundaries
are the 33.3 and 66.7 percentiles, as in q1937 and q1939. Also prints the share of fibres with D - K = +1 under Poisson counts.
Usage: py q1942_thirds_table.py > paper_neuron/q1942_out.txt
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "paper_neuron")


def main():
    a = {x["path"]: x for x in json.load(open(os.path.join(P, "q1926_rows.json")))}
    b = {x["path"]: x for x in json.load(open(os.path.join(P, "q1937_rows.json")))}
    m = {x["path"]: x for x in json.load(open(os.path.join(P, "q1939_rows.json")))}
    # 15 Oct 23:3x: fourth column, the rate-dependent count of q1950 (Amendments 2294-2295); the other columns are unchanged
    r = {x["path"]: x for x in json.load(open(os.path.join(P, "q1950_rows.json")))}
    paths = list(m)
    assert len(paths) == 265 and set(paths) == set(a) == set(b) == set(r)
    F = np.array([m[p]["F"] for p in paths]); q = np.percentile(F, [33.3, 66.7])
    groups = (("most regular", F < q[0]), ("middle", (F >= q[0]) & (F < q[1])), ("least regular", F >= q[1]),
              ("all", np.ones(len(F), bool)))
    for name, s in groups:
        ps = [p for p, ss in zip(paths, s) if ss]
        fr = [np.mean([a[p]["D"] <= K for p, K in ((p, src[p]["K"]) for p in ps)]) for src in (a, b, m, r)]
        print("Reading: %-13s n %3d, Fano %.3f to %.3f, D <= K: Poisson %.3f, binomial %.3f, matched %.3f, rate-matched %.3f"
              % (name, len(ps), F[s].min(), F[s].max(), fr[0], fr[1], fr[2], fr[3]))
    hi = [p for p in paths if a[p]["A"] >= 8]
    print("Reading: fibres with A >= 8: %d; D <= K: Poisson %.3f, binomial %.3f, matched %.3f, rate-matched %.3f" % (
        len(hi), np.mean([a[p]["D"] <= a[p]["K"] for p in hi]), np.mean([a[p]["D"] <= b[p]["K"] for p in hi]),
        np.mean([a[p]["D"] <= m[p]["K"] for p in hi]), np.mean([a[p]["D"] <= r[p]["K"] for p in hi])))
    dk = np.array([a[p]["D"] - a[p]["K"] for p in paths])
    print("Reading: Poisson D - K distribution %s" % {int(v): int((dk == v).sum()) for v in np.unique(dk)})
    print("Reading: Poisson D - K = +1 in %d of %d fibres (%.3f); D - K = 0 in %d" % ((dk == 1).sum(), len(dk), (dk == 1).mean(), (dk == 0).sum()))


if __name__ == "__main__":
    main()
