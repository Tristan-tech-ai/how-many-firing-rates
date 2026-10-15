"""q1926 (15 Oct 2026): task P2STEPS, registered in Amendment 2272 before any PH recording was analysed.

Prediction (paper_neuron/LAUGHLIN_DERIVATION.md, Proposition L1, Barlow frame). For any stimulus law, the infomax tuning curve of a
Poisson fibre takes exactly K(A, lam) distinct rates, where K is the number of mass points of the capacity-achieving input on
[lam, lam + A]. Evaluated at any set of sound levels, such a curve shows at most K distinct rates. A graded curve shows as many distinct
rates as the data can resolve.
Data: DANDI 001262 PH recordings (census q1895; tones at the best frequency, 5 to 8 levels, 20 to 40 repeats), counts in 0 <= t < 50 ms
(the window of Amendments 2264, 2267, 2271), levels from the PH_ABI<level> keys, the silent condition excluded.
Selection: >= 4 levels, >= 20 repeats at every level, A >= 1 spike per window.
Per fibre: means m_l per level; lam = min m_l, A = max m_l - lam; K = support_exact.support(A, lam)["N"] (the cutting-plane KKT tool of
paper 2). D = 1 + number of adjacent level pairs (ordered by dB) whose counts differ, by the exact conditional binomial test of two
Poisson means (sum of counts split by the repeat numbers), Holm-corrected at 0.05 over the adjacent pairs of the fibre.
VERDICT (H_steps, registered): PASS if D <= K in at least 80 percent of the selected fibres; FAIL otherwise. VOID if fewer than 20
fibres are selected; INCOMPLETE if the census is missing.
Readings: distribution of D - K; fraction with D <= K split by A below or above 5; quartiles of A, lam and K.
Usage: py q1926_ph_steps.py [--registration-run]
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from support_exact import support

CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
OUT = os.path.join(HERE, "paper_neuron", "q1926_rows.json")
W = 0.050


def count_distinct(levels, counts):
    """levels sorted ascending; counts: list of arrays of trial counts. Returns D."""
    p = []
    for a, b in zip(counts[:-1], counts[1:]):
        n1, n2 = int(np.sum(a)), int(np.sum(b))
        if n1 + n2 == 0:
            p.append(1.0); continue
        frac = len(a) / (len(a) + len(b))
        p.append(binomtest(n1, n1 + n2, frac).pvalue)
    order = np.argsort(p); m = len(p); sig = 0
    for i, j in enumerate(order):                     # Holm step-down
        if p[j] <= 0.05 / (m - i):
            sig += 1
        else:
            break
    return 1 + sig


def fibre(rec):
    sp = rec.get("spikes", {})
    keys = [k for k in sp if k.startswith("PH_ABI")]
    if len(keys) < 4:
        return None
    lev = sorted((float(re.sub(r"[^0-9.\-]", "", k[len("PH_ABI"):])), k) for k in keys)
    counts = [np.array([sum(1 for t in tr if 0.0 <= t < W) for tr in sp[k]], float) for _, k in lev]
    if min(len(c) for c in counts) < 20:
        return None
    means = np.array([c.mean() for c in counts]); lam = float(means.min()); A = float(means.max() - lam)
    if A < 1.0:
        return None
    return [l for l, _ in lev], counts, lam, A


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    frac = float(np.mean([r["D"] <= r["K"] for r in rows]))
    return ("PASS" if frac >= 0.8 else "FAIL"), "fraction with D <= K %.3f over %d fibres (bar >= 0.8)" % (frac, len(rows))


def registration_run():
    rng = np.random.default_rng(2272); ok = True
    levels = list(np.arange(0, 70, 10.0))                          # 7 levels
    for A, lam in ((3.0, 0.5), (8.0, 0.5), (15.0, 1.0)):
        K = support(A, lam)["N"]
        # staircase: K values at quantiles, assigned monotonically to the 7 levels
        vals = np.linspace(lam, lam + A, K); idx = np.minimum((np.arange(7) * K) // 7, K - 1); stair = vals[idx]
        graded = np.linspace(lam, lam + A, 7)
        res = {}
        for name, rates in (("staircase", stair), ("graded", graded)):
            ok_count = 0
            for _ in range(40):
                counts = [rng.poisson(r, 40).astype(float) for r in rates]
                ok_count += count_distinct(levels, counts) <= K
            res[name] = ok_count / 40
        print("control A = %g, lam = %g, K = %d (40 repeats, 7 levels, 40 synthetic fibres): fraction D <= K staircase %.2f, graded %.2f"
              % (A, lam, K, res["staircase"], res["graded"]))
        ok &= res["staircase"] >= 0.9
        if A >= 8:
            ok &= res["graded"] <= 0.5
    good = [{"D": 2, "K": 3}] * 30; bad = [{"D": 5, "K": 3}] * 30
    d = (decide(good)[0], decide(bad)[0], decide(good[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID); INCOMPLETE is returned by main() without the census" % d)
    ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1926: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1926 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS):
        print("VERDICT q1926: INCOMPLETE (census missing)"); return 3
    rows = []
    for rec in (x for x in json.load(open(CENSUS)) if "error" not in x):
        f = fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        rows.append({"path": rec["path"], "lam": lam, "A": A, "K": int(support(A, lam)["N"]), "D": int(count_distinct(levels, counts)),
                     "n_levels": len(levels)})
    json.dump(rows, open(OUT, "w"))
    if rows:
        dk = np.array([r["D"] - r["K"] for r in rows]); As = np.array([r["A"] for r in rows])
        vals, cnt = np.unique(dk, return_counts=True)
        print("Reading: fibres %d; D - K distribution %s" % (len(rows), dict(zip(vals.tolist(), cnt.tolist()))))
        for name, m in (("A < 5", As < 5), ("A >= 5", As >= 5)):
            if m.any():
                print("Reading: %s: %d fibres, fraction D <= K %.3f" % (name, int(m.sum()), float(np.mean(dk[m] <= 0))))
        q = lambda k: tuple(np.percentile([r[k] for r in rows], [25, 50, 75]))
        print("Reading: A quartiles %.2f %.2f %.2f; lam quartiles %.2f %.2f %.2f; K quartiles %g %g %g" % (*q("A"), *q("lam"), *q("K")))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1926 (Amendment 2272, H_steps): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
