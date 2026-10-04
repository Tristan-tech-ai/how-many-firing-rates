"""Arm 3 refinement: exact one-shot MAP codebooks on a finer rate grid. Deterministic."""

import numpy as np, itertools, json, math, sys
from scipy.stats import poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T, L = 0.05, 60.0


def run(M, step):
    rates = np.arange(0.0, L + 1e-9, step)
    Y = np.arange(0, 41)
    P = poisson.pmf(Y[None, :], (rates * T)[:, None])
    best = (2.0, None)
    for c in itertools.combinations(range(len(rates)), M):
        eps = 1.0 - P[list(c)].max(axis=0).sum() / M
        if eps < best[0]:
            best = (eps, c)
    return best[0], [float(rates[i]) for i in best[1]]


out = {}
for M, step in ((3, 0.5), (4, 1.0), (5, 1.0)):
    eps, code = run(M, step)
    counts = [round(x * T, 2) for x in code]
    print(f"M={M} step={step} Hz  eps*={eps:.6f}  code={code} Hz  = mean counts {counts}")
    out[f"M{M}"] = {"step": step, "eps": eps, "code": code, "counts": counts}
json.dump(out, open("results_oneshot_refine.json", "w"), indent=1)
