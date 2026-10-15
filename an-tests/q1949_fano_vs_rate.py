"""q1949_fano_vs_rate.py (15 Oct 2026, task P2WEAKAUDIT step 1b): does the Fano factor of the fibres fall with the mean count faster
than the matched Conway-Maxwell-Poisson count (one zeta per fibre, q1939) predicts? Readings only.
For every fibre of the discreteness test and every level with a mean count of at least 1: observed Fano factor, the Fano factor of the
matched count at that mean (fano_at(m, zeta_fibre)), and that of the binomial count with n = ceil(lam + A) (1 - m/n).
Readings by thirds of the fibre's median Fano factor: mean observed minus predicted Fano at the fibre's lowest and highest such level,
and the mean within-fibre slope of observed Fano against the mean count compared with the slope of each model."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1926_ph_steps as Q  # noqa: E402
from q1939_ph_steps_compoisson import fano_at  # noqa: E402

r48 = {r["path"]: r for r in json.load(open(os.path.join(HERE, "paper_neuron", "q1948_rows.json")))}
r39 = {r["path"]: r for r in json.load(open(os.path.join(HERE, "paper_neuron", "q1939_rows.json")))}
rows = []
for rec in json.load(open(Q.CENSUS)):
    if rec.get("path") not in r48:
        continue
    lev, counts, lam, A = Q.fibre(rec)
    zeta = r39[rec["path"]]["nu"]; n = int(np.ceil(lam + A))
    pts = [(c.mean(), c.var(ddof=1) / c.mean()) for c in counts if c.mean() >= 1]
    if len(pts) < 2:
        continue
    pts.sort()
    m = np.array([p[0] for p in pts]); fo = np.array([p[1] for p in pts])
    fc = np.array([fano_at(x, zeta) for x in m]); fb = 1 - m / n
    s = lambda y: float(np.polyfit(m, y, 1)[0])
    rows.append(dict(F=r48[rec["path"]]["F"], lo_c=fo[0] - fc[0], hi_c=fo[-1] - fc[-1], lo_b=fo[0] - fb[0], hi_b=fo[-1] - fb[-1],
                     so=s(fo), sc=s(fc), sb=s(fb)))
F = np.array([r["F"] for r in rows]); cut = np.quantile(F, [1 / 3, 2 / 3])
for name, sel in (("most regular third", F <= cut[0]), ("middle third", (F > cut[0]) & (F <= cut[1])), ("least regular third", F > cut[1]), ("all", F == F)):
    S = [r for r, s_ in zip(rows, sel) if s_]
    g = lambda k: float(np.mean([r[k] for r in S]))
    print("Reading: %s (%d fibres): observed minus matched Fano at lowest level %+.3f, at highest %+.3f; observed minus binomial %+.3f, %+.3f; "
          "slope of Fano per spike: observed %+.4f, matched %+.4f, binomial %+.4f"
          % (name, len(S), g("lo_c"), g("hi_c"), g("lo_b"), g("hi_b"), g("so"), g("sc"), g("sb")))
