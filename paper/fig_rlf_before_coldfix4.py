"""fig_rlf.py (15 Oct 2026, task P2FIG): Fig 1 of paper 2, one recorded rate-level function with both fitted shapes and the optimal
staircase. The fibre is chosen by a rule fixed before any plot was made: among the 50 fibres of paper_neuron/q1925_rows.json whose fitted
peak count A_sq is closest to the median A_sq, take the one whose d (loglik of the predicted shape minus that of the sigmoid) is the
median of those 50 (the lower middle one for an even count). The fits are redone with the functions of q1925 (same data, same fit),
and the staircase is Proposition 1 for a Gaussian level law with the fitted mu and sigma: rate r0 + x_j where Phi((L - mu)/sigma) lies
in (W_{j-1}, W_j], with x_j and w_j the mass points and weights of the capacity-achieving input on [r0, r0 + A] (support_exact).
Output: paper_neuron/figs/fig_rlf.pdf and paper_neuron/figs/fig_rlf.json (every plotted parameter)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
sys.path.insert(0, TW)
import q1925_rlf_shape as Q  # noqa: E402
from support_exact import support  # noqa: E402
from scipy.stats import norm  # noqa: E402

rows = json.load(open(os.path.join(HERE, "q1925_rows.json")))
med_a = float(np.median([r["A_sq"] for r in rows]))
near = sorted(rows, key=lambda r: abs(r["A_sq"] - med_a))[:50]
near = sorted(near, key=lambda r: r["d"])
pick = near[(len(near) - 1) // 2]
rec = next(x for x in json.load(open(Q.CENSUS)) if x.get("path") == pick["path"])
L, y, spont = Q.select(rec)
a, b = Q.fit(Q.m_sq, L, y), Q.fit(Q.m_log, L, y)
r0, A, mu, sg = (float(v) for v in a.x)
sup = support(A, r0)
W = np.cumsum(sup["w"]); W = W / W[-1]
grid = np.linspace(L.min() - 5, L.max() + 5, 600)
u = norm.cdf((grid - mu) / sg)
idx = np.searchsorted(W, u, side="left").clip(0, len(W) - 1)
stair = r0 + np.array(sup["pts"])[idx]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
fig, ax = plt.subplots(figsize=(5.2, 3.4))
rng = np.random.default_rng(0)
ax.plot(L + rng.uniform(-0.6, 0.6, len(L)), y + rng.uniform(-0.12, 0.12, len(y)), ".", color="0.7", ms=3, label="single-trial counts")
levs = np.unique(L)
ax.plot(levs, [y[L == v].mean() for v in levs], "o", color="k", ms=4, label="mean count per level")
ax.plot(grid, Q.m_sq(a.x, grid), "-", color="C0", lw=1.6, label="predicted shape (fit)")
ax.plot(grid, Q.m_log(b.x, grid), "--", color="C3", lw=1.2, label="symmetric sigmoid (fit)")
ax.step(grid, stair, where="post", color="C2", lw=1.4, label="optimal staircase, %d levels" % sup["N"])
ax.set_xlabel("sound level (dB)"); ax.set_ylabel("spike count in 50 ms")
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fig_rlf.pdf"))
out = {"path": pick["path"], "rule": "median d among the 50 fibres with A_sq closest to median A_sq", "median_A_sq": med_a,
       "d_stored": pick["d"], "d_refit": float(b.fun - a.fun), "r0": r0, "A": A, "mu": mu, "sigma": sg,
       "log_params": [float(v) for v in b.x], "K": sup["N"], "points": sup["pts"], "weights": sup["w"], "levels": int(len(levs)),
       "trials": int(len(y))}
json.dump(out, open(os.path.join(HERE, "figs", "fig_rlf.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("path", "d_stored", "d_refit", "r0", "A", "mu", "sigma", "K", "levels", "trials")}))
