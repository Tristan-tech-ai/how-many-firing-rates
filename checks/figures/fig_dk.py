"""fig_dk.py (15 Oct 2026, task P2DKFIG; third draft reading, finding 96): the discreteness result as a figure.
(A) D against the Poisson K and (B) D against the matched Conway-Maxwell-Poisson K for the 265 phase-locking fibres (q1926_rows.json,
q1939_rows.json), with a small fixed jitter and the line D = K; points on or below the line meet the bound.
(C) one fibre, chosen by a rule fixed here before any plot was made: among the fibres with D = K under the matched K and D > K under
the Poisson K, the one whose peak count A is the median of that set (lower median). Its mean counts per sound level with standard
errors; neighbouring levels that the exact conditional binomial test with Holm's correction (q1926.count_distinct) does not separate
share a colour, so the colours are the D distinct rates.
Outputs figs/fig_dk.pdf and figs/fig_dk.json (the chosen fibre and the counts behind panels A and B).
Usage: py paper_neuron/fig_dk.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, ROOT)
from q1926_ph_steps import fibre, CENSUS, count_distinct  # noqa: E402

a = json.load(open(os.path.join(H, "q1926_rows.json")))
m = {r["path"]: r for r in json.load(open(os.path.join(H, "q1939_rows.json")))}
paths = [r["path"] for r in a]
D = np.array([r["D"] for r in a]); Kp = np.array([r["K"] for r in a]); Km = np.array([m[p]["K"] for p in paths])
A = np.array([r["A"] for r in a])

cand = [i for i in range(len(a)) if D[i] == Km[i] and D[i] > Kp[i]]
order = sorted(cand, key=lambda i: A[i])
pick = order[(len(order) - 1) // 2]
rec = next(x for x in json.load(open(CENSUS)) if x.get("path") == paths[pick])
levels, counts, lam, Af = fibre(rec)
assert count_distinct(levels, counts) == D[pick]

# groups: replicate the Holm step-down of count_distinct to mark which neighbouring pairs are separated
p = []
for c1, c2 in zip(counts[:-1], counts[1:]):
    n1, n2 = int(np.sum(c1)), int(np.sum(c2))
    p.append(1.0 if n1 + n2 == 0 else binomtest(n1, n1 + n2, len(c1) / (len(c1) + len(c2))).pvalue)
order_p = np.argsort(p); sep = np.zeros(len(p), bool)
for i, j in enumerate(order_p):
    if p[j] <= 0.05 / (len(p) - i):
        sep[j] = True
    else:
        break
group = np.concatenate([[0], np.cumsum(sep)])
assert group[-1] + 1 == D[pick]

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
# 16 Oct 2026 (P2FIGFILES): PLOS figure rules: Arial, Times or Symbol at 8 to 12 pt, at most 7.5 in wide; TrueType embedded
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "pdf.fonttype": 42})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.5, 3.2))
rng = np.random.default_rng(0)
jx, jy = rng.uniform(-0.18, 0.18, len(D)), rng.uniform(-0.18, 0.18, len(D))
for ax, K, title, frac in ((a1, Kp, "(A) Poisson counts", np.mean(D <= Kp)), (a2, Km, "(B) matched counts", np.mean(D <= Km))):
    ok = D <= K
    ax.scatter(K[ok] + jx[ok], D[ok] + jy[ok], s=7, color="0.35", alpha=0.6, lw=0, label="D ≤ K")   # P2REFEREE7: the sign, not "<="
    ax.scatter(K[~ok] + jx[~ok], D[~ok] + jy[~ok], s=7, color="C3", alpha=0.7, lw=0, label="D > K")
    hi = max(D.max(), K.max()) + 0.6
    ax.plot([0.5, hi], [0.5, hi], color="k", lw=0.8)
    ax.set_xlim(0.5, min(hi, 10.5)); ax.set_ylim(0.5, 7.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))   # P2REFEREE7: K is an integer (panel B showed 2.5, 7.5)
    ax.set_xlabel("steps K of the optimal staircase"); ax.set_ylabel("distinct mean counts D")
    ax.set_title("%s:\n%.0f%% with D \u2264 K" % (title, 100 * frac), fontsize=10, loc="left")   # P2REFEREE7: two lines, the titles of A and B touched
a1.legend(fontsize=8, frameon=False, loc="upper left")
mu = np.array([c.mean() for c in counts]); se = np.array([c.std(ddof=1) / np.sqrt(len(c)) for c in counts])
for g in range(group[-1] + 1):
    idx = np.where(group == g)[0]
    a3.errorbar(np.array(levels)[idx], mu[idx], yerr=se[idx], fmt="o", ms=4, color="C%d" % g, capsize=2)
a3.set_xlabel("sound level (dB)"); a3.set_ylabel("mean count in 50 ms")
a3.set_title("(C) one fibre: D = %d,\nK = %d (Poisson), %d (matched)" % (D[pick], Kp[pick], Km[pick]), fontsize=10, loc="left")
fig.tight_layout()
os.makedirs(os.path.join(H, "figs"), exist_ok=True)
fig.savefig(os.path.join(H, "figs", "fig_dk.pdf"))
out = {"rule": "D == K_matched and D > K_poisson; lower-median A", "candidates": len(cand), "path": paths[pick], "A": float(Af),
       "lam": float(lam), "D": int(D[pick]), "K_poisson": int(Kp[pick]), "K_matched": int(Km[pick]), "levels": list(map(float, levels)),
       "means": mu.tolist(), "groups": group.tolist(), "frac_poisson": float(np.mean(D <= Kp)), "frac_matched": float(np.mean(D <= Km)),
       "matched_K_above_10_not_shown": int(np.sum(Km > 10)), "matched_K_above_10_all_meet": bool(np.all(D[Km > 10] <= Km[Km > 10]))}
json.dump(out, open(os.path.join(H, "figs", "fig_dk.json"), "w"), indent=1)
print({k: out[k] for k in ("candidates", "A", "lam", "D", "K_poisson", "K_matched", "frac_poisson", "frac_matched")})
