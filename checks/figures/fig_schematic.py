"""fig_schematic.py (15 Oct 2026, P2COLDFIX finding 59): schematic Fig 1 of paper 2.
(a) a Gaussian stimulus distribution with the quantiles W_j of the optimal input at A = 10 (lambda = 0) marked;
(b) the infomax staircases of Proposition 1 for that stimulus law at A = 3, 10 and 25 (lambda = 0), with the large-A envelope A F_S(s)^2;
(c) the number of steps K(A) against A: below A = 18 from support_exact.support with points closer than 0.2 in the Fisher coordinate
    merged (the q1928 control: the merged count changes from 4 to 5 at the computed A_5 = 18.1366), from A = 18 on from the computed
    count changes A_K of gate_truth/table2_primary.json (support_exact splits points above about A = 28 and is not used there), with
    the proved asymptotic lower bound 0.056 A^(2/3) of Theorem 3. Panel (b) uses the merged points at A = 3, 10 and 25.
Every plotted number is written to paper_neuron/figs/fig_schematic.json."""
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from support_exact import support as _support  # noqa: E402


def support(A, lam):
    r = _support(A, lam); p = np.array(r['pts']); w = np.array(r['w'])
    o = np.argsort(p); p, w = p[o], w[o]; t = 2 * np.sqrt(p + lam)
    P, Wt = [p[0]], [w[0]]
    for i in range(1, len(p)):
        if t[i] - 2 * np.sqrt(P[-1] + lam) < 0.2:
            P[-1] = (P[-1] * Wt[-1] + p[i] * w[i]) / (Wt[-1] + w[i]); Wt[-1] += w[i]
        else:
            P.append(p[i]); Wt.append(w[i])
    return {'N': len(P), 'pts': [float(x) for x in P], 'w': [float(x) for x in Wt]}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

s = np.linspace(-3, 3, 1201)
F = norm.cdf(s)
out = {"stimulus": "standard normal", "staircases": {}, "K_of_A": {}}
plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})   # 15 Oct 21:2x: draft reading 2, finding 17
# 16 Oct 2026 (P2FIGFILES): PLOS figure rules: Arial, Times or Symbol at 8 to 12 pt, at most 7.5 in wide; TrueType embedded
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Arial", "pdf.fonttype": 42})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.5, 3.3))
sup10 = support(10.0, 0.0)
W10 = np.cumsum(sup10["w"]) / np.sum(sup10["w"])
a1.plot(s, norm.pdf(s), color="k")
for j, wj in enumerate(W10[:-1], start=1):
    q = norm.ppf(wj)
    a1.axvline(q, color="C2", lw=0.8, ls="--")
a1.set_xlabel("stimulus s"); a1.set_ylabel("probability density")
a1.set_title("(A) stimulus distribution,\nquantiles for A = 10", fontsize=10, loc="left")
for A, c in ((3.0, "C0"), (10.0, "C2"), (25.0, "C3")):
    sp = support(A, 0.0)
    W = np.cumsum(sp["w"]) / np.sum(sp["w"])
    idx = np.searchsorted(W, F, side="left").clip(0, len(W) - 1)
    a2.step(s, np.array(sp["pts"])[idx] / A, where="post", color=c, lw=1.3, label="A = %g, %d steps" % (A, sp["N"]))
    out["staircases"]["%g" % A] = {"K": sp["N"], "points": sp["pts"], "weights": sp["w"]}
a2.plot(s, F ** 2, color="0.4", lw=1.0, ls=":", label="large-A envelope")
a2.set_xlabel("stimulus s"); a2.set_ylabel("rate / peak count")
a2.legend(fontsize=8, frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.2, ncol=2, columnspacing=0.8)   # P2FIGFILES: two columns above the curves   # 16 Oct: draft reading 4, p5 (legend over the steps)
a2.set_xlim(-3.2, 3.2); a2.set_ylim(-0.05, 1.42); a2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a2.set_title("(B) infomax tuning curves", fontsize=10, loc="left")
T = json.load(open(os.path.join(os.path.dirname(HERE), "gate_truth", "table2_primary.json")))
AK = sorted((int(k), v["A_K"]) for k, v in T.items() if k.isdigit())
As = np.concatenate([np.arange(1.0, 18.0, 0.25), np.arange(18.0, 61.0, 0.25)])
Ks = [support(A, 0.0)["N"] if A < 18.0 else 4 + sum(1 for k, a in AK if a <= A) for A in As]
out["K_of_A"] = {"A": As.tolist(), "K": Ks}
a3.step(As, Ks, where="post", color="k", lw=1.3, label="optimal K")
a3.plot(As, 0.056 * As ** (2.0 / 3.0), color="C1", lw=1.0, label="bound 0.056 $A^{2/3}$")   # P2REFEREE7: exponent set as math
a3.set_xlabel("peak count A"); a3.set_ylabel("number of steps K")
a3.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.16), handlelength=1.2)   # P2FIGFILES: at 7.5 in, between the steps and the bound
a3.set_title("(C) more spikes allow\nmore steps", fontsize=10, loc="left")
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "fig_schematic.pdf"))
out["W_at_A10"] = W10.tolist()
json.dump(out, open(os.path.join(HERE, "figs", "fig_schematic.json"), "w"), indent=1)
print("K at A=3,10,25:", [out["staircases"][k]["K"] for k in ("3", "10", "25")], "K(60):", Ks[-1], "K(17.75):", Ks[list(As).index(17.75)])
