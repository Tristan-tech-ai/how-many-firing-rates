"""Figures for paper 2 (4 Oct 2026), house style figstyle.py.
fig_wall.pdf    : gate_truth/table2_primary.json (built from primary output files): sqrt(A_K) vs A_g(K); c(K) measured vs closure.
fig_front.pdf   : _archive/lanes/pareto_front/results.json (cost-constrained BA for the long-window front; best-found search
                  under a direct rate constraint for the short-window front, n = 50, eps = 0.01, T = 50 ms, rates 0-60 Hz).
fig_graded.pdf  : temporal_within/results_glm_recert2.json (glm_recert2.py: exact test of every binary code, 0.5 Hz grid,
                  A = 3): advantage of the best graded code over the best binary code against the energy budget.
Run from this folder: py make_figs2.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figstyle import apply, OK, WIDTH

apply()
HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.join(HERE, "..", "..")

# ---- fig_wall ----
# Table II values come from gate_truth/build_table2.py, which reads only primary output files (4 Oct 2026, 17:1x)
T2 = json.load(open(os.path.join(TW, "gate_truth", "table2_primary.json"), encoding="utf-8"))
K = list(range(5, 25))
Ag = [T2[str(k)]["A_g"] for k in K]
AK = [T2[str(k)]["A_K"] for k in K]
c = [T2[str(k)]["c"] for k in K]
clo = [T2[str(k)]["closure"] for k in K]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(WIDTH, 4.2))
a1.plot([3.5, 20], [3.5, 20], color="0.7", lw=0.6, ls=":")
a1.plot(Ag, [x ** 0.5 for x in AK], "o", ms=3, color=OK["blue"])
a1.text(13.2, 10.6, r"$\sqrt{A_K} = A_g$ (dotted)", fontsize=8, color="0.35", rotation=33)
a1.set_xlabel(r"Gaussian half-length $A_g(K)$"); a1.set_ylabel(r"Poisson $\sqrt{A_K}$")
a1.text(-0.22, 1.02, "(a)", transform=a1.transAxes, fontsize=10, fontweight="bold", va="top")
a2.plot(K, clo, "-", color="0.4", lw=1.0, label="chain closure $E(g_{\\rm eff})/2$")
a2.plot(K[:16], c[:16], "o", ms=3.5, color=OK["blue"], label="measured")
a2.plot(K[16:], c[16:], "s", ms=4.5, mfc="none", mec=OK["vermillion"], mew=1.0, label="predicted before measured")
a2.set_xlabel(r"number of rates $K$"); a2.set_ylabel(r"wall constant $c(K)$")
a2.legend(loc="lower right")
a2.text(-0.22, 1.02, "(b)", transform=a2.transAxes, fontsize=10, fontweight="bold", va="top")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_wall.pdf")); plt.close(fig)

# ---- fig_front ----
d = json.load(open(os.path.join(TW, "..", "_archive", "lanes", "pareto_front", "results.json"), encoding="utf-8"))
fam = d["comp"]["family"]; glm = d["glm"]["count"]
cols = {0.3: OK["blue"], 0.5: OK["sky"], 1.0: OK["black"], 2.0: OK["orange"]}   # fig-gate: selection (Fano factors drawn; keys of results.json comp.family)
fig, (b1, b2) = plt.subplots(2, 1, figsize=(WIDTH, 4.4))
for f in (0.3, 0.5, 1.0, 2.0):   # fig-gate: selection
    r = fam[str(f)]; E = np.array(r["E"])
    b1.plot(E, np.array(r["C_asym"]), ls="--", lw=0.9, color=cols[f])
    y = np.array(r["byN"]["50"]["logM_finN"]) / 50.0
    b1.plot(E, y, ls="-", lw=1.3, color=cols[f])
    b1.text(E[-1] + 0.4, np.array(r["C_asym"])[-1], f"Fano {f}", fontsize=8, va="center")
b1.axhline(0, color="0.7", lw=0.5)
b1.plot([], [], ls="--", color="0.3", lw=0.9, label="long windows ($C$)")
b1.plot([], [], ls="-", color="0.3", lw=1.3, label="50 windows, $\\epsilon = 0.01$")
b1.set_xlim(0, 33); b1.set_xlabel("mean firing rate (Hz)"); b1.set_ylabel("nats per window")
b1.legend(loc="upper left")
b1.text(-0.22, 1.02, "(a)", transform=b1.transAxes, fontsize=10, fontweight="bold", va="top")
for f, col in ((0.5, OK["sky"]), (1.0, OK["black"])):
    r = fam[str(f)]
    b2.plot(np.array(r["E"]), np.array(r["C_asym"]), "-", lw=1.2, color=col, label=f"COM-Poisson, Fano {f}")
g = glm["amp8_tau4"]
b2.plot(np.array(g["E"]), np.array(g["C_asym"]), "-", lw=1.6, color=OK["vermillion"], label="GLM, induced Fano 0.46")
g0 = glm["h0"]
b2.plot(np.array(g0["E"]), np.array(g0["C_asym"]), ":", lw=1.2, color=OK["vermillion"], label="GLM, no history")
b2.set_xlim(0, 33); b2.set_xlabel("mean firing rate (Hz)"); b2.set_ylabel(r"long-window $C$ (nats per window)")
b2.legend(loc="lower right")
b2.text(-0.22, 1.02, "(b)", transform=b2.transAxes, fontsize=10, fontweight="bold", va="top")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_front.pdf")); plt.close(fig)

# ---- fig_graded ----
d = json.load(open(os.path.join(TW, "results_glm_recert2.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
for k, lab, mk, col in (("h0", "no history", "o", OK["black"]), ("amp2_tau4", "amplitude 2", "s", OK["blue"]),
                        ("amp4_tau4", "amplitude 4", "^", OK["green"]), ("amp8_tau4", "amplitude 8", "D", OK["vermillion"])):
    rows = d[k]; E = [r["E"] for r in rows]; gp = [1e3 * max(r["gap"], 0.0) for r in rows]
    ax.plot(E, gp, marker=mk, ms=2.8, lw=0.9, color=col, label=lab)
ax.set_xlabel("energy budget, mean rate (Hz)"); ax.set_ylabel(r"graded advantage ($10^{-3}$ nats)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_graded.pdf")); plt.close(fig)
print("written fig_wall, fig_front, fig_graded")
