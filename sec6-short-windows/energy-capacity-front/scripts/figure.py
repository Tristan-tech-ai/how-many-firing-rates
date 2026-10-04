"""Builds the figure from results.json. ONLY real computed points are plotted (no schematic curves)."""

import json, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open("results.json", encoding="utf-8"))
comp, glm = d["comp"], d["glm"]
fam = comp["family"]
FANOS = comp["fanos"]
EPS = comp["eps"]
KAPPA, BETA = 0.71e9, 0.34e9  # Attwell-Laughlin / Kostal-Shinomoto linear budget (ATP/spike, ATP/s)

cmap = plt.get_cmap("viridis")
col = {f: cmap(i / (len(FANOS) - 1)) for i, f in enumerate(FANOS)}
fig, AX = plt.subplots(2, 2, figsize=(13.5, 10.4))
fig.suptitle(
    "Energy vs capacity for single neurons: asymptotic capacity vs finite-blocklength log M*(n, $\\epsilon$)\n"
    f"COM-Poisson count channels (0-60 Hz alphabet, 50 ms window), $\\epsilon$ = {EPS}. "
    "All points computed exactly; no schematic curves.",
    fontsize=11.5,
)

# ---------------- Panel A: the energy-capacity front ----------------
ax = AX[0, 0]
n = 10
for f in FANOS:
    r = fam[str(f)]
    E = np.array(r["E"])
    ax.plot(E, np.array(r["byN"][str(n)]["logM_asym"]), ls="--", lw=1.2, color=col[f], alpha=0.75)
    ax.plot(E, np.array(r["byN"][str(n)]["logM_finN"]), ls="-", lw=2.0, color=col[f])
# zero-crossings: below the FIRST, no model transmits; between first and last, only the low-Fano ones do
zcs = [fam[str(f)]["byN"][str(n)]["zero_crossing_Hz"] for f in FANOS]
zmin, zmax = np.nanmin(zcs), np.nanmax(zcs)
ax.axvspan(0, zmin, color="crimson", alpha=0.11)
ax.axvspan(zmin, zmax, color="orange", alpha=0.07)
ax.text(
    zmin / 2, 6.6, "no model\ntransmits\nanything", ha="center", va="center", fontsize=7.5, color="crimson"
)
ax.text(
    (zmin + zmax) / 2,
    6.6,
    "only the most reliable\n(low-Fano) neurons transmit",
    ha="center",
    va="center",
    fontsize=7.5,
    color="darkorange",
)
ax.set_xlabel("mean firing rate (Hz)  —  energy (Attwell-Laughlin: linear in rate)")
ax.set_ylabel("log M*  (nats, per $n$=10 windows = 500 ms)")
ax.set_title(
    "A. The front. dashed = asymptotic $nC$;  solid = finite-blocklength $nC-\\sqrt{nV}Q^{-1}(\\epsilon)$",
    fontsize=9.5,
)
h = [Line2D([], [], color=col[f], lw=2, label=f"Fano {f}") for f in FANOS]
h += [
    Line2D([], [], color="k", ls="--", lw=1.2, label="asymptotic"),
    Line2D([], [], color="k", ls="-", lw=2, label="finite-N (n=10)"),
]
ax.legend(handles=h, fontsize=7.2, ncol=2, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(0, 29)

# ---------------- Panel B: the threshold and its 1/n scaling ----------------
ax = AX[0, 1]
r = fam["1.0"]
E = np.array(r["E"])
ax.plot(E, np.array(r["C_asym"]) * 1.0, color="k", ls="--", lw=1.4, label="asymptotic  $C$ (per window)")
for n2, c in zip(comp["NS"], ["#d62728", "#1f77b4", "#2ca02c"]):
    y = np.array(r["byN"][str(n2)]["logM_finN"]) / n2
    ax.plot(E, y, color=c, lw=2, label=f"finite-N, n={n2}  (per window)")
    z = r["byN"][str(n2)]["zero_crossing_Hz"]
    if np.isfinite(z):
        ax.axvline(z, color=c, ls=":", lw=1.2)
        ax.annotate(
            f"n={n2}: {z:.1f} Hz\n$N_{{sp}}$={n2*z*comp['T']:.1f}",
            (z, 0.06 + 0.045 * comp["NS"].index(n2)),
            fontsize=7,
            color=c,
            ha="left",
        )
    else:
        ax.annotate(
            f"n={n2}: never transmits\n(max $N_{{sp}}$={n2*E.max()*comp['T']:.1f} < {comp['Qinv_sq']:.1f})",
            (12, 0.015),
            fontsize=7,
            color=c,
            ha="left",
        )
ax.axhline(0, color="grey", lw=0.7)
ax.set_title(
    "B. Poisson neuron: a transmission THRESHOLD the asymptote does not have.\n"
    f"Threshold sits at a fixed total spike budget $N_{{sp}}\\approx Q^{{-1}}(\\epsilon)^2$={comp['Qinv_sq']:.2f}",
    fontsize=9.5,
)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("log M* per window (nats)")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(0, 29)

# ---------------- Panel C: the GLM line, at matched readout and matched dispersion ----------------
ax = AX[1, 0]
gl = glm["count"]["amp8_tau4"]
gE = np.array(gl["E"])
gC = np.array(gl["C_asym"])
for f, sty in ((0.5, "-"), (1.0, "-")):
    r2 = fam[str(f)]
    ax.plot(
        np.array(r2["E"]), np.array(r2["C_asym"]), sty, color=col[f], lw=2, label=f"COM-Poisson, Fano {f}"
    )
ax.plot(gE, gC, color="crimson", lw=2.6, label="GLM (refractory kernel)\ninduced Fano 0.46")
ax.plot(
    np.array(glm["count"]["h0"]["E"]),
    np.array(glm["count"]["h0"]["C_asym"]),
    color="crimson",
    lw=1.3,
    ls=":",
    label="GLM, zero history",
)
ax.set_title(
    "C. Does history-dependence improve the tradeoff?\nAbove Poisson, but BELOW a count model of the same Fano.",
    fontsize=9.5,
)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("asymptotic $C$ per window (nats)")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25)
ax.set_xlim(0, 29)

# ---------------- Panel D: where should a neuron sit? bits per ATP ----------------
ax = AX[1, 1]
r = fam["1.0"]
E = np.array(r["E"])
W = BETA + KAPPA * E
a = np.array(r["byN"]["10"]["logM_asym"]) / 10.0
for n2, c in zip([10, 50], ["#1f77b4", "#2ca02c"]):
    b = np.array(r["byN"][str(n2)]["logM_finN"]) / n2
    e = b / W * 1e9
    ax.plot(E, e, color=c, lw=2, label=f"finite-N, n={n2}")
    i = int(np.nanargmax(e))
    ax.plot(E[i], e[i], "o", color=c, ms=7)
    ax.annotate(
        f"{E[i]:.0f} Hz", (E[i], e[i]), textcoords="offset points", xytext=(4, 5), fontsize=8, color=c
    )
ea = a / W * 1e9
ax.plot(E, ea, color="k", ls="--", lw=1.5, label="asymptotic")
i = int(np.nanargmax(ea))
ax.plot(E[i], ea[i], "ko", ms=7)
ax.annotate(f"{E[i]:.1f} Hz", (E[i], ea[i]), textcoords="offset points", xytext=(4, 5), fontsize=8)
ax.axvspan(2, 5, color="grey", alpha=0.16)
ax.text(
    3.5, ax.get_ylim()[1] * 0.55, "measured\ncortical\n2-5 Hz", ha="center", fontsize=7.5, color="dimgrey"
)
ax.set_title(
    "D. Efficiency on the REAL A&L budget ($W=\\beta+\\kappa r$).\n"
    "Asymptotic optimum ~1 Hz (near measured); finite-N optimum 10-24 Hz.",
    fontsize=9.5,
)
ax.set_xlabel("mean firing rate (Hz)")
ax.set_ylabel("nats per $10^9$ ATP (log M*/W)")
ax.legend(fontsize=7.5)
ax.grid(alpha=0.25)
ax.set_xlim(0, 29)

fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig("pareto_front.png", dpi=175)
print("wrote pareto_front.png")
