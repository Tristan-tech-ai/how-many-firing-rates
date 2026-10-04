"""Q181: the growth-conjecture verdict from the first DIRECT transition, and the pre-registered predictions
for the next one. Inputs: the measured 59-point birth (the 58-point optimal at n = 679, not at 681; linear
zero of the centre signal), the q173 chain (two-wall law, closure c(K), kappa = 0.42), the K = 40 Gaussian
anchor A_g(40) = 29.41 (q143) and the certified binomial transitions n*(25), n*(26) (Table VII).
Everything printed is computed here; nothing is typed in from memory except the anchors named above."""

import os, sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q173_conjecture_table as q  # noqa: prints its table on import; harmless

KAPPA, A40 = 0.42, q.A40


def c_of(g):
    return (0.9840 - 0.2347 * g) / 2


def h_of_n(n):
    return np.pi * np.sqrt(n) / 2


def Ag_from_nstar(K, n, g):
    """invert the two-wall law: A_g = h_F - 2c - kappa (h_F - sqrt n)/K"""
    h = h_of_n(n)
    return h - 2 * c_of(g) - KAPPA * (h - np.sqrt(n)) / K


def nstar_from_Ag(K, A, g):
    c = c_of(g)
    h = A + 2 * c
    for _ in range(80):
        n = (2 * h / np.pi) ** 2
        h = A + 2 * c + KAPPA * (h - np.sqrt(n)) / K
    return (2 * h / np.pi) ** 2


# --- the measurement: centre signal of the 58-point at 681 and 679 (q179 walk, forty digits)
s681, s679 = 4.154e-15, -2.493e-14
n59 = 681 - 2 * s681 / (s681 - s679)
print(
    f"\n59-point birth: linear zero of the 58-point centre signal between 681 and 679 -> n*(59) = {n59:.2f}"
)
g59 = q.Ag_of(60, "4/3") - q.Ag_of(58, "4/3")  # g_eff differs by < 0.01 among the columns
A59 = Ag_from_nstar(59, n59, g59)
print(f"g_eff(59) used {g59:.4f}, c = {c_of(g59):.4f}; implied Gaussian amplitude A_g(59) = {A59:.3f}")
for lab, A in (
    ("A^(4/3)", q.Ag_conj(59, "4/3")),
    ("A log A", q.Ag_conj(59, "log")),
    ("1.22 held", q.Ag_conj(59, "1.22")),
):
    print(
        f"   column {lab:10s}: A_g(59) = {A:.3f}, n*(59) = {nstar_from_Ag(59, A, g59):.1f}, miss = {nstar_from_Ag(59, A, g59) - n59:+.1f}"
    )
gam_40_59 = np.log(59 / 40) / np.log(A59 / A40)
print(f"secant exponent d ln K / d ln A_g, K = 40 -> 59 (anchor A_g(40) = {A40:.2f}): {gam_40_59:.4f}")
# sensitivity to the anchor and to the measurement
for dA in (-0.2, 0.2):
    print(f"   anchor A_g(40) {dA:+.1f} -> secant {np.log(59/40)/np.log(A59/(A40+dA)):.4f}")
for dn in (-1, 1):
    print(f"   n*(59) {dn:+d} -> secant {np.log(59/40)/np.log(Ag_from_nstar(59, n59+dn, g59)/A40):.4f}")
# --- the binomial's own coordinate, no Gaussian anchor: K vs h_F between the certified transitions and this one
n25, n26 = 174.934, 186.386
h26, h59 = h_of_n(n26), h_of_n(n59)
print(f"\nbare Fisher coordinate: K = 26 born at h_F = {h26:.3f}, K = 59 at h_F = {h59:.3f}")
print(f"   secant d ln K / d ln h_F, 26 -> 59: {np.log(59/26)/np.log(h59/h26):.4f}")
print(f"   local at K = 25.5 from the plateau length: {2/(25.5*np.log(n26/n25)):.4f}")
# --- pre-registered predictions for the 69-point birth, each hypothesis carried from the MEASURED A_g(59)
g69 = q.Ag_of(70, "4/3") - q.Ag_of(68, "4/3")
print(f"\nPREDICTIONS for n*(69) from A_g(59) = {A59:.3f} (g_eff(69) = {g69:.4f}, c = {c_of(g69):.4f}):")
hyp = {
    "A^(4/3) locally (1.333)": 4 / 3,
    "exponent rising (1.29)": 1.29,
    "A log A locally (1+1/ln A)": 1 + 1 / np.log(A59),
    "secant 40->59 held (%.3f)" % gam_40_59: gam_40_59,
    "1.22 held": 1.22,
}
for lab, gam in hyp.items():
    A69 = A59 * (69 / 59) ** (1 / gam)
    print(f"   {lab:32s}: A_g(69) = {A69:.3f}, n*(69) = {nstar_from_Ag(69, A69, g69):.1f}")
print(
    "   the original q173 columns anchored at K = 40:  ",
    {lab: round(q.nstar(69, w)[0], 1) for lab, w in (("4/3", "4/3"), ("log", "log"), ("1.22", "1.22"))},
)
print(
    "\nmeasurement resolution: +-1 in n from the step; chain error (kappa +-50 %, c +-0.02) about +-3 in n at 870."
)
