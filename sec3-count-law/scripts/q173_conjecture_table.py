"""Q173: the two growth conjectures and the measured Gaussian staircase, written as predictions for the
symmetric binomial transitions n*(K) at K = 55..72, so the thirty-digit transitions can be compared the
moment they land (Amendments 251, 253b, 253c).

Chain of the prediction, every step from a file:
  A_g(K) for K <= 40 from q143 (results_q143_Ag_all.json; spacing flags fail from 41) and Table II below 26;
  the two-wall law read forward: h_F(K) = A_g(K) + 2 c(K) + kappa (h_F - sqrt n) / K, solved for h_F, then
  n*(K) = (2 h_F / pi)^2; c(K) from the closure law (0.9840 - 0.2347 g_eff)/2 with g_eff extrapolated as
  A_g(K+1) - A_g(K-1), kappa = 0.42;
  beyond K = 40, A_g(K) from each conjecture normalised at K = 40, A_g = 29.41:
     A^(4/3):  K = 40 (A_g / 29.41)^(4/3)     -> A_g(K) = 29.41 (K/40)^(3/4)
     A log A:  K = c A_g ln A_g, c = 40 / (29.41 ln 29.41)  -> A_g(K) by inversion
     local exponent 1.22 held:  A_g(K) = 29.41 (K/40)^(1/1.22)
The measured lower bounds are printed alongside.
"""

import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
q143 = json.load(open(os.path.join(HERE, "results_q143_Ag_all.json")))["A_g"]
Ag = {int(k): float(v) for k, v in q143.items() if int(k) <= 40}
table2 = {
    5: 4.02128,
    6: 5.05816,
    7: 6.03983,
    8: 6.97912,
    9: 7.88348,
    10: 8.75930,
    11: 9.60970,
    12: 10.43922,
    13: 11.24861,
    14: 12.04221,
    15: 12.81870,
    16: 13.58200,
    17: 14.33240,
    18: 15.07100,
    19: 15.79860,
    20: 16.51610,
    21: 17.22400,
    22: 17.92310,
    23: 18.61381,
    24: 19.29666,
    25: 19.9719,
}
Ag.update(table2)
A40 = Ag[40]
c_log = 40.0 / (A40 * np.log(A40))


def Ag_conj(K, which):
    if which == "4/3":
        return A40 * (K / 40.0) ** 0.75
    if which == "1.22":
        return A40 * (K / 40.0) ** (1 / 1.22)
    # A log A: solve c A ln A = K
    A = A40
    for _ in range(60):
        f = c_log * A * np.log(A) - K
        df = c_log * (np.log(A) + 1)
        A -= f / df
    return A


def Ag_of(K, which):
    return Ag[K] if K in Ag else Ag_conj(K, which)


def nstar(K, which):
    # c(K) from the closure law with g_eff = A_g(K+1) - A_g(K-1)
    g = Ag_of(K + 1, which) - Ag_of(K - 1, which)
    c = (0.9840 - 0.2347 * g) / 2
    A_g = Ag_of(K, which)
    # h_F = A_g + 2c + 0.42 (h_F - sqrt n)/K with n = (2 h_F/pi)^2  -> fixed point in h_F
    h = A_g + 2 * c
    for _ in range(50):
        n = (2 * h / np.pi) ** 2
        h = A_g + 2 * c + 0.42 * (h - np.sqrt(n)) / K
    return (2 * h / np.pi) ** 2, A_g, c


meas = {
    38: "N(350) = 38 (30-digit)",
    55: "N(700) >= 55 (30-digit)",
    57: "N(750) >= 57 (float, above floor)",
    58: "N(760) >= 58 (30-digit: K=57 curv +1.6e-10)",
    66: "N(1000) >= 66 (30-digit)",
    68: "N(1030) >= 68 (float)",
    70: "N(1030) >= 70 (float, provisional)",
}
print(
    f"{'K':>3} | {'n* A^(4/3)':>11} {'n* A log A':>11} {'n* gamma=1.22':>13} | {'A_g used':>9} {'c(K)':>6} | measured"
)
for K in list(range(36, 41)) + list(range(54, 62)) + list(range(64, 78)):
    r1 = nstar(K, "4/3")
    r2 = nstar(K, "log")
    r3 = nstar(K, "1.22")
    src = "q143/T2" if K in Ag else "conj"
    print(
        f"{K:3d} | {r1[0]:11.1f} {r2[0]:11.1f} {r3[0]:13.1f} | {r1[1]:9.3f} {r1[2]:6.3f} | {meas.get(K, '')}   [{src}]"
    )
print(
    "\nA fixed n sits in the plateau K with n*(K) <= n < n*(K+1). Table VII (measured): n*(25) = 174.934, n*(26) = 186.386;"
)
print(
    "check: ",
    {K: round(nstar(K, "4/3")[0], 2) for K in (25, 26)},
    " (the two-wall law with kappa = 0.42, no conjecture involved at K <= 40)",
)
