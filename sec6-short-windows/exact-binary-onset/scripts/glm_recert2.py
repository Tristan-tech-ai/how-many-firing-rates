"""
Recertify Part B capacity-code classification on a FINE grid (0.5 Hz) with a continuum-sense certificate:
BINARY iff a 2-level code {0} U {j, j+1} (time-sharing between ADJACENT grid levels allowed = one interior level)
reaches C_BA within 1e-7. GRADED iff no such code does (a genuine third level is needed).
Then recount Part B contrast rows using the saved transmitting masks. Deterministic.
"""

import numpy as np, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G1 = os.path.join(HERE, "..", "glm_contrib1")
sys.path.insert(0, G1)
from glm import glm_count_P
from core import mean_rate_cost, asym_front

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
T = 0.05
K = 121
RATES = np.linspace(0, 60, K)
KB = 25
E_TARGETS = np.concatenate([np.linspace(0.4, 10, 13), np.linspace(11.5, 29, 11)])
KERNELS = [
    (None, "h0"),
    ({"amp": 2.0, "tau_ms": 4.0}, "amp2_tau4"),
    ({"amp": 4.0, "tau_ms": 4.0}, "amp4_tau4"),
    ({"amp": 8.0, "tau_ms": 4.0}, "amp8_tau4"),
]


def Dkl(P, Q):
    m = P > 0
    return float((P[m] * (np.log(P[m]) - np.log(Q[m]))).sum())


def ba_restricted(Ps, cs, E):
    """max sum w D(P_i||Q) s.t. sum w = 1, sum w c <= E, w >= 0 -- SLSQP on a 3-point support (concave)."""
    from scipy.optimize import minimize

    n = len(Ps)
    H = np.array([Dkl(p, p) for p in Ps])  # zero; keep interface simple
    Hs = np.array([float((p[p > 0] * np.log(p[p > 0])).sum()) for p in Ps])

    def negC(w):
        Q = w @ Ps
        lq = np.log(np.where(Q > 0, Q, 1e-300))
        return -float(w @ (Hs - Ps @ lq))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, {"type": "ineq", "fun": lambda w: E - w @ cs}]
    best = 0.0
    for w0 in (np.ones(n) / n, np.array([0.7, 0.15, 0.15]), np.array([0.4, 0.3, 0.3])):
        r = minimize(
            negC,
            w0,
            method="SLSQP",
            bounds=[(0, 1)] * n,
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 400},
        )
        if r.success or True:
            w = np.clip(r.x, 0, 1)
            w /= w.sum()
            if w @ cs <= E + 1e-9:
                best = max(best, -negC(w))
    return best


old = json.load(open(os.path.join(G1, "resultsB.json")))
out = {}
tot = {"contrast": 0, "no_contrast": 0, "rows": 0}
for h, tag in KERNELS:
    Pc = glm_count_P(RATES, h, T, KB)
    cost = mean_rate_cost(Pc, T)
    rows = asym_front(Pc, cost, E_TARGETS[E_TARGETS <= cost.max()])
    rec = []
    for r in rows:
        E, C_BA, q = float(r["rate"]), float(r["C"]), np.asarray(r["q"])
        supp = np.where(q > 1e-4)[0]
        clusters = []
        for i in supp:
            if clusters and i - clusters[-1][-1] <= 1:
                clusters[-1].append(int(i))
            else:
                clusters.append([int(i)])
        # best 2-level code: {0} + one interior pair (j, j+1); search pairs near the BA clusters and globally coarse
        cand = set()
        for c in clusters:
            for i in c:
                cand.update([max(1, i - 1), i, min(K - 2, i)])
        cand.update(range(1, K - 1, 4))
        best = -1.0
        for j in sorted(c for c in cand if 1 <= c <= K - 2):
            C2 = ba_restricted(Pc[[0, j, j + 1]], cost[[0, j, j + 1]], E)
            best = max(best, C2)
        gap = C_BA - best
        cert = "BINARY" if gap <= 1e-7 else "GRADED"
        rec.append(
            {
                "E": E,
                "C_BA": C_BA,
                "n_clusters_BA": len(clusters),
                "cluster_rates": [float(np.mean(RATES[c])) for c in clusters],
                "best2_C": best,
                "gap": gap,
                "cert": cert,
            }
        )
    out[tag] = rec
    ng = sum(1 for x in rec if x["cert"] == "GRADED")
    print(
        f"{tag:<10}: certified GRADED (continuum sense) {ng}/{len(rec)} rows;  BA clusters per row: {[x['n_clusters_BA'] for x in rec]}"
    )
    print(
        f"            gaps (nats) where GRADED: {[f'{x['gap']:.1e}' for x in rec if x['cert']=='GRADED'][:12]}"
    )
    # contrast recount with old transmitting masks (same E ordering)
    orec = old["kernels"][tag]
    for n in ("10", "50"):
        F = np.array(orec["byN"][n]["F"], float)
        valid = np.isfinite(F)
        m = min(len(valid), len(rec))
        for i in range(m):
            if not valid[i]:
                continue
            tot["rows"] += 1
            if rec[i]["cert"] == "GRADED":
                tot["contrast"] += 1
            else:
                tot["no_contrast"] += 1
print(
    f"\nPART B contrast recount (transmitting rows x n, capacity certified on 0.5 Hz grid, adjacent merged):"
)
print(
    f"   contrast {tot['contrast']} / {tot['rows']}  ({100*tot['contrast']/max(tot['rows'],1):.0f}%)   no-contrast {tot['no_contrast']}    (old: 60 / 130 = 46%)"
)
out["contrast_recount"] = tot
json.dump(out, open(os.path.join(HERE, "results_glm_recert2.json"), "w"), indent=1)
print("[saved]")
