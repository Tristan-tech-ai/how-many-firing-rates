"""
Q33, the Gaussian side of the correspondence: N_Gauss(sqrt A) against N_Poisson(A). 2026-09-06. EXPLORATORY
(float64, not a certificate).

Pre-registered in WALL_ANALYSIS.md (addendum 2026-09-06). The variance-stabilising map t = 2 sqrt(x) sends
Poisson(x) on [0, A] to roughly Gaussian noise of unit variance on t in [0, 2 sqrt A], an interval of length
2 sqrt A, which is the amplitude-constrained Gaussian channel |u| <= A_g with A_g = sqrt A. So the test is
N_Gauss(A_g = sqrt A) - N_Poisson(A) at the twelve certified A. Readings, fixed before running:
    constant difference (0 or 1) at all twelve  -> lane 3 (correspondence as a theorem) is the target;
    difference growing with A                   -> lane 3 dies, lane 2 (zero counting) is next;
    oscillating difference                      -> the correspondence is asymptotic and the constant matters.

Method. Y = u + Z, Z ~ N(0,1), |u| <= A_g. D(u) = int phi(y-u) [log phi(y-u) - log Q(y)] dy with
Q = sum_k w_k phi(y - u_k); D'(u) = int phi(y-u)(y-u)[log phi(y-u) - log Q(y)] dy. KKT equalities
D(u_k) = D(u_0), sum w = 1, D'(u_i) = 0 at interior atoms, solved by Levenberg-Marquardt from the continuation
guess; trapezoid on a 0.05 grid (the integrands are Gaussian-smooth, so the rule converges super-geometrically).
Continuation upward in A_g in steps of 0.05 with the boundary atoms pinned at +-A_g. When the KKT check fails
(max of D - C over a 2001-point grid above 1e-9), the atom nearest the violation is SPLIT into a pair if the
violation is within 0.5 of an interior atom (a SPLIT move; it was tried in q33_gauss_support_split_FAILED.py and destabilised the
continuation above A_g = 8.8, so it is NOT used), otherwise a new atom is inserted at the violation with its
mirror, which is the move that produced the clean band 4.5 to 12.0 twice. Atoms closer than 0.05 are merged
and negligible weights dropped. A step is CLEAN when the residual is below 1e-10, the check passes, all weights
exceed 1e-5, atoms are 0.1 apart and the solution is symmetric to 1e-6; counts are reported only from clean
steps. Control: the 2 -> 3 transition must sit at A_g = 1.665 (Smith 1971). The first version of this script
inserted instead of splitting and blew up between A_g = 2.95 and 4.15 and again above 12; the clean band it
produced (4.5 to 12.0) agrees with this version where both exist.
Output: results_q33_gauss_support.json, the target table, and the transition table with the offset in sqrt A.
"""

import os, sys, json, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
LOG2PI = np.log(2 * np.pi)


def grid(Ag, dy=0.05):
    return np.arange(-Ag - 10, Ag + 10 + dy / 2, dy), dy


def logphi(y, u):
    z = y[None, :] - np.asarray(u)[:, None]
    return -0.5 * z * z - 0.5 * LOG2PI


def D_and_Dp(u_query, us, ws, y, dy):
    P = np.exp(logphi(y, us))
    Q = ws @ P
    lQ = np.log(np.maximum(Q, 1e-300))
    lPq = logphi(y, u_query)
    Pq = np.exp(lPq)
    g = lPq - lQ[None, :]
    D = (Pq * g).sum(1) * dy
    Dp = (Pq * (y[None, :] - np.asarray(u_query)[:, None]) * g).sum(1) * dy
    return D, Dp


def residual(v, Ag, K, y, dy):
    ws = v[:K]
    us = np.concatenate(([-Ag], v[K:], [Ag]))
    D, Dp = D_and_Dp(us, us, ws, y, dy)
    return np.concatenate((D[1:] - D[0], [ws.sum() - 1.0], Dp[1:-1]))


def solve(Ag, us0, ws0):
    y, dy = grid(Ag)
    K = len(us0)
    v0 = np.concatenate((ws0, us0[1:-1]))
    r = least_squares(
        residual, v0, args=(Ag, K, y, dy), method="lm", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=300
    )
    ws = r.x[:K]
    inner = r.x[K:]
    order = np.argsort(np.concatenate(([-Ag], inner, [Ag])))
    us = np.concatenate(([-Ag], inner, [Ag]))[order]
    ws = ws[order]
    D, _ = D_and_Dp(us, us, ws, y, dy)
    C = float(ws @ D)
    xq = np.linspace(-Ag, Ag, 2001)
    Dq, _ = D_and_Dp(xq, us, ws, y, dy)
    viol = float(Dq.max() - C)
    at = float(xq[Dq.argmax()])
    return us, ws, C, viol, at, float(np.abs(r.fun).max())


def clean_up(us, ws, merge_tol=0.05, w_min=1e-6):
    pairs = sorted(zip(us.tolist(), ws.tolist()))
    out = []
    for u, w in pairs:
        if out and abs(u - out[-1][0]) < merge_tol:
            u0, w0 = out[-1]
            out[-1] = ((u0 * w0 + u * w) / (w0 + w) if w0 + w > 0 else (u0 + u) / 2, w0 + w)
        else:
            out.append((u, w))
    K = len(out)
    out = [(u, w) for i, (u, w) in enumerate(out) if w > w_min or i in (0, K - 1)]
    us = np.array([u for u, _ in out])
    ws = np.array([max(w, 1e-9) for _, w in out])
    ws = ws / ws.sum()
    return us, ws


def continuation(targets, Ag0, Ag_max, us0, ws0, step=0.05, tol=1e-9, say=print):
    us = np.array(us0, float)
    ws = np.array(ws0, float)
    Ag = Ag0
    hist = []
    tgt = {}
    tq = sorted(targets)
    while Ag <= Ag_max + 1e-9:
        for _ in range(8):
            us, ws = clean_up(us, ws)
            us, ws, C, viol, at, res = solve(Ag, us, ws)
            if viol <= tol:
                break
            new = [at] if min(abs(us - at)) > 0.05 else []
            if abs(at) > 0.05 and min(abs(us + at)) > 0.05:
                new.append(-at)
            if not new:
                break
            pairs = [(float(u), float(w) * (1 - 0.02 * len(new))) for u, w in zip(us, ws)] + [
                (float(a), 0.02) for a in new
            ]
            pairs.sort()
            us = np.array([u for u, _ in pairs])
            ws = np.array([w for _, w in pairs])
        us, ws = clean_up(us, ws)
        us, ws, C, viol, at, res = solve(Ag, us, ws)
        K = len(us)
        sym = float(np.abs(us + us[::-1]).max() + np.abs(ws - ws[::-1]).max())
        clean = bool(
            res < 1e-10 and viol <= tol and ws.min() > 1e-5 and np.diff(us).min() > 0.1 and sym < 1e-6
        )
        hist.append(
            {
                "Ag": round(Ag, 4),
                "K": K,
                "C": C,
                "viol": viol,
                "res": res,
                "sym": sym,
                "clean": clean,
                "us": us.tolist(),
                "ws": ws.tolist(),
            }
        )
        while tq and tq[0] <= Ag + 1e-9:
            t = tq.pop(0)
            tgt[t] = {"Ag": round(Ag, 4), "K": K, "clean": clean}
        if (
            abs(Ag * 2 - round(Ag * 2)) < 1e-9
            or (len(hist) > 1 and hist[-1]["K"] != hist[-2]["K"])
            or not clean
        ):
            say(
                f"   Ag = {Ag:6.2f}  K = {K:2d}  C = {C:.6f}  max(D - C) = {viol:+.2e}  KKT resid {res:.1e}  asym {sym:.1e}  {'clean' if clean else 'UNCLEAN'}   atoms {np.round(us, 3).tolist()}"
            )
        Ag_new = Ag + step
        us = us * (Ag_new / Ag)
        Ag = Ag_new
    return hist, tgt


if __name__ == "__main__":
    A_poisson = {
        27.9452: 5,
        28.362: 6,
        39.612: 6,
        40.028: 7,
        52.5555: 7,
        66.6505: 8,
        97.9375: 10,
        100.0: 11,
        115.2578: 11,
        133.355: 12,
        150.0: 13,
        152.0: 13,
    }
    poisson_trans = {
        6: (28.150, 28.300),
        7: (39.612, 40.028),
        8: (52.556, 52.701),
        9: (66.650, 67.057),
        10: (81.850, 81.9068),
        11: (97.938, 98.344),
        12: (115.258, 115.310),
        13: (133.355, 133.692),
    }
    targets = {float(np.sqrt(A)): (A, K) for A, K in A_poisson.items()}
    t0 = time.time()
    say = lambda *a: print(*a, flush=True)
    print("control: continuation from two atoms at A_g = 1.2", flush=True)
    h0, _ = continuation([], 1.2, 2.2, [-1.2, 1.2], [0.5, 0.5], say=say)
    t23 = next((r["Ag"] for r in h0 if r["K"] == 3 and r["clean"]), None)
    print(f"   first clean 3-atom amplitude on the 0.05 grid = {t23} (Smith 1971: 1.665)\n", flush=True)
    print("main run from the clean 5-atom solution at A_g = 4.5", flush=True)
    hist, tgt = continuation(
        list(targets.keys()), 4.5, 13.0, [-4.5, -1.8409, 0.0, 1.8409, 4.5], [0.2] * 5, say=say
    )
    print(
        f"\n{'A (Poisson)':>12s} {'N_Poisson':>9s} {'A_g = sqrt A':>12s} {'N_Gauss':>8s} {'difference':>10s}"
    )
    rows = []
    for ag in sorted(targets):
        A, KP = targets[ag]
        KG = tgt[ag]["K"]
        cl = tgt[ag]["clean"]
        rows.append({"A": A, "N_poisson": KP, "Ag": ag, "N_gauss": KG, "diff": KG - KP, "clean": cl})
        print(f"{A:12.4f} {KP:9d} {ag:12.4f} {KG:8d} {KG - KP:10d}   {'clean' if cl else 'UNCLEAN'}")
    gtrans = {}
    for r in hist:
        if r["clean"] and r["K"] not in gtrans:
            gtrans[r["K"]] = r["Ag"]
    print(
        f"\n{'K':>3s} {'Gauss: first A_g with K':>24s} {'Poisson: sqrt(A) at K-1 -> K':>30s} {'offset (Poisson - Gauss)':>24s}"
    )
    offs = []
    for K in sorted(gtrans):
        if K in poisson_trans:
            lo, hi = poisson_trans[K]
            pm = float(np.sqrt((lo + hi) / 2))
            offs.append(pm - gtrans[K])
            print(f"{K:3d} {gtrans[K]:24.2f} {pm:30.3f} {pm - gtrans[K]:24.3f}")
    json.dump(
        {"rows": rows, "gauss_transitions": gtrans, "offsets": offs, "control_2to3": t23, "history": hist},
        open(os.path.join(HERE, "results_q33_gauss_support.json"), "w"),
        indent=1,
    )
    diffs = [r["diff"] for r in rows if r["clean"]]
    print(
        f"\ndifferences at clean targets: {diffs};  offsets in sqrt A: mean {np.mean(offs):.3f}, sd {np.std(offs):.3f}   [{time.time()-t0:.0f}s]"
    )
    if len(set(diffs)) == 1:
        print("reading: constant difference; lane 3 (correspondence as a theorem) is the target")
    elif all(b >= a for a, b in zip(diffs, diffs[1:])) and diffs[-1] > diffs[0]:
        print("reading: difference grows with A; lane 3 dies, lane 2 next")
    else:
        print("reading: difference oscillates; the correspondence is asymptotic and the constant matters")
