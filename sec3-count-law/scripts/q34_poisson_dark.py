"""
Q34, the offset c as a function of dark current lambda. 2026-09-06, EXPLORATORY (float64, not a certificate).

Pre-registered in WALL_ANALYSIS.md (Q33 refinement). With dark current, Y ~ Poisson(x + lambda), 0 <= x <= A.
The variance-stabilised variable t = 2 sqrt(x + lambda) runs over an interval of half-length
h(A, lambda) = sqrt(A + lambda) - sqrt(lambda). The offset against the Gaussian channel is
    c(lambda, K) = h(A_K(lambda), lambda) - A_g,K,
with A_K(lambda) the Poisson amplitude where the count becomes K and A_g,K the bisected Gaussian transition of
results_q33_refine_transitions.json. Readings, fixed before running: c decreasing toward 0 with lambda
supports the discreteness reading (the low end of the Poisson channel is where the stabilising map fails);
c flat in lambda says the offset is geometric and the reading is wrong.
CONTROL: at lambda = 0 the float64 transitions must land inside or next to the Amendment 68 brackets.

Method. D(x) = sum_y P(x,y)[log P(x,y) - log Q(y)], P = Poisson(x + lambda), D'(x) = sum_y P (y/(x+lambda) - 1)
[log P - log Q]. KKT equalities at the atoms, boundary atoms pinned at 0 and A, least squares from the
continuation guess (step 1 in A from the certified five-atom solution at A = 27.9452), an atom inserted at the
violation when max(D - C) > 1e-9 on a 3001-point grid, atoms closer than 0.1 merged, negligible weights
dropped; transitions bisected on the violation of the (K-1)-atom solution.
"""
import os, sys, json, time
import numpy as np
from scipy.special import gammaln
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")


def ny_of(A, lam):
    m = A + lam
    return int(m + 18 * np.sqrt(m) + 50)


def logP(x, lam, y):
    mu = np.asarray(x, float)[:, None] + lam
    with np.errstate(divide="ignore"):
        lp = y[None, :] * np.log(np.where(mu > 0, mu, 1.0)) - mu - gammaln(y[None, :] + 1)
    lp = np.where(mu > 0, lp, np.where(y[None, :] == 0, 0.0, -np.inf))
    return lp


def D_and_Dp(xq, xs, ws, lam, y):
    P = np.exp(logP(xs, lam, y)); Q = ws @ P
    lQ = np.log(np.maximum(Q, 1e-300))
    lPq = logP(xq, lam, y); Pq = np.exp(lPq)
    g = np.where(Pq > 0, lPq - lQ[None, :], 0.0)
    D = (Pq * g).sum(1)
    mu = np.asarray(xq, float)[:, None] + lam
    Dp = (Pq * (y[None, :] / np.where(mu > 0, mu, 1.0) - 1) * g).sum(1)
    return D, Dp


def residual(v, A, lam, K, y):
    ws = v[:K]; xs = np.concatenate(([0.0], v[K:], [A]))
    D, Dp = D_and_Dp(xs, xs, ws, lam, y)
    return np.concatenate((D[1:] - D[0], [ws.sum() - 1.0], Dp[1:-1]))


def solve(A, lam, xs0, ws0):
    y = np.arange(ny_of(A, lam))
    K = len(xs0)
    v0 = np.concatenate((ws0, xs0[1:-1]))
    r = least_squares(residual, v0, args=(A, lam, K, y), method="lm", xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=400)
    ws = r.x[:K]; inner = r.x[K:]
    order = np.argsort(np.concatenate(([0.0], inner, [A])))
    xs = np.concatenate(([0.0], inner, [A]))[order]; ws = ws[order]
    D, _ = D_and_Dp(xs, xs, ws, lam, y)
    C = float(ws @ D)
    xq = np.linspace(0, A, 3001)
    Dq, _ = D_and_Dp(xq, xs, ws, lam, y)
    viol = float(Dq.max() - C); at = float(xq[Dq.argmax()])
    return xs, ws, C, viol, at, float(np.abs(r.fun).max())


def clean_up(xs, ws, A, merge_tol=0.1, w_min=1e-6):
    pairs = sorted(zip(xs.tolist(), ws.tolist()))
    out = []
    for u, w in pairs:
        if out and abs(u - out[-1][0]) < merge_tol:
            u0, w0 = out[-1]
            out[-1] = ((u0 * w0 + u * w) / (w0 + w) if w0 + w > 0 else (u0 + u) / 2, w0 + w)
        else:
            out.append((u, w))
    K = len(out)
    out = [(u, w) for i, (u, w) in enumerate(out) if w > w_min or i in (0, K - 1)]
    out[0] = (0.0, out[0][1]); out[-1] = (A, out[-1][1])
    xs = np.array([u for u, _ in out]); ws = np.array([max(w, 1e-9) for _, w in out]); ws = ws / ws.sum()
    return xs, ws


def step_solve(A, lam, xs, ws, tol=1e-9):
    for _ in range(8):
        xs, ws = clean_up(xs, ws, A)
        xs, ws, C, viol, at, res = solve(A, lam, xs, ws)
        if viol <= tol:
            break
        if min(abs(xs - at)) < 0.1:
            break
        pairs = sorted([(float(u), float(w) * 0.98) for u, w in zip(xs, ws)] + [(at, 0.02)])
        xs = np.array([u for u, _ in pairs]); ws = np.array([w for _, w in pairs])
    xs, ws = clean_up(xs, ws, A)
    xs, ws, C, viol, at, res = solve(A, lam, xs, ws)
    clean = bool(res < 1e-10 and viol <= tol and ws.min() > 1e-5 and np.diff(xs).min() > 0.2)
    return xs, ws, C, viol, res, clean


def viol_of(A, lam, xs, ws):
    x = np.array(xs) * (A / xs[-1]); x[0] = 0.0; x[-1] = A
    _, _, _, viol, _, _ = solve(A, lam, x, np.array(ws))
    return viol


def transitions(lam, xs0, ws0, A0, A_max=140.0, step=1.0, say=print):
    xs, ws = np.array(xs0, float), np.array(ws0, float)
    A = A0
    prev = None; trans = {}; bad = 0
    while A <= A_max + 1e-9:
        xs, ws, C, viol, res, clean = step_solve(A, lam, xs, ws)
        K = len(xs)
        bad = 0 if clean else bad + 1
        if bad >= 3:
            say(f"   lambda = {lam:4.1f}  continuation lost at A = {A:.2f}; stopping this lambda")
            break
        if prev is not None and clean and prev["clean"] and K == prev["K"] + 1:
            lo, hi = prev["A"], A
            b = prev
            v_lo, v_hi = viol_of(lo, lam, b["xs"], b["ws"]), viol_of(hi, lam, b["xs"], b["ws"])
            if v_lo <= 1e-9 < v_hi:
                for _ in range(14):
                    mid = (lo + hi) / 2
                    if viol_of(mid, lam, b["xs"], b["ws"]) <= 1e-9:
                        lo = mid
                    else:
                        hi = mid
                trans[K] = (lo + hi) / 2
                say(f"   lambda = {lam:4.1f}  K = {K:2d} at A = {trans[K]:9.4f}   (h = {np.sqrt(trans[K] + lam) - np.sqrt(lam):.4f})")
            else:
                say(f"   lambda = {lam:4.1f}  K = {K:2d} between A = {lo} and {hi}: bracket does not straddle ({v_lo:.1e}, {v_hi:.1e})")
        if clean:
            prev = {"A": A, "K": K, "xs": xs.tolist(), "ws": ws.tolist(), "clean": clean}
        else:
            say(f"   lambda = {lam:4.1f}  A = {A:7.2f}  K = {K:2d}  UNCLEAN (viol {viol:+.1e}, resid {res:.1e})")
        A_new = A + step
        xs = xs * (A_new / A); xs[0] = 0.0
        A = A_new
    return trans


if __name__ == "__main__":
    e = json.load(open(os.path.join(HERE, "results_q32_exact40_all.json")))["5 to 6 lower"]
    xs0 = [float(v) for v in e["x"]]; ws0 = [float(v) for v in e["w"]]; A0 = e["A"]
    gauss = {int(k): v["Ag_transition"] for k, v in json.load(open(os.path.join(HERE, "results_q33_refine_transitions.json")))["transitions"].items()}
    brackets = {6: (28.150, 28.300), 7: (39.612, 40.028), 8: (52.556, 52.701), 9: (66.650, 67.057),
                10: (81.850, 81.9068), 11: (97.938, 98.344), 12: (115.258, 115.310), 13: (133.355, 133.692)}
    lams = [float(v) for v in sys.argv[1:]] or [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    out = {}
    t0 = time.time()
    say = lambda *a: print(*a, flush=True)
    for lam in lams:
        tr = transitions(lam, xs0, ws0, A0, say=say)
        row = {}
        for K, A_K in tr.items():
            h = float(np.sqrt(A_K + lam) - np.sqrt(lam))
            row[K] = {"A": A_K, "h": h, "c": (h - gauss[K]) if K in gauss else None}
            if lam == 0.0 and K in brackets:
                lo, hi = brackets[K]
                say(f"      control K = {K}: float64 transition {A_K:.4f} vs certified bracket ({lo}, {hi}] -> {'inside' if lo - 0.05 <= A_K <= hi + 0.05 else 'OUTSIDE'}")
        out[str(lam)] = row
        cs = [r["c"] for r in row.values() if r["c"] is not None]
        say(f"   lambda = {lam}: c over K = {[round(v, 3) for v in cs]}  mean {np.mean(cs):.3f}   [{time.time()-t0:.0f}s]\n")
        json.dump(out, open(os.path.join(HERE, "results_q34_poisson_dark.json"), "w"), indent=1)
    print(f"{'lambda':>7s} " + " ".join(f"{'c(K=%d)' % K:>9s}" for K in range(6, 14)) + f" {'mean c':>8s}")
    for lam, row in out.items():
        print(f"{float(lam):7.1f} " + " ".join(f"{row[K]['c']:9.3f}" if K in row and row[K]["c"] is not None else f"{'-':>9s}" for K in range(6, 14))
              + f" {np.mean([r['c'] for r in row.values() if r['c'] is not None]):8.3f}")
