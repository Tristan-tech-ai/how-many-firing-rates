"""
Q89: Gaussian optimal input from scratch: Blahut-Arimoto on a fine symmetric input grid, then clustering,
symmetric Newton refinement (q84 functions) and a violation check. Independent of births and splits.
Usage: py q89_gauss_ba.py <A_g list, comma separated> [<grid step, default 0.02>]
"""

import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import q33_gauss_support as qg

src = (
    open(os.path.join(HERE, "q84_gauss_symmetric_chain.py"), encoding="utf-8")
    .read()
    .split("# start: the K0 state from file")[0]
)
ns = {"__file__": os.path.join(HERE, "q84_gauss_symmetric_chain.py")}
_argv = sys.argv
sys.argv = ["x", "0", "0"]
exec(src, ns)
sys.argv = _argv
solve_fixed, violation, full_from_half = ns["solve_fixed"], ns["violation"], ns["full_from_half"]
say = lambda *a: print(*a, flush=True)
t0 = time.time()


def blahut_arimoto(Ag, step, iters=20000, tol=1e-9):
    y, dy = qg.grid(Ag)
    grid = np.arange(-Ag, Ag + 1e-9, step)
    n = len(grid)
    logP = qg.logphi(y, grid)  # (n, ny)
    P = np.exp(logP)
    p = np.full(n, 1.0 / n)
    for it in range(iters):
        Q = p @ P
        lQ = np.log(np.maximum(Q, 1e-300))
        D = ((P * (logP - lQ[None, :])).sum(1)) * dy
        C = float(p @ D)
        gap = D.max() - C
        if gap < tol:
            break
        p = p * np.exp(D - C)
        p /= p.sum()
        if it % 2000 == 1999:
            say(
                f"      BA it {it+1}: gap {gap:.2e}, mass > 1e-4 on {(p > 1e-4).sum()} points   [{time.time()-t0:.0f}s]"
            )
    return grid, p, D, C, gap, it + 1


def clusters_of(grid, p, thresh=1e-5, merge=0.3):
    idx = np.where(p > thresh)[0]
    cl = [[idx[0]]]
    for i in idx[1:]:
        if grid[i] - grid[cl[-1][-1]] < merge:
            cl[-1].append(i)
        else:
            cl.append([i])
    u = np.array([np.average(grid[c], weights=p[c]) for c in cl])
    w = np.array([p[c].sum() for c in cl])
    return u, w / w.sum()


step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
out = {}
for Ag in [float(v) for v in sys.argv[1].split(",")]:
    grid, p, D, C, gap, its = blahut_arimoto(Ag, step)
    u, w = clusters_of(grid, p)
    K = len(u)
    say(
        f"A_g = {Ag:.3f}: BA gap {gap:.1e} after {its} iterations; {K} clusters: u = {np.round(u, 3).tolist()}, w = {np.round(w, 4).tolist()}   [{time.time()-t0:.0f}s]"
    )
    # symmetric refinement
    m = K // 2
    right_u = u[len(u) - m :]
    right_w = w[len(w) - m :]
    wc = float(w[(K - 1) // 2]) if K % 2 == 1 else None
    wcf, wf, uf, res, itn = solve_fixed(
        K, Ag, wc, np.array(right_w), np.array(right_u[:-1]) if m > 0 else np.array([])
    )
    uu, ww = full_from_half(K, wcf, wf, uf, Ag)
    vmax, up = violation(K, Ag, wcf, wf, uf)
    say(
        f"   refined: residual {res:.0e} ({itn} steps); max(D - C) = {vmax:+.1e} at u = {up:.3f}; min w {ww.min():.4f}; N = {K if vmax < 1e-9 else 'UNRESOLVED'}   [{time.time()-t0:.0f}s]"
    )
    out[str(Ag)] = {"N": int(K), "u": uu.tolist(), "w": ww.tolist(), "viol": float(vmax), "res": float(res)}
    json.dump(out, open(os.path.join(HERE, "results_q89_gauss_ba.json"), "w"), indent=1)
say("done")
