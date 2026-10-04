"""
Core machinery for the energy-vs-capacity Pareto front.

Primitives:
  CV_from_P(P, q)      exact C, V for ANY conditional matrix P[symbol, output]  (kernel ported verbatim from
                       the validated infomax_energy/energy_probe.py exact_CV, so COM-Poisson and GLM share it)
  comp_P(...)          COM-Poisson count channel
  ba_capacity_cost     cost-constrained Blahut-Arimoto -> GLOBALLY OPTIMAL asymptotic front (convex problem)
  front_asymptotic     n*C(E) via BA + bisection on the cost multiplier
  front_finiteN        max{nC - sqrt(nV)Qinv : rate <= E}, non-convex -> best-found (enumerate + polish + BA seed)

The front uses a DIRECT rate constraint (rate <= E), never a Lagrangian sweep: a Lagrangian can only reach the
concave hull, and the finite-N objective is not concave, so a Lagrangian jumps across bands of energy (this is
what produced the spurious "bifurcation / no optimum at intermediate rates" in the prior infomax_energy run).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from family import com_pmf

Qinv = lambda e: float(norm.ppf(1 - e))


# ---------------------------------------------------------------- exact information quantities
def CV_from_P(P, q):
    """Exact (C, V) for conditional matrix P[x, y] and input q. V = Var of the information density under the
    JOINT distribution (matches the paper's convention; reproduces C=0.5944, V=0.2224)."""
    P = np.asarray(P, float)
    q = np.asarray(q, float)
    q = q / q.sum()
    Pn = q @ P
    ok = (P > 0) & (Pn[None, :] > 0)
    I = np.zeros_like(P)
    np.divide(P, Pn[None, :], out=I, where=ok)
    np.log(I, out=I, where=ok)
    I[~ok] = 0.0
    W = q[:, None] * P
    C = float((W * I).sum())
    return C, float((W * I * I).sum()) - C * C


_pmf_cache = {}


def comp_P(rates_hz, fano, T=0.05, S=None):
    """COM-Poisson count channel. Row x = count pmf for mean mu_x = rates_hz[x]*T at the given Fano."""
    rates_hz = np.asarray(rates_hz, float)
    mu = rates_hz * T
    if S is None:
        S = int(mu.max() + 15 * np.sqrt(mu.max() * max(fano, 1.0) + 1) + 20)
    rows = []
    for m in mu:
        if m <= 1e-12:  # silence symbol: exact delta at zero count
            r = np.zeros(S)
            r[0] = 1.0
        else:
            k = (round(float(m), 6), round(float(fano), 6), S)
            if k not in _pmf_cache:
                _pmf_cache[k] = com_pmf(m, fano, S)[0]
            r = _pmf_cache[k]
        rows.append(r)
    return np.vstack(rows)


def mean_rate_cost(P, T=0.05, counts=None):
    """Energy per symbol = REALISED mean firing rate (Hz) = E[spike count | x] / T.
    Load-bearing for the GLM: a refractory kernel suppresses spikes, so a symbol with nominal rate 60 Hz may
    actually fire at 40 Hz. Charging the nominal rate would overstate the GLM's energy by ~50% and understate
    its efficiency. For COM-Poisson the realised mean equals the nominal by construction (checked)."""
    P = np.asarray(P, float)
    if counts is None:
        counts = np.arange(P.shape[1])
    return (P @ np.asarray(counts, float)) / T


def word_counts(Kbins):
    """Spike count of each binary word (for mean_rate_cost on a word alphabet)."""
    idx = np.arange(1 << Kbins, dtype=np.int64)
    pc = np.zeros(1 << Kbins, dtype=np.int64)
    for b in range(Kbins):
        pc += (idx >> b) & 1
    return pc


def fano_of_P(P, q):
    """Induced count Fano of the OUTPUT mixture-weighted per-symbol counts (mean over used symbols)."""
    y = np.arange(P.shape[1])
    q = np.asarray(q, float)
    q = q / q.sum()
    m = P @ y
    v = P @ (y * y) - m**2
    use = (q > 1e-6) & (m > 1e-9)
    if not use.any():
        return float("nan")
    return float((q[use] * (v[use] / m[use])).sum() / q[use].sum())


# ---------------------------------------------------------------- asymptotic front (globally optimal)
def ba_capacity_cost(P, cost, s, iters=4000, tol=1e-12, q0=None):
    """Blahut-Arimoto for max_q [ I(q) - s*sum_x q_x cost_x ]. Convex problem -> global optimum.
    Returns (q, I, mean_cost). q0 warm-starts (used by the parametric sweep)."""
    P = np.asarray(P, float)
    cost = np.asarray(cost, float)
    K = P.shape[0]
    q = (
        np.ones(K) / K
        if q0 is None
        else np.maximum(np.asarray(q0, float), 1e-12) / np.sum(np.maximum(q0, 1e-12))
    )
    prev = -np.inf
    for _ in range(iters):
        Qy = q @ P
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where((P > 0) & (Qy[None, :] > 0), np.log(P / np.where(Qy > 0, Qy, 1)[None, :]), 0.0)
        D = (P * R).sum(axis=1)  # D(P_x || Q)
        t = np.log(np.maximum(q, 1e-300)) + D - s * cost
        t -= t.max()
        q = np.exp(t)
        q /= q.sum()
        obj = float(q @ D) - s * float(q @ cost)
        if abs(obj - prev) < tol:
            break
        prev = obj
    C, _ = CV_from_P(P, q)
    return q, C, float(q @ cost)


def ba_front_parametric(P, cost, s_list, iters=1500, tol=1e-10):
    """Warm-started sweep of the cost multiplier s. Because the capacity-cost function is CONCAVE, this
    parametric sweep traces the ENTIRE asymptotic front (every point has a supporting line), and each point is
    the global optimum for its own s. Returns rows sorted by mean cost. This is the 'parametric plot' of the front.
    """
    P = np.asarray(P, float)
    cost = np.asarray(cost, float)
    rows = []
    q = None
    for s in sorted(s_list, reverse=True):  # high s (low energy) -> low s, warm-started
        q, C, e = ba_capacity_cost(P, cost, s, iters=iters, tol=tol, q0=q)
        V = CV_from_P(P, q)[1]
        rows.append({"s": float(s), "q": q.copy(), "C": float(C), "V": float(V), "rate": float(e)})
    rows.sort(key=lambda r: r["rate"])
    return rows


def ba_at_budget(P, cost, E, s_hi=200.0, iters=60):
    """Globally-optimal q maximising I subject to mean cost <= E (bisection on the multiplier s)."""
    q0, C0, e0 = ba_capacity_cost(P, cost, 0.0)
    if e0 <= E + 1e-12:
        return q0, C0, e0  # unconstrained optimum already feasible
    lo, hi = 0.0, s_hi
    best = None
    for _ in range(iters):
        s = 0.5 * (lo + hi)
        q, C, e = ba_capacity_cost(P, cost, s)
        if e > E:
            lo = s
        else:
            hi = s
            best = (q, C, e)
        if hi - lo < 1e-10:
            break
    if best is None:
        q, C, e = ba_capacity_cost(P, cost, s_hi)
        best = (q, C, e)
    return best


# ---------------------------------------------------------------- finite-N front (non-convex, best-found)
def _softmax(t):
    e = np.exp(t - t.max())
    return e / e.sum()


def front_point(P, cost, E, obj, p_grid=41, seed_qs=(), polish=True, allow_silence=True):
    """max{ obj(C,V,rate) : rate <= E } over inputs q. Best-found.
    Candidates: silence-only; binary {silence, j} up to the constraint boundary; coarse ternary; supplied seeds
    (e.g. the globally-optimal BA input); then Powell polish under a barrier. Exhaustive over the sparse family
    the optima are known to live in (validated in infomax_energy against a 6-restart/18-init global search).
    """
    P = np.asarray(P, float)
    cost = np.asarray(cost, float)
    K = P.shape[0]
    best = None

    def consider(q):
        nonlocal best
        q = np.asarray(q, float)
        if q.sum() <= 0:
            return
        q = q / q.sum()
        if not allow_silence and int((q > 1e-6).sum()) < 2:
            return  # a 1-symbol code transmits nothing
        r = float(q @ cost)
        if r > E + 1e-9:
            return
        C, V = CV_from_P(P, q)
        o = obj(C, V, r)
        if best is None or o > best["obj"]:
            best = {"q": q.copy(), "C": C, "V": V, "rate": r, "obj": o}

    if allow_silence:
        q = np.zeros(K)
        q[0] = 1.0
        consider(q)  # silence only -> obj 0 (always achievable)
    for j in range(1, K):
        if cost[j] <= 0:
            continue
        pmax = min(1.0, E / cost[j])
        for p in np.linspace(0.0, pmax, p_grid):
            q = np.zeros(K)
            q[0] = 1 - p
            q[j] = p
            consider(q)
    for j in range(1, K):  # coarse ternary
        for k in range(j + 1, K):
            for pj in (0.1, 0.2, 0.3, 0.4, 0.5):
                for pk in (0.1, 0.2, 0.3, 0.4, 0.5):
                    if pj + pk > 1.0:
                        continue
                    if pj * cost[j] + pk * cost[k] > E + 1e-12:
                        continue
                    q = np.zeros(K)
                    q[0] = 1 - pj - pk
                    q[j] = pj
                    q[k] = pk
                    consider(q)
    for sq in seed_qs:
        consider(sq)

    if polish and best is not None:
        BIG = 1e3
        t0 = np.log(np.maximum(best["q"], 1e-12))

        def neg(t):
            q = _softmax(t)
            C, V = CV_from_P(P, q)
            r = float(q @ cost)
            return -(obj(C, V, r) - BIG * max(0.0, r - E))

        r = minimize(neg, t0, method="Powell", options={"maxiter": 900, "xtol": 1e-7, "ftol": 1e-10})
        consider(_softmax(r.x))
    return best


def reduce_outputs(P, decimals=12):
    """EXACT alphabet reduction: merge outputs whose normalised conditional profile over x is identical.
    If P(y1|x)=a*u_x and P(y2|x)=b*u_x then the likelihood ratio, hence the information density, is the same
    for both, so C and V are preserved exactly (verified to 5e-17 on the GLM word channel)."""
    P = np.asarray(P, float)
    s = P.sum(axis=0)
    keep = s > 1e-300
    P = P[:, keep]
    s = s[keep]
    prof = np.round(P / s[None, :], decimals)
    _, inv = np.unique(prof.T, axis=0, return_inverse=True)
    G = int(inv.max()) + 1
    out = np.zeros((P.shape[0], G))
    for g in range(G):
        out[:, g] = P[:, inv == g].sum(axis=1)
    return out


def asym_front(P, rates_hz, E_targets, tol_E=1e-4):
    """Asymptotic (capacity-cost) front at prescribed energies. For each E, bisect the cost multiplier s so the
    BA optimum sits exactly at mean rate E (or take the unconstrained optimum if it is already cheaper). Each
    solve is a global optimum (convex problem), warm-started along the sweep. Computed ONCE per channel: the
    asymptotic front does not depend on the blocklength n."""
    cost = np.asarray(rates_hz, float)
    q_un, C_un, e_un = ba_capacity_cost(P, cost, 0.0, iters=3000, tol=1e-11)
    rows = []
    qw = None
    for E in np.sort(np.asarray(E_targets, float)):
        if E >= e_un - 1e-9:  # unconstrained optimum already affordable
            q, C, e = q_un, C_un, e_un
        else:
            lo, hi = 0.0, 1.0
            while True:  # grow the bracket until rate(hi) <= E
                q, C, e = ba_capacity_cost(P, cost, hi, iters=1200, tol=1e-11, q0=qw)
                if e <= E or hi > 1e4:
                    break
                hi *= 4.0
            for _ in range(45):
                s = 0.5 * (lo + hi)
                q, C, e = ba_capacity_cost(P, cost, s, iters=1200, tol=1e-11, q0=qw)
                if e > E:
                    lo = s
                else:
                    hi = s
                if abs(e - E) < tol_E:
                    break
            q, C, e = ba_capacity_cost(P, cost, hi, iters=2000, tol=1e-11, q0=qw)
            qw = q
        rows.append(
            {
                "E": float(E),
                "q": np.asarray(q).copy(),
                "C": float(C),
                "V": float(CV_from_P(P, q)[1]),
                "rate": float(e),
            }
        )
    return rows


def build_fronts(P, rates_hz, n, eps, s_list=None, T=0.05):
    """Asymptotic front by warm-started parametric BA sweep (globally optimal at every point, and the sweep
    covers the whole front because the capacity-cost function is concave). The finite-N front is then evaluated
    at the SAME energies via the direct rate constraint (best-found; seeded with the BA optimum and neighbours).
    y is log M* in nats: n*C  and  n*C - sqrt(n*V)*Qinv(eps)."""
    cost = np.asarray(rates_hz, float)  # energy = mean firing rate in Hz (A&L: linear)
    Qi = Qinv(eps)
    if s_list is None:
        s_list = np.concatenate([[0.0], np.geomspace(2e-4, 1.0, 34)])
    rows = ba_front_parametric(P, cost, s_list)
    seen, keep = set(), []
    for r in rows:  # drop duplicate energies from the sweep
        k = round(r["rate"], 6)
        if k in seen:
            continue
        seen.add(k)
        keep.append(r)
    rows = keep
    out = {
        k: []
        for k in (
            "E",
            "logM_asym",
            "logM_finN",
            "C_asym",
            "V_asym",
            "C_finN",
            "V_finN",
            "rate_finN",
            "nsupp_asym",
            "nsupp_finN",
            "fano_finN",
            "fano_asym",
        )
    }
    for i, r in enumerate(rows):
        E = r["rate"]
        seeds = [r["q"]] + [rows[j]["q"] for j in (i - 1, i + 1) if 0 <= j < len(rows)]
        fa = front_point(
            P, cost, E, lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi, seed_qs=tuple(seeds)
        )
        out["E"].append(float(E))
        out["logM_asym"].append(float(n * r["C"]))
        out["C_asym"].append(float(r["C"]))
        out["V_asym"].append(float(r["V"]))
        out["nsupp_asym"].append(int((r["q"] > 1e-3).sum()))
        out["fano_asym"].append(fano_of_P(P, r["q"]))
        if fa is None:
            for k, v in (
                ("logM_finN", 0.0),
                ("C_finN", 0.0),
                ("V_finN", 0.0),
                ("rate_finN", 0.0),
                ("nsupp_finN", 0),
                ("fano_finN", float("nan")),
            ):
                out[k].append(v)
        else:
            out["logM_finN"].append(float(fa["obj"]))
            out["C_finN"].append(float(fa["C"]))
            out["V_finN"].append(float(fa["V"]))
            out["rate_finN"].append(float(fa["rate"]))
            out["nsupp_finN"].append(int((fa["q"] > 1e-3).sum()))
            out["fano_finN"].append(fano_of_P(P, fa["q"]))
    return {k: np.asarray(v) for k, v in out.items()}


def timeshare_hull(E, C, V, n, eps):
    """Finite-N achievable front WITH time-sharing: split n uses between two operating points i, j.
    log M = n[aCi+(1-a)Cj] - sqrt(n[aVi+(1-a)Vj]) Qinv. Upper envelope over pairs and mixtures, at energy
    a*Ei+(1-a)*Ej. Returns the hull evaluated on the same E grid. This is the honest 'is it still a front if
    you allow mixing' object -- the asymptotic front is automatically concave, the finite-N one need not be.
    """
    Qi = Qinv(eps)
    E = np.asarray(E, float)
    C = np.asarray(C, float)
    V = np.asarray(V, float)
    P = len(E)
    alphas = np.linspace(0, 1, 21)
    best = np.full(P, -np.inf)
    for i in range(P):
        for j in range(P):
            for a in alphas:
                e = a * E[i] + (1 - a) * E[j]
                c = a * C[i] + (1 - a) * C[j]
                v = a * V[i] + (1 - a) * V[j]
                val = n * c - np.sqrt(max(n * v, 0.0)) * Qi
                idx = np.searchsorted(E, e, side="right") - 1  # attribute to the largest grid E <= e
                if 0 <= idx < P and val > best[idx]:
                    best[idx] = val
    return np.maximum.accumulate(np.where(np.isfinite(best), best, -np.inf))
