"""Q135: symmetric binomial channel Bin(n, p) on p in [0, 1], mirror-reduced KKT system in mpmath (dps 30), continuation
in n.  Unknowns as in q134: right-half interior positions u_1 < .. < u_m (x = n p, > n/2), their weights, the endpoint
weight (endpoint x = n, mirrored at 0), and the centre weight for odd K.  Equations: D(u_j) = D(endpoint), D'(u_j) = 0,
D(centre) = D(endpoint) for odd K, total mass 1.
Solver: float pre-solve (scipy hybr, robust) + mp Newton polish (finite-difference Jacobian, damped).  Near a birth the
released system is nearly singular (the bifurcation), so the newborn coordinate is HELD and found by a bracketed root
search of the dropped equation: odd K near an insertion -> hold the centre weight wc, g(wc) = D(centre) - D(end);
even K near a split -> hold the innermost pair position u_0, g(u_0) = D'(u_0).
Transitions: n*(K+1) = zero crossing (linear in n) of the K-point's centre violation D(1/2) - C (even K, insertion) or
of its centre curvature D''(1/2) (odd K, split), both from the K-point at consecutive integers.
usage: py q135_symmetric_binomial_mp.py <n0> <K0> <n_max>   (start: q134 record at n0 if present, else the wall-gap guess)
"""

import sys, os, json, time
import numpy as np
from scipy.optimize import root
from mpmath import mp, mpf, log, exp, loggamma, sqrt, asin, pi, matrix, lu_solve

mp.dps = 30
MAX_DX = mpf("0.5")
FLOAT_ONLY = (
    os.environ.get("FLOAT_ONLY") == "1"
)  # skip the mp polish: float solutions (residual ~1e-13) are accepted, births still by held brackets
TOLR = mpf("1e-9") if FLOAT_ONLY else mpf("1e-18")
JUMP = mpf(
    "1.5"
)  # x units: a solve whose innermost atom moves more than this from the last state is a branch jump
HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
say = lambda *a: print(*a, flush=True)
t0 = time.time()
n0, K0, nmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
_src = (
    open(os.path.join(HERE, "q134_symmetric_binomial_mirror.py"), encoding="utf-8")
    .read()
    .split("\nu, w, wc, res = solve(n0")[0]
)
if os.environ.get("FLOAT_ONLY") == "1":
    _src = _src.replace('"maxfev": 200000', '"maxfev": 4000').replace('"maxiter": 100000', '"maxiter": 3000')
_a = sys.argv
sys.argv = ["x", str(n0), str(K0), str(nmax)]
ns = {"__file__": os.path.join(HERE, "q134_symmetric_binomial_mirror.py")}
exec(_src, ns)
sys.argv = _a
fsolve, fguess, fanalyse, fheld, ffull, fstats = (
    ns["solve"],
    ns["guess"],
    ns["analyse"],
    ns["solve_held"],
    ns["full"],
    ns["stats"],
)

# ---------------- mp layer
_LB = {}


def lbinom(n):
    if n not in _LB:
        _LB[n] = [loggamma(n + 1) - loggamma(y + 1) - loggamma(n - y + 1) for y in range(n + 1)]
    return _LB[n]


def logP(n, x):
    LB = lbinom(n)
    lp = log(x / n)
    lq = log(1 - x / n)
    return [LB[y] + y * lp + (n - y) * lq for y in range(n + 1)]


def fullQ(n, u, w, we, wc):
    Q = [mpf(0)] * (n + 1)
    LPs = []
    for uj, wj in zip(u, w):
        L = logP(n, uj)
        LPs.append(L)
        for y in range(n + 1):
            e = exp(L[y])
            Q[y] += wj * e
            Q[n - y] += wj * e
    Q[n] += we
    Q[0] += we
    Lc = None
    if wc is not None:
        Lc = logP(n, mpf(n) / 2)
        for y in range(n + 1):
            Q[y] += wc * exp(Lc[y])
    return Q, LPs, Lc


def D_Dp(n, x, L, lQ):
    D = mpf(0)
    Dp = mpf(0)
    for y in range(n + 1):
        p = exp(L[y])
        g = L[y] - lQ[y]
        D += p * g
        Dp += p * (y / x - (n - y) / (n - x)) * g
    return D, Dp


def residual(n, u, w, we, wc, hold_centre=False, drop_dp0=False):
    Q, LPs, Lc = fullQ(n, u, w, we, wc)
    lQ = [log(q) for q in Q]
    Dend = -lQ[n]
    eqD = []
    eqDp = []
    for j, (uj, L) in enumerate(zip(u, LPs)):
        D, Dp = D_Dp(n, uj, L, lQ)
        eqD.append(D - Dend)
        if not (drop_dp0 and j == 0):
            eqDp.append(Dp)
    if wc is not None and not hold_centre:
        Dc, _ = D_Dp(n, mpf(n) / 2, Lc, lQ)
        eqD.append(Dc - Dend)
    mass = 2 * sum(w) + 2 * we + (wc if wc is not None else 0) - 1
    return eqD + eqDp + [mass]


def probe(n, u, w, we, wc):
    """centre violation D(1/2) - C and centre curvature D''(1/2); C = D(endpoint); also D'(u_0)"""
    Q, LPs, Lc = fullQ(n, u, w, we, wc)
    lQ = [log(q) for q in Q]
    C = -lQ[n]
    d = mpf(n) * mpf("1e-3")
    vals = []
    for xc in (mpf(n) / 2 - d, mpf(n) / 2, mpf(n) / 2 + d):
        D, _ = D_Dp(n, xc, logP(n, xc), lQ)
        vals.append(D - C)
    _, dp0 = D_Dp(n, u[0], LPs[0], lQ)
    return vals[1], (vals[0] - 2 * vals[1] + vals[2]) / d**2, dp0


def newton_gen(F, v, feas, steps=40, tol=mpf("1e-22"), npos=0):
    """damped Newton with a finite-difference Jacobian; npos = number of leading position unknowns (step cap)"""
    r = F(v)
    res = max(abs(t) for t in r)
    k = 0
    N = len(v)
    while res > tol and k < steps:
        J = matrix(N, N)
        for i in range(N):
            h = mpf("1e-14") * max(1, abs(v[i]))
            vp = list(v)
            vp[i] += h
            rp = F(vp)
            for j in range(N):
                J[j, i] = (rp[j] - r[j]) / h
        dv = lu_solve(J, matrix([-t for t in r]))
        lam = mpf(1)
        dmax = max([abs(dv[i]) for i in range(npos)] + [mpf(0)])
        if dmax > MAX_DX:
            lam = MAX_DX / dmax
        ok = False
        for _ in range(40):
            vn = [v[i] + lam * dv[i] for i in range(N)]
            if feas(vn):
                rn = F(vn)
                resn = max(abs(t) for t in rn)
                if resn < res * (1 - lam / 10):
                    ok = True
                    break
            lam /= 2
        if not ok:
            break
        v, r, res = vn, rn, resn
        k += 1
    return v, res, k


def to_mp(a):
    return [mpf(repr(float(q))) for q in a]


def floats(u, w, we, wc):
    return (
        np.array([float(t) for t in u]),
        np.array([float(t) for t in w] + [float(we)]),
        (float(wc) if wc is not None else None),
    )


CENTRE_MARGIN = mpf(
    "1.5"
)  # x units (~0.3 Fisher): with a centre atom present the innermost pair must stay off the centre


def feasible_uw(n, uu, ww, wwe, wwc, margin=mpf(0)):
    m = len(uu)
    return (
        all(uu[i] < uu[i + 1] for i in range(m - 1))
        and (m == 0 or (uu[0] > mpf(n) / 2 + margin and uu[-1] < n))
        and all(t > 0 for t in ww)
        and wwe > 0
        and (wwc is None or wwc > 0)
    )


def solve_full(n, u, w, we, wc, hold_c=None, margin_x=None):
    """full system (or centre weight held): float pre-solve then mp polish; returns u, w, we, wc, res, steps"""
    uf, wf, wcf = floats(u, w, we, wc)
    resf = 1.0
    uf2 = wf2 = wcf2 = None
    try:
        if hold_c is None:
            uf2, wf2, wcf2, resf = fsolve(n, uf, wf, wcf)
        else:
            uf2, wf2, wcf2, resf = fheld(n, uf, wf, float(hold_c))
    except Exception:
        pass
    margin = (
        margin_x
        if margin_x is not None
        else (CENTRE_MARGIN if (hold_c is not None or wc is not None) else mpf(0))
    )
    okf = (
        uf2 is not None
        and resf < 1e-8
        and np.all(np.isfinite(uf2))
        and wf2.min() > 0
        and (wcf2 is None or wcf2 > 0)
        and np.all(np.diff(uf2) > 0)
        and uf2[0] > n / 2 + float(margin)
        and uf2[-1] < n
    )
    if okf:
        u0, w0, we0 = to_mp(uf2), to_mp(wf2[:-1]), mpf(repr(float(wf2[-1])))
        wc0 = hold_c if hold_c is not None else (mpf(repr(float(wcf2))) if wcf2 is not None else None)
    else:
        u0, w0, we0, wc0 = list(u), list(w), we, (hold_c if hold_c is not None else wc)
        say(
            f"      [float stage failed] resf {resf:.1e}; "
            + (
                f"min w {wf2.min():.2e}, u0 - n/2 {uf2[0] - n / 2:.3f}, ordered {bool(np.all(np.diff(uf2) > 0))}, wc {wcf2}"
                if uf2 is not None
                else "exception"
            )
        )
        if margin_x is not None:
            return (
                list(u),
                list(w),
                we,
                wc,
                mpf(1),
                0,
            )  # even-K pair collapsed: leave it to the pair-held fallback
    m = len(u0)
    odd = wc0 is not None and hold_c is None

    def unpack(v):
        return (
            v[:m],
            v[m : 2 * m],
            v[2 * m],
            (v[2 * m + 1] if odd else (hold_c if hold_c is not None else None)),
        )

    def F(v):
        uu, ww, wwe, wwc = unpack(v)
        return residual(n, uu, ww, wwe, wwc, hold_centre=(hold_c is not None))

    def feas(v):
        uu, ww, wwe, wwc = unpack(v)
        return feasible_uw(n, uu, ww, wwe, wwc, margin)

    v = list(u0) + list(w0) + [we0] + ([wc0] if odd else [])
    if FLOAT_ONLY and okf:
        return list(u0), list(w0), we0, wc0, mpf(repr(float(resf))), 0
    v, res, k = newton_gen(F, v, feas, steps=(40 if okf else 120), npos=m)
    uu, ww, wwe, wwc = unpack(v)
    return list(uu), list(ww), wwe, wwc, res, k


def solve_pair_held(n, u0, u_rest, w, we):
    """even K with the innermost right atom held at u0 (its D' equation dropped): float pre-solve then mp polish"""
    m = len(u_rest) + 1
    u0f = float(u0)

    def Ff(v):
        x, ww = ffull(n, np.concatenate([[u0f], v[: m - 1]]), v[m - 1 : 2 * m], None)
        D, Dp, _ = fstats(n, x, ww)
        K = len(x)
        h = K // 2
        right = np.arange(h, K)
        return np.concatenate([D[right[:-1]] - D[right[-1]], Dp[right[1:-1]], [ww.sum() - 1]])

    vf = np.concatenate([[float(q) for q in u_rest], [float(q) for q in w], [float(we)]])
    resf = 1.0
    vf2 = vf
    try:
        sol = root(Ff, vf, method="hybr", options={"xtol": 1e-14, "maxfev": (4000 if FLOAT_ONLY else 200000)})
        resf = np.abs(Ff(sol.x)).max()
        vf2 = sol.x
    except Exception:
        pass
    okf = (
        resf < 1e-8
        and np.all(np.isfinite(vf2))
        and vf2[m - 1 :].min() > 0
        and np.all(np.diff(np.concatenate([[u0f], vf2[: m - 1]])) > 0)
        and vf2[m - 2] < n
    )
    v = to_mp(vf2) if okf else (list(u_rest) + list(w) + [we])
    if FLOAT_ONLY and okf:
        return [u0] + list(v[: m - 1]), list(v[m - 1 : 2 * m - 1]), v[2 * m - 1], mpf(repr(float(resf))), 0

    def unpack(v):
        return [u0] + list(v[: m - 1]), v[m - 1 : 2 * m - 1], v[2 * m - 1]

    def F(v):
        uu, ww, wwe = unpack(v)
        return residual(n, uu, ww, wwe, None, drop_dp0=True)

    def feas(v):
        uu, ww, wwe = unpack(v)
        return feasible_uw(n, uu, ww, wwe, None)

    v, res, k = newton_gen(F, v, feas, steps=(40 if okf else 120), npos=m - 1)
    uu, ww, wwe = unpack(v)
    return list(uu), list(ww), wwe, res, k


def illinois(g, a, b, ga, gb, tol=mpf("1e-22"), it_max=40):
    """regula falsi (Illinois) on a bracket with ga * gb < 0; g returns (value, state); state[-1] is the solve residual"""
    side = 0
    sc = None
    for it in range(it_max):
        c = b - gb * (b - a) / (gb - ga)
        gc, sc = g(c)
        if abs(gc) < tol or sc[-1] > TOLR:
            return c, gc, sc, it + 1
        if gc * gb < 0:
            a, ga = b, gb
            side = 0
        else:
            ga = ga / 2 if side == 1 else ga
            side = 1
        b, gb = c, gc
    return b, gb, sc, it_max


def solve_centre_bracket(n, u, w, we, wc_lo, wc_hi=None):
    """odd K near an insertion: hold wc, root of g(wc) = D(centre) - D(end) in wc.  g > 0 at small wc (the K-point's
    violation) and decreases through zero at the physical weight; the bracket is found by an upward scan with
    continuation from the last converged state (failed solves shrink the step and never decide a sign)"""
    last = [u, w, we]

    def g(wc):
        u2, w2, we2, _, res2, k2 = solve_full(n, last[0], last[1], last[2], None, hold_c=wc)
        if res2 < TOLR and abs(u2[0] - last[0][0]) > JUMP:
            res2 = mpf(1)  # branch jump: treat as a failed solve
        if res2 < TOLR:
            last[0], last[1], last[2] = u2, w2, we2  # continuation in wc
        Q, LPs, Lc = fullQ(n, u2, w2, we2, wc)
        lQ = [log(q) for q in Q]
        Dc, _ = D_Dp(n, mpf(n) / 2, Lc, lQ)
        say(
            f"      [centre bracket] wc = {mp.nstr(wc, 5)}: g = {mp.nstr(Dc + lQ[n], 4)}, solve residual {mp.nstr(res2, 2)} ({k2} steps), u0 - n/2 = {mp.nstr(u2[0] - mpf(n) / 2, 4)}"
        )
        return Dc + lQ[n], (u2, w2, we2, res2)

    a = wc_lo
    ga, sa = g(a)
    tries = 0
    while (sa[-1] > TOLR or ga <= 0) and tries < 8:
        a = a / 3
        ga, sa = g(a)
        tries += 1
    if sa[-1] > TOLR or ga <= 0:
        return None
    b, gb = a, ga
    f = mpf("1.3")
    tries = 0
    found = False
    while tries < 30:
        c = b * f
        gc, sc = g(c)
        tries += 1
        if sc[-1] > TOLR:
            f = sqrt(f)
            continue
        if gc < 0:
            found = True
            break
        b, gb = c, gc
    if not found:
        return None
    wc, gc, sc, it = illinois(g, b, c, gb, gc)
    return sc[0], sc[1], sc[2], wc, max(abs(gc), sc[3]), it


def solve_pair_bracket(n, u_rest, w, we, d_lo):
    """even K near a split: hold the innermost pair position (half-separation d, Fisher units), root of g(d) = D'(u_0).
    g > 0 for d below the physical separation (the pair is pushed outward) and turns negative beyond it; upward scan
    with continuation from the last converged state, failed solves shrink the step and never decide a sign"""
    last = [u_rest, w, we]

    def g(d):
        u0 = xF(n, pi * sqrt(mpf(n)) / 2 + d)
        uu, ww, wwe, res2, k2 = solve_pair_held(n, u0, last[0], last[1], last[2])
        if res2 < TOLR and len(last[0]) > 0 and abs(uu[1] - last[0][0]) > JUMP:
            res2 = mpf(1)
        if res2 < TOLR:
            last[0], last[1], last[2] = uu[1:], ww, wwe
        Q, LPs, Lc = fullQ(n, uu, ww, wwe, None)
        lQ = [log(q) for q in Q]
        _, dp0 = D_Dp(n, uu[0], LPs[0], lQ)
        say(
            f"      [pair bracket] d = {mp.nstr(d, 5)}: g = D'(u0) = {mp.nstr(dp0, 4)}, solve residual {mp.nstr(res2, 2)} ({k2} steps)"
        )
        return dp0, (uu, ww, wwe, res2)

    a = d_lo
    ga, sa = g(a)
    tries = 0
    while (sa[-1] > TOLR or ga <= 0) and tries < 8:
        a = a / 2
        ga, sa = g(a)
        tries += 1
    if sa[-1] > TOLR or ga <= 0:
        return None
    b, gb = a, ga
    f = mpf("1.3")
    tries = 0
    found = False
    while tries < 30:
        c = b * f
        gc, sc = g(c)
        tries += 1
        if sc[-1] > TOLR:
            f = sqrt(f)
            continue
        if gc < 0:
            found = True
            break
        b, gb = c, gc
    if not found:
        return None
    d, gc, sc, it = illinois(g, b, c, gb, gc)
    return sc[0], sc[1], sc[2], d, max(abs(gc), sc[3]), it


def tF(n, x):
    return 2 * sqrt(mpf(n)) * asin(sqrt(x / n))


def xF(n, t):
    return mpf(n) * mp.sin(t / (2 * sqrt(mpf(n)))) ** 2


def analyse_float(n, u, w, we, wc):
    uf, wf, wcf = floats(u, w, we, wc)
    return fanalyse(n, uf, wf, wcf)  # C, vmax, tpk, ppk, curv, tf, Dc


# ---------------- start state
K = K0
n = n0
src = os.path.join(HERE, "results_q134_sym_bin_from93.json")
rec = None
rec135 = None
import glob

for fn in sorted(glob.glob(os.path.join(HERE, "results_q135_sym_bin_from*.json"))):
    for r in json.load(open(fn)).get("states", []):
        if r["n"] == n0 and r["K"] == K0 and "u" in r:
            rec135 = r
if rec135 is None and os.path.exists(src):
    for r in json.load(open(src)):
        if r["n"] == n0 and r["K"] == K0:
            rec = r
if rec is not None:
    t = np.concatenate([[0.0], np.cumsum(rec["gaps"])])
    x = n0 * np.sin(np.clip(t / (2 * np.sqrt(n0)), 0, np.pi / 2)) ** 2
    h = K0 // 2
    odd = K0 % 2 == 1
    uf = x[h + (1 if odd else 0) : -1]
    wf = np.ones(len(uf) + 1) / K0
    wcf = (1.0 / K0) if odd else None
    say(f"start from the q134 record at n = {n0}, K = {K0} (gaps -> positions; weights re-solved)")
else:
    uf, wf, wcf = fguess(n0, K0)
    say("start from the wall-gap guess")
d_last = None
n_split = None
if rec135 is not None:
    u = [mpf(v) for v in rec135["u"]]
    w = [mpf(v) for v in rec135["w"]]
    we = mpf(rec135["we"])
    wc = mpf(rec135["wc_str"]) if rec135.get("wc_str") else None
    u, w, we, wc, res, k = solve_full(n0, u, w, we, wc)
    if wc is None and rec135.get("pair_d") is not None:
        d_last = mpf(repr(rec135["pair_d"]))
        n_split = n0 - 2.0
    say(f"start from the q135 record at n = {n0}, K = {K0} (mp state re-polished)")
else:
    u, w, we, wc, res, k = solve_full(
        n0,
        to_mp(uf),
        to_mp(wf[:-1]),
        mpf(repr(float(wf[-1]))),
        (mpf(repr(float(wcf))) if wcf is not None else None),
    )
say(f"start n = {n}, K = {K}: residual {mp.nstr(res, 3)} in {k} mp steps   [{time.time()-t0:.0f}s]")
if res > TOLR:
    say("start not converged; stop")
    sys.exit(1)

# ---------------- continuation
out = []
transitions = []
prev_probe = None
while n < nmax:
    n1 = n + 1
    tt = [tF(n, x) for x in u]
    u1 = [xF(n1, t) for t in tt]
    young_pair = wc is None and d_last is not None and d_last < (mpf("0.3") if FLOAT_ONLY else mpf("0.6"))
    young_centre = wc is not None and wc < (mpf("0.01") if FLOAT_ONLY else mpf("0.02"))
    if young_centre:
        r = solve_centre_bracket(n1, u1, w, we, wc / 2)
        if r is None:
            say(f"n = {n1}: K = {K} centre bracket not found; stop")
            break
        u1, w1, we1, wc1, res, k = r
        how = f"centre-held bracket ({k} it)"
    elif young_pair:
        d_g = d_last * sqrt(mpf(n1 - n_split) / mpf(n - n_split))
        r = solve_pair_bracket(n1, u1[1:], w, we, d_g / 2)
        if r is None:
            say(f"n = {n1}: K = {K} pair bracket not found; stop")
            break
        u1, w1, we1, d1, res, k = r
        wc1 = None
        how = f"pair-held bracket ({k} it)"
    else:
        u1g = list(u1)
        mx = (u[0] - mpf(n) / 2) / 2 if (wc is None and len(u) > 0) else None
        u1, w1, we1, wc1, res, k = solve_full(n1, u1, w, we, wc, margin_x=mx)
        how = f"full ({k} mp steps)"
        if res > TOLR and wc is None and d_last is not None:
            r = solve_pair_bracket(n1, u1g[1:], w, we, d_last * mpf("0.7"))
            if r is not None:
                u1, w1, we1, d1, res, k = r
                wc1 = None
                how = f"pair-held fallback ({k} it)"
    if res > TOLR:
        say(f"n = {n1}: K = {K} solve failed (residual {mp.nstr(res, 3)}, {how}); stop")
        break
    if wc1 is None and d_last is not None:
        d_last = tF(n1, u1[0]) - pi * sqrt(mpf(n1)) / 2
    Dc, curv, dp0 = probe(n1, u1, w1, we1, wc1)
    born = ""
    # transition of THIS K-point: zero crossing of the centre violation (even K) or the centre curvature (odd K)
    sig = Dc if wc1 is None else curv
    if prev_probe is not None and prev_probe[0] == K and prev_probe[1] < 0 < sig:
        n_star = n + float(-prev_probe[1] / (sig - prev_probe[1]))
        transitions.append(
            {
                "K_new": K + 1,
                "type": "insertion" if wc1 is None else "split",
                "n_star": n_star,
                "sig_prev": float(prev_probe[1]),
                "sig_now": float(sig),
            }
        )
        say(
            f"   >>> transition to K = {K + 1} ({transitions[-1]['type']}) at n* = {n_star:.3f} (signal {float(prev_probe[1]):+.2e} at {n} -> {float(sig):+.2e} at {n1})"
        )
    prev_probe = (K, sig)
    if wc1 is None and Dc > 0:  # insertion of a centre atom
        r = solve_centre_bracket(n1, u1, w1, we1, mpf("1e-3"))
        if r is not None and r[4] < TOLR:
            u1, w1, we1, wc1 = r[0], r[1], r[2], r[3]
            K += 1
            d_last = None
            born = f" INSERTION (D(1/2)-C was {mp.nstr(Dc, 3)}); newborn weight {mp.nstr(wc1, 5)}"
            prev_probe = None
        else:
            born = " centre insertion failed"
    elif wc1 is not None and curv > 0:  # split of the centre atom into a mirror pair
        r = solve_pair_bracket(n1, u1, [wc1 / 2] + list(w1), we1, mpf("0.02"))
        if r is not None and r[4] < TOLR:
            u1, w1, we1 = r[0], r[1], r[2]
            wc1 = None
            K += 1
            d_last = r[3]
            n_split = transitions[-1]["n_star"] if transitions and transitions[-1]["K_new"] == K else n1 - 0.5
            born = f" SPLIT (curv was {mp.nstr(curv, 3)}); pair half-separation {mp.nstr(d_last, 5)}"
            prev_probe = None
        else:
            born = " split failed"
    C, vmax, tpk, ppk, curv_f, tf, Dc_f = analyse_float(n1, u1, w1, we1, wc1)
    Dc, curv, dp0 = probe(n1, u1, w1, we1, wc1)
    u, w, we, wc, n = u1, w1, we1, wc1, n1
    wmin = float(min(min(w), we, wc if wc is not None else 1))
    say(
        f"n = {n}: K = {K} [{how}], pair d = {(float(d_last) if (wc is None and d_last is not None) else float('nan')):.4f}, C = {C:.6f}, min w {wmin:.2e}, wc {(float(wc) if wc is not None else float('nan')):.4e}, centre D-C {mp.nstr(Dc, 3)}, curv {mp.nstr(curv, 3)}, off-atom max D-C {vmax:+.2e} at t = {tpk:.3f}{born}   [{time.time()-t0:.0f}s]"
    )
    out.append(
        {
            "n": n,
            "K": K,
            "C": float(C),
            "w_min": wmin,
            "wc": (float(wc) if wc is not None else None),
            "pair_d": (float(d_last) if (wc is None and d_last is not None) else None),
            "centre_DC": float(Dc),
            "curv": float(curv),
            "off_atom_max": float(vmax),
            "gaps": np.round(np.diff(tf), 4).tolist(),
            "born": born,
            "u": [mp.nstr(t, 25) for t in u],
            "w": [mp.nstr(t, 25) for t in w],
            "we": mp.nstr(we, 25),
            "wc_str": (mp.nstr(wc, 25) if wc is not None else None),
        }
    )
    json.dump(
        {"states": out, "transitions": transitions},
        open(os.path.join(HERE, f"results_q135_sym_bin_from{n0}.json"), "w"),
        indent=1,
    )
say(f"done [{time.time()-t0:.0f}s]; transitions: {transitions}")
