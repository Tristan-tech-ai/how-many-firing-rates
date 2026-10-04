"""
Q84: the Gaussian birth chain in the SYMMETRIC parametrisation (right half only), which removes the
antisymmetric zero mode that stalls the full-system Newton at a centre birth (pitchfork). Float64, central
difference Jacobian, feasibility-scaled full Newton step.
Odd K: centre atom (weight wc) + m = (K-1)/2 mirrored atoms at +u_j (weights w_j), u_m = A_g.
Even K: m = K/2 mirrored atoms at +u_j, u_m = A_g.
Fixed-A_g unknowns: odd: wc, w_1..w_m, u_1..u_{m-1}; even: w_1..w_m, u_1..u_{m-1}.
Equations: D(u_j) = D(u_ref) for all atoms but the reference, total weight = 1, D'(u_j) = 0 for the interior
mirrored atoms (the centre atom has D' = 0 by symmetry).
Births: even K -> odd K+1 by a centre atom (weight held at eps, A_g free); odd K -> even K+2 by a mirror
pair at +-u_peak (their common weight held at eps). A_g(next) by extrapolation of A_g(eps) to 0.
Usage: py q84_gauss_symmetric_chain.py <K_start> <K_end>
"""

import os, sys, json, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import q33_gauss_support as qg

say = lambda *a: print(*a, flush=True)
t0 = time.time()
K0, K1 = int(sys.argv[1]), int(sys.argv[2])
Q = {}
for fn in ["results_q55_gauss18.json", "results_q67_gauss22.json", "results_q77_gauss_growth.json"] + sorted(
    glob.glob(os.path.join(HERE, "results_q84_gauss_chain_*.json"))
):
    try:
        Q.update(json.load(open(os.path.join(HERE, fn))))
    except Exception:
        pass


def full_from_half(K, wc, w, u, Ag):
    """u: interior right positions u_1..u_{m-1}; returns full sorted arrays."""
    ur = np.concatenate([u, [Ag]])
    wr = w
    if K % 2 == 1:
        uu = np.concatenate([-ur[::-1], [0.0], ur])
        ww = np.concatenate([wr[::-1], [wc], wr])
    else:
        uu = np.concatenate([-ur[::-1], ur])
        ww = np.concatenate([wr[::-1], wr])
    return uu, ww


def newton(F, v0, feas, steps=200, tol=1e-12):
    v = np.array(v0, float)
    hist = []
    for it in range(steps):
        r = F(v)
        nr = np.abs(r).max()
        hist.append(nr)
        if nr < tol:
            return v, nr, hist
        n = len(v)
        J = np.empty((n, n))
        for j in range(n):
            h = 1e-7 * max(1.0, abs(v[j]))
            vp = v.copy()
            vp[j] += h
            vm = v.copy()
            vm[j] -= h
            J[:, j] = (F(vp) - F(vm)) / (2 * h)
        try:
            dv = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return v, nr, hist
        v = v - feas(v, dv) * dv
    return v, np.abs(F(v)).max(), hist


def residual_full(K, wc, w, u, Ag, y, dy):
    uu, ww = full_from_half(K, wc, w, u, Ag)
    Dv, Dp = qg.D_and_Dp(uu, uu, ww, y, dy)
    m = len(w)
    c0 = (K - 1) // 2 if K % 2 == 1 else None
    right = uu[len(uu) - m :]
    Dr = Dv[len(uu) - m :]
    Dpr = Dp[len(uu) - m :]
    if K % 2 == 1:
        eqD = Dr - Dv[c0]  # m equations: D(u_j) = D(0)
        mass = wc + 2 * w.sum() - 1
    else:
        eqD = Dr[:-1] - Dr[-1]  # m-1 equations
        mass = 2 * w.sum() - 1
    eqDp = Dpr[:-1]  # interior mirrored atoms
    return np.concatenate([eqD, [mass], eqDp])


def solve_fixed(K, Ag, wc, w, u):
    y, dy = qg.grid(Ag)
    m = len(w)

    def unpack(v):
        if K % 2 == 1:
            return v[0], v[1 : 1 + m], v[1 + m :]
        return None, v[:m], v[m:]

    def F(v):
        wc_, w_, u_ = unpack(v)
        return residual_full(K, wc_, w_, u_, Ag, y, dy)

    def feas(v, dv):
        wc_, w_, u_ = unpack(v)
        dwc, dw, du = unpack(dv)
        a = 1.0
        ws_ = np.concatenate([[wc_], w_]) if K % 2 == 1 else w_
        dws = np.concatenate([[dwc], dw]) if K % 2 == 1 else dw
        for k in range(len(ws_)):
            if dws[k] > 0:
                a = min(a, 0.9 * ws_[k] / dws[k])
        pos = np.concatenate([[0.0], u_, [Ag]])
        dpos = np.concatenate([[0.0], du, [0.0]])
        for i in range(len(pos) - 1):
            gap = pos[i + 1] - pos[i]
            dg = dpos[i + 1] - dpos[i]
            if dg > 0:
                a = min(a, 0.8 * gap / dg)
        return a

    v0 = np.concatenate([[wc], w, u]) if K % 2 == 1 else np.concatenate([w, u])
    v, res, hist = newton(F, v0, feas)
    wc_, w_, u_ = unpack(v)
    return wc_, w_, u_, res, len(hist)


def unfold(K, Ag0, wc, w, u, eps, held, hold_centre, trace=False):
    """Generic: right-half weights in `held` (indices into w) fixed at eps; the centre weight fixed at eps if
    hold_centre (odd K only); A_g free. Unknown vector: [wc if odd and not held] + free weights + u + [Ag]."""
    y, dy = qg.grid(Ag0 + 1.5)
    m = len(w)
    free = [k for k in range(m) if k not in held]
    has_wc = (K % 2 == 1) and (not hold_centre)

    def unpack(v):
        i = 0
        wc_ = None
        if K % 2 == 1:
            if hold_centre:
                wc_ = eps
            else:
                wc_ = v[0]
                i = 1
        w_ = np.empty(m)
        w_[free] = v[i : i + len(free)]
        w_[held] = eps
        u_ = v[i + len(free) : -1]
        Ag = v[-1]
        return wc_, w_, u_, Ag

    def F(v):
        wc_, w_, u_, Ag = unpack(v)
        return residual_full(K, wc_, w_, u_, Ag, y, dy)

    def feas(v, dv):
        a = 1.0
        nw = (1 if has_wc else 0) + len(free)
        for k in range(nw):
            if dv[k] > 0:
                a = min(a, 0.9 * v[k] / dv[k])
        wc_, w_, u_, Ag = unpack(v)
        pos = np.concatenate([[0.0], u_, [Ag]])
        dpos = np.concatenate([[0.0], dv[nw:-1], [dv[-1]]])
        for i in range(len(pos) - 1):
            gap = pos[i + 1] - pos[i]
            dg = dpos[i + 1] - dpos[i]
            if dg > 0:
                a = min(a, 0.8 * gap / dg)
        return a

    v0 = ([wc] if has_wc else []) + list(np.asarray(w)[free]) + list(u) + [Ag0]
    v, res, hist = newton(F, np.array(v0, float), feas)
    if trace:
        say("      hist: " + " ".join(f"{h:.1e}" for h in hist[:12]) + (" ..." if len(hist) > 12 else ""))
    wc_, w_, u_, Ag = unpack(v)
    return Ag, wc_, w_, u_, res, len(hist)


def violation(K, Ag, wc, w, u):
    uu, ww = full_from_half(K, wc, w, u, Ag)
    y, dy = qg.grid(Ag)
    uq = np.linspace(0, Ag, 20001)
    Dat, _ = qg.D_and_Dp(uu, uu, ww, y, dy)
    C = float(ww @ Dat)
    Dq, _ = qg.D_and_Dp(uq, uu, ww, y, dy)
    v = Dq - C
    i = v.argmax()
    return float(v[i]), float(uq[i])


def probe_next(K, wc, w, u, Ag):
    found = None
    h = 0.01
    A_start_probe = Ag
    up = None
    for _ in range(600):
        Ag_probe = round(Ag + h, 6)
        wc_p, w_p, u_p, res_p, _n = solve_fixed(K, Ag_probe, wc, w, u * (Ag_probe / Ag))
        if res_p > 1e-10:
            h = h / 2
            if h < 0.004:
                say(f"   probe at {Ag_probe:.3f} failed even at step {h:.4f} (residual {res_p:.0e})")
                break
            continue
        h = min(h * 1.5, 0.01 if Ag_probe < A_start_probe + 0.25 else 0.02)
        vmax_p, up = violation(K, Ag_probe, wc_p, w_p, u_p)
        wc, w, u, Ag = wc_p, w_p, u_p, Ag_probe
        if _ % 10 == 9:
            say(
                f"      probe A_g = {Ag_probe:.2f}: viol {vmax_p:+.1e} at u = {up:.3f}; min w {min(w_p.min(), wc_p if wc_p is not None else 1):.4f}"
            )
        if vmax_p > 1e-11:
            found = Ag_probe
            break
    return found, h, up, wc, w, u, Ag


# start: the K0 state from file, symmetrised
if f"gauss_K{K0}_us" in Q:
    uu0 = np.array(Q[f"gauss_K{K0}_us"])
    ww0 = np.array(Q[f"gauss_K{K0}_ws"])
    Ag = float(uu0[-1])
else:
    G = json.load(open(os.path.join(HERE, "results_q33_gauss_support.json")))["history"]
    rows = [r for r in G if r["clean"] and r["K"] == K0]
    b = max(rows, key=lambda r: r["Ag"])
    uu0 = np.array(b["us"])
    ww0 = np.array(b["ws"])
    Ag = float(uu0[-1])
    Q[f"Ag_{K0+1}"] = Ag + 0.02
    say(f"start from the q33 history: K = {K0} at A_g = {Ag:.3f}")
K = K0
m = K // 2
right_u = uu0[len(uu0) - m :]
right_w = ww0[len(ww0) - m :]
wc = float(ww0[(K - 1) // 2]) if K % 2 == 1 else None
w = np.array(right_w)
u = np.array(right_u[:-1])
wc, w, u, res, it = solve_fixed(K, Ag, wc, w, u)
say(f"start K = {K} at A_g = {Ag:.4f}: symmetric residual {res:.0e} ({it} steps)")
out = {}
if f"Ag_{K0+1}" in Q:
    Ag_next = float(Q[f"Ag_{K0+1}"])
    A_bracket = (Ag_next - 0.02, Ag_next)
else:
    found, h, up, wc, w, u, Ag = probe_next(K, wc, w, u, Ag)
    if found is None:
        say("no first transition found by probing; stop")
        sys.exit(0)
    say(f"first transition just below {found:.3f} (peak at u = {up:+.3f})")
    Ag_next = found
    A_bracket = (found - h, found)
while K < K1:
    A_try = Ag_next + 0.007
    while Ag < A_try - 1e-9:
        A_step = min(Ag + 0.1, A_try)
        wc, w, u, res, it = solve_fixed(K, A_step, wc, w, u * (A_step / Ag))
        Ag = A_step
        if res > 1e-10:
            break
    vmax, u_ins = violation(K, A_try, wc, w, u)
    say(
        f"K = {K} at A_g = {A_try:.4f}: residual {res:.0e} ({it} steps); max(D - C) = {vmax:+.2e} at u = {u_ins:+.4f}   [{time.time()-t0:.0f}s]"
    )
    if res > 1e-10 or vmax <= 0:
        say("   no violating point; stop")
        break
    if u_ins < 0.05:
        if K % 2 == 1:
            say("   centre already occupied but peak at centre; stop")
            break
        Knew = K + 1
        wc2 = 1e-3
        w2 = w * ((1 - 1e-3) / (2 * w.sum()))
        u2 = u.copy()
        held = []
        hold_centre = True
        ins = 0.0
    else:
        Knew = K + 2
        ins = u_ins
        if K % 2 == 1:
            ins = 0.05  # odd K: the pair is born next to the centre (a split); start there
        pos_all = np.concatenate([u, [A_try]])
        idx = int(np.searchsorted(pos_all, ins))
        u_full = np.insert(pos_all, idx, ins)
        w_full = np.insert(w, idx, 1e-3)
        u2 = u_full[:-1]
        w2 = w_full
        wc2 = (wc - 2e-3) if (K % 2 == 1) else wc
        held = [idx]
        hold_centre = False
    unf = lambda Ag0, wc_, w_, u_, eps, trace=False: unfold(
        Knew, Ag0, wc_, w_, u_, eps, held, hold_centre, trace
    )
    say(f"   inserting {'centre atom' if hold_centre else 'mirror pair at +-%.4f' % ins} -> K = {Knew}")
    A_eps = {}
    Ag_cur, wc_cur, w_cur, u_cur = A_try, wc2, w2, u2
    for eps in (1e-3, 5e-4, 2.5e-4):
        Ag_e, wc_e, w_e, u_e, r_e, it_e = unf(Ag_cur, wc_cur, w_cur, u_cur, eps, trace=True)
        say(f"   eps = {eps}: residual {r_e:.0e} ({it_e} steps), A_g = {Ag_e:.6f}")
        if r_e > 1e-10 and eps == 1e-3:
            say("   retrying the first rung at eps = 2.5e-4")
            Ag_e, wc_e, w_e, u_e, r_e, it_e = unf(Ag_cur, wc_cur, w_cur, u_cur, 2.5e-4, trace=True)
            say(f"   eps = 0.00025 (retry): residual {r_e:.0e} ({it_e} steps), A_g = {Ag_e:.6f}")
            if r_e <= 1e-10:
                A_eps[2.5e-4] = Ag_e
                Ag_cur, wc_cur, w_cur, u_cur = Ag_e, wc_e, w_e, u_e
                Ag_e, wc_e, w_e, u_e, r_e, it_e = unf(Ag_cur, wc_cur, w_cur, u_cur, 5e-4, trace=True)
                say(f"   eps = 0.0005 (after retry): residual {r_e:.0e} ({it_e} steps), A_g = {Ag_e:.6f}")
                if r_e <= 1e-10:
                    A_eps[5e-4] = Ag_e
                    Ag_cur, wc_cur, w_cur, u_cur = Ag_e, wc_e, w_e, u_e
                    S1e3 = (Ag_e, wc_e, w_e, u_e)
            break
        if r_e > 1e-10:
            break
        A_eps[eps] = Ag_e
        Ag_cur, wc_cur, w_cur, u_cur = Ag_e, wc_e, w_e, u_e
        if eps == 1e-3:
            S1e3 = (Ag_e, wc_e, w_e, u_e)
    if len(A_eps) < 2:
        say("   unfolding failed; stop")
        break
    es = np.array(list(A_eps))
    As = np.array([A_eps[k] for k in es])
    cf = np.polyfit(es, As, 1)
    say(
        f"   A_g({Knew}) by unfolding = {cf[1]:.6f}; dA/deps = {cf[0]:.3f}; newborn position after the ladder = {(0.0 if hold_centre else u_cur[held[0]]):.4f}"
    )
    out[f"Ag_{Knew}_unfold"] = float(cf[1])
    try:
        lo_b, hi_b = A_bracket
        if not (lo_b - 0.02 <= cf[1] <= hi_b + 0.02):
            say(
                f"   WARNING: unfolding transition {cf[1]:.5f} outside the probe bracket [{lo_b:.4f}, {hi_b:.4f}]: spurious branch; using the bracket midpoint"
            )
            out[f"Ag_{Knew}_unfold"] = float(0.5 * (lo_b + hi_b))
            out[f"Ag_{Knew}_spurious_unfold"] = float(cf[1])
    except NameError:
        pass
    # climb from the eps = 1e-3 state (not from the lowest rung)
    Ag_cur, wc_cur, w_cur, u_cur = S1e3
    if hold_centre:
        # centre birth: leave the eps ladder at 1e-3 and walk the (K+1)-state at fixed A_g in small steps
        Ag_w = Ag_cur
        wc_w, w_w, u_w = wc_cur, w_cur, u_cur
        for dA in [0.01] * 10 + [0.05] * 6:
            A_n = round(Ag_w + dA, 6)
            wc_n, w_n, u_n, r_n, it_n = solve_fixed(Knew, A_n, wc_w, w_w, u_w * (A_n / Ag_w))
            if r_n > 1e-10 or wc_n < 5e-4:
                say(
                    f"   fixed walk after the centre birth stopped at {A_n:.4f} (residual {r_n:.0e}, centre w {wc_n:.5f})"
                )
                break
            Ag_w, wc_w, w_w, u_w = A_n, wc_n, w_n, u_n
        say(f"   centre weight {wc_w:.5f} at A_g = {Ag_w:.4f} after the fixed walk")
        Ag_cur, wc_cur, w_cur, u_cur = Ag_w, wc_w, w_w, u_w
    else:
        eps = 1e-3
        while eps < 3.2e-2:
            eps = min(eps * 1.25, 3.2e-2)
            Ag_e, wc_e, w_e, u_e, r_e, it_e = unf(Ag_cur, wc_cur, w_cur, u_cur, eps)
            if r_e > 1e-10:
                say(f"   climb stopped at eps = {eps} (residual {r_e:.0e} after {it_e} steps)")
                break
            Ag_cur, wc_cur, w_cur, u_cur = Ag_e, wc_e, w_e, u_e
    if (not hold_centre) and Knew % 2 == 1 and u_cur[held[0]] < 0.15:
        # a SPLIT of the centre atom: hand the mass over until the centre is nearly empty, then drop it
        say(f"   split of the centre atom detected (pair at u = {u_cur[held[0]]:.4f}); handing over the mass")
        eps = 1e-3
        while True:
            total = wc_cur + 2 * eps
            eps_try = min(eps * 1.25, 0.5 * total - 2e-4)
            if eps_try <= eps:
                break
            Ag_e, wc_e, w_e, u_e, r_e, it_e = unf(Ag_cur, wc_cur, w_cur, u_cur, eps_try)
            if r_e > 1e-10:
                say(f"   hand-over stopped at eps = {eps_try:.5f} (residual {r_e:.0e})")
                break
            eps = eps_try
            Ag_cur, wc_cur, w_cur, u_cur = Ag_e, wc_e, w_e, u_e
        say(
            f"   centre weight {wc_cur:.5f} at pair weight {eps:.5f}, A_g = {Ag_cur:.6f}; dropping the centre -> K = {Knew - 1}"
        )
        A_split = Ag_cur
        Knew = Knew - 1
        w_cur = np.array(w_cur, float)
        w_cur[held[0]] = eps + wc_cur / 2
        wc_cur = None
        out[f"Ag_{Knew}_unfold"] = float(A_split)
        # the separated branch by the PAIR-WEIGHT ladder (A_g free), the natural continuation for a light pair
        eps_p = float(w_cur[held[0]])
        unf2 = lambda Ag0, w_, u_, e: unfold(Knew, Ag0, None, w_, u_, e, held, False)
        ok_ladder = False
        while eps_p < 0.025:
            e_try = min(eps_p * 1.25, 0.025)
            Ag_e, wc_e, w_e, u_e, r_e, it_e = unf2(Ag_cur, w_cur, u_cur, e_try)
            if r_e > 1e-10:
                say(f"   pair ladder stopped at eps = {e_try:.5f} (residual {r_e:.0e} after {it_e} steps)")
                break
            eps_p = e_try
            Ag_cur, w_cur, u_cur = Ag_e, w_e, u_e
            ok_ladder = True
        say(
            f"   pair at u = {u_cur[held[0]]:.4f} with weight {eps_p:.5f} at A_g = {Ag_cur:.6f} after the pair ladder"
        )
        if not ok_ladder:
            say("   no separated branch found; stop")
            break
    wc_f, w_f, u_f, res_f, it_f = solve_fixed(Knew, Ag_cur, wc_cur, w_cur, u_cur)
    uu_f, ww_f = full_from_half(Knew, wc_f, w_f, u_f, Ag_cur)
    say(
        f"   K = {Knew} state at A_g = {Ag_cur:.4f}: residual {res_f:.0e} ({it_f} steps), min w {ww_f.min():.4f}   [{time.time()-t0:.0f}s]"
    )
    if res_f > 1e-10:
        say("   fixed solve failed; stop")
        break
    out[f"gauss_K{Knew}_us"] = uu_f.tolist()
    out[f"gauss_K{Knew}_ws"] = ww_f.tolist()
    out[f"gauss_K{Knew}_Ag"] = float(Ag_cur)
    json.dump(out, open(os.path.join(HERE, f"results_q84_gauss_chain_{K0}_{K1}.json"), "w"), indent=1)
    K, wc, w, u, Ag = Knew, wc_f, w_f, u_f, Ag_cur
    found, h, up, wc, w, u, Ag = probe_next(K, wc, w, u, Ag)
    if found is None:
        say("   no next transition found by probing; stop")
        break
    say(f"   next transition just below {found:.3f} (peak at u = {up:+.3f})")
    Ag_next = found
    A_bracket = (found - h, found)
say(f"done [{time.time()-t0:.0f}s]")
