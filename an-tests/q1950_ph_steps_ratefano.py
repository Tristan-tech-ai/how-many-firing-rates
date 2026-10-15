"""q1950 (15 Oct 2026): task P2WEAKAUDIT step 2, H_stepsR, registered in Amendment 2294 before any K of this model was computed on data.

Why. q1949 (Reading 551) showed that within each fibre the Fano factor falls with the mean count (slope -0.0356 per spike), about ten
times faster than the matched Conway-Maxwell-Poisson count of q1939 with one zeta per fibre (-0.0038), so that count is too regular at
low counts and too noisy at high counts. This is the third dispersion model of the discreteness test; the earlier results (Poisson
0.517, binomial 0.883, matched 0.823) stay as reported.
Channel. Per fibre (q1926 selection and counts, 0-50 ms), the least-squares line Fano(m) = a + b m through the levels with mean count
>= 1 (if fewer than two such levels: the flat line at the fibre's median Fano of q1936). The target Fano at rate r is a + b r clamped
to [0.05, 1]. At rate r > 0 the count is Conway-Maxwell-Poisson in mean form with zeta(r) in [1, 50] solving Fano = target (zeta 1
where the target is 1; zeta 50 where even zeta 50 leaves the Fano above the target, counted as capped); zeta(r) is solved on 41 rates
and interpolated linearly to the grid. Rate 0 gives the count 0.
K: cutting plane of q1939 on 801 rates in [lam, lam + A], tolerance 1e-8, mass points of weight > 1e-6, merged when closer than 0.2 in
2 sqrt(r). D: q1926, unchanged.
VERDICT (H_stepsR, registered): PASS if D <= K in at least 80 percent of the fibres (bar unchanged); FAIL otherwise; VOID below 20 fibres.
Readings: fraction by thirds of the median Fano, by A >= 8 and A < 8, distribution of K - K_matched (q1939), share of capped grid rates,
quartiles of a and b.
Usage: py q1950_ph_steps_ratefano.py [--registration-run]
"""
import json
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support_exact import _solve_w  # noqa: E402
from q1939_ph_steps_compoisson import k_cmp  # noqa: E402

CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
R26 = os.path.join(HERE, "paper_neuron", "q1926_rows.json")
R36 = os.path.join(HERE, "paper_neuron", "q1936_rows.json")
R39 = os.path.join(HERE, "paper_neuron", "q1939_rows.json")
OUT = os.path.join(HERE, "paper_neuron", "q1950_rows.json")
ZCAP = 50.0
FLO, FHI = 0.05, 1.0
TAIL = 1e-12


def cmp_rows(rs, zs):
    """Rows of the mean-form COM-Poisson pmf, rate rs[i] with exponent zs[i]; rate 0 gives the count 0."""
    rs = np.asarray(rs, float); zs = np.asarray(zs, float)
    rmax = float(rs.max()); ymax = int(np.ceil(rmax + 12 * np.sqrt(rmax + 1) + 30))
    while True:
        y = np.arange(ymax + 1); lf = gammaln(y + 1)
        lo = np.full(len(rs), -400.0); hi = np.full(len(rs), 400.0)
        for _ in range(110):
            t = 0.5 * (lo + hi); lw = t[:, None] * y[None, :] - zs[:, None] * lf[None, :]
            lw -= lw.max(1, keepdims=True); p = np.exp(lw); p /= p.sum(1, keepdims=True)
            big = p @ y > rs; hi = np.where(big, t, hi); lo = np.where(big, lo, t)
        t = 0.5 * (lo + hi); lw = t[:, None] * y[None, :] - zs[:, None] * lf[None, :]
        lw -= lw.max(1, keepdims=True); p = np.exp(lw); p /= p.sum(1, keepdims=True)
        p[rs == 0] = 0.0; p[rs == 0, 0] = 1.0
        if p[:, -1].max() < TAIL:
            break
        ymax *= 2
    pos = rs > 0
    assert np.max(np.abs(p[pos] @ y - rs[pos])) < 1e-8, "mean not matched"
    return p


def fano_rows(rs, zs):
    p = cmp_rows(rs, zs); y = np.arange(p.shape[1]); mu = p @ y
    return (p @ y ** 2 - mu ** 2) / mu


def zeta_for(rs, target):
    """Vectorized bisection in log zeta on [1, ZCAP]; returns zeta and the capped mask."""
    rs = np.asarray(rs, float); target = np.asarray(target, float)
    z = np.ones(len(rs)); capped = np.zeros(len(rs), bool)
    act = target < 1.0 - 1e-12
    if act.any():
        fcap = fano_rows(rs[act], np.full(act.sum(), ZCAP))
        cap_here = fcap > target[act]
        lo = np.zeros(act.sum()); hi = np.full(act.sum(), np.log(ZCAP))
        for _ in range(45):
            mid = 0.5 * (lo + hi); f = fano_rows(rs[act], np.exp(mid))
            above = f > target[act]; lo = np.where(above, mid, lo); hi = np.where(above, hi, mid)
        zz = np.exp(0.5 * (lo + hi)); zz[cap_here] = ZCAP
        z[act] = zz; capped[np.where(act)[0][cap_here]] = True
    return z, capped


def line_for(counts, Fmed):
    pts = [(c.mean(), c.var(ddof=1) / c.mean()) for c in counts if c.mean() >= 1]
    if len(pts) < 2:
        return float(Fmed), 0.0, True
    m = np.array([p[0] for p in pts]); f = np.array([p[1] for p in pts])
    b, a = np.polyfit(m, f, 1)
    return float(a), float(b), False


def channel(A, lam, a, b, ngrid=801):
    rs = np.linspace(lam, lam + A, ngrid)
    coarse = np.linspace(max(lam, 1e-3), lam + A, 41)
    tc = np.clip(a + b * coarse, FLO, FHI)
    zc, capc = zeta_for(coarse, tc)
    zs = np.interp(rs, coarse, zc)
    return rs, cmp_rows(rs, zs), float(np.mean(capc))


def k_from_rows(rs, P, tol=1e-8, maxit=40):
    logP = np.log(np.where(P > 0, P, 1e-300)); Hx = (P * logP).sum(1)
    S = [0, len(rs) - 1]
    for _ in range(maxit):
        w, C = _solve_w(P[S], Hx[S]); Q = w @ P[S]; lq = np.log(np.where(Q > 0, Q, 1e-300)); D = Hx - P @ lq
        k = int(D.argmax())
        if D[k] - C <= tol or k in S:
            break
        S.append(k)
    pts = np.sort([rs[i] for i, wi in zip(S, w) if wi > 1e-6]); t = 2 * np.sqrt(pts)
    return 1 + int(np.sum(np.diff(t) >= 0.2))


def k_rate(A, lam, a, b):
    rs, P, capshare = channel(A, lam, a, b)
    return k_from_rows(rs, P), capshare


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean([r["D"] <= r["K"] for r in rows]))
    return ("PASS" if f >= 0.8 else "FAIL"), "D <= K in %.3f of %d fibres (bar 0.80)" % (f, len(rows))


def fibre_lines():
    from q1926_ph_steps import fibre
    r26 = {r["path"]: r for r in json.load(open(R26))}; r36 = {r["path"]: r for r in json.load(open(R36))}
    out = []
    for rec in json.load(open(CENSUS)):
        if "error" in rec or rec.get("path") not in r26:
            continue
        f = fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        a, b, flat = line_for(counts, r36[rec["path"]]["median_fano"])
        out.append((rec["path"], levels, counts, lam, A, a, b, flat))
    return out, r26, r36


def registration_run():
    ok = True
    # 1. the channel with a constant zeta reproduces the q1939 K
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        for z0 in (1.0, 3.0):
            rs = np.linspace(lam, lam + A, 801); kk = k_from_rows(rs, cmp_rows(rs, np.full(801, z0))); k39 = k_cmp(A, lam, z0)
            print("control: A %.0f lam %.1f zeta %.0f: K from rows %d, q1939 K %d" % (A, lam, z0, kk, k39)); ok &= kk == k39
    # 2. zeta_for hits the target Fano
    rs = np.array([1.0, 3.0, 6.0, 12.0]); tg = np.array([0.6, 0.4, 0.3, 0.2]); z, cap = zeta_for(rs, tg); fz = fano_rows(rs, z)
    print("control: zeta_for targets %s -> Fano %s, capped %s" % (tg, np.round(fz, 6), cap)); ok &= bool(np.all(np.abs(fz - tg) < 1e-6)) and not cap.any()
    # 3. a falling Fano line gives at least the K of the flat line through its value at the mean (direction reading, not a criterion)
    # 4. graded control at the median line of the fibres (Fano statistics only; no K or D of data is read)
    lines, _, _ = fibre_lines(); a_med = float(np.median([l[5] for l in lines])); b_med = float(np.median([l[6] for l in lines]))
    print("control: median line of the %d fibres: a %.4f, b %.4f" % (len(lines), a_med, b_med))
    from q1926_ph_steps import count_distinct
    rng = np.random.default_rng(2294); LEV = list(np.arange(0, 70, 10.0))
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        K, capshare = k_rate(A, lam, a_med, b_med)
        vals = np.linspace(lam, lam + A, K); idx = np.minimum((np.arange(7) * K) // 7, K - 1); stair = vals[idx]
        graded = np.linspace(lam, lam + A, 7); res = {}
        for name, rates in (("staircase", stair), ("graded", graded)):
            zz, _ = zeta_for(np.maximum(rates, 1e-3), np.clip(a_med + b_med * rates, FLO, FHI))
            P = cmp_rows(rates, zz); y = np.arange(P.shape[1])
            res[name] = float(np.mean([count_distinct(LEV, [rng.choice(y, size=40, p=row / row.sum()).astype(float) for row in P]) <= K
                                       for _ in range(200)]))
        print("control: A %.0f lam %.1f, K (median line) %d, capped share %.3f: D <= K in %.3f of staircase and %.3f of graded fibres"
              % (A, lam, K, capshare, res["staircase"], res["graded"]))
        ok &= res["staircase"] >= 0.80 and res["graded"] <= 0.30
    rows = [{"D": 2, "K": 3}] * 25; bad = [{"D": 4, "K": 3}] * 25
    d = (decide(rows)[0], decide(bad)[0], decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1950: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1950 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (CENSUS, R26, R36, R39)):
        print("VERDICT q1950: INCOMPLETE (input missing)"); return 3
    lines, r26, r36 = fibre_lines(); r39 = {r["path"]: r for r in json.load(open(R39))}
    rows = []
    for path, levels, counts, lam, A, a, b, flat in lines:
        K, capshare = k_rate(A, lam, a, b)
        rows.append({"path": path, "D": r26[path]["D"], "K": K, "K_matched": r39[path]["K"], "K_poisson": r26[path]["K"], "A": A,
                     "lam": lam, "a": a, "b": b, "flat": flat, "capshare": capshare, "F": r36[path]["median_fano"]})
        json.dump(rows, open(OUT, "w"))
    if rows:
        F = np.array([r["F"] for r in rows]); q = np.quantile(F, [1 / 3, 2 / 3])
        for name, m in (("most regular", F <= q[0]), ("middle", (F > q[0]) & (F <= q[1])), ("least regular", F > q[1])):
            sub = [r for r, mm in zip(rows, m) if mm]
            print("Reading: %s third, fibres %d, D <= K in %.3f" % (name, len(sub), float(np.mean([r["D"] <= r["K"] for r in sub]))))
        for name, sub in (("A >= 8", [r for r in rows if r["A"] >= 8]), ("A < 8", [r for r in rows if r["A"] < 8])):
            print("Reading: %s, fibres %d, D <= K in %.3f" % (name, len(sub), float(np.mean([r["D"] <= r["K"] for r in sub]))))
        dk = np.array([r["K"] - r["K_matched"] for r in rows])
        print("Reading: K - K_matched values %s" % {int(v): int(np.sum(dk == v)) for v in np.unique(dk)})
        print("Reading: mean capped share of the 41 rates %.3f; flat lines %d; quartiles of a %s, of b %s" % (
            float(np.mean([r["capshare"] for r in rows])), sum(r["flat"] for r in rows),
            np.round(np.quantile([r["a"] for r in rows], [0.25, 0.5, 0.75]), 3), np.round(np.quantile([r["b"] for r in rows], [0.25, 0.5, 0.75]), 4)))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1950 (Amendment 2294, H_stepsR): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
