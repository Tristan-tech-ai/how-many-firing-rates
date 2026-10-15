"""q1939 (15 Oct 2026): task P2FANOMATCH, to be registered in Amendment 2290 before any K of a recorded fibre is computed.

Question. Amendment 2289 found that the binomial channel of H_stepsF (q1937) is the most regular binomial count with the fibre's range
and is not matched to each fibre's dispersion (more regular than the fibre in 158 fibres, less regular in 102). Does D <= K hold when K
is computed for a count whose Fano factor at the fibre's mean count equals the fibre's measured Fano factor?
Channel. Conway-Maxwell-Poisson count in mean form: at rate r the count has P(y) proportional to exp(t y) / (y!)^nu, with t chosen so
that the mean is exactly r (r = 0 gives the count 0). nu = 1 is the Poisson count; nu > 1 is less variable. For each fibre nu is the
root in [1, NU_CAP] of Fano(nu; mbar) = F, where mbar is the mean of the fibre's level means (q1926 selection, counts in 0-50 ms) and F
its median Fano factor over levels with mean count >= 1 (paper_neuron/q1936_rows.json). If F >= 1, nu = 1 (Poisson). If even nu = NU_CAP
leaves the Fano factor above F, nu = NU_CAP (the most regular count of this family considered; such fibres are counted as capped).
K is the number of mass points of the capacity-achieving input on [lam, lam + A], found by the cutting-plane method of q1937 with this
kernel on a grid of 801 rates; points closer than 0.2 in the coordinate 2 sqrt(r) are merged. D is taken from paper_neuron/q1926_rows.json.
VERDICT (H_stepsM, registered; the bar of Amendment 2272 is unchanged): PASS if D <= K in at least 80 percent of the fibres; FAIL
otherwise. VOID if fewer than 20 fibres; INCOMPLETE if an input file is missing.
Readings: the fraction with D <= K by thirds of F, the distribution of K - K_Poisson, the number of capped fibres and their fraction
with D <= K, and the fraction with D <= K among fibres not capped.
Usage: py q1939_ph_steps_compoisson.py [--registration-run]
"""
import json
import os
import sys

import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from support_exact import _solve_w, support  # noqa: E402

CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
R26 = os.path.join(HERE, "paper_neuron", "q1926_rows.json")
R36 = os.path.join(HERE, "paper_neuron", "q1936_rows.json")
OUT = os.path.join(HERE, "paper_neuron", "q1939_rows.json")
NU_CAP = 50.0
TAIL = 1e-12


def cmp_pmf(rs, nu):
    """Rows: COM-Poisson pmf with mean exactly rs[i] (to 1e-9), truncated where the last retained point has mass < TAIL."""
    rs = np.atleast_1d(np.asarray(rs, float)); rmax = float(rs.max())
    ymax = int(np.ceil(rmax + 12 * np.sqrt(rmax + 1) + 30))
    while True:
        y = np.arange(ymax + 1); lf = gammaln(y + 1)
        lo = np.full(len(rs), -400.0); hi = np.full(len(rs), 400.0)
        for _ in range(120):
            t = 0.5 * (lo + hi); lw = t[:, None] * y[None, :] - nu * lf[None, :]
            lw -= lw.max(1, keepdims=True); p = np.exp(lw); p /= p.sum(1, keepdims=True)
            m = p @ y; big = m > rs; hi = np.where(big, t, hi); lo = np.where(big, lo, t)
        t = 0.5 * (lo + hi); lw = t[:, None] * y[None, :] - nu * lf[None, :]
        lw -= lw.max(1, keepdims=True); p = np.exp(lw); p /= p.sum(1, keepdims=True)
        p[rs == 0] = 0.0; p[rs == 0, 0] = 1.0
        if p[:, -1].max() < TAIL:
            break
        ymax *= 2
    assert np.max(np.abs(p @ y - rs)) < 1e-9, "mean not matched"
    return p


def fano_at(m, nu):
    p = cmp_pmf([m], nu)[0]; y = np.arange(len(p)); mu = p @ y
    return float(p @ (y - mu) ** 2 / mu)


def nu_for(F, mbar):
    if F >= 1.0:
        return 1.0, False
    if fano_at(mbar, NU_CAP) > F:
        return NU_CAP, True
    lo, hi = 1.0, NU_CAP
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fano_at(mbar, mid) > F:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), False


def k_cmp(A, lam, nu, ngrid=801, tol=1e-8, maxit=40):
    rs = np.linspace(lam, lam + A, ngrid); P = cmp_pmf(rs, nu); logP = np.log(np.where(P > 0, P, 1e-300)); Hx = (P * logP).sum(1)
    S = [0, ngrid - 1]
    for _ in range(maxit):
        w, C = _solve_w(P[S], Hx[S]); Q = w @ P[S]; lq = np.log(np.where(Q > 0, Q, 1e-300)); D = Hx - P @ lq
        k = int(D.argmax())
        if D[k] - C <= tol or k in S:
            break
        S.append(k)
    pts = np.sort([rs[i] for i, wi in zip(S, w) if wi > 1e-6]); t = 2 * np.sqrt(pts)
    return 1 + int(np.sum(np.diff(t) >= 0.2))


def k_poisson_merged(A, lam):
    p = np.sort(np.array(support(A, lam)["pts"], float)); t = 2 * np.sqrt(p + lam)
    return 1 + int(np.sum(np.diff(t) >= 0.2))


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean([r["D"] <= r["K"] for r in rows]))
    return ("PASS" if f >= 0.8 else "FAIL"), "D <= K in %.3f of %d fibres (bar 0.80)" % (f, len(rows))


def registration_run():
    ok = True
    for m in (0.7, 3.0, 12.0):
        f1 = fano_at(m, 1.0); print("control: Fano at nu = 1, mean %.1f: %.6f (Poisson 1)" % (m, f1)); ok &= abs(f1 - 1) < 1e-6
    fs = [fano_at(3.0, nu) for nu in (1, 1.5, 2, 3, 5, 10)]
    print("control: Fano at mean 3 for nu = 1, 1.5, 2, 3, 5, 10: %s" % ", ".join("%.3f" % f for f in fs))
    ok &= all(b < a for a, b in zip(fs, fs[1:]))
    nu, cap = nu_for(0.4, 3.0); f = fano_at(3.0, nu)
    print("control: nu_for(F = 0.4, mbar = 3) = %.4f (capped %s), Fano there %.6f" % (nu, cap, f)); ok &= abs(f - 0.4) < 1e-6 and not cap
    for A, lam in ((3.0, 0.5), (8.0, 0.5), (15.0, 1.0)):
        kp, kc = k_poisson_merged(A, lam), k_cmp(A, lam, 1.0)
        print("control: A %.1f lam %.1f: Poisson K %d, COM-Poisson K at nu = 1: %d" % (A, lam, kp, kc)); ok &= kp == kc
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        nus = (1.0, 1.5, 2.0, 3.0, 5.0); ks = [k_cmp(A, lam, nu) for nu in nus]
        print("control: A %.0f lam %.1f, K at nu = %s: %s" % (A, lam, nus, ks))
        ok &= all(b >= a for a, b in zip(ks, ks[1:])) and ks[-1] > ks[0]
    rows = [{"D": 2, "K": 3}] * 25; bad = [{"D": 4, "K": 3}] * 25
    d = (decide(rows)[0], decide(bad)[0], decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1939: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1939 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (CENSUS, R26, R36)):
        print("VERDICT q1939: INCOMPLETE (input missing)"); return 3
    from q1926_ph_steps import fibre
    r26 = {r["path"]: r for r in json.load(open(R26))}; r36 = {r["path"]: r for r in json.load(open(R36))}
    rows = []
    for rec in json.load(open(CENSUS)):
        if "error" in rec or rec.get("path") not in r26:
            continue
        f = fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        mbar = float(np.mean([c.mean() for c in counts])); F = r36.get(rec["path"], {}).get("median_fano")
        if F is None:
            continue
        nu, capped = nu_for(F, mbar)
        K = k_cmp(A, lam, nu) if nu > 1.0 else k_poisson_merged(A, lam)
        rows.append({"path": rec["path"], "D": r26[rec["path"]]["D"], "K_poisson": r26[rec["path"]]["K"], "K": K, "nu": nu,
                     "capped": capped, "F": F})
    json.dump(rows, open(OUT, "w"))
    if rows:
        F = np.array([r["F"] for r in rows]); q = np.percentile(F, [33.3, 66.7])
        for name, m in (("low F", F < q[0]), ("mid F", (F >= q[0]) & (F < q[1])), ("high F", F >= q[1])):
            sub = [r for r, mm in zip(rows, m) if mm]
            print("Reading: %s third, fibres %d, D <= K in %.3f" % (name, len(sub), float(np.mean([r["D"] <= r["K"] for r in sub]))))
        dk = np.array([r["K"] - r["K_poisson"] for r in rows])
        print("Reading: K - K_Poisson values %s" % {int(v): int(np.sum(dk == v)) for v in np.unique(dk)})
        cap = [r for r in rows if r["capped"]]; unc = [r for r in rows if not r["capped"]]
        print("Reading: capped fibres %d, D <= K in %s; not capped %d, D <= K in %s" % (
            len(cap), "%.3f" % np.mean([r["D"] <= r["K"] for r in cap]) if cap else "n/a",
            len(unc), "%.3f" % np.mean([r["D"] <= r["K"] for r in unc]) if unc else "n/a"))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1939 (Amendment 2290, H_stepsM): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
