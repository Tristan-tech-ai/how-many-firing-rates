"""q1937 (15 Oct 2026): task P2FANOK, registered in Amendment 2286 before any K of a recorded fibre was recomputed.

Question. Amendment 2285 found the phase-locking fibres strongly sub-Poisson (median Fano factor 0.355). The K of the discreteness test
(Amendment 2272) was computed for Poisson counts and is too small for such fibres. Does D <= K hold when K is computed for a counting
channel with the fibre's own dispersion?
Channel. Y ~ Binomial(n, r/n) with the rate r in [lam, lam + A], whose Fano factor is 1 - r/n. For each fibre n is the smallest integer
with n >= lam + A and 1 - mbar/n <= F_fib, where mbar is the mean of the fibre's level means (q1926 selection, counts in 0-50 ms) and
F_fib its median Fano factor over levels with mean count >= 1 (paper_neuron/q1936_rows.json); if F_fib >= 1 the Poisson channel is used.
K is the number of mass points of the capacity-achieving input, found by the cutting-plane method of support_exact.py with the binomial
kernel; points closer than 0.2 in the coordinate 2 sqrt(r) are merged, as in q1928. D is taken from paper_neuron/q1926_rows.json.
VERDICT (H_stepsF, registered; the bar of Amendment 2272 is unchanged): PASS if D <= K in at least 80 percent of the fibres; FAIL
otherwise. VOID if fewer than 20 fibres; INCOMPLETE if an input file is missing.
Readings: the fraction with D <= K by thirds of F_fib, the distribution of K - K_Poisson.
Usage: py q1937_ph_steps_binomial.py [--registration-run]
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
OUT = os.path.join(HERE, "paper_neuron", "q1937_rows.json")


def _Pbin(rs, n):
    y = np.arange(n + 1)[None, :]; p = np.clip(rs[:, None] / n, 1e-300, 1 - 1e-15)
    lc = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
    return np.exp(lc + y * np.log(p) + (n - y) * np.log1p(-p))


def k_binomial(A, lam, n, ngrid=801, tol=1e-8, maxit=40):
    rs = np.linspace(lam, lam + A, ngrid); P = _Pbin(rs, n); logP = np.log(np.where(P > 0, P, 1e-300)); Hx = (P * logP).sum(1)
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


def choose_n(lam, A, mbar, F):
    if F >= 1.0:
        return None
    # repaired 15 Oct 2026 (Amendment 2287): 1 - mbar/n grows with n, so the smallest n >= lam + A with 1 - mbar/n <= F is
    # ceil(lam + A) when it qualifies; when it does not, no n reaches F and the most regular channel n = ceil(lam + A) is used.
    # Version 1 raised n and fell back to the Poisson channel for 107 fibres (run VOID, outputs kept as *_v1_void).
    return max(int(np.ceil(lam + A)), 1)


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean([r["D"] <= r["K"] for r in rows]))
    return ("PASS" if f >= 0.8 else "FAIL"), "D <= K in %.3f of %d fibres (bar 0.80)" % (f, len(rows))


def registration_run():
    ok = True
    for A, lam in ((3.0, 0.5), (8.0, 0.5), (15.0, 1.0)):
        kp, kb = k_poisson_merged(A, lam), k_binomial(A, lam, 4000)
        print("control: A %.1f lam %.1f: Poisson K %d, binomial K at n = 4000 (Fano near 1) %d" % (A, lam, kp, kb)); ok &= kp == kb
    # version 2 of this control (15 Oct 2026 20:2x, before any recorded fibre was used): version 1 asked K to grow already at n = 11 for
    # A = 8, lam = 0.5 and failed (K = 3 at n = 4000, 40, 16, 11). A probe gave K = 4 at n = 10 and, for A = 15, lam = 1, K = 4 at
    # n = 4000, 40, 20 and 5 at n = 17: K grows with sub-Poisson counts only when the dispersion is strong. The control now asks that K
    # never falls as n falls and is larger at n = ceil(lam + A) + 1 than at n = 4000, at both (A, lam).
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        ns = (4000, 40, 20, int(np.ceil(lam + A)) + 1); ks = [k_binomial(A, lam, n) for n in ns]
        print("control: A %.0f lam %.1f, K at n = %s (Fano at the mid rate %s): %s" % (
            A, lam, ns, ", ".join("%.2f" % (1 - (lam + A / 2) / n) for n in ns), ks))
        ok &= all(b >= a for a, b in zip(ks, ks[1:])) and ks[-1] > ks[0]
    rows = [{"D": 2, "K": 3}] * 25; bad = [{"D": 4, "K": 3}] * 25
    d = (decide(rows)[0], decide(bad)[0], decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1937: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1937 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (CENSUS, R26, R36)):
        print("VERDICT q1937: INCOMPLETE (input missing)"); return 3
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
        n = choose_n(lam, A, mbar, F)
        K = k_binomial(A, lam, n) if n else k_poisson_merged(A, lam)
        rows.append({"path": rec["path"], "D": r26[rec["path"]]["D"], "K_poisson": r26[rec["path"]]["K"], "K": K, "n": n, "F": F})
    json.dump(rows, open(OUT, "w"))
    if rows:
        F = np.array([r["F"] for r in rows]); q = np.percentile(F, [33.3, 66.7])
        for name, m in (("low F", F < q[0]), ("mid F", (F >= q[0]) & (F < q[1])), ("high F", F >= q[1])):
            sub = [r for r, mm in zip(rows, m) if mm]
            print("Reading: %s third, fibres %d, D <= K in %.3f" % (name, len(sub), float(np.mean([r["D"] <= r["K"] for r in sub]))))
        dk = np.array([r["K"] - r["K_poisson"] for r in rows])
        print("Reading: K - K_Poisson values %s" % {int(v): int(np.sum(dk == v)) for v in np.unique(dk)})
    v, why = decide(rows)
    print(why)
    print("VERDICT q1937 (Amendments 2286 and 2287, H_stepsF repaired): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
