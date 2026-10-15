"""q1953 (15 Oct 2026): task P2WEAKAUDIT step 6, H_stepsL, a per-fibre likelihood test of discreteness that replaces the count D of
adjacent significant pairs (q1926), whose power is low near peak count 8 when the tested levels cover only part of the range
(Reading 553, q1952 registration run). Registered only after the controls below pass; amendment 2297 if they do.

Per fibre (q1926 selection, counts in 0-50 ms, levels ordered by sound level, n_l presentations, means m_l):
  V(m) = max(m, 0.05) * clamp(a + b m, 0.05, 1), the variance of the fibre's count model (q1950 Fano line a, b).
  T = min over splits of the ordered levels into at most K contiguous groups of  sum_l n_l (m_l - mu_g)^2 / V(mu_g),
  mu_g the presentation-weighted mean of group g. K: the rate-dependent K of q1950 (q1950_rows.json).
  p: parametric bootstrap under the fitted split (counts drawn from the q1950 channel at the group means mu_g, same n_l), B draws,
  p = (1 + #{T* >= T}) / (B + 1). The fibre is consistent with a staircase of at most K steps if p >= 0.05.
VERDICT (H_stepsL): PASS if at least 80 percent of the fibres are consistent; FAIL otherwise; VOID below 20 fibres.
Readings: the fraction for A >= 8 and A < 8 and by thirds of the median Fano; the same statistic with K + 1 and K - 1.
Usage: py q1953_ph_steps_lik.py [--registration-run]
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1950_ph_steps_ratefano as R  # noqa: E402

OUT = os.path.join(HERE, "paper_neuron", "q1953_rows.json")
R50 = os.path.join(HERE, "paper_neuron", "q1950_rows.json")
ALPHA = 0.05


def var_fn(m, a, b):
    m = np.asarray(m, float)
    return np.maximum(m, 0.05) * np.clip(a + b * m, R.FLO, R.FHI)


def splits(L, kmax):
    for k in range(1, min(kmax, L) + 1):
        for cuts in itertools.combinations(range(1, L), k - 1):
            yield (0,) + cuts + (L,)


def stat(means, ns, K, a, b):
    best, bmu = np.inf, None
    for s in splits(len(means), K):
        mu = np.empty(len(means)); t = 0.0
        for i, j in zip(s[:-1], s[1:]):
            g = float(np.dot(ns[i:j], means[i:j]) / ns[i:j].sum()); mu[i:j] = g
        t = float(np.sum(ns * (means - mu) ** 2 / var_fn(mu, a, b)))
        if t < best:
            best, bmu = t, mu
    return best, bmu


def sampler(rates, a, b):
    rates = np.asarray(rates, float)
    z, _ = R.zeta_for(np.maximum(rates, 1e-3), np.clip(a + b * rates, R.FLO, R.FHI))
    P = R.cmp_rows(rates, z); y = np.arange(P.shape[1]); P = P / P.sum(1, keepdims=True)
    return P, y


def pvalue(counts, K, a, b, B, rng):
    ns = np.array([len(c) for c in counts], float); means = np.array([c.mean() for c in counts])
    T, mu = stat(means, ns, K, a, b)
    P, y = sampler(mu, a, b); ge = 0
    for _ in range(B):
        sim = np.array([rng.choice(y, size=int(n), p=P[i]).mean() for i, n in enumerate(ns)])
        ge += stat(sim, ns, K, a, b)[0] >= T
    return (1 + ge) / (B + 1), T


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    f = float(np.mean([r["p"] >= ALPHA for r in rows]))
    return ("PASS" if f >= 0.8 else "FAIL"), "consistent with at most K steps in %.3f of %d fibres (bar 0.80)" % (f, len(rows))


def registration_run(nfib=100, B=100):
    ok = True; rng = np.random.default_rng(2297); a, b = 0.7034, -0.0355
    for A, lam in ((8.0, 0.5), (15.0, 1.0)):
        Kfull, _ = R.k_rate(A, lam, a, b); vals = np.linspace(lam, lam + A, Kfull)
        for cover in ("full", "upper"):
            lo_t = lam if cover == "full" else lam + 1.0
            tested = np.linspace(lo_t, lam + A, 7)
            stair = vals[np.argmin(np.abs(tested[:, None] - vals[None, :]), axis=1)]
            res = {}
            for name, rates in (("staircase", stair), ("graded", tested)):
                P, y = sampler(rates, a, b); rej = []
                for _ in range(nfib):
                    cs = [rng.choice(y, size=40, p=P[i]).astype(float) for i in range(7)]
                    mlo, mhi = min(c.mean() for c in cs), max(c.mean() for c in cs)
                    K, _ = R.k_rate(mhi - mlo, mlo, a, b)            # K from the observed range, as on data
                    rej.append(pvalue(cs, K, a, b, B, rng)[0] < ALPHA)
                res[name] = float(np.mean(rej))
            print("control: A %.0f lam %.1f, %s range: rejected %.3f of staircase and %.3f of graded fibres (K full range %d)"
                  % (A, lam, cover, res["staircase"], res["graded"], Kfull))
            ok &= res["staircase"] <= 0.10 and res["graded"] >= 0.70
    rows = [{"p": 0.5}] * 25; bad = [{"p": 0.01}] * 25
    d = (decide(rows)[0], decide(bad)[0], decide(rows[:5])[0])
    print("control decide(): %s %s %s (expected PASS FAIL VOID)" % d); ok &= d == ("PASS", "FAIL", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1953: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1953 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not all(os.path.exists(p) for p in (R.CENSUS, R.R26, R.R36, R50)):
        print("VERDICT q1953: INCOMPLETE (input missing)"); return 3
    lines, r26, r36 = R.fibre_lines(); r50 = {r["path"]: r for r in json.load(open(R50))}
    rng = np.random.default_rng(2297); rows = []
    for path, levels, counts, lam, A, a, b, flat in lines:
        K = r50[path]["K"]
        p, T = pvalue(counts, K, a, b, 200, rng)
        p1, _ = pvalue(counts, K + 1, a, b, 200, rng); pm, _ = pvalue(counts, max(K - 1, 1), a, b, 200, rng)
        rows.append({"path": path, "K": K, "T": T, "p": p, "p_Kplus1": p1, "p_Kminus1": pm, "A": A, "F": r36[path]["median_fano"]})
        json.dump(rows, open(OUT, "w"))
    if rows:
        f = lambda S, k="p": float(np.mean([r[k] >= ALPHA for r in S]))
        for name, sub in (("A >= 8", [r for r in rows if r["A"] >= 8]), ("A < 8", [r for r in rows if r["A"] < 8])):
            print("Reading: %s, fibres %d, consistent %.3f" % (name, len(sub), f(sub)))
        F = np.array([r["F"] for r in rows]); q = np.percentile(F, [33.3, 66.7])
        for name, m in (("most regular", F < q[0]), ("middle", (F >= q[0]) & (F < q[1])), ("least regular", F >= q[1])):
            sub = [r for r, mm in zip(rows, m) if mm]
            print("Reading: %s third, fibres %d, consistent %.3f" % (name, len(sub), f(sub)))
        print("Reading: consistent with K + 1 steps %.3f, with K - 1 steps %.3f" % (f(rows, "p_Kplus1"), f(rows, "p_Kminus1")))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1953 (Amendment 2297, H_stepsL): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
