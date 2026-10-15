"""q1935 (15 Oct 2026): task P2SQRTL, registered in Amendment 2282 before any fit of the new shape to data.

Why. With a spontaneous count r0 the small-noise infomax argument makes
sqrt(r) uniform on [sqrt(r0), sqrt(r0 + A)], so for a Gaussian level law the large-A rate-level function is
    M_ex:  r(L) = (sqrt(r0) + (sqrt(r0 + A) - sqrt(r0)) Phi((L - mu)/sigma))^2,
while H_L1 (Amendment 2271, q1925) tested M_sq: r0 + A Phi((L - mu)/sigma)^2, the form for r0 small compared with A. All three shapes
(M_ex, M_sq and the symmetric logistic M_log) have four parameters. Data, selection, window, fit (Poisson maximum likelihood, best of 12
fixed starts, L-BFGS-B) and bounds are those of q1925, copied as text.
Known-answer check: the refitted d_sq = loglik(M_sq) - loglik(M_log) must equal the value stored by the registered run of q1925 in
paper_neuron/q1925_rows.json within 1e-6 for every fibre (same data, same deterministic fit); otherwise VOID.
VERDICT (H_sqrt, registered): PASS if all three hold:
 (a) the mean over fibres of d_ex = loglik(M_ex) - loglik(M_log) has a bootstrap 95 percent lower bound above 0 (2000 resamples of
     fibres, seed 2282), and the median fitted sigma of M_ex lies in [6, 20] dB;
 (b) the mean over fibres of g = loglik(M_ex) - loglik(M_sq) is >= 0;
 (c) the mean of g over the fibres whose spontaneous count (mean count at the lowest level) is at or above the median exceeds the
     mean of g over the fibres below the median.
FAIL otherwise. VOID if fewer than 20 fibres are selected, more than 10 percent of the fits fail, or the known-answer check fails;
INCOMPLETE if the census or the q1925 rows are missing.
Readings: bootstrap lower bound of mean g, fraction of fibres with g > 0, mean g by spontaneous half, quartiles of the fitted r0.
Usage: py q1935_rlf_sqrt_shape.py [--registration-run]
"""
import json
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
ROWS1925 = os.path.join(HERE, "paper_neuron", "q1925_rows.json")
OUT = os.path.join(HERE, "paper_neuron", "q1935_rows.json")
W = 0.050
BOUNDS = [(0.0, 50.0), (0.0, 100.0), (-20.0, 120.0), (0.5, 60.0)]


def m_sq(p, L):
    r0, A, mu, sg = p
    return r0 + A * norm.cdf((L - mu) / sg) ** 2


def m_ex(p, L):
    r0, A, mu, sg = p
    a = np.sqrt(max(r0, 0.0)); b = np.sqrt(max(r0, 0.0) + max(A, 0.0))
    return (a + (b - a) * norm.cdf((L - mu) / sg)) ** 2


def m_log(p, L):
    r0, A, mu, s = p
    return r0 + A / (1.0 + np.exp(-np.clip((L - mu) / s, -50, 50)))


def nll(p, model, L, y):
    r = np.maximum(model(p, L), 1e-9)
    return float(np.sum(r - y * np.log(r) + gammaln(y + 1)))


def fit(model, L, y):
    lo, hi = float(L.min()), float(L.max()); best = None
    ymin, ymax = max(float(np.min(y)), 0.0), float(np.max(y))
    for mu in np.linspace(lo, hi, 4):
        for sc in (3.0, 10.0, 25.0):
            p0 = [min(ymin + 0.01, 49), max(ymax - ymin, 0.1), mu, sc]
            r = minimize(nll, p0, args=(model, L, y), method="L-BFGS-B", bounds=BOUNDS)
            if r.success or r.status in (0, 2):
                if best is None or r.fun < best.fun:
                    best = r
    return best


def fibre_data(rec):
    sp = rec.get("spikes", {}); L, y = [], []
    keys = [k for k in sp if k.startswith("RLF_ABI")]
    for k in keys:
        lev = float(re.sub(r"[^0-9.\-]", "", k[len("RLF_ABI"):]))
        for tr in sp[k]:
            L.append(lev); y.append(sum(1 for t in tr if 0.0 <= t < W))
    return keys, np.array(L), np.array(y, float)


def select(rec):
    keys, L, y = fibre_data(rec)
    if len(keys) < 8:
        return None
    levs = np.unique(L); reps = [np.sum(L == v) for v in levs]
    if min(reps) < 5:
        return None
    means = np.array([y[L == v].mean() for v in levs])
    if means.max() - means.min() < 1.0:
        return None
    return L, y, means[np.argmin(levs)]


def boot_lower(d, n=2000, seed=2282):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5))


def decide(rows, n_fail, known_ok=True):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    if n_fail > 0.1 * (len(rows) + n_fail):
        return "VOID", "more than 10 percent of fits failed"
    if not known_ok:
        return "VOID", "known-answer check against q1925 failed"
    d = np.array([r["d_ex"] for r in rows]); g = np.array([r["g"] for r in rows]); sp = np.array([r["spont"] for r in rows])
    lo = boot_lower(d); med = float(np.median([r["sigma_ex"] for r in rows]))
    hi_m = sp >= np.median(sp)
    gh, gl = float(g[hi_m].mean()), float(g[~hi_m].mean()) if (~hi_m).any() else float("nan")
    a = lo > 0 and 6.0 <= med <= 20.0
    b = float(g.mean()) >= 0.0
    c = bool((~hi_m).any()) and gh > gl
    why = ("(a) mean d_ex %.4f, bootstrap 95%% lower bound %.4f (bar > 0), median sigma_ex %.2f dB (bar [6, 20]): %s; "
           "(b) mean g %.4f (bar >= 0): %s; (c) mean g high-spontaneous half %.4f vs low half %.4f (bar high > low): %s"
           % (float(d.mean()), lo, med, a, float(g.mean()), b, gh, gl, c))
    return ("PASS" if (a and b and c) else "FAIL"), why


def synth(rng, n, gen):
    levels = np.arange(0, 81, 3.0); L = np.repeat(levels, 10); rows = []
    for i in range(n):
        model, p = gen(i)
        y = rng.poisson(model(p, L)).astype(float)
        e, q, b = fit(m_ex, L, y), fit(m_sq, L, y), fit(m_log, L, y)
        rows.append({"d_ex": b.fun - e.fun, "g": q.fun - e.fun, "sigma_ex": float(e.x[3]), "spont": float(np.mean(y[L == 0]))})
    return rows


def registration_run():
    rng = np.random.default_rng(2)
    ok = True
    # positive: M_ex data, half the fibres with a low and half with a high spontaneous count
    pos = synth(rng, 200, lambda i: (m_ex, [0.2 if i % 2 else 4.0, 8.0, 40.0, 12.0]))
    v = decide(pos, 0); print("control: M_ex data, r0 0.2 or 4.0 (200 synthetic fibres): %s, %s" % v); ok &= v[0] == "PASS"
    # negative 1: logistic data must fail (a)
    neg1 = synth(rng, 200, lambda i: (m_log, [0.2 if i % 2 else 4.0, 8.0, 40.0, 5.0]))
    v = decide(neg1, 0); print("control: M_log data: %s, %s" % v); ok &= v[0] == "FAIL"
    # negative 2: M_sq data with a high spontaneous count must fail (b)
    neg2 = synth(rng, 200, lambda i: (m_sq, [0.2 if i % 2 else 4.0, 8.0, 40.0, 12.0]))
    v = decide(neg2, 0); print("control: M_sq data: %s, %s" % v); ok &= v[0] == "FAIL"
    base = {"d_ex": 1.0, "g": 0.1, "sigma_ex": 12.0}
    good = [dict(base, spont=float(i % 2), g=0.1 + 0.2 * (i % 2)) for i in range(30)]
    flat_c = [dict(base, spont=float(i % 2), g=0.3 - 0.2 * (i % 2)) for i in range(30)]
    neg_b = [dict(base, spont=float(i % 2), g=-0.5 + 0.1 * (i % 2)) for i in range(30)]
    bad_a = [dict(r, d_ex=-1.0) for r in good]
    d = (decide(good, 0)[0], decide(bad_a, 0)[0], decide(neg_b, 0)[0], decide(flat_c, 0)[0], decide(good[:5], 0)[0],
         decide(good, 10)[0], decide(good, 0, known_ok=False)[0])
    print("control decide(): %s %s %s %s %s %s %s (expected PASS FAIL FAIL FAIL VOID VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "FAIL", "FAIL", "VOID", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1935: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1935 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS) or not os.path.exists(ROWS1925):
        print("VERDICT q1935: INCOMPLETE (census or q1925 rows missing)"); return 3
    ref = {r["path"]: r["d"] for r in json.load(open(ROWS1925))}
    recs = [x for x in json.load(open(CENSUS)) if "error" not in x]
    rows, n_fail, known_ok, worst = [], 0, True, 0.0
    for rec in recs:
        s = select(rec)
        if s is None:
            continue
        L, y, spont = s
        e, q, b = fit(m_ex, L, y), fit(m_sq, L, y), fit(m_log, L, y)
        if e is None or q is None or b is None:
            n_fail += 1; continue
        d_sq = b.fun - q.fun
        if rec["path"] not in ref:
            known_ok = False
        else:
            worst = max(worst, abs(d_sq - ref[rec["path"]]))
        rows.append({"path": rec["path"], "d_ex": b.fun - e.fun, "g": q.fun - e.fun, "d_sq": d_sq, "sigma_ex": float(e.x[3]),
                     "r0_ex": float(e.x[0]), "spont": float(spont)})
    known_ok = known_ok and worst <= 1e-6 and len(rows) == len(ref)
    json.dump(rows, open(OUT, "w"))
    print("Reading: known-answer check, largest |d_sq - q1925 d| = %.3g over %d fibres (q1925 rows %d)" % (worst, len(rows), len(ref)))
    if rows:
        g = np.array([r["g"] for r in rows]); r0 = np.array([r["r0_ex"] for r in rows])
        print("Reading: mean g %.4f, bootstrap lower bound %.4f, fraction g > 0 %.3f; r0_ex quartiles %.2f %.2f %.2f"
              % (float(g.mean()), boot_lower(g), float(np.mean(g > 0)), *np.percentile(r0, [25, 50, 75])))
    v, why = decide(rows, n_fail, known_ok)
    print(why)
    print("VERDICT q1935 (Amendment 2282, H_sqrt): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
