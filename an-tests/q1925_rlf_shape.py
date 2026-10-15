"""q1925 (15 Oct 2026): task P2LAUGHLIN step 3, registered in Amendment 2271 before any rate-level function was fitted.

Prediction (paper_neuron/LAUGHLIN_DERIVATION.md, P1, Barlow frame). Natural sound levels are distributed roughly as a symmetric bell in
dB (Escabi et al., J. Neurosci. 2003; local text papers/escabi2003.txt: "roughly symmetric bell-shaped distribution", 90th-percentile
range about 30 dB for environmental sounds and about 50 dB for vocalizations and speech, i.e. a Gaussian SD of about 9 to 15 dB). For a
Gaussian level law N(mu, sigma^2) the infomax rate-level function of a Poisson fibre in the large-A form (McDonnell and Stocks 2008,
Eq. 5) is
    M_sq:  r(L) = r0 + A Phi((L - mu)/sigma)^2,
an asymmetric sigmoid, against the usual symmetric description
    M_log: r(L) = r0 + A/(1 + exp(-(L - mu)/s)).
Both have four parameters. Data: DANDI 001262 RLF recordings (census q1895), counts in 0 <= t < 50 ms per trial (the window of
Amendments 2264 and 2267), levels from the RLF_ABI<level> keys in dB, the silent condition excluded. Selection: >= 8 levels, >= 5
repeats at every level, and max - min of the level means >= 1 spike per window. Fit: Poisson maximum likelihood on the trial counts,
best of 12 fixed starts per model (scipy L-BFGS-B, bounds r0 in [0, 50], A in [0, 100], mu in [-20, 120] dB, sigma and s in [0.5, 60] dB).
Design note: the per-fibre comparison has little power at 10 repeats (controls before registration: mean log-likelihood difference
+0.15 with SD 0.48 per fibre on M_sq data, -0.16 with SD 0.77 on M_log data), so the verdict pools the fibres.
VERDICT (H_L1, registered): PASS if (a) the mean over fibres of d = loglik(M_sq) - loglik(M_log) has a bootstrap 95 percent lower bound
(2000 resamples of fibres, seed 2271) above 0 AND (b) the median fitted sigma of M_sq lies in [6, 20] dB; FAIL otherwise. VOID if fewer than 20 fibres are selected or more than 10 percent of
the fits fail; INCOMPLETE if the census is missing.
Readings: fraction preferring M_sq, median log-likelihood difference, quartiles of sigma and s, the same split by spontaneous count
(level-0 mean below or above 0.5 per window).
Usage: py q1925_rlf_shape.py [--registration-run]
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
OUT = os.path.join(HERE, "paper_neuron", "q1925_rows.json")
W = 0.050
BOUNDS = [(0.0, 50.0), (0.0, 100.0), (-20.0, 120.0), (0.5, 60.0)]


def m_sq(p, L):
    r0, A, mu, sg = p
    return r0 + A * norm.cdf((L - mu) / sg) ** 2


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


def boot_lower(d, n=2000, seed=2271):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5))


def decide(rows, n_fail):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    if n_fail > 0.1 * (len(rows) + n_fail):
        return "VOID", "more than 10 percent of fits failed"
    d = [r["d"] for r in rows]; lo = boot_lower(d); med = float(np.median([r["sigma"] for r in rows]))
    ok = lo > 0 and 6.0 <= med <= 20.0
    return ("PASS" if ok else "FAIL"), ("mean d %.4f, bootstrap 95%% lower bound %.4f (bar > 0); median sigma %.2f dB (bar [6, 20])"
                                         % (float(np.mean(d)), lo, med))


def registration_run():
    rng = np.random.default_rng(1)
    levels = np.arange(0, 81, 3.0); L = np.repeat(levels, 10)
    ok = True; wins = []
    for truth, model, p in (("sq", m_sq, [0.5, 8.0, 40.0, 12.0]), ("log", m_log, [0.5, 8.0, 40.0, 5.0])):
        rows = []
        for _ in range(200):
            y = rng.poisson(model(p, L)).astype(float)
            a, b = fit(m_sq, L, y), fit(m_log, L, y)
            rows.append({"d": b.fun - a.fun, "sigma": float(a.x[3])})
        v = decide(rows, 0); wins.append(v[0])
        print("control: data from M_%s (200 synthetic fibres, 27 levels x 10 repeats): %s, %s" % (truth, v[0], v[1]))
    ok &= wins[0] == "PASS" and wins[1] == "FAIL"
    good = [{"d": 1.0, "sigma": 12.0}] * 30; bad = [{"d": -1.0, "sigma": 12.0}] * 30; off = [{"d": 1.0, "sigma": 40.0}] * 30
    d = (decide(good, 0)[0], decide(bad, 0)[0], decide(off, 0)[0], decide(good[:5], 0)[0], decide(good, 10)[0])
    print("control decide(): %s %s %s %s %s (expected PASS FAIL FAIL VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "FAIL", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1925: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1925 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS):
        print("VERDICT q1925: INCOMPLETE (census missing)"); return 3
    recs = [x for x in json.load(open(CENSUS)) if "error" not in x]
    rows, n_fail = [], 0
    for rec in recs:
        s = select(rec)
        if s is None:
            continue
        L, y, spont = s
        a, b = fit(m_sq, L, y), fit(m_log, L, y)
        if a is None or b is None:
            n_fail += 1; continue
        rows.append({"path": rec["path"], "d": b.fun - a.fun, "sigma": float(a.x[3]), "s": float(b.x[3]), "spont": float(spont),
                     "A_sq": float(a.x[1])})
    json.dump(rows, open(OUT, "w"))
    if rows:
        dd = np.array([r["d"] for r in rows]); sg = np.array([r["sigma"] for r in rows]); ss = np.array([r["s"] for r in rows])
        print("Reading: fibres %d, failed fits %d; median log-likelihood difference (M_sq minus M_log, in favour of M_sq) %.3f"
              % (len(rows), n_fail, float(np.median(dd))))
        print("Reading: sigma quartiles %.2f %.2f %.2f dB; s quartiles %.2f %.2f %.2f dB" % (*np.percentile(sg, [25, 50, 75]), *np.percentile(ss, [25, 50, 75])))
        for name, m in (("low spontaneous", [r["spont"] < 0.5 for r in rows]), ("high spontaneous", [r["spont"] >= 0.5 for r in rows])):
            m = np.array(m)
            if m.any():
                print("Reading: %s fibres %d, fraction preferring M_sq %.3f, median sigma %.2f dB" % (name, int(m.sum()), float(np.mean(dd[m] > 0)), float(np.median(sg[m]))))
    v, why = decide(rows, n_fail)
    print(why)
    print("VERDICT q1925 (Amendment 2271, H_L1): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
