"""q1936 (15 Oct 2026): task P2FANO, registered in Amendment 2284 before any Fano factor of a recorded fibre was computed.

Question (junior draft reading, finding 44). The discreteness test of Amendment 2272 (q1926) computes K for Poisson counts and compares
neighbouring sound levels with an exact conditional binomial test, which assumes Poisson counts. If the counts of a fibre vary more than
Poisson counts do, that test finds extra differences; if they vary less, the number of rates the noise allows changes. Are the counts
of the phase-locking fibres close enough to Poisson for that test?
Data: the fibres of q1926 (same census, same selection imported from q1926_ph_steps.fibre: at least 4 sound levels, at least 20
presentations per level, range of level means at least 1 spike per window; counts in 0 <= t < 50 ms).
Statistic: for every fibre and sound level with a mean count of at least 1, the Fano factor F = sample variance (ddof 1) / mean of the
counts across presentations. The verdict uses the median of F over all such fibre-levels.
VERDICT (H_fano, registered): PASS if the median Fano factor lies in [0.8, 1.2]; FAIL otherwise. VOID if fewer than 20 fibres or
fewer than 50 fibre-levels qualify; INCOMPLETE if the census is missing.
Readings: quartiles of F, the fraction of fibre-levels with F above 1.5 and below 0.67, the median F of fibres with D > K and with
D <= K (D and K from paper_neuron/q1926_rows.json).
Usage: py q1936_ph_fano.py [--registration-run]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
ROWS1926 = os.path.join(HERE, "paper_neuron", "q1926_rows.json")
OUT = os.path.join(HERE, "paper_neuron", "q1936_rows.json")


def fano_levels(counts):
    out = []
    for c in counts:
        m = float(np.mean(c))
        if m >= 1.0 and len(c) >= 2:
            out.append(float(np.var(c, ddof=1) / m))
    return out


def decide(fanos, n_fibres):
    if n_fibres < 20 or len(fanos) < 50:
        return "VOID", "too few fibres (%d) or fibre-levels (%d)" % (n_fibres, len(fanos))
    med = float(np.median(fanos))
    return ("PASS" if 0.8 <= med <= 1.2 else "FAIL"), "median Fano factor %.4f over %d fibre-levels of %d fibres (bar [0.8, 1.2])" % (
        med, len(fanos), n_fibres)


def synth(rng, kind, n_fib=265):
    fan, nf = [], 0
    for _ in range(n_fib):
        means = np.linspace(1.0, 10.0, 6); reps = 30; counts = []
        for m in means:
            if kind == "poisson":
                c = rng.poisson(m, reps)
            elif kind == "over":       # negative binomial with Fano 1.6: var = m + m^2/k, k = m / 0.6
                k = m / 0.6; c = rng.negative_binomial(k, k / (k + m), reps)
            else:                      # binomial with Fano 0.6: var = m (1 - p), p = 0.4
                n = int(round(m / 0.4)); c = rng.binomial(n, m / n, reps)
            counts.append(c.astype(float))
        fan += fano_levels(counts); nf += 1
    return fan, nf


def registration_run():
    rng = np.random.default_rng(2284); ok = True
    for kind, want in (("poisson", "PASS"), ("over", "FAIL"), ("under", "FAIL")):
        f, n = synth(rng, kind); v = decide(f, n)
        print("control: %s fibres: %s, %s" % (kind, v[0], v[1])); ok &= v[0] == want
    d = (decide([1.0] * 60, 25)[0], decide([1.6] * 60, 25)[0], decide([0.6] * 60, 25)[0], decide([1.0] * 60, 5)[0], decide([1.0] * 10, 25)[0])
    print("control decide(): %s %s %s %s %s (expected PASS FAIL FAIL VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "FAIL", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1936: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1936 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS):
        print("VERDICT q1936: INCOMPLETE (census missing)"); return 3
    from q1926_ph_steps import fibre
    recs = [x for x in json.load(open(CENSUS)) if "error" not in x]
    rows, allf = [], []
    for rec in recs:
        f = fibre(rec)
        if f is None:
            continue
        levels, counts, lam, A = f
        fl = fano_levels(counts)
        rows.append({"path": rec["path"], "fanos": fl, "median_fano": float(np.median(fl)) if fl else None})
        allf += fl
    json.dump(rows, open(OUT, "w"))
    if allf:
        a = np.array(allf)
        print("Reading: Fano quartiles %.3f %.3f %.3f; fraction above 1.5 %.3f, below 0.67 %.3f" % (
            *np.percentile(a, [25, 50, 75]), float(np.mean(a > 1.5)), float(np.mean(a < 0.67))))
        if os.path.exists(ROWS1926):
            dk = {r["path"]: r["D"] > r["K"] for r in json.load(open(ROWS1926))}
            ex = [r["median_fano"] for r in rows if r["median_fano"] is not None and dk.get(r["path"]) is True]
            ok = [r["median_fano"] for r in rows if r["median_fano"] is not None and dk.get(r["path"]) is False]
            if ex and ok:
                print("Reading: median of per-fibre median Fano, fibres with D > K %.3f (%d), with D <= K %.3f (%d)" % (
                    float(np.median(ex)), len(ex), float(np.median(ok)), len(ok)))
    v, why = decide(allf, len(rows))
    print(why)
    print("VERDICT q1936 (Amendment 2284, H_fano): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
