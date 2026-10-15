"""q1957 (16 Oct 2026): task P2SHAPEALT, to be registered (Amendment 2303) before any rate-level function is fitted with the new model.

Why. The registered shape test H_L1 (q1925, Amendment 2271) compared the predicted rate-level form
    M_sq:  r(L) = r0 + A Phi((L - mu)/sigma)^2
with one symmetric sigmoid only. A fourth draft reading noted that this shows asymmetry, not the predicted form. The test here compares
M_sq with an asymmetric curve whose asymmetry has the SAME direction (slow start, abrupt end), the complementary log-log sigmoid
    M_cll: r(L) = r0 + A (1 - exp(-exp((L - mu)/s))),
which also has four parameters. M_sq comes from Barlow's principle with a Gaussian level law; M_cll comes from no theory here.
Data, window, selection and fits are those of q1925 (functions imported unchanged): DANDI 001262 RLF recordings, counts in
0 <= t < 50 ms, >= 8 levels, >= 5 repeats, range of level means >= 1 spike; Poisson maximum likelihood, best of 12 fixed starts,
the same bounds (s in [0.5, 60] dB).
VERDICT (H_asym, to be registered): PASS if the mean over fibres of d = loglik(M_sq) - loglik(M_cll) has a bootstrap 95 percent lower
bound (2000 resamples of fibres, seed 2303) above 0; FAIL otherwise. VOID if fewer than 20 fibres are selected or more than 10 percent
of the fits fail; INCOMPLETE if the census is missing.
Readings (not part of the rule): fraction of fibres with d > 0, median d, the same comparison against the standard Gompertz curve
r0 + A exp(-exp(-(L - mu)/s)) (asymmetry of the opposite direction), and the split by spontaneous count as in q1925.
Usage: py q1957_rlf_shape_asym.py [--registration-run]
"""
import json
import os
import sys

import numpy as np

import q1925_rlf_shape as Q

OUT = os.path.join(Q.HERE, "paper_neuron", "q1957_rows.json")


def m_cll(p, L):
    r0, A, mu, s = p
    return r0 + A * (1.0 - np.exp(-np.exp(np.clip((L - mu) / s, -50, 50))))


def m_gom(p, L):
    r0, A, mu, s = p
    return r0 + A * np.exp(-np.exp(np.clip(-(L - mu) / s, -50, 50)))


def boot_lower(d, n=2000, seed=2303):
    return Q.boot_lower(d, n, seed)


def decide(rows, n_fail):
    if len(rows) < 20:
        return "VOID", "fewer than 20 fibres"
    if n_fail > 0.1 * (len(rows) + n_fail):
        return "VOID", "more than 10 percent of fits failed"
    d = [r["d"] for r in rows]; lo = boot_lower(d)
    return ("PASS" if lo > 0 else "FAIL"), ("mean d %.4f, bootstrap 95%% lower bound %.4f (bar > 0)" % (float(np.mean(d)), lo))


def registration_run():
    rng = np.random.default_rng(2303)
    levels = np.arange(0, 81, 3.0); L = np.repeat(levels, 10)
    ok = True; got = []
    for truth, model, p in (("sq", Q.m_sq, [0.5, 8.0, 40.0, 12.0]), ("cll", m_cll, [0.5, 8.0, 45.0, 8.0])):
        rows = []
        for _ in range(200):
            y = rng.poisson(model(p, L)).astype(float)
            a, b = Q.fit(Q.m_sq, L, y), Q.fit(m_cll, L, y)
            rows.append({"d": b.fun - a.fun})
        v = decide(rows, 0); got.append(v[0])
        print("control: data from M_%s (200 synthetic fibres, 27 levels x 10 repeats): %s, %s" % (truth, v[0], v[1]))
    ok &= got == ["PASS", "FAIL"]
    good = [{"d": 1.0}] * 30; bad = [{"d": -1.0}] * 30
    d = (decide(good, 0)[0], decide(bad, 0)[0], decide(good[:5], 0)[0], decide(good, 10)[0])
    print("control decide(): %s %s %s %s (expected PASS FAIL VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1957: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1957 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(Q.CENSUS):
        print("VERDICT q1957: INCOMPLETE (census missing)"); return 3
    recs = [x for x in json.load(open(Q.CENSUS)) if "error" not in x]
    rows, n_fail = [], 0
    for rec in recs:
        s = Q.select(rec)
        if s is None:
            continue
        L, y, spont = s
        a, b, c = Q.fit(Q.m_sq, L, y), Q.fit(m_cll, L, y), Q.fit(m_gom, L, y)
        if a is None or b is None:
            n_fail += 1; continue
        rows.append({"path": rec["path"], "d": b.fun - a.fun, "d_gom": (c.fun - a.fun) if c is not None else None, "spont": float(spont)})
    json.dump(rows, open(OUT, "w"))
    if rows:
        dd = np.array([r["d"] for r in rows])
        print("Reading: fibres %d, failed fits %d; fraction with d > 0 %.3f; median d %.4f" % (len(rows), n_fail, float(np.mean(dd > 0)), float(np.median(dd))))
        dg = np.array([r["d_gom"] for r in rows if r["d_gom"] is not None])
        if len(dg):
            print("Reading: against the standard Gompertz curve: fibres %d, mean %.4f, lower bound %.4f, fraction > 0 %.3f" % (len(dg), float(dg.mean()), boot_lower(dg), float(np.mean(dg > 0))))
        for name, m in (("low spontaneous", [r["spont"] < 0.5 for r in rows]), ("high spontaneous", [r["spont"] >= 0.5 for r in rows])):
            m = np.array(m)
            if m.any():
                print("Reading: %s fibres %d, mean d %.4f, fraction d > 0 %.3f" % (name, int(m.sum()), float(dd[m].mean()), float(np.mean(dd[m] > 0))))
    v, why = decide(rows, n_fail)
    print(why)
    print("VERDICT q1957 (Amendment 2303, H_asym): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
