"""
Re-run every bracket endpoint certificate with the repaired argument. 2026-09-06.

The independent audit found that the cubic safe radius used |D'''| <= M3 on intervals up to 3.5 times larger than
where M3 was sampled, and that two gaps at A = 100 were skipped as "covered" on the strength of it. My repaired
version (q32_certificate_v2.py) makes the radius self-consistent and never skips a gap; it passes at A = 100
with worst slacks matching the auditor's own corrected copy. Every endpoint certified under the old code
(Amendments 53, 54, 58, plus A = 150) inherits the flaw and is re-run here. Any that fails drops back to a
sampled check and is reported as such.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q32_certificate_v2 import analyse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_all():
    sols = {}
    srcs = (
        ("results_q32_descend_low.json", lambda d: [(r["A"], r["K"], r["x"], r["w"]) for r in d["rows"]]),
        ("results_q32_descend_final.json", lambda d: [(r["A"], r["K"], r["x"], r["w"]) for r in d]),
        ("results_q32_descend.json", lambda d: [(r["A"], r["K"], r["x"], r["w"]) for r in d["rows"]]),
        ("results_q32_merge.json", lambda d: [(d["A"], d["K"], d["x"], d["w"])]),
        (
            "results_q32_A150.json",
            lambda d: [(150.0, len(v["x"]), v["x"], v["w"]) for k, v in d.items() if "insert" in k],
        ),
        ("results_q32_endpoints.json", lambda d: [(v["A"], v["K"], v["x"], v["w"]) for v in d.values()]),
        (
            "results_q32_continuation2.json",
            lambda d: [
                (r["A"], r["K"], r["x"], r["w"])
                for r in (d["rows"] if isinstance(d, dict) and "rows" in d else d)
                if r.get("x")
            ],
        ),
        (
            "results_q32_split.json",
            lambda d: [(float(k), v["N"], v["x"], v["w"]) for k, v in d.items() if v.get("x")],
        ),
        (
            "results_q32_probe_rest.json",
            lambda d: [
                (v["A"] if "A" in v else float(k[:-4]), v["K"], v["x"], v["w"])
                for k, v in d.items()
                if k.endswith("_sol")
            ],
        ),
    )
    for fn, get in srcs:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        try:
            for A, K, x, w in get(json.load(open(p))):
                sols[(round(float(A), 3), int(K))] = (float(A), x, w)
        except Exception as e:
            print(f"   [skip {fn}: {e}]")
    return sols


S = load_all()
TARGETS = [
    ("5 to 6 lower", 27.945, 5),
    ("5 to 6 upper", 28.362, 6),
    ("6 to 7 lower", 39.612, 6),
    ("6 to 7 upper", 40.028, 7),
    ("7 to 8 lower", 52.5555, 7),
    ("8 to 9 lower", 66.6505, 8),
    ("10 to 11 lower", 97.9375, 10),
    ("11 to 12 lower", 115.258, 11),
    ("12 to 13 lower", 133.355, 12),
    ("13 to 14 lower", 152.0, 13),
    ("A = 150", 150.0, 13),
]
res = {}
for label, A, K in TARGETS:
    key = next(((a, k) for (a, k) in S if k == K and abs(a - A) < 0.02), None)
    if key is None:
        print(f"{label:16s} A = {A:9.4f} K = {K:2d}   solution not on disk")
        res[label] = None
        continue
    Av, x, w = S[key]
    t0 = time.time()
    ok = bool(analyse(Av, x, w, nsamp=160, quiet=True))
    res[label] = ok
    print(
        f"{label:16s} A = {Av:9.4f} K = {K:2d}   {'passes with self-consistent radii' if ok else 'FAILS'}   [{time.time()-t0:.0f}s]",
        flush=True,
    )
    json.dump(res, open(os.path.join(HERE, "results_q32_recertify_v2.json"), "w"), indent=1)
bad = [k for k, v in res.items() if v is False]
print(f"\n{'all endpoints pass under the repaired argument' if not bad else 'FAIL: ' + str(bad)}")
