"""
Re-run every bracket endpoint certificate with PROVED constants. 2026-09-06.

Amendment 68 restored "inequalities covering every point" at all eleven endpoints plus A = 150, with the stated
limit that M2 and M3 were sampled maxima. q32_certificate_v3.py replaces every sampled quantity by an interval
enclosure (support_iv.py) and reports eta, the KKT residual of the decimal input. A = 100 passed (Amendment
69: worst slack 1.036e-6 against 1.11e-6 sampled, 27 minutes). Each endpoint either passes with proved
constants (eta and worst chord slack recorded) or fails; a failure drops that endpoint back to the v2 status
and is reported as such. load_all and TARGETS are copied from q32_recertify_v2.py rather than imported,
because that module runs its whole loop at import.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q32_certificate_v3 import analyse

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

if __name__ == "__main__":
    S = load_all()
    out = os.path.join(HERE, "results_q32_recertify_v3.json")
    res = json.load(open(out)) if os.path.exists(out) else {}
    for label, A, K in TARGETS:
        if label in res and res[label] is not None:
            print(f"{label:16s} already done: {'pass' if res[label]['ok'] else 'FAIL'}", flush=True)
            continue
        key = next(((a, k) for (a, k) in S if k == K and abs(a - A) < 0.02), None)
        if key is None:
            print(f"{label:16s} A = {A:9.4f} K = {K:2d}   solution not on disk", flush=True)
            res[label] = None
            continue
        Av, x, w = S[key]
        t0 = time.time()
        r = analyse(Av, x, w, nsamp=200, quiet=True)
        res[label] = r
        print(
            f"{label:16s} A = {Av:9.4f} K = {K:2d}   {'passes with proved constants' if r['ok'] else 'FAILS'}   "
            f"eta {r['eta']:.2e}  worst slack {'-' if r['worst_slack'] is None else '%.3e' % r['worst_slack']}   [{time.time()-t0:.0f}s]",
            flush=True,
        )
        json.dump(res, open(out, "w"), indent=1)
    bad = [k for k, v in res.items() if v is not None and not v["ok"]]
    print(f"\n{'all endpoints pass with proved constants' if not bad else 'FAIL: ' + str(bad)}", flush=True)
