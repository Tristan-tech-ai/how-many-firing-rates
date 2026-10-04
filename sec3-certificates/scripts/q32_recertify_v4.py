"""
Exact-point certificate (q32_certificate_v4.py) at every bracket endpoint. 2026-09-06.

Inputs: results_q32_exact40_all.json (40-digit KKT points with exact boundary atoms, and the Krawczyk ladder
per endpoint). For each endpoint whose Krawczyk test passed, the certificate is run with the parameters as
the smallest passing box; a pass proves D(x; F*) <= C(F*) on [0, A] for the exact KKT point F*, hence
N*(A) = K at that A (with the cited KKT-sufficiency and uniqueness theorems). Output:
results_q32_recertify_v4.json; resumable.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q32_certificate_v4 import analyse_box

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    src = json.load(open(os.path.join(HERE, "results_q32_exact40_all.json")))
    out_p = os.path.join(HERE, "results_q32_recertify_v4.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for label, rec in src.items():
        if label in res and res[label] is not None:
            print(f"{label:16s} already done: {'PROVED' if res[label]['ok'] else 'NOT PROVED'}", flush=True)
            continue
        if rec.get("rho") is None:
            print(f"{label:16s} Krawczyk did not pass at any rho; skipped", flush=True)
            res[label] = None
            continue
        t0 = time.time()
        r = analyse_box(rec["A"], rec["x"], rec["w"], rec["rho"], quiet=True)
        r["krawczyk"] = rec["krawczyk"][rec["rho"]]
        res[label] = r
        print(
            f"{label:16s} A = {rec['A']:9.4f} K = {rec['K']:2d} rho = {rec['rho']}   {'PROVED for the exact point' if r['ok'] else 'NOT PROVED'}   "
            f"worst slack {'-' if r['worst_slack'] is None else '%.3e' % r['worst_slack']}   [{time.time()-t0:.0f}s]",
            flush=True,
        )
        json.dump(res, open(out_p, "w"), indent=1)
    bad = [k for k, v in res.items() if v is not None and not v["ok"]]
    print(
        f"\n{'N*(A) = K PROVED at every endpoint' if not bad else 'NOT PROVED at: ' + str(bad)}", flush=True
    )
