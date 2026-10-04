"""
Re-run piece (i) alone, with both gaps closed, at every endpoint already certified. 2026-09-06.
Both statuses: the float input (v3, Amendment 69) and the Krawczyk box (v4, Amendment 70). Pieces (ii) and
(iii) are unaffected by the fix (the exempt regions are unchanged when the right run still reaches the same
point), so a pass here completes the earlier certificates without re-running them.
Output: results_q32_piece_i_fix.json.
"""

import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mpmath import mp, mpf, mpi, iv
from support_iv import Channel
from q32_piece_i import piece_i
from q32_recertify_v3 import load_all, TARGETS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
quiet = lambda *a, **k: None

if __name__ == "__main__":
    out = {}
    S = load_all()
    ex = json.load(open(os.path.join(HERE, "results_q32_exact40_all.json")))
    items = [("A = 100 (control)", 100.0, 11)] + TARGETS
    for label, A, K in items:
        t0 = time.time()
        rec = {}
        if label.startswith("A = 100"):
            d = json.load(open(os.path.join(HERE, "results_q32_A100_mp2.json")))["11"]
            Av, x, w = 100.0, d["x"], d["w"]
            e = json.load(open(os.path.join(HERE, "results_q32_A100_exact40.json")))
        else:
            key = next(((a, k) for (a, k) in S if k == K and abs(a - A) < 0.02), None)
            Av, x, w = S[key]
            e = ex[label]
        ch = Channel(Av, x, w, 40)
        _, _, ok_f, info_f = piece_i(ch, ch.xp[-1], say=quiet)
        rec["float"] = {"ok": ok_f, **info_f}
        mp.dps = 40
        iv.dps = 40
        rho = mpf(e.get("rho") or "1e-30")
        xt = [mpf(v) for v in e["x"]]
        wt = [mpf(v) for v in e["w"]]
        Kx = len(xt)
        Xb = [iv.mpf(0)] + [mpi(xt[i] - rho, xt[i] + rho) for i in range(1, Kx - 1)] + [iv.mpf(xt[-1])]
        Wb = [mpi(v - rho, v + rho) for v in wt]
        chb = Channel(Av, Xb, Wb, 40)
        _, _, ok_b, info_b = piece_i(chb, xt[-1], say=quiet)
        rec["box"] = {"ok": ok_b, **info_b}
        out[label] = rec
        print(
            f"{label:18s} A = {Av:9.4f} K = {K:2d}  float: {'ok' if ok_f else 'FAIL'} sliver {info_f['sliver_bound']:.3f} L {info_f['left_margin']:.3f} R {info_f['right_margin']:.3f}   "
            f"box: {'ok' if ok_b else 'FAIL'} sliver {info_b['sliver_bound']:.3f} L {info_b['left_margin']:.3f} R {info_b['right_margin']:.3f}   [{time.time()-t0:.0f}s]",
            flush=True,
        )
        json.dump(out, open(os.path.join(HERE, "results_q32_piece_i_fix.json"), "w"), indent=1)
    bad = [k for k, v in out.items() if not (v["float"]["ok"] and v["box"]["ok"])]
    print(
        "piece (i) with both gaps closed holds at every endpoint" if not bad else f"FAIL at {bad}", flush=True
    )
