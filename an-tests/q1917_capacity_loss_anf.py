"""q1917 (15 Oct 2026): route A of paper 2 on gerbil auditory-nerve fibres (task P2ANFA), registered in Amendment 2267 before any
loss of these data was computed. Same quantity and instrument as q1913 (Amendment 2265; functions imported unchanged), new data.

Data: DANDI 001262 (Heeringa 2024, Sci. Data 11:411), census q1895 (data_ext/dandi001262/census.json), recording type RLF
(tone bursts at the fibre's best frequency, one condition per sound level). Spike times are relative to the trial start.
Window (fixed before any loss is read, equal to the A1 window of q1913): spikes with 0 <= t < 0.050 s.
Conditions: the RLF_ABI<level> keys (the RLF_silent key is excluded from the conditions; it enters one Reading only).
Selection: one recording per NWB file; at least 8 levels with at least 5 repeats at every level; A >= 0.05 (as q1913).
Per recording: r_c = mean count in level c, lam = min_c r_c, A = max_c r_c - lam;
  I_meas = I(C; Y) with C uniform over the levels and Y | c ~ Poisson(r_c);
  Cap = capacity of the Poisson channel with input in [lam, lam + A] (q1913.capacity, gap 1e-7 nats, upper bound used);
  loss = 1 - I_meas / Cap.
VERDICT (H_anf, registered): PASS if the median loss over the selected recordings is >= 0.2 (the prediction written in the queue
from the A1 result), FAIL otherwise; VOID if fewer than 20 recordings are selected or any capacity run does not reach its gap;
INCOMPLETE if the census file is missing.
Readings printed: number selected; quartiles of loss, A, lam and of the number of levels; the median loss with RLF_silent added as
a condition; the fraction of recordings with loss < 0.01 (the A1 bar of H_flat).
Usage: py q1917_capacity_loss_anf.py [--registration-run]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1913_capacity_loss_a1_v2 as Q

CENSUS = os.path.join(HERE, "data_ext", "dandi001262", "census.json")
OUT = os.path.join(HERE, "paper_neuron", "q1917_rows.json")
W = 0.050
MIN_LEVELS, MIN_REPS, MIN_A = 8, 5, 0.05


def level_means(rec, with_silent=False):
    sp = rec.get("spikes", {})
    keys = [k for k in sp if k.startswith("RLF_ABI") or (with_silent and k == "RLF_silent")]
    out = {}
    for k in keys:
        trials = sp[k]
        out[k] = (len(trials), float(np.mean([sum(1 for t in tr if 0.0 <= t < W) for tr in trials])) if trials else 0.0)
    return out


def select(rec):
    m = level_means(rec)
    if len(m) < MIN_LEVELS or min(n for n, _ in m.values()) < MIN_REPS:
        return None
    r = np.array([v for _, v in m.values()])
    lam = float(r.min()); A = float(r.max() - lam)
    return None if A < MIN_A else r


def row(rec, r):
    lam = float(r.min()); A = float(r.max() - lam)
    Cap, _, ok = Q.capacity(lam, A)
    I = Q.mutual_info(r, np.full(len(r), 1.0 / len(r)))
    rs = np.array([v for _, v in level_means(rec, with_silent=True).values()])
    Is = Q.mutual_info(rs, np.full(len(rs), 1.0 / len(rs)))
    Caps, _, oks = Q.capacity(float(rs.min()), float(rs.max() - rs.min()))
    return {"path": rec["path"], "levels": len(r), "lam": lam, "A": A, "I": I, "Cap": Cap, "ok": bool(ok and oks),
            "loss": 1 - I / Cap, "loss_silent": 1 - Is / Caps}


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 recordings"
    if not all(x["ok"] for x in rows):
        return "VOID", "a capacity run did not reach its gap"
    med = float(np.median([x["loss"] for x in rows]))
    return ("PASS" if med >= 0.2 else "FAIL"), "median loss %.4f over %d recordings (bar >= 0.2)" % (med, len(rows))


def registration_run():
    """Synthetic records only. (a) an on-off fibre: 12 levels, 6 at rate 0.2 and 6 at rate 3.2, counts exactly equal to the rate
    pattern in expectation (spike times placed inside the window) -> loss small; (b) an evenly graded fibre on the same range ->
    loss larger; (c) a record with 7 levels and one with 4 repeats are rejected by select(); spikes at t >= 0.05 are not counted;
    (d) decide() gives PASS on 30 rows with loss 0.3, FAIL with loss 0.1, VOID on 10 rows and on a row with ok False."""
    ok = True

    def rec_from_rates(rates, reps=10, extra_late=3):
        sp = {}
        for i, rr in enumerate(rates):
            n = int(round(rr * reps))
            trials = [[] for _ in range(reps)]
            for j in range(n):
                trials[j % reps].append(0.001 + 0.04 * (j / max(n, 1)))
            for tr in trials:
                tr.extend([0.06 + 0.01 * q for q in range(extra_late)])
            sp["RLF_ABI%d" % (3 * i)] = trials
        sp["RLF_silent"] = [[0.1] for _ in range(reps)]
        return {"path": "synthetic", "spikes": sp}

    ra = rec_from_rates([0.2] * 6 + [3.2] * 6)
    rb = rec_from_rates(list(np.linspace(0.2, 3.2, 12)))
    xa, xb = select(ra), select(rb)
    ok &= xa is not None and xb is not None
    if ok:
        Ra, Rb = row(ra, xa), row(rb, xb)
        print("control (a) on-off fibre: means %s, loss %.4f" % (np.round(xa[[0, -1]], 3).tolist(), Ra["loss"]))
        print("control (b) graded fibre: loss %.4f" % Rb["loss"])
        ok &= abs(xa.min() - 0.2) < 1e-12 and abs(xa.max() - 3.2) < 1e-12   # late spikes excluded
        ok &= Ra["loss"] < Rb["loss"] and Ra["ok"] and Rb["ok"]
    r7 = rec_from_rates([0.2, 3.2] * 3 + [1.0]); r4 = rec_from_rates([0.2] * 6 + [3.2] * 6, reps=4)
    ok &= select(r7) is None and select(r4) is None
    print("control (c) 7 levels rejected: %s; 4 repeats rejected: %s" % (select(r7) is None, select(r4) is None))
    hi = [{"loss": 0.3, "ok": True}] * 30; lo = [{"loss": 0.1, "ok": True}] * 30
    bad = hi[:29] + [{"loss": 0.3, "ok": False}]
    d = (decide(hi)[0], decide(lo)[0], decide(hi[:10])[0], decide(bad)[0])
    print("control (d) decide(): %s %s %s %s (expected PASS FAIL VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1917: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1917 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CENSUS):
        print("VERDICT q1917: INCOMPLETE (census missing)"); return 3
    recs = [x for x in json.load(open(CENSUS)) if "error" not in x]
    rows = []
    for rec in recs:
        r = select(rec)
        if r is not None:
            rows.append(row(rec, r))
            if len(rows) % 50 == 0:
                print("progress: %d recordings done" % len(rows), flush=True)
    json.dump(rows, open(OUT, "w"))
    if rows:
        L = np.array([x["loss"] for x in rows])
        q = lambda k: tuple(np.percentile([x[k] for x in rows], [25, 50, 75]))
        print("Reading: selected %d of %d records" % (len(rows), len(recs)))
        print("Reading: loss quartiles %.4f %.4f %.4f; A quartiles %.3f %.3f %.3f; lam quartiles %.3f %.3f %.3f; levels quartiles %g %g %g"
              % (*q("loss"), *q("A"), *q("lam"), *q("levels")))
        print("Reading: median loss with RLF_silent as a condition %.4f" % float(np.median([x["loss_silent"] for x in rows])))
        print("Reading: fraction with loss < 0.01: %.3f" % float(np.mean(L < 0.01)))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1917 (Amendment 2267, H_anf): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
