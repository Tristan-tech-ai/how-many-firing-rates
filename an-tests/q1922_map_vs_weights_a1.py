"""q1922 (15 Oct 2026): task P2BARLOW, registered in Amendment 2269 before any output was read. Separates the rate map from the
stimulus weights in route A (who chooses the input distribution).

Per A1 unit (same cache, window and selection as q1913; the interval capacity Cap is read from paper_neuron/q1913_rows.npy, written
by the q1913 run of Amendment 2266, after a control that the recomputed lam and A of every unit equal the stored ones):
  r_c    = mean count in the 5 tone conditions (the measured rate set);
  I_unif = I(C; Y) with uniform weights (route A, stored as I);
  C_set  = capacity of the Poisson channel restricted to inputs in the measured rate set, weights free (Blahut-Arimoto on 5 points,
           stopped when max_c D(c) - I < 1e-10 nats; the upper bound max_c D(c) is used);
  loss_free = 1 - C_set/Cap;  share = (C_set - I_unif)/(Cap - I_unif), the part of the route A loss due to the uniform weights.
Expectation stated before the run: both ends lam and lam + A are measured rates, and for A < 3.3679 (160 of 161 units) the
capacity-achieving input sits on {lam, lam + A} when lam = 0 (with lam > 0 this is expected but not proved here), so loss_free is
expected near 0; the test checks the decomposition, and the share is the reading of interest.
VERDICT (H_map, registered): PASS if the median loss_free over the units is below 0.02, FAIL otherwise; VOID if the control on lam
and A fails, fewer than 20 units, or a Blahut-Arimoto run misses its gap; INCOMPLETE if the cache or the stored rows are missing.
Readings: quartiles of loss_free and of share; number of units with loss_free > 0.05.
Usage: py q1922_map_vs_weights_a1.py [--registration-run]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1913_capacity_loss_a1_v2 as Q

ROWS = os.path.join(HERE, "paper_neuron", "q1913_rows.npy")
GAP = 1e-10


def cap_set(rates, itmax=2000000):
    x = np.asarray(rates, float)
    ymax = int(x.max() + 12 * np.sqrt(x.max() + 1) + 30)
    P = Q.pois_matrix(x, ymax)
    p = np.full(len(x), 1.0 / len(x))
    for _ in range(itmax):
        D, I, U = Q._gap_stats(P, p)
        if U - I < GAP:
            return U, True
        p = p * np.exp(D); p /= p.sum()
    return U, False


def unit_means(cache=Q.CACHE):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    ends = np.append(sidx[1:], len(sp)); conds = np.unique(tt); out = {}
    for k in range(len(sidx)):
        s = sp[sidx[k]:ends[k]]
        c = np.searchsorted(s, mo + Q.W) - np.searchsorted(s, mo)
        out[k] = np.array([c[tt == u].mean() for u in conds])
    return out


def decide(rows, control_ok=True):
    if not control_ok:
        return "VOID", "control on lam and A failed"
    if len(rows) < 20:
        return "VOID", "fewer than 20 units"
    if not all(r["ok"] for r in rows):
        return "VOID", "a Blahut-Arimoto run missed its gap"
    med = float(np.median([r["loss_free"] for r in rows]))
    return ("PASS" if med < 0.02 else "FAIL"), "median loss_free %.5f over %d units (bar < 0.02)" % (med, len(rows))


def registration_run():
    ok = True
    # (a) two rates at the ends with A = 0.4, lam = 0: free weights reach the interval capacity
    Cap, _, _ = Q.capacity(0.0, 0.4)
    Cs, conv = cap_set([0.0, 0.1, 0.2, 0.3, 0.4])
    la = 1 - Cs / Cap
    # (b) a rate set that misses the top end loses capacity
    Cs2, conv2 = cap_set([0.0, 0.1, 0.2])
    lb = 1 - Cs2 / Cap
    print("control (a) graded set with both ends, A = 0.4: loss_free %.2e; (b) set without the top end: loss_free %.3f" % (la, lb))
    ok &= conv and conv2 and abs(la) < 1e-6 and lb > 0.1
    good = [{"loss_free": 0.001, "ok": True}] * 30; bad = [{"loss_free": 0.05, "ok": True}] * 30
    d = (decide(good)[0], decide(bad)[0], decide(good[:5])[0], decide(good, control_ok=False)[0])
    print("control decide(): %s %s %s %s (expected PASS FAIL VOID VOID); INCOMPLETE is returned by main() when inputs are missing" % d)
    ok &= d == ("PASS", "FAIL", "VOID", "VOID")
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1922: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1922 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not (os.path.exists(Q.CACHE) and os.path.exists(ROWS)):
        print("VERDICT q1922: INCOMPLETE (cache or stored rows missing)"); return 3
    stored = np.load(ROWS, allow_pickle=True); means = unit_means()
    control_ok = True; rows = []
    for s in stored:
        r = means[s["unit"]]
        lam, A = float(r.min()), float(r.max() - r.min())
        control_ok &= abs(lam - s["lam"]) < 1e-12 and abs(A - s["A"]) < 1e-12
        Cs, conv = cap_set(r)
        rows.append({"unit": s["unit"], "A": A, "I_unif": s["I"], "C_set": Cs, "Cap": s["Cap"], "ok": conv,
                     "loss_free": 1 - Cs / s["Cap"], "share": (Cs - s["I"]) / (s["Cap"] - s["I"])})
    if rows:
        lf = np.array([x["loss_free"] for x in rows]); sh = np.array([x["share"] for x in rows])
        print("Reading: units %d; loss_free quartiles %.5f %.5f %.5f; share quartiles %.4f %.4f %.4f; units with loss_free > 0.05: %d"
              % (len(rows), *np.percentile(lf, [25, 50, 75]), *np.percentile(sh, [25, 50, 75]), int(np.sum(lf > 0.05))))
    v, why = decide(rows, control_ok)
    print(why)
    print("VERDICT q1922 (Amendment 2269, H_map): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
