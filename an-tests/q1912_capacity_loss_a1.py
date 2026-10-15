"""q1912 (15 Oct 2026): route A of paper 2 (task P2LOSS), registered in Amendment 2264 before any data output was read.

Question. The registered discrete-levels test failed (Q8): mouse A1 units use graded rates across the 5 tone conditions although
the theory makes two levels optimal at their peak counts. Does a graded map cost little information (a flat objective), or a lot?
Quantity, per unit of a1/LA11_ses1.nwb (cache a1/a1_cache.npz; same counts as q8_levels_a1.py: 50 ms from tone onset, units with
A >= 0.05, the Q8 rule): rates r_c = mean count in condition c (5 tone frequencies), lam = min_c r_c, A = max_c r_c - lam;
  I_meas = I(C; Y) with C uniform over the 5 conditions and Y | c ~ Poisson(r_c);
  Cap = capacity of the Poisson channel with input in [lam, lam + A] (Blahut-Arimoto on 401 grid points; the iteration stops when the
        upper bound max_x D(x) minus the lower bound I is below 1e-7 nats, and the upper bound is used);
  loss = 1 - I_meas/Cap.
VERDICT (H_flat, registered): PASS if loss < 0.01 for at least half of the selected units, FAIL otherwise; VOID if fewer than 20 units
are selected or any BA run does not reach its gap; INCOMPLETE if the cache is missing.
Readings printed: the median and quartiles of loss, A and lam; the theory curve loss_unif(A, lam) for the uniform continuous input on
[lam, lam + A] at A in {0.3, 1, 3, 10, 30} and lam/A in {0, 0.5}.
Usage: py q1912_capacity_loss_a1.py [--registration-run]
"""
import os
import sys
import numpy as np
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "a1", "a1_cache.npz")
W = 0.05
GAP = 1e-7


def pois_matrix(x, ymax):
    y = np.arange(ymax + 1)
    with np.errstate(divide="ignore"):
        lx = np.where(x[:, None] > 0, np.log(np.maximum(x[:, None], 1e-300)), -np.inf)
    L = -x[:, None] + y[None, :] * lx - gammaln(y + 1)[None, :]
    L[:, 0] = -x
    return np.exp(L)


def mutual_info(rates, probs):
    ymax = int(max(rates) + 12 * np.sqrt(max(rates) + 1) + 30)
    P = pois_matrix(np.asarray(rates, float), ymax)
    q = probs @ P
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(P > 0, P * (np.log(P) - np.log(q[None, :])), 0.0)
    return float(probs @ T.sum(1))


def capacity(lam, A, n=401, itmax=200000):
    x = lam + A * np.linspace(0, 1, n)
    ymax = int(lam + A + 12 * np.sqrt(lam + A + 1) + 30)
    P = pois_matrix(x, ymax)
    p = np.full(n, 1.0 / n)
    for it in range(itmax):
        q = p @ P
        with np.errstate(divide="ignore", invalid="ignore"):
            D = np.where(P > 0, P * (np.log(P) - np.log(q[None, :])), 0.0).sum(1)
        I = float(p @ D); U = float(D.max())
        if U - I < GAP:
            return U, I, True
        p = p * np.exp(D); p /= p.sum()
    return U, I, False


def unit_losses(cache=CACHE):
    z = np.load(cache, allow_pickle=True)
    sp, sidx, tt, mo = z["spikes"], z["sidx"], z["trial_type"], z["move_onset_time"]
    ends = np.append(sidx[1:], len(sp)); conds = np.unique(tt)
    rows = []
    for k in range(len(sidx)):
        s = sp[sidx[k]:ends[k]]
        c = np.searchsorted(s, mo + W) - np.searchsorted(s, mo)
        means = np.array([c[tt == u].mean() for u in conds])
        lam = float(means.min()); A = float(means.max() - lam)
        if A < 0.05:
            continue
        Cap, Ilo, ok = capacity(lam, A)
        I = mutual_info(means, np.full(len(means), 1.0 / len(means)))
        rows.append({"unit": k, "lam": lam, "A": A, "I": I, "Cap": Cap, "ok": ok, "loss": 1 - I / Cap})
    return rows


def decide(rows):
    if len(rows) < 20:
        return "VOID", "fewer than 20 units"
    if not all(r["ok"] for r in rows):
        return "VOID", "a capacity run did not reach its gap"
    frac = np.mean([r["loss"] < 0.01 for r in rows])
    return ("PASS" if frac >= 0.5 else "FAIL"), "fraction of units with loss < 0.01: %.3f (%d units)" % (frac, len(rows))


def registration_run():
    """Synthetic maps only. (a) 5 conditions placed at the capacity-achieving support with its weights approximated by repetition
    -> loss near 0; (b) an evenly graded map -> loss well above 0.01; (c) decide() on 30 copies of (a) -> PASS, of (b) -> FAIL,
    on 10 rows -> VOID."""
    ok = True
    lam, A = 0.1, 0.4
    Cap, _, conv = capacity(lam, A)
    # binary support {lam, lam + A}; best weight on the top level by a fine search
    ws = np.linspace(0.05, 0.95, 181)
    Ib = max(mutual_info([lam, lam + A], np.array([1 - w, w])) for w in ws)
    graded = lam + A * np.linspace(0, 1, 5)
    Ig = mutual_info(graded, np.full(5, 0.2))
    la, lb = 1 - Ib / Cap, 1 - Ig / Cap
    print("control (a) binary map at the extremes, best weights: loss %.5f (capacity converged %s)" % (la, conv))
    print("control (b) evenly graded 5-level map: loss %.4f" % lb)
    ok &= la < 0.01 and lb > 0.01 and conv
    ra = [{"loss": la, "ok": True}] * 30; rb = [{"loss": lb, "ok": True}] * 30
    ok &= decide(ra)[0] == "PASS" and decide(rb)[0] == "FAIL" and decide(ra[:10])[0] == "VOID"
    print("control decide(): PASS on (a), FAIL on (b), VOID on 10 rows:", decide(ra)[0], decide(rb)[0], decide(ra[:10])[0])
    return ok


def main():
    if os.environ.get("TIGHT_SOFT_EXIT"):
        print("VERDICT q1912: VOID (refuses TIGHT_SOFT_EXIT)"); os._exit(2)
    if "--registration-run" in sys.argv:
        ok = registration_run()
        print("VERDICT q1912 registration run: %s" % ("CONTROLS PASS" if ok else "CONTROLS FAIL"))
        return 0 if ok else 2
    if not os.path.exists(CACHE):
        print("VERDICT q1912: INCOMPLETE (cache missing)"); return 3
    rows = unit_losses()
    L = np.array([r["loss"] for r in rows]); As = np.array([r["A"] for r in rows]); ls = np.array([r["lam"] for r in rows])
    print("Reading: units %d; loss quartiles %.4f %.4f %.4f; A quartiles %.3f %.3f %.3f; lam quartiles %.3f %.3f %.3f"
          % (len(rows), *np.percentile(L, [25, 50, 75]), *np.percentile(As, [25, 50, 75]), *np.percentile(ls, [25, 50, 75])))
    for Aa in (0.3, 1, 3, 10, 30):
        for f in (0.0, 0.5):
            lam = f * Aa; Cap, _, _ = capacity(lam, Aa)
            xs = lam + Aa * np.linspace(0, 1, 401)
            Iu = mutual_info(xs, np.full(401, 1 / 401))
            print("Reading: theory, uniform input on [lam, lam + A], A = %g, lam/A = %.1f: loss %.4f" % (Aa, f, 1 - Iu / Cap))
    np.save(os.path.join(HERE, "paper_neuron", "q1912_rows.npy"), np.array(rows, dtype=object))
    v, why = decide(rows)
    print(why)
    print("VERDICT q1912 (Amendment 2264, H_flat): %s" % v)
    return {"PASS": 0, "FAIL": 0, "VOID": 2}[v]


if __name__ == "__main__":
    sys.exit(main())
