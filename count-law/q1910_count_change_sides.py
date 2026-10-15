"""q1910 = q1896 copied as text and set by arguments: py q1910_count_change_sides.py RUNDIR A_BELOW A_ABOVE. The two certified sides of
a located count change (RUNDIR/A25_result.json from q1899), states and certificates in RUNDIR/sides/. Original docstring of q1896:
q1896 (15 Oct 2026): the two certified sides of the count change A_25 = 412.4271 (q1893/q1894, Amendment 2258).
 K = 24 side: the ascent of q480 at A = 412.40 seeded from the K = 24 bracket state of p2k25 (lo).
 K = 25 side: at A = 412.46 a 25th atom is inserted at the birth point of the K = 24 bracket state (hi, xpeak), with
   weight 0.003 taken from its two neighbours in proportion; the ascent of q480 on the 25-atom support follows.
Both states then go to the ball-arithmetic certificate q470_support_certificate.py (128 bits, 8 pieces), whose own
last line states the result. Output p2k25/sides/. RESUME: rerun; finished steps are skipped."""
import json
import os
import subprocess
import sys

from mpmath import mp, mpf

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import q1893_a25_walk as W

mp.dps = 40
RUNDIR = sys.argv[1]; A_LO = float(sys.argv[2]); A_HI = float(sys.argv[3])
OUT = os.path.join(HERE, RUNDIR, "sides")
W.OUT = OUT


def insert(st, xnew, Anew, eps):
    A1 = mpf(st["A"]); An = mpf(Anew)
    x = [mpf(t) * An / A1 for t in st["x"]]; w = [mpf(t) for t in st["w"]]
    xn = mpf(repr(xnew)) * An / A1
    k = max(i for i in range(len(x)) if x[i] < xn)
    e = mpf(eps); a, b = w[k], w[k + 1]
    w[k] -= e * a / (a + b); w[k + 1] -= e * b / (a + b)
    x.insert(k + 1, xn); w.insert(k + 1, e)
    return {"A": mp.nstr(An, 40), "K": st["K"] + 1, "x": [mp.nstr(t, 40) for t in x], "w": [mp.nstr(t, 40) for t in w]}


def certify(path, tag):
    log = os.path.join(OUT, "cert_%s.log" % tag)
    if os.path.exists(log) and "VERDICT" in open(log, encoding="utf-8", errors="replace").read():
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); env.pop("TIGHT_SOFT_EXIT", None)
    with open(log, "w", encoding="utf-8") as lf:
        subprocess.run([sys.executable, os.path.join(HERE, "q470_support_certificate.py"), "128", "8", os.path.relpath(path, HERE)],
                       cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = json.load(open(os.path.join(HERE, RUNDIR, "A25_result.json")))
    lo = W.load(os.path.join(HERE, r["lo"]["file"])); hi = W.load(os.path.join(HERE, r["hi"]["file"]))
    out = {}
    K = lo["K"]
    p24 = os.path.join(OUT, "state_K%d_A%.4f.json" % (K, A_LO))
    if not os.path.exists(p24):
        st, info = W.ascent(W.seed_at(A_LO, lo), "K%d_%.4f" % (K, A_LO)); assert st is not None, info
        W.save_state(st, p24); out["below"] = info
    p25 = os.path.join(OUT, "state_K%d_A%.4f.json" % (K + 1, A_HI))
    if not os.path.exists(p25):
        st, info = W.ascent(insert(hi, r["hi"]["xpeak"], A_HI, "0.003"), "K%d_%.4f" % (K + 1, A_HI)); assert st is not None, info
        W.save_state(st, p25); out["above"] = info
    for p in (p24, p25):
        st = W.load(p); V, xp, k = W.excess(st)
        out[os.path.basename(p)] = {"V": V, "minw": float(min(mpf(t) for t in st["w"])), "K": st["K"]}
    json.dump(out, open(os.path.join(OUT, "sides.json"), "w"), indent=1)
    certify(p24, "below"); certify(p25, "above")
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
