"""q1899 = q1893 with the run set by environment (see the comment under HERE). Original docstring of q1893:
q1893 (15 Oct 2026): A_25 of the amplitude-constrained Poisson channel, the count change K = 24 -> 25.

REGISTRATION (PREREG_Q32.md, Amendment 126, 9 Sep 2026 12:40): A_25 = 412.43 (primary, chain-length corrected),
also 412.48 (E_inf law, Amendment 127) and 412.35 (uncorrected); tolerance 0.15 either way. The first attempt was
VOID (Amendment 250: Newton from a 6.5 per cent rescale of the K = 24 state landed outside its basin). Its two
recorded next tries were a Gaussian-mapped guess or a walk in small steps from A = 387.237. This script does the
walk, with the maximiser of Amendments 622-623 (q480_ascent.py, trust-region ascent of I on a fixed support,
160 bits) in place of the root-finders that stalled.

METHOD.
 1. Start: the stored 40-digit K = 24 state results_q66_K24_A387.237_exact40.json (origin atom, 22 interior
    atoms, wall atom at A). Alphabet truncated at Ymax = A + 16 sqrt(A) + 60, as in q470/q480.
 2. Walk up in A. Seed at the new A by linear extrapolation in A of (x_k / A, w_k) from the last two accepted
    states (rescale only for the first step). Run q480 (plain mode, 160 bits, up to 80 steps, target max |g|
    below 1e-20). Accept when max |g| < 1e-16, all weights > 1e-12 and positions ordered; otherwise halve the
    step (step starts at 0.5, grows by 1.5 up to 3.0; below 1e-3 the walk is STUCK).
 3. At every accepted state: off-support excess V(A) = max over [0, A] of D(x) - D(0), with D the divergence
    of the output law at input x from the output law of the state. Grid of 60 points in every gap between
    atoms (double precision), Brent refinement of the best gap's peak (double), then D at the peak in mpmath
    at 40 digits from the 40-digit state. At a stationary K-atom state D = D(0) at every atom, so V < 0 means
    the state is the capacity-achieving input (KKT) and V > 0 means a 25th atom is due.
 4. When V changes sign between two accepted states: Illinois regula falsi on V(A), each new A seeded from the
    nearer bracket state, until the bracket is narrower than 2e-6. A_25 = linear interpolation of V in the
    final bracket.
 5. A weight below 1e-12 on the way is a DEATH (a different event); the walk stops and records it. A cap at
    A = 430 ends the walk if V never changes sign.
CHECKPOINT AND RESUME. Directory p2k25/: every accepted state p2k25/state_A<A>.json, the list p2k25/walk.json,
and at the end p2k25/A25_result.json. A restart reads walk.json and continues from the last accepted states.

BLIND READOUT. The walker prints counts and phases only (no A, no V); the verdict is the last line of
q1894_a25_verdict.py, run on p2k25/A25_result.json.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
from mpmath import mp, mpf

HERE = os.path.dirname(os.path.abspath(__file__))
# CONTROL MODE (env Q1893_CONTROL=1): the same instrument re-locates the MEASURED A_24 = 385.381 (newborn-weight
# extrapolation, PREREG_Q32.md near line 2787) on the K = 23 branch from the stored state at A = 378.000.
# q1899 (15 Oct 2026): q1893 copied as text with the run set by environment (Amendment 2261): Q1899_START (state file),
# Q1899_OUT (directory), Q1899_CAP (largest A), Q1899_H0 (first step). DELTA_T is 0.08 here (0.15 in q1893), so that a
# split birth with an offset below 0.15 Fisher units at larger K is also seen. Nothing else differs from q1893.
CONTROL = False
OUT = os.path.join(HERE, os.environ["Q1899_OUT"])
WALK = os.path.join(OUT, "walk.json")
RESULT = os.path.join(OUT, "A25_result.json")
START = os.path.join(HERE, os.environ["Q1899_START"])
H0, HMAX, HMIN = float(os.environ.get("Q1899_H0", "0.5")), 3.0, 1e-3
ACAP = float(os.environ["Q1899_CAP"])
BRACKET_TOL = 2e-6
GACC, WMIN = 1e-16, 1e-12
DELTA_T = mpf("0.08")
mp.dps = 40
say = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def ymax(A):
    return int(A + 16 * np.sqrt(A) + 60)


def load(path):
    d = json.load(open(path))
    d = d[-1] if isinstance(d, list) else d
    return {"A": str(d["A"]), "K": int(d["K"]), "x": [str(t) for t in d["x"]], "w": [str(t) for t in d["w"]]}


def save_state(st, path):
    json.dump(st, open(path, "w"), indent=1)


def excess(st, npts=24):
    """V = max over [0, A] of D(x) - D(0), all in mpmath at 40 digits (the bulk ripple is of order 1e-13, below the
    double-precision noise of D, so no double-precision ranking is used). Grid of npts points in every gap, then
    golden-section refinement of the three highest gaps. Returns (V as a 30-digit string, peak x, gap index)."""
    A = float(st["A"]); X = [mpf(t) for t in st["x"]]; W = [mpf(t) for t in st["w"]]; N = ymax(A) + 1
    LG = [mp.loggamma(y + 1) for y in range(N)]
    lw = [mp.log(t) for t in W]
    mlogQ = []
    for y in range(N):
        terms = [lw[0]] if y == 0 else []
        for i in range(1, len(X)):
            terms.append(lw[i] - X[i] + y * mp.log(X[i]) - LG[y])
        mx = max(terms); mlogQ.append(mx + mp.log(sum(mp.exp(t - mx) for t in terms)))
    mC = -mlogQ[0]

    def E(xx):
        lx = mp.log(xx); s = mpf(0); sq = float(xx) ** 0.5
        for y in range(max(0, int(float(xx) - 18 * sq - 20)), min(N, int(float(xx) + 18 * sq + 20) + 1)):
            l = -xx + y * lx - LG[y]; s += mp.exp(l) * (l - mlogQ[y])
        return s - mC
    # V = max of E over [0, A] minus the Fisher neighbourhoods |t - t_k| < DELTA_T of the atoms (t = 2 sqrt(x)).
    # The births of the Poisson channel are tangencies 0.18-0.20 (split) or 0.56-0.61 (insertion) Fisher units from
    # the nearest atom (Amendment 129), so with DELTA_T = 0.15 every birth lies in the searched set; V is continuous
    # in A and its first zero on the K = 24 branch is the count change.
    T = [2 * mp.sqrt(t) for t in X]
    peaks = []
    for k in range(len(X) - 1):
        ta, tb = T[k] + DELTA_T, T[k + 1] - DELTA_T
        if tb <= ta:
            continue
        g = [ta + (tb - ta) * j / (npts - 1) for j in range(npts)]
        v = [E((s / 2) ** 2) for s in g]
        j = max(range(npts), key=lambda i: v[i]); peaks.append((v[j], j, k, ta, tb))
    peaks.sort(key=lambda z: z[0], reverse=True)
    best = None
    gr = (mp.sqrt(5) - 1) / 2
    for vj, j, k, ta, tb in peaks[:3]:
        hstep = (tb - ta) / (npts - 1)
        if j == 0 or j == npts - 1:                 # maximum on the edge of the excluded neighbourhood
            sp, vp = ta + hstep * j, vj
        else:
            a_, b_ = ta + hstep * (j - 1), ta + hstep * (j + 1)
            Et = lambda s: E((s / 2) ** 2)
            c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_); fc, fd = Et(c_), Et(d_)
            for _ in range(45):
                if fc > fd:
                    b_, d_, fd = d_, c_, fc; c_ = b_ - gr * (b_ - a_); fc = Et(c_)
                else:
                    a_, c_, fc = c_, d_, fd; d_ = a_ + gr * (b_ - a_); fd = Et(d_)
            sp, vp = (c_, fc) if fc > fd else (d_, fd)
        if best is None or vp > best[0]:
            best = (vp, float((sp / 2) ** 2), k)
    return mp.nstr(best[0], 30), best[1], best[2]


def ascent(seed, tag):
    sp = os.path.join(OUT, "seed_%s.json" % tag)
    save_state(seed, sp)
    env = dict(os.environ, OUTTAG="_asc", MODE="plain", PYTHONIOENCODING="utf-8")
    env.pop("TIGHT_SOFT_EXIT", None)
    rel = os.path.relpath(sp, HERE)
    with open(os.path.join(OUT, "ascent_%s.log" % tag), "w", encoding="utf-8") as lf:
        rc = subprocess.run([sys.executable, os.path.join(HERE, "q480_ascent.py"), rel, "160", "80", "0.02", "1e-20"],
                            cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=7200).returncode
    op = sp.replace(".json", "_asc.json")
    if rc != 0 or not os.path.exists(op):
        return None, {"rc": rc, "reason": "no output"}
    d = json.load(open(op))
    st = {"A": d["A"], "K": d["K"], "x": d["x"], "w": d["w"]}
    xs = [mpf(t) for t in st["x"]]; ws = [mpf(t) for t in st["w"]]
    ok = (float(d["gmax"]) < GACC and min(ws) > WMIN and all(xs[i + 1] > xs[i] for i in range(len(xs) - 1)))
    info = {"rc": rc, "gmax": float(d["gmax"]), "resF": float(d["residual"]), "minw": float(min(ws))}
    return (st if ok else None), info


def seed_at(Anew, s1, s0=None):
    """Seed at Anew from s1 (and s0): extrapolate x/A and w linearly in A; renormalise weights."""
    An = mpf(Anew); A1 = mpf(s1["A"]); u1 = [mpf(t) / A1 for t in s1["x"]]; w1 = [mpf(t) for t in s1["w"]]
    if s0 is None:
        u, w = u1, w1
    else:
        A0 = mpf(s0["A"]); u0 = [mpf(t) / A0 for t in s0["x"]]; w0 = [mpf(t) for t in s0["w"]]
        f = (An - A1) / (A1 - A0)
        u = [a + f * (a - b) for a, b in zip(u1, u0)]; w = [a + f * (a - b) for a, b in zip(w1, w0)]
    if min(w) <= 0:
        u, w = u1, w1
    sw = sum(w); w = [t / sw for t in w]
    x = [mpf(0)] + [t * An for t in u[1:-1]] + [An]
    return {"A": mp.nstr(An, 40), "K": s1["K"], "x": [mp.nstr(t, 40) for t in x], "w": [mp.nstr(t, 40) for t in w]}


def record(walk, st, info, tag):
    path = os.path.join(OUT, "state_%s.json" % tag)
    save_state(st, path)
    V, xp, k = excess(st)
    walk["states"].append({"A": st["A"], "file": os.path.relpath(path, HERE), "V": V, "xpeak": xp, "gap": k, **info})
    json.dump(walk, open(WALK, "w"), indent=1)
    return float(V)


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "walker.pid"), "w").write(str(os.getpid()))
    if os.path.exists(RESULT):
        say("result exists; nothing to do"); return 0
    walk = json.load(open(WALK)) if os.path.exists(WALK) else {"states": [], "h": H0, "phase": "walk", "events": []}
    if not walk["states"]:
        st = load(START)
        record(walk, st, {"rc": 0, "gmax": None, "resF": None, "minw": float(min(mpf(t) for t in st["w"]))}, "start")
        say("start state recorded (count 1)")
    S = walk["states"]
    while True:
        S = walk["states"]
        if walk["phase"] == "walk":
            last = S[-1]
            if float(last["V"]) > 0 and len(S) >= 2 and float(S[-2]["V"]) < 0:
                walk["phase"] = "root"; walk["lo"], walk["hi"] = len(S) - 2, len(S) - 1; walk["side"] = 0
                json.dump(walk, open(WALK, "w"), indent=1); say("sign change seen; root phase starts"); continue
            if float(last["V"]) > 0 and len(S) == 1:
                res = {"status": "start_not_optimal"}; json.dump(res, open(RESULT, "w"), indent=1); say("start state has V > 0"); return 0
            if float(last["A"]) >= ACAP:
                json.dump({"status": "no_change_below_cap", "cap": ACAP, "states": len(S)}, open(RESULT, "w"), indent=1)
                say("cap reached without a sign change"); return 0
            h = walk["h"]
            s1 = load(os.path.join(HERE, last["file"])); s0 = load(os.path.join(HERE, S[-2]["file"])) if len(S) >= 2 else None
            Anew = min(float(s1["A"]) + h, ACAP)
            tag = "%.9f" % Anew
            st, info = ascent(seed_at(Anew, s1, s0), tag)
            if st is None:
                walk["events"].append({"A_try": tag, "info": info}); walk["h"] = h / 2
                json.dump(walk, open(WALK, "w"), indent=1)
                say("ascent rejected (count %d); step halved" % len(walk["events"]))
                if walk["h"] < HMIN:
                    json.dump({"status": "stuck", "states": len(S), "events": walk["events"][-5:]}, open(RESULT, "w"), indent=1)
                    say("STUCK"); return 0
                continue
            if info["minw"] < 1e-8:
                walk["events"].append({"A_try": tag, "info": info, "note": "small weight"})
            record(walk, st, info, tag); walk["h"] = min(h * 1.5, HMAX); json.dump(walk, open(WALK, "w"), indent=1)
            say("accepted state count %d" % len(walk["states"]))
        else:
            lo, hi = S[walk["lo"]], S[walk["hi"]]
            Alo, Ahi = mpf(lo["A"]), mpf(hi["A"]); Vlo, Vhi = mpf(lo["V"]), mpf(hi["V"])
            walk.setdefault("eff", [lo["V"], hi["V"]])
            if Ahi - Alo < BRACKET_TOL:
                A25 = Alo - Vlo * (Ahi - Alo) / (Vhi - Vlo)
                res = {"status": "located", "A25": mp.nstr(A25, 20), "lo": lo, "hi": hi,
                       "peak_t": float(2 * np.sqrt(hi["xpeak"])), "n_states": len(S), "n_rejected": len(walk["events"])}
                json.dump(res, open(RESULT, "w"), indent=1); say("located; result written"); return 0
            wl, wh = mpf(walk["eff"][0]), mpf(walk["eff"][1])          # Illinois: effective end values
            Anew = Alo - wl * (Ahi - Alo) / (wh - wl)
            Anew = min(max(Anew, Alo + (Ahi - Alo) * mpf("0.02")), Ahi - (Ahi - Alo) * mpf("0.02"))
            near = lo if (Anew - Alo) < (Ahi - Anew) else hi
            s1 = load(os.path.join(HERE, near["file"]))
            tag = mp.nstr(Anew, 15)
            st, info = ascent(seed_at(float(Anew), s1), tag)
            if st is None:
                json.dump({"status": "stuck_in_root", "info": info}, open(RESULT, "w"), indent=1); say("STUCK in root phase"); return 0
            V = record(walk, st, info, tag)
            idx = len(walk["states"]) - 1
            if V < 0:
                walk["lo"] = idx; walk["eff"][0] = st_V = walk["states"][idx]["V"]
                if walk["side"] == -1: walk["eff"][1] = mp.nstr(mpf(walk["eff"][1]) / 2, 30)
                walk["side"] = -1
            else:
                walk["hi"] = idx; walk["eff"][1] = walk["states"][idx]["V"]
                if walk["side"] == 1: walk["eff"][0] = mp.nstr(mpf(walk["eff"][0]) / 2, 30)
                walk["side"] = 1
            json.dump(walk, open(WALK, "w"), indent=1)
            say("root step done (count %d)" % len(walk["states"]))


if __name__ == "__main__":
    sys.exit(main())
