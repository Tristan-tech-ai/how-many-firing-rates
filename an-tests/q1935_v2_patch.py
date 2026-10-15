"""q1935_v2_patch.py (15 Oct 2026 19:2x): version 2 of the H_sqrt rule of q1935, written after the controls of version 1 failed and
before any fit to recorded data (q1935 has never been run on data). Version 1 is kept as q1935_rlf_sqrt_shape_v1.py (sha256 prefix
7bbb213449110563) and its control output as paper_neuron/q1935_regrun_v1.txt.
Audit of the version 1 controls: on data drawn from the exact shape M_ex, criterion (c) came out reversed (mean g 0.041 in the high-
spontaneous half against 0.122 in the low half) and (a) had a lower bound of -0.009; on logistic data, (b) passed (mean g 0.051). So (c)
is not a property of M_ex, (b) does not separate M_ex from the logistic, and (a) lacked power with the toy parameters (r0 0.2 or 4,
A = 8, 200 fibres). Version 2 tests the referee question directly and draws the control fibres from the fitted parameters of the 950
real fibres of the registered q1925 run (spontaneous count, peak count A_sq and sigma; the logistic scale s from the same rows), so the
controls have the size and the parameter spread of the real test. No bar is moved toward success: the new rule asks for two lower
bounds above zero, which is stronger than version 1's mean >= 0 for g."""
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "q1935_rlf_sqrt_shape.py")
s = open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(a, b):
    global s
    a2, b2 = a.replace("\n", nl), b.replace("\n", nl)
    assert s.count(a2) == 1, a[:60]
    s = s.replace(a2, b2)


rep("""VERDICT (H_sqrt, registered): PASS if all three hold:
 (a) the mean over fibres of d_ex = loglik(M_ex) - loglik(M_log) has a bootstrap 95 percent lower bound above 0 (2000 resamples of
     fibres, seed 2282), and the median fitted sigma of M_ex lies in [6, 20] dB;
 (b) the mean over fibres of g = loglik(M_ex) - loglik(M_sq) is >= 0;
 (c) the mean of g over the fibres whose spontaneous count (mean count at the lowest level) is at or above the median exceeds the
     mean of g over the fibres below the median.
FAIL otherwise.""", """VERDICT (H_sqrt, version 2, registered): PASS if both hold (bootstrap: 2000 resamples of fibres, seed 2282):
 (a) the mean over fibres of g = loglik(M_ex) - loglik(M_sq) has a bootstrap 95 percent lower bound above 0 (the exact form describes
     the fibres better than the form registered in H_L1);
 (b) the mean over fibres of d_ex = loglik(M_ex) - loglik(M_log) has a bootstrap 95 percent lower bound above 0, and the median fitted
     sigma of M_ex lies in [6, 20] dB.
FAIL otherwise. Version 1 (criteria on mean g >= 0 and on the spontaneous halves) failed its own controls; see q1935_v2_patch.py.""")
rep("""    d = np.array([r["d_ex"] for r in rows]); g = np.array([r["g"] for r in rows]); sp = np.array([r["spont"] for r in rows])
    lo = boot_lower(d); med = float(np.median([r["sigma_ex"] for r in rows]))
    hi_m = sp >= np.median(sp)
    gh, gl = float(g[hi_m].mean()), float(g[~hi_m].mean()) if (~hi_m).any() else float("nan")
    a = lo > 0 and 6.0 <= med <= 20.0
    b = float(g.mean()) >= 0.0
    c = bool((~hi_m).any()) and gh > gl
    why = ("(a) mean d_ex %.4f, bootstrap 95%% lower bound %.4f (bar > 0), median sigma_ex %.2f dB (bar [6, 20]): %s; "
           "(b) mean g %.4f (bar >= 0): %s; (c) mean g high-spontaneous half %.4f vs low half %.4f (bar high > low): %s"
           % (float(d.mean()), lo, med, a, float(g.mean()), b, gh, gl, c))
    return ("PASS" if (a and b and c) else "FAIL"), why""",
"""    d = np.array([r["d_ex"] for r in rows]); g = np.array([r["g"] for r in rows]); sp = np.array([r["spont"] for r in rows])
    lo_g, lo_d = boot_lower(g), boot_lower(d); med = float(np.median([r["sigma_ex"] for r in rows]))
    hi_m = sp >= np.median(sp)
    a = lo_g > 0
    b = lo_d > 0 and 6.0 <= med <= 20.0
    why = ("(a) mean g %.4f, bootstrap 95%% lower bound %.4f (bar > 0): %s; (b) mean d_ex %.4f, bootstrap 95%% lower bound %.4f (bar > 0), "
           "median sigma_ex %.2f dB (bar [6, 20]): %s; reading: mean g high-spontaneous half %.4f, low half %.4f"
           % (float(g.mean()), lo_g, a, float(d.mean()), lo_d, med, b, float(g[hi_m].mean()),
              float(g[~hi_m].mean()) if (~hi_m).any() else float("nan")))
    return ("PASS" if (a and b) else "FAIL"), why""")
rep("""def registration_run():
    rng = np.random.default_rng(2)
    ok = True
    # positive: M_ex data, half the fibres with a low and half with a high spontaneous count
    pos = synth(rng, 200, lambda i: (m_ex, [0.2 if i % 2 else 4.0, 8.0, 40.0, 12.0]))
    v = decide(pos, 0); print("control: M_ex data, r0 0.2 or 4.0 (200 synthetic fibres): %s, %s" % v); ok &= v[0] == "PASS"
    # negative 1: logistic data must fail (a)
    neg1 = synth(rng, 200, lambda i: (m_log, [0.2 if i % 2 else 4.0, 8.0, 40.0, 5.0]))
    v = decide(neg1, 0); print("control: M_log data: %s, %s" % v); ok &= v[0] == "FAIL"
    # negative 2: M_sq data with a high spontaneous count must fail (b)
    neg2 = synth(rng, 200, lambda i: (m_sq, [0.2 if i % 2 else 4.0, 8.0, 40.0, 12.0]))
    v = decide(neg2, 0); print("control: M_sq data: %s, %s" % v); ok &= v[0] == "FAIL"
    base = {"d_ex": 1.0, "g": 0.1, "sigma_ex": 12.0}
    good = [dict(base, spont=float(i % 2), g=0.1 + 0.2 * (i % 2)) for i in range(30)]
    flat_c = [dict(base, spont=float(i % 2), g=0.3 - 0.2 * (i % 2)) for i in range(30)]
    neg_b = [dict(base, spont=float(i % 2), g=-0.5 + 0.1 * (i % 2)) for i in range(30)]
    bad_a = [dict(r, d_ex=-1.0) for r in good]
    d = (decide(good, 0)[0], decide(bad_a, 0)[0], decide(neg_b, 0)[0], decide(flat_c, 0)[0], decide(good[:5], 0)[0],
         decide(good, 10)[0], decide(good, 0, known_ok=False)[0])
    print("control decide(): %s %s %s %s %s %s %s (expected PASS FAIL FAIL FAIL VOID VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "FAIL", "FAIL", "VOID", "VOID", "VOID")
    return ok""",
"""def registration_run():
    # version 2: control fibres carry the fitted parameters of the 950 real fibres of the registered q1925 run (r0 = spontaneous count,
    # A = A_sq, sigma; logistic scale s), mu = 40 dB, 27 levels x 10 repeats; one synthetic fibre per real fibre and per model
    if not os.path.exists(ROWS1925):
        print("control: q1925 rows missing"); return False
    par = json.load(open(ROWS1925))
    rng = np.random.default_rng(2)
    ok = True
    pos = synth(rng, len(par), lambda i: (m_ex, [par[i]["spont"], par[i]["A_sq"], 40.0, par[i]["sigma"]]))
    v = decide(pos, 0); print("control: M_ex data (%d fibres with the q1925 parameters): %s, %s" % (len(par), v[0], v[1])); ok &= v[0] == "PASS"
    neg1 = synth(rng, len(par), lambda i: (m_sq, [par[i]["spont"], par[i]["A_sq"], 40.0, par[i]["sigma"]]))
    v = decide(neg1, 0); print("control: M_sq data: %s, %s" % v); ok &= v[0] == "FAIL"
    neg2 = synth(rng, len(par), lambda i: (m_log, [par[i]["spont"], par[i]["A_sq"], 40.0, par[i]["s"]]))
    v = decide(neg2, 0); print("control: M_log data: %s, %s" % v); ok &= v[0] == "FAIL"
    base = {"d_ex": 1.0, "g": 0.5, "sigma_ex": 12.0}
    good = [dict(base, spont=float(i % 2), g=0.5 + 0.1 * (i % 3)) for i in range(30)]
    bad_a = [dict(r, g=-0.5) for r in good]
    bad_b = [dict(r, d_ex=-1.0) for r in good]
    bad_s = [dict(r, sigma_ex=40.0) for r in good]
    d = (decide(good, 0)[0], decide(bad_a, 0)[0], decide(bad_b, 0)[0], decide(bad_s, 0)[0], decide(good[:5], 0)[0],
         decide(good, 10)[0], decide(good, 0, known_ok=False)[0])
    print("control decide(): %s %s %s %s %s %s %s (expected PASS FAIL FAIL FAIL VOID VOID VOID)" % d)
    ok &= d == ("PASS", "FAIL", "FAIL", "FAIL", "VOID", "VOID", "VOID")
    return ok""")
open(P, "w", encoding="utf-8", newline="").write(s)
print("ok")
