"""Pins that could fail. Run BEFORE any front is trusted (VERIFIER_DISCIPLINE §0: ask what the check does if
the thing is false). Every assertion here fails loudly on broken machinery."""

import numpy as np, json, sys
from scipy.stats import poisson, binom
from core import CV_from_P, comp_P, ba_capacity_cost, ba_at_budget, front_point, Qinv
from glm import glm_word_P, glm_count_P, glm_induced_fano

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
T = 0.05
K = 12
rates = np.linspace(0, 60, K)
res = {}

# ---- PIN 1: reproduce the paper's Poisson numbers C=0.5944, V=0.2224 -------------------------------------
P1 = comp_P(rates, 1.0, T)
q, C, e = ba_capacity_cost(P1, rates, 0.0)
V = CV_from_P(P1, q)[1]
res["pin1_paper_CV"] = {
    "C": C,
    "V": V,
    "match_C": abs(C - 0.5944) < 2e-3,
    "match_V": abs(V - 0.2224) < 2e-3,
    "opt_rate_Hz": e,
    "support": int((q > 1e-3).sum()),
}
print(
    f"PIN1 paper numbers: C={C:.4f} (0.5944) V={V:.4f} (0.2224) -> "
    f"{res['pin1_paper_CV']['match_C'] and res['pin1_paper_CV']['match_V']}   opt rate={e:.2f} Hz"
)

# ---- PIN 2: COM-Poisson nu=1 IS Poisson ------------------------------------------------------------------
from family import com_pmf

p, m, f = com_pmf(3.0, 1.0, 40)
d = float(np.abs(p - poisson.pmf(np.arange(len(p)), 3.0)).max())
res["pin2_comp_is_poisson"] = {"max_abs_diff": d, "ok": d < 1e-12}
print(f"PIN2 COM-Poisson nu=1 vs Poisson: max|diff|={d:.2e} -> {d < 1e-12}")

# ---- PIN 3 (FALSIFIER): duplicate symbols must give C = 0 ------------------------------------------------
Pdup = np.vstack([P1[5], P1[5]])
Cdup, _ = CV_from_P(Pdup, np.array([0.5, 0.5]))
res["pin3_falsifier"] = {"C_dup": Cdup, "fires": abs(Cdup) < 1e-12}
print(f"PIN3 falsifier (identical symbols): C={Cdup:.2e} -> fires={abs(Cdup) < 1e-12}")

# ---- PIN 4: GLM zero-history counts == Binomial exactly ---------------------------------------------------
Kb = 14
dt = T / Kb
Pc0 = glm_count_P(rates, h_ms=None, T=T, Kbins=Kb)  # h=None -> zero history
pspk = 1 - np.exp(-rates * dt)
Pbin = np.vstack([binom.pmf(np.arange(Kb + 1), Kb, pp) for pp in pspk])
d4 = float(np.abs(Pc0 - Pbin).max())
res["pin4_glm_zero_history_is_binomial"] = {"max_abs_diff": d4, "ok": d4 < 1e-12}
print(f"PIN4 GLM(h=0) counts vs Binomial: max|diff|={d4:.2e} -> {d4 < 1e-12}")

# ---- PIN 5: GLM word-level C == count-level C when h=0 ----------------------------------------------------
# With no history the spike placement given the count is uniform and independent of x, so timing carries zero
# information. A bug in the 2^K word enumerator breaks this.
Pw0 = glm_word_P(rates, h_ms=None, T=T, Kbins=Kb)
qt = np.ones(K) / K
Cw, Vw = CV_from_P(Pw0, qt)
Cc, Vc = CV_from_P(Pc0, qt)
res["pin5_word_eq_count_zero_history"] = {
    "C_word": Cw,
    "C_count": Cc,
    "dC": abs(Cw - Cc),
    "dV": abs(Vw - Vc),
    "ok": abs(Cw - Cc) < 1e-10 and abs(Vw - Vc) < 1e-10,
}
print(
    f"PIN5 GLM(h=0) word C={Cw:.10f} vs count C={Cc:.10f}  |dC|={abs(Cw-Cc):.2e} |dV|={abs(Vw-Vc):.2e} "
    f"-> {res['pin5_word_eq_count_zero_history']['ok']}"
)

# ---- PIN 6: GLM(h=0) -> Poisson channel as dt -> 0 --------------------------------------------------------
conv = []
for Kb2 in (10, 25, 50, 100, 200):
    Pc = glm_count_P(rates, h_ms=None, T=T, Kbins=Kb2)
    qq, Cg, _ = ba_capacity_cost(Pc, rates, 0.0)
    conv.append({"Kbins": Kb2, "dt_ms": 1000 * T / Kb2, "C": float(Cg), "gap_vs_poisson": float(Cg - 0.5944)})
    print(
        f"PIN6 GLM(h=0) Kbins={Kb2:>4} dt={1000*T/Kb2:>5.2f} ms  C={Cg:.4f}  gap vs Poisson {Cg-0.5944:+.4f}"
    )
mono = all(
    abs(conv[i + 1]["gap_vs_poisson"]) < abs(conv[i]["gap_vs_poisson"]) + 1e-9 for i in range(len(conv) - 1)
)
res["pin6_glm_to_poisson"] = {
    "rows": conv,
    "monotone_convergence": bool(mono),
    "final_gap": conv[-1]["gap_vs_poisson"],
}
print(
    f"PIN6 converges monotonically toward the Poisson channel: {mono} (final gap {conv[-1]['gap_vs_poisson']:+.4f})"
)

# ---- PIN 7: BA is globally optimal -> the sparse enumerator must NEVER beat it on the asymptotic objective -
viol = []
for E in (2.0, 5.0, 10.0, 20.0, 40.0):
    qa, Ca, ea = ba_at_budget(P1, rates, E)
    fe = front_point(P1, rates, E, lambda C, V, r: C)  # enumerator on the SAME (asymptotic) objective
    viol.append({"E": E, "C_BA": float(Ca), "C_enum": float(fe["C"]), "enum_minus_BA": float(fe["C"] - Ca)})
    print(f"PIN7 E={E:>5.1f} Hz  C_BA={Ca:.6f}  C_enum={fe['C']:.6f}  enum-BA={fe['C']-Ca:+.2e}")
worst = max(v["enum_minus_BA"] for v in viol)
res["pin7_enum_never_beats_BA"] = {"rows": viol, "worst_excess": worst, "ok": worst < 1e-6}
print(f"PIN7 enumerator never beats globally-optimal BA: {worst < 1e-6} (worst excess {worst:+.2e})")

# ---- PIN 8: asymptotic front must be concave & non-decreasing (it is a capacity-cost function) -------------
Eg = np.linspace(0.5, 45, 25)
Cs = np.array([ba_at_budget(P1, rates, E)[1] for E in Eg])
inc = bool(np.all(np.diff(Cs) > -1e-9))
sec = np.diff(Cs, 2)
conc = bool(np.all(sec < 1e-7))
res["pin8_asym_front_shape"] = {"nondecreasing": inc, "concave": conc, "worst_second_diff": float(sec.max())}
print(f"PIN8 asymptotic front non-decreasing={inc} concave={conc} (worst 2nd diff {sec.max():+.2e})")

ok = all(
    [
        res["pin1_paper_CV"]["match_C"],
        res["pin1_paper_CV"]["match_V"],
        res["pin2_comp_is_poisson"]["ok"],
        res["pin3_falsifier"]["fires"],
        res["pin4_glm_zero_history_is_binomial"]["ok"],
        res["pin5_word_eq_count_zero_history"]["ok"],
        res["pin7_enum_never_beats_BA"]["ok"],
        res["pin8_asym_front_shape"]["nondecreasing"],
        res["pin8_asym_front_shape"]["concave"],
    ]
)
res["ALL_PINS_PASS"] = bool(ok)
json.dump(res, open("pins.json", "w"), indent=1)
print("\nALL PINS PASS:", ok)
if not ok:
    raise SystemExit("PIN FAILURE - do not trust any front until fixed")
