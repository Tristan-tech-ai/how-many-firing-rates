"""Derived numbers of paper 2, computed only from primary output files (no number is typed here except definitions such as the
peak rate 60 Hz, the window 50 ms, epsilon = 0.01 and the constant 0.4002 taken from the companion paper).
Writes paper2_derived.json; paragraphs of paper2.tex that print these numbers declare "paper2_derived.json" in their comment.
"""
import os, re, json, glob, math
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
ROOT = os.path.dirname(TW)
J = lambda *p: json.load(open(os.path.join(*p), encoding="utf-8"))
out = {}

# ---- certificates (Table V and the three later ones) ----
t5 = J(HERE, "table5_primary.json")
As = [r["A_source"] for r in t5]
Ks = [r["K"] for r in t5]
out["certificates_in_table"] = len(t5)
later = []
for pat in ["q470_v5.log", "q470_A1144_K46_v11.log", "q470_A2000_K66_y20_c*.log"]:
    files = sorted(glob.glob(os.path.join(TW, pat)))
    txt = "".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    m = [float(x) for x in re.findall(r"worst (?:free-stretch )?margin[^0-9]*?([0-9.]+e-\d+)", txt)]
    later.append({"logs": [os.path.basename(f) for f in files], "PASS": ("PASS" in txt or "PROVED" in txt), "worst_margin": min(m) if m else None})
out["later_certificates"] = later
out["certificates_total"] = len(t5) + sum(1 for x in later if x["PASS"])
out["table_A_min"], out["table_A_max"] = min(As), max(As)
out["table_K_min"], out["table_K_max"] = min(Ks), max(Ks)
out["table_K_missing"] = [k for k in range(min(Ks), max(Ks) + 1) if k not in Ks]
i = min(range(len(t5)), key=lambda j: t5[j]["slack_source"])
out["smallest_margin_32"], out["smallest_margin_K"] = t5[i]["slack_source"], t5[i]["K"]
byK = {}
for r in t5:
    byK.setdefault(r["K"], []).append(r["A_source"])
iv = {k: (min(v), max(v)) for k, v in byK.items() if len(v) >= 2}
out["intervals_count"] = len(iv)
out["intervals_total_length"] = sum(b - a for a, b in iv.values())
out["certified_span"] = max(As) - min(As)

# ---- Barletta-Dytso published staircases ----
st = J(TW, "data_barletta_dytso", "staircases.json")
pairs = {lam: len(v["A"]) - 1 for lam, v in st.items()}
out["bd_pairs_total"] = sum(pairs.values())
out["bd_pairs_lambda0"] = pairs["0.0"]
out["bd_falls"] = sum(1 for v in st.values() for a, b in zip(v["N"], v["N"][1:]) if b < a)
out["bd_Amax_lambda0"] = max(st["0.0"]["A"])
out["bd_grid_step"] = round(st["0.0"]["A"][1] - st["0.0"]["A"][0], 6)
# transitions without baseline (A where N first increases) and the sqrt(A) and A^(2/3) fits over A >= 20
A0, N0 = st["0.0"]["A"], st["0.0"]["N"]
tr = [(a, k) for a, k in zip(A0, N0) if a >= 20]          # every published optimum with A >= 20 (as in Amendment 618)
def fit(f):
    xs = [f(a) for a, _ in tr]; ys = [k for _, k in tr]; n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx
bs, as_ = fit(math.sqrt)
b23, a23 = fit(lambda a: a ** (2 / 3))
out["bd_rows_A_ge_20_count"] = len(tr)
out["fit_sqrt"] = {"slope": bs, "intercept": as_, "at_1144.25": bs * math.sqrt(1144.2477) + as_, "at_2000": bs * math.sqrt(2000) + as_}
out["fit_two_thirds"] = {"slope": b23, "intercept": a23, "at_1144.25": b23 * 1144.2477 ** (2 / 3) + a23, "at_2000": b23 * 2000 ** (2 / 3) + a23}
out["law_0.4002"] = {str(a): 0.4002 * a ** (2 / 3) for a in (311.368737, 1144.2477, 2000)}

# ---- Gaussian staircase from the 128-bit relocations: count minus 0.4002 L^(4/3) ----
ag = {}
for f in glob.glob(os.path.join(TW, "centre_scripts", "c60_*K*.json")):
    m = re.fullmatch(r"c60_(gate_|small_)?K(\d+)\.json", os.path.basename(f))
    if m:
        d = J(f)
        if "A_t" in d:
            ag.setdefault(int(m.group(2)) + 1, float(d["A_t"]))
def NG(L):
    return max([k for k, a in ag.items() if a <= L] + [3])
out["gauss_excess_at_count_changes"] = {str(K): {"L": ag[K], "excess": K - 0.4002 * ag[K] ** (4 / 3)} for K in (15, 24)}

# ---- short windows ----
B = J(ROOT, "glm_contrib1", "resultsB.json")
B3 = J(ROOT, "glm_contrib1", "resultsB3.json")
nB = sum(bn["n_valid"] for v in B["kernels"].values() for bn in v["byN"].values())
nB10 = sum(v["byN"]["10"]["n_valid"] for v in B["kernels"].values())
rows3 = [r for a in B3["arms"].values() for r in a["rows"]]
out["short_cases_partB"], out["short_cases_partB_n10"], out["short_cases_B3"] = nB, nB10, len(rows3)
out["short_cases_total"] = nB + len(rows3)
out["short_all_binary"] = all(s == 2 for v in B["kernels"].values() for bn in v["byN"].values() for s in bn["supp_finN_transmitting"]) and \
    all(r["supp_dag"] == 2 for r in rows3)
out["B3_A_range"] = [min(r["A_nominal"] for r in rows3), max(r["A_nominal"] for r in rows3)]
out["B3_n_range"] = [min(r["n"] for r in rows3), max(r["n"] for r in rows3)]
out["B3_T_ms_range"] = [min(r["T_ms"] for r in rows3), max(r["T_ms"] for r in rows3)]
out["B3_A_above_first_change"] = sorted({r["A_nominal"] for r in rows3 if r["A_nominal"] > 3.3679})
out["B3_max_spikes_per_decision"] = B3["protocol"]["T_TOTAL"] * B3["protocol"]["peak_Hz"]
h0 = [r for r in rows3 if r.get("fano_top") is not None]
out["fano_h0_A3"] = [r["fano_top"] for r in B3["arms"]["h0"]["rows"] if abs(r["A_nominal"] - 3.0) < 1e-9][0]
qinv = NormalDist().inv_cdf(1 - 0.01)
out["Qinv_0.01"] = qinv
out["phi_50_spikes"] = qinv / math.sqrt(50)
out["spikes_phi_0.3"] = (qinv / 0.3) ** 2
out["spikes_one_message"] = qinv ** 2
out["phi_30_spikes_with_ratio_1.12"] = qinv / math.sqrt(30 * 1.12)
out["peak_rate_for_first_change_50ms_Hz"] = 3.3679 / 0.05
out["KS_cap_A_50ms"] = 50 * 0.05

# ---- graded budgets at A = 3 (results_glm_recert2.json) ----
G = J(TW, "results_glm_recert2.json")
best = max(((k, r) for k, rs in G.items() if k != "contrast_recount" for r in rs), key=lambda kr: kr[1]["C_BA"] - kr[1]["best2_C"])
out["graded_max_gain"] = best[1]["C_BA"] - best[1]["best2_C"]
out["graded_max_gain_rel_pct"] = out["graded_max_gain"] / best[1]["C_BA"] * 100
out["graded_max_gain_at"] = {"kernel": best[0], "E": best[1]["E"]}

# ---- baseline (dark current) at K = 11 ----
D = J(TW, "results_q106_dc_K11.json")
agK = ag[11]
out["c11_lambda0"] = math.sqrt(D["poisson_A"]) - agK
for key, v in D.items():
    if key.startswith("dc_") and isinstance(v, dict) and "A_K" in v:
        lam = float(key[3:])
        out[f"c11_lambda{lam:g}"] = math.sqrt(v["A_K"] + lam) - math.sqrt(lam) - agK

# ---- small structural facts read from primary files ----
out["bd_comparison_rows"] = len(J(TW, "results_q111_bd_comparison.json"))
t2 = J(HERE, "table2_primary.json")
out["table2_K_range"] = [min(int(k) for k in t2 if k.isdigit()), max(int(k) for k in t2 if k.isdigit())]
st0 = st["0.0"]
out["bd_change_3_to_4_lambda0"] = next(a for a, n0, n1 in zip(st0["A"][1:], st0["N"], st0["N"][1:]) if n0 == 3 and n1 == 4)
out["B3_peak_Hz"] = B3["protocol"]["peak_Hz"]
out["B3_eps"] = B3["protocol"]["EPS"]
out["B3_T_ms_at_A3"] = [r["T_ms"] for r in B3["arms"]["h0"]["rows"] if abs(r["A_nominal"] - 3.0) < 1e-9][0]
later_K = {}
for f in ("q470_v5.log", "q470_A1144_K46_v11.log", "q470_A2000_K66_y20_c1-12.log"):
    txt = open(os.path.join(TW, f), encoding="utf-8", errors="replace").read()
    mK = re.search(r"\bK\s*=\s*(\d+)", txt)
    mA = re.search(r"\bA\s*=\s*([\d.]+)", txt)
    later_K[f] = {"K": int(mK.group(1)) if mK else None, "A": float(mA.group(1)) if mA else None}
out["later_certificate_K_A"] = later_K

# ---- the closure estimators and the worked case (Table II inputs) ----
E = lambda g: (0.9840 - 0.2347 * g) / 2
est = {"symmetric": [], "one_sided": [], "mean": []}
for K in range(5, 25):
    c = t2[str(K)]["c"]
    est["symmetric"].append(c - E(ag[K + 1] - ag[K - 1]))
    est["one_sided"].append(c - E(2 * (ag[K + 1] - ag[K])))
    est["mean"].append(c - E(2 * ag[K] / (K - 1)))
out["estimator_rms_K5_24"] = {k: math.sqrt(sum(x * x for x in v) / len(v)) for k, v in est.items()}
out["worked_case_K15"] = {"A_g14": ag[14], "A_g15": ag[15], "A_g16": ag[16], "g_eff": ag[16] - ag[14], "E": 2 * E(ag[16] - ag[14]),
                          "c": E(ag[16] - ag[14]), "sum": ag[15] + E(ag[16] - ag[14]), "measured_sqrtA15": math.sqrt(t2["15"]["A_K"])}
# chain lengths read from the chain script itself (q75_halfline_Eg.py: "m, n_t = 30, 10"): free levels m, frozen tail n_t,
# one level at silence; equations: D equal at the m free levels and at the first frozen one (m + 1), D' = 0 at the free levels (m)
_m = re.search(r"^m, n_t = (\d+), (\d+)", open(os.path.join(TW, "q75_halfline_Eg.py"), encoding="utf-8").read(), re.M)
_mf, _nt = int(_m.group(1)), int(_m.group(2))
out["chain_levels"] = {"free": _mf, "frozen": _nt, "total": 1 + _mf + _nt, "equations": 2 * _mf + 1, "positions": _mf, "weights": _mf + 1}

# ---- Lagrangian sweep (glm_contrib1/resultsA2.json) ----
A2 = J(ROOT, "glm_contrib1", "resultsA2.json")["byN"]["10"]
rates = sorted(A2["achieved_rate_Hz"])
gi = max(range(len(rates) - 1), key=lambda j: rates[j + 1] - rates[j])
out["lagrangian_n10"] = {"multipliers": len(A2["lams"]), "gap_low_Hz": rates[gi], "gap_high_Hz": rates[gi + 1]}

# ---- energy-capacity fronts (_archive/lanes/pareto_front/results.json) ----
import numpy as np
PF = J(ROOT, "_archive", "lanes", "pareto_front", "results.json")
P0m = {r["tag"]: r for r in PF["glm"]["P0_mechanism"]}
out["glm_realised_top_rate_Hz"] = {k: v["realised_top_rate_Hz"] for k, v in P0m.items()}
out["glm_induced_fano_top"] = {k: v["induced_fano_top"] for k, v in P0m.items()}
out["glm_P0_top"] = {k: v["P0_at_top_symbol"] for k, v in P0m.items()}
out["timing_bonus_max"] = max(max(v["timing_bonus_vs_E"]) for v in PF["glm"]["word"].values())
Eg = [3, 6, 10, 15, 20, 25]
cnt = PF["glm"]["count"]["amp8_tau4"]
fam = PF["comp"]["family"]
glm8 = np.interp(Eg, cnt["E"], cnt["C_asym"])
poi = np.interp(Eg, fam["1.0"]["E"], fam["1.0"]["C_asym"])
com5 = np.interp(Eg, fam["0.5"]["E"], fam["0.5"]["C_asym"])
out["front_E_Hz"] = Eg
out["glm8_minus_poisson"] = list(glm8 - poi)
out["com05_minus_glm8"] = list(com5 - glm8)
out["glm8_minus_poisson_range"] = [float(min(glm8 - poi)), float(max(glm8 - poi))]
out["com05_minus_glm8_range"] = [float(min(com5 - glm8)), float(max(com5 - glm8))]
cp = PF["comp"]
out["front_config"] = {"fanos": [min(cp["fanos"]), max(cp["fanos"])], "NS": cp["NS"], "eps": cp["eps"], "T_ms": cp["T"] * 1000,
                       "rates_Hz": [min(cp["rates_Hz"]), max(cp["rates_Hz"])]}

# ---- on-off losses (results_onoff_loss.json, written by onoff_loss.py) ----
p = os.path.join(TW, "results_onoff_loss.json")
if os.path.exists(p):
    O = J(p)
    out["onoff"] = O
# ---- support for the low-precision numbers of paper 2 (added 4 Oct, second gate pass) ----
# Table II residual column (closure minus measured wall constant), from table2_primary.json
dK = {int(k): v["diff"] for k, v in t2.items() if k.isdigit()}
rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))
out["table2_resid"] = {"rms_K5_20": rms([dK[k] for k in range(5, 21)]), "rms_K5_24": rms([dK[k] for k in range(5, 25)]),
                       "at_K6": dK[6], "at_K24": dK[24], "min": min(dK.values()), "argmin_K": min(dK, key=dK.get),
                       "max": max(dK.values()), "first_positive_K": min(k for k in dK if dK[k] > 0),
                       "rising_from_K6": all(dK[k + 1] > dK[k] for k in range(6, 24)),
                       "K21": dK[21], "K22": dK[22], "mean_K21_22": (dK[21] + dK[22]) / 2}
# g_eff extrapolation of the chain law beyond its fitted range 1.43 <= g <= 2.35 (Gaussian A_g up to K = 26 needed)
out["geff_below_fit_range"] = {str(K): 1.43 - (ag[K + 1] - ag[K - 1]) for K in range(19, 26) if K + 1 in ag and K - 1 in ag}
# wall constant at K = 11 with a baseline, from every q106 dark-current log line "dc lam: A_11 = ...; L = ..."
c11 = {0.0: out["c11_lambda0"]}
for f in sorted(glob.glob(os.path.join(TW, "results_q106_dc_K11*.log"))):
    for m in re.finditer(r"^dc ([\d.]+): A_11 = ([\d.]+) .*?; L = ([\d.]+);", open(f, encoding="utf-8", errors="replace").read(), re.M):
        c11[float(m.group(1))] = float(m.group(3)) / 2 - agK
out["c11_by_lambda"] = {f"{k:g}": v for k, v in sorted(c11.items())}
out["c11_removed_fraction_lambda0.001"] = 1 - c11[0.001] / c11[0.0]
out["dc_lambdas_K11_closed"] = sorted(k for k in c11 if k >= 0.01)
out["dc_lambdas_K15_closed"] = sorted({float(m.group(1)) for f in glob.glob(os.path.join(TW, "results_q106_dc_K15*.log"))
                                       for m in re.finditer(r"^dc ([\d.]+): A_15 = ", open(f, encoding="utf-8", errors="replace").read(), re.M)})
# chain law against the published Barletta-Dytso transitions with a baseline (results_q111_bd_comparison.json:
# [lambda, K, A, measured half-length excess..., predicted]); deviation = measured excess - predicted excess
BDc = J(TW, "results_q111_bd_comparison.json")
out["bd_comparison"] = {}
for lam in sorted({r[0] for r in BDc}):
    dev = [r[5] - r[6] for r in BDc if r[0] == lam]
    out["bd_comparison"][f"{lam:g}"] = {"changes": len(dev), "rms": rms(dev), "max_abs": max(abs(x) for x in dev),
                                       "K_min": min(r[1] for r in BDc if r[0] == lam), "K_max": max(r[1] for r in BDc if r[0] == lam)}
out["bd_comparison_max_abs"] = max(v["max_abs"] for v in out["bd_comparison"].values())
out["bd_comparison_below_K5"] = sum(1 for r in BDc if r[1] < 5)
# Table III summary ranges (table3_primary.json, built from the q8_verdict.py outputs)
t3 = {k: v for k, v in J(HERE, "table3_primary.json").items() if k != "summary"}
five = [k for k in t3 if not k.startswith("mouse A1")]
gaps = {k: abs(t3[k]["gap_data_cont"]) for k in t3}
out["table3_summary"] = {"growth_tuned_min_five": min(t3[k]["growth_tuned_pct"] for k in five),
                         "growth_tuned_max_five": max(t3[k]["growth_tuned_pct"] for k in five),
                         "A1_tuned_reaching_max_pct": t3["mouse A1 (ceiling 5)"]["tuned_reaching_max_pct"],
                         "ratio_min": min(t3[k]["ratio_data_disc"] for k in t3), "ratio_max": max(t3[k]["ratio_data_disc"] for k in t3),
                         "within_0.2_count": sum(1 for v in gaps.values() if v <= 0.2),
                         "gap_rest_min": min(v for v in gaps.values() if v > 0.2), "gap_rest_max": max(v for v in gaps.values() if v > 0.2),
                         "plateau2_min": min(t3[k]["plateau2_low_pct"] for k in t3), "plateau2_max": max(t3[k]["plateau2_low_pct"] for k in t3)}
R1 = J(TW, "results_q8_a1.json")
_rows = [u["W"][W] for u in R1["units"] for W in [str(w) for w in R1["config"]["W"]] if "Ncv_by_m" in u["W"].get(W, {})]
_pw = sorted(r["A"] for r in _rows if r["powered"])
out["A1_median_peak_count_powered"] = (_pw[(len(_pw) - 1) // 2] + _pw[len(_pw) // 2]) / 2
# graded budgets at A = 3 over the four kernels (results_glm_recert2.json)
gk = {}
for k, rs in G.items():
    if k == "contrast_recount":
        continue
    Es = {round(r["E"], 6) for r in rs}
    gE = {round(r["E"], 6) for r in rs if r["cert"] == "GRADED"}
    gk[k] = {"distinct_budgets": len(Es), "graded_distinct": len(gE), "graded_E_min": min(gE) if gE else None,
             "graded_E_max": max(gE) if gE else None, "max_BA_below_binary": max([-r["gap"] for r in rs if r["gap"] < 0] or [0])}
out["graded_by_kernel"] = gk
out["graded_distinct_total"] = sum(v["graded_distinct"] for v in gk.values())
out["budgets_distinct_total"] = sum(v["distinct_budgets"] for v in gk.values())
out["max_BA_below_binary"] = max(v["max_BA_below_binary"] for v in gk.values())
# energy-capacity fronts: Fano ordering at comparable energies, and the bits-per-ATP optimum (Attwell-Laughlin budget,
# beta and kappa as in _archive/lanes/pareto_front/figure.py)
famc = PF["comp"]["family"]; FAN = PF["comp"]["fanos"]
fin_t = fin_o = 0; fin_bad = []
for n in PF["comp"]["NS"]:
    for i in range(len(famc["1.0"]["E"])):
        tr = [(f, famc[str(f)]["byN"][str(n)]["logM_finN"][i]) for f in FAN]
        tr = [(f, v) for f, v in tr if v is not None and np.isfinite(v) and v > 0]
        if len(tr) >= 2:
            fin_t += 1
            if all(a[1] >= b[1] for a, b in zip(tr, tr[1:])):
                fin_o += 1
            else:
                fin_bad.append({"n": n, "E": famc["1.0"]["E"][i], "pair": [a[0] for a, b in zip(tr, tr[1:]) if a[1] < b[1]] +
                                [b[0] for a, b in zip(tr, tr[1:]) if a[1] < b[1]]})
asy_t = asy_o = 0; asy_bad = []
for i in range(len(famc["1.0"]["E"])):
    es = [famc[str(f)]["E"][i] for f in FAN]
    if max(es) - min(es) > 1e-3:          # not a comparable energy (the sweep reached different rates)
        continue
    vs = [famc[str(f)]["C_asym"][i] for f in FAN]
    asy_t += 1
    if all(a >= b for a, b in zip(vs, vs[1:])):
        asy_o += 1
    else:
        asy_bad.append({"E": es[0], "pairs": [[FAN[j], FAN[j + 1]] for j in range(len(FAN) - 1) if vs[j] < vs[j + 1]]})
out["front_order"] = {"finite_testable": fin_t, "finite_ordered": fin_o, "finite_exceptions": fin_bad,
                      "asym_comparable": asy_t, "asym_ordered": asy_o, "asym_exceptions": asy_bad}
KAP, BET = 0.71e9, 0.34e9
r1 = famc["1.0"]; E1 = np.array(r1["E"]); W1 = BET + KAP * E1
ea = np.array(r1["byN"]["10"]["logM_asym"]) / 10.0 / W1
ia = int(np.nanargmax(ea))
e50 = np.array(r1["byN"]["50"]["logM_finN"]) / 50.0 / W1
out["atp_optimum"] = {"asym_Hz": float(E1[ia]), "asym_neighbours_Hz": [float(E1[ia - 1]), float(E1[ia + 1])], "grid_points": len(E1),
                      "finite_n50_Hz": float(E1[int(np.nanargmax(e50))])}
# later certificates: sweep box radii (q470 logs "sweep box r = ...")
out["later_box_radii"] = []
for pat in ["q470_v5.log", "q470_A1144_K46_v11.log", "q470_A2000_K66_y20_c*.log"]:
    txt = "".join(open(f, encoding="utf-8", errors="replace").read() for f in sorted(glob.glob(os.path.join(TW, pat))))
    out["later_box_radii"].append(float(re.search(r"sweep box r = ([\d.e+-]+):", txt).group(1)))
out["last_margin_over_box"] = later[2]["worst_margin"] / out["later_box_radii"][2]
# the 32 certificates of Table V: Krawczyk box radius rho of each matched record (field "rho" next to "A", "K", "worst_slack")
def _recs(d):
    if isinstance(d, dict):
        if {"K", "A", "worst_slack"} <= set(d):
            yield d
        for v in d.values():
            yield from _recs(v)
    elif isinstance(d, list):
        for v in d:
            yield from _recs(v)
_rho = []
for r in t5:
    rec = [x for x in _recs(J(TW, r["source"])) if int(x["K"]) == r["K"] and abs(float(x["A"]) - r["A_source"]) < 1e-9]
    if rec and rec[0].get("rho") is not None:
        _rho.append(float(rec[0]["rho"]))
out["table5_box_radius"] = {"found": len(_rho), "max": max(_rho) if _rho else None, "min": min(_rho) if _rho else None}
# q470 alphabet truncation: Ymax = A + YSIG sqrt(A) + 60, read back from each log's "alphabet 0..Ymax" and its A
out["later_trunc_sigma"] = []
for f in ("q470_v5.log", "q470_A1144_K46_v11.log", "q470_A2000_K66_y20_c1-12.log"):
    txt = open(os.path.join(TW, f), encoding="utf-8", errors="replace").read()
    m = re.search(r"A = ([\d.]+), alphabet 0\.\.(\d+)", txt)
    a_, y_ = float(m.group(1)), int(m.group(2))
    out["later_trunc_sigma"].append((y_ - a_ - 60) / math.sqrt(a_))
# chain law E(g) = 0.9840 - 0.2347 g against the chain's own excess on 1.43 <= g <= 2.35 (results_q75b_Eg.json)
_q = J(TW, "results_q75b_Eg.json")
_gs = [(float(k), v["E"]) for k, v in _q.items() if 1.43 - 1e-9 <= float(k) <= 2.35 + 1e-9]
out["chain_law_rms"] = rms([e - (0.9840 - 0.2347 * g) for g, e in _gs])
_n = len(_gs); _mx = sum(g for g, _ in _gs) / _n; _my = sum(e for _, e in _gs) / _n
_b = sum((g - _mx) * (e - _my) for g, e in _gs) / sum((g - _mx) ** 2 for g, _ in _gs)
out["chain_law_lsq_refit"] = {"points": _n, "intercept": _my - _b * _mx, "slope": _b}
# second rate of the code from the chain at g = 1.55 (q108 logs "mu at atoms 1..4 [x1, ...]")
out["second_rate_chain"] = {}
for f in ("results_q108_dc_chain_g1.55.log", "results_q108_dc_chain_g1.55_ext.log"):
    for m in re.finditer(r"^lambda = ([\d.]+): .*?mu at atoms 1\.\.4 \[([\d.]+)", open(os.path.join(TW, f), encoding="utf-8", errors="replace").read(), re.M):
        out["second_rate_chain"].setdefault(m.group(1), float(m.group(2)))

json.dump(out, open(os.path.join(HERE, "paper2_derived.json"), "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    if k not in ("onoff", "bd_transitions_A_ge_20"):
        print(k, "=", v)
