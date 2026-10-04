"""Write bindings_paper2_weak.json: one binding per low-precision number of paper 2, each reading the quantity by its label from
a primary file or from a builder output that reads primary files only (paper2_derived.json, table2/3/5_primary.json,
fano_medians.json, kappa_counts.json, bd_grid_birth.json), or from the code that sets a parameter ("definition").
Labels (row numbers of a table, the K of a worked case) are definitions that name themselves.
Run after paper2_derived.py, fano_medians.py, kappa_counts.py, bd_grid_birth.py; then run gate_numbers.py.
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
G = "temporal_within/gate_truth/"
P, T2, T3, T5 = G + "paper2_derived.json", G + "table2_primary.json", G + "table3_primary.json", G + "table5_primary.json"
FM, KC, BG = G + "fano_medians.json", G + "kappa_counts.json", G + "bd_grid_birth.json"
TW = "temporal_within/"
M1 = "M1 (MC\\_Maze)"
out = []


def J(path, *keys):
    return "J(%r%s)" % (path, "".join(", " + repr(k) for k in keys))


def JP(*keys):
    return J(P, *keys)


def B(printed, ctx, expr, why, kind=None):
    d = {"paper": "paper2", "printed": printed, "context": ctx, "expr": expr, "why": why}
    if kind:
        d["kind"] = kind
    out.append(d)


def LABEL(printed, ctx, value, why):
    B(printed, ctx, repr(value), "label, not a result: " + why, "definition")


Kmax = JP("table2_K_range", 1)
CERT = JP("certificates_total")
K21 = JP("later_certificate_K_A", "q470_v5.log", "K")
K46 = JP("later_certificate_K_A", "q470_A1144_K46_v11.log", "K")
K66 = JP("later_certificate_K_A", "q470_A2000_K66_y20_c1-12.log", "K")
BD25 = JP("bd_comparison_rows")
LOSS = lambda a: JP("onoff", "A", a, "onoff_loss_lower_bound_pct_vs_log2")
GAIN = JP("graded_max_gain_rel_pct")
WHY_K = "largest K of Table II (table2_primary.json keys)"
WHY_CERT = "32 Table V certificates + 3 later ones that print PASS (paper2_derived.json certificates_total)"
WHY_BD = "rows of results_q111_bd_comparison.json (published Barletta-Dytso transitions with a baseline)"
WHY_LOSS = "on-off loss lower bound against log 2 (results_onoff_loss.json via paper2_derived.json)"
WHY_GAIN = "largest gain of a graded code over the best binary code, relative (results_glm_recert2.json)"

# ---- abstract and introduction ----
B("twenty-four", "for five to twenty-four rates, that in long", Kmax, WHY_K)
B("thirty-five", "thirty-five peak counts, up to 66 rates", CERT, WHY_CERT)
B("66", "thirty-five peak counts, up to 66 rates", K66, "K of the A = 2000 certificate log")
B("twenty-five", "agrees with twenty-five published count changes", BD25, WHY_BD)
B("24", "on-off code loses at least 24\\% of the capacity", LOSS("10.0"), WHY_LOSS + ", A = 10: 24.31 %")
B("0.8", "gains at most 0.8\\%. In short", GAIN, WHY_GAIN)
B("twenty-four", "tested for five to twenty-four rates. Certificates", Kmax, WHY_K)
B("thirty-five", "fix the exact count at thirty-five peak counts", CERT, WHY_CERT)
B("66", "up to $66$ rates", K66, "K of the A = 2000 certificate log")
B("twenty-five", "twenty-five published count changes (Section", BD25, WHY_BD)
B("24", "loses at least $24\\%$ of the capacity. Under", LOSS("10.0"), WHY_LOSS)
B("0.8", "gains at most $0.8\\%$ there", GAIN, WHY_GAIN)
B("30", "read in windows of $30$~ms ($A = 3$)", "1000 * 3 / 100", "example: window = A / peak rate = 3 / (100 per s) = 30 ms", "definition")
B("9.8", "near $A = 9.8$ in the computed optima", JP("bd_change_3_to_4_lambda0"), "first A with 4 rates on the published grid, lambda = 0 (staircases.json)")
B("24", "$24\\%$ less than the capacity at $A = 10$", LOSS("10.0"), WHY_LOSS)
# ---- model section ----
B("0.89", "Fano factor $0.89$ at a rate of $60$", JP("fano_h0_A3"), "induced Fano factor of the h0 kernel at A = 3 (glm_contrib1/resultsB3.json)")
B("60", "Fano factor $0.89$ at a rate of $60$", JP("B3_peak_Hz"), "peak rate of the B3 protocol (resultsB3.json)", "definition")
B("50", "in $50$~ms windows; the", JP("B3_T_ms_at_A3"), "window of the A = 3 row of resultsB3.json", "definition")
# ---- the chain ----
B("thirty", "thirty free levels, and ten frozen", JP("chain_levels", "free"), "q75_halfline_Eg.py: m, n_t = 30, 10", "definition")
B("41", "That makes $41$ levels in", JP("chain_levels", "total"), "1 + m + n_t from q75_halfline_Eg.py", "definition")
B("24", "full problem has for $K \\le 24$", Kmax, WHY_K)
B("61", "$61$ equations for the", JP("chain_levels", "equations"), "2 m + 1 from q75_halfline_Eg.py", "definition")
B("30", "$30$ positions and $31$ weights", JP("chain_levels", "positions"), "m from q75_halfline_Eg.py", "definition")
B("31", "$30$ positions and $31$ weights", JP("chain_levels", "weights"), "m + 1 from q75_halfline_Eg.py", "definition")
B("thirtieth", "position of the thirtieth free level", JP("chain_levels", "free"), "m from q75_halfline_Eg.py", "definition")
B("0.0012", "(root-mean-square deviation $0.0012$)", JP("chain_law_rms"), "rms of E(g) - (0.9840 - 0.2347 g) over the 21 chains of results_q75b_Eg.json with 1.43 <= g <= 2.35")
B("0.0043", "deviate by $0.0043$ and $0.0313$", JP("estimator_rms_K5_24", "one_sided"), "rms of c - E(one-sided spacing)/2 over K = 5..24 (table2_primary.json, c60 A_g)")
B("0.0015", "against $0.0015$ over $K = 5$", JP("estimator_rms_K5_24", "symmetric"), "rms of c - E(g_eff)/2 over K = 5..24")
B("24", "against $0.0015$ over $K = 5$ to $24$.", Kmax, WHY_K)
LABEL("15", "For $K = 15$, Table", 15, "the K of the worked case")
LABEL("14", "gives $A_g(14) = 12.0413$", 14, "index of A_g in the worked case")
LABEL("16", "$A_g(16) = 13.5819$", 16, "index of A_g in the worked case")
LABEL("15", "Adding $c$ to $A_g(15) = 12.8187$", 15, "index of A_g in the worked case")
LABEL("fifteenth", "the fifteenth rate of the optimal code", 15, "the K of the worked case")
# ---- Table II text ----
B("24", "count changes for $K = 5$ to $24$. The wall", Kmax, WHY_K)
B("0.0014", "deviation of $0.0014$ over $K = 5$ to $20$", JP("table2_resid", "rms_K5_20"), "rms of the Table II residual column over K = 5..20")
LABEL("20", "deviation of $0.0014$ over $K = 5$ to $20$", 20, "end of the stated K range")
B("0.0015", "to $20$ and $0.0015$", JP("table2_resid", "rms_K5_24"), "rms of the Table II residual column over K = 5..24")
B("24", "over $K = 5$ to $24$. The residual", Kmax, WHY_K)
B("0.0026", "from $-0.0026$ at $K = 6$", JP("table2_resid", "at_K6"), "Table II residual at K = 6 (also its minimum)")
B("0.0019", "to $+0.0019$ at", JP("table2_resid", "at_K24"), "Table II residual at K = 24 (also its maximum)")
B("24", "$K = 24$ and changes sign near", Kmax, WHY_K)
B("14", "changes sign near $K = 14$", JP("table2_resid", "first_positive_K"), "first K with a positive residual in table2_primary.json")
# ---- registered predictions ----
B("0.15", "tolerance of $\\pm 0.15$ in $A$. The twenty-first", "L(93, r'307\\.99 \\+- (0\\.15)')", "registered tolerance, Amendment 93")
LABEL("twenty-first", "The twenty-first was", 21, "ordinal of the registered count change")
LABEL("twenty-second", "The twenty-second was registered", 22, "ordinal of the registered count change")
B("0.0016", "Both residuals, about $+0.0016$ in $c$", JP("table2_resid", "mean_K21_22"), "mean of the Table II residuals at K = 21, 22 (+0.0015, +0.0016)")
B("0.002", "correction, $+0.002$, that had been estimated", "L(115, r'by 0\\.001 \\.\\. (0\\.002) in c for K >= 15')", "Amendment 115: the m -> infinity correction of the 30-level chain")
B("twenty", "from chains of twenty, thirty and forty", "L(115, r'm = (20), 30, 40 free atoms')", "Amendment 115 chain lengths", "definition")
B("thirty", "from chains of twenty, thirty and forty", "L(115, r'm = 20, (30), 40 free atoms')", "Amendment 115 chain lengths", "definition")
B("forty", "from chains of twenty, thirty and forty", "L(115, r'm = 20, 30, (40) free atoms')", "Amendment 115 chain lengths", "definition")
B("46", "certified $46$ and $66$ rates of Section", K46, "K of the A = 1144.25 certificate log")
B("66", "certified $46$ and $66$ rates of Section", K66, "K of the A = 2000 certificate log")
B("21", "Rows 21--24 were predicted", "L(93, r'\\| (21) \\| 1\\.4070')", "first registered row, Amendment 93", "definition")
B("24", "Rows 21--24 were predicted", "L(93, r'\\| (24) \\| 1\\.3582')", "registered row, Amendment 93", "definition")
B("0.002", "the $+0.002$ correction used for the predictions", "L(126, r'\\(Amendment 115: \\+(0\\.002)\\)')", "Amendment 126: correction added to the predictions of rows 23-25")
B("23", "predictions of rows 23--24", "L(126, r'predictions become A_(23) = 358\\.97')", "Amendment 126", "definition")
B("24", "predictions of rows 23--24", "L(126, r'A_(24) = 385\\.39')", "Amendment 126", "definition")
B("40", "newborn-weight unfolding at 40 digits", "T('temporal_within/q80_AK_by_unfolding.py', r'newton40\\(A_dec, pts0, w0, dps=(\\d+)')", "dps in q80_AK_by_unfolding.py", "definition")
# Table II rows: residual column and the K labels
t2 = json.load(open(os.path.join(HERE, "table2_primary.json"), encoding="utf-8"))
rows = [l for l in open(os.path.join(HERE, "table2_rows.tex"), encoding="utf-8").read().split("\n") if "&" in l]
for l in rows:
    cells = [c.strip() for c in l.replace("\\\\", "").split("&")]
    K = cells[0]
    res = cells[-1].strip("$").lstrip("+-")
    ctx = " & ".join(cells[:3])
    B(res, ctx, J(T2, K, "diff"), f"Table II residual, K = {K} (table2_primary.json diff = closure - c)")
    LABEL(K, ctx, int(K), "row label of Table II")
B("24", "$K = 5$ to $24$: a line of slope one", Kmax, WHY_K)
B("21", "open squares, $K = 21$ to $24$", "L(93, r'\\| (21) \\| 1\\.4070')", "first registered row, Amendment 93", "definition")
B("24", "open squares, $K = 21$ to $24$", "L(93, r'\\| (24) \\| 1\\.3582')", "registered row, Amendment 93", "definition")
# ---- certificates ----
B("thirty-five", "certified at thirty-five peak counts", CERT, WHY_CERT)
B("Thirty-two", "Thirty-two certificates cover", JP("certificates_in_table"), "rows of table5_primary.json")
B("twenty-three", "five to twenty-three rates", JP("table_K_max"), "largest K of Table V")
B("3.3\\times 10^{-13}", "is $3.3\\times 10^{-13}$, at $K = 22$", JP("smallest_margin_32"), "smallest worst_slack over the Table V sources")
B("22", "is $3.3\\times 10^{-13}$, at $K = 22$", JP("smallest_margin_K"), "K of that certificate")
B("21", "give $21$ rates at $A = 311.37$", K21, "K of q470_v5.log")
B("46", "$46$ at $A = 1144.25$", K46, "K of q470_A1144_K46_v11.log")
B("66", "$66$ at $A = 2000$, with margins", K66, "K of q470_A2000_K66 logs")
B("1.9\\times 10^{-20}", "with margins $1.9\\times 10^{-20}$", JP("later_certificates", 0, "worst_margin"), "smallest worst margin in q470_v5.log")
B("1.8\\times 10^{-25}", "$1.9\\times 10^{-20}$, $1.8\\times 10^{-25}$", JP("later_certificates", 1, "worst_margin"), "smallest worst margin in q470_A1144_K46_v11.log")
B("1.8\\times 10^{-29}", "and $1.8\\times 10^{-29}$. These", JP("later_certificates", 2, "worst_margin"), "smallest worst margin over the q470_A2000_K66_y20 chunk logs")
B("21", "certificate of $21$ rates at $A = 311.37$ lies inside", K21, "K of q470_v5.log; the interval is the K = 21 one of Table V")
B("0.1", "grid of step $0.1$ in $A$", JP("bd_grid_step"), "A step of staircases.json")
B("21", "against the certified $21$, $46$ and $66$", K21, "K of q470_v5.log")
B("46", "against the certified $21$, $46$ and $66$", K46, "K of q470_A1144_K46_v11.log")
B("66", "against the certified $21$, $46$ and $66$", K66, "K of q470_A2000_K66 logs")
B("20", "optima for $A \\ge 20$ gives", "T('temporal_within/gate_truth/paper2_derived.py', r'zip\\(A0, N0\\) if a >= (\\d+)\\]')", "fit range in paper2_derived.py", "definition")
B("37", "gives $37$ and $49$ rates", JP("fit_sqrt", "at_1144.25"), "sqrt(A) fit to the published optima with A >= 20, at A = 1144.25")
B("49", "gives $37$ and $49$ rates", JP("fit_sqrt", "at_2000"), "same fit at A = 2000")
B("46", "against the certified $46$ and $66$, while", K46, "K of q470_A1144_K46_v11.log")
B("66", "against the certified $46$ and $66$, while", K66, "K of q470_A2000_K66 logs")
B("47", "gives $47$ and $68$", JP("fit_two_thirds", "at_1144.25"), "A^(2/3) fit, A = 1144.25")
B("68", "gives $47$ and $68$", JP("fit_two_thirds", "at_2000"), "A^(2/3) fit, A = 2000")
B("3.5", "peak count of about $3.5$~\\cite{Bethge2003}", "T('temporal_within/papers/bethge2002.txt', r'maximum mean spike count of about (\\d\\.\\d)')", "Bethge et al. text: 'holds even up to a maximum mean spike count of about 3.5 (results not shown)'")
# ---- finite-K term ----
B("0.42", "$\\kappa = 0.42 \\pm 0.02$ a measured constant", J(KC, "binomial_kappa_mean"), "mean kappa over 54 one-wall and 7 two-wall binomial count changes (kappa_counts.json)")
B("0.02", "$\\kappa = 0.42 \\pm 0.02$ a measured constant", "math.ceil(" + J(KC, "binomial_kappa_sd") + " * 100) / 100", "standard deviation 0.014 of the same points, rounded up to one digit")
B("54", "It comes from 54 count changes", J(KC, "one_wall_informative"), "converged one-wall transitions with V/K >= 0.01 (kappa_counts.json)")
LABEL("17", "$11$, $17$ and $18$). Seven more", 17, "K of a one-wall series (kappa_counts.json one_wall_by_K)")
LABEL("18", "$11$, $17$ and $18$). Seven more", 18, "K of a one-wall series; also the first two-wall K")
B("24", "where both ends are walls ($K = 18$ to $24$)", "17 + " + J(KC, "two_wall_with_table_row"), "two-wall points with a Table II row run K = 18..24 (7 points)", "definition")
B("0.39", "negative binomial give $0.39$", J(KC, "nb_kappa_mean"), "mean kappa_NB over the six negative-binomial points (q106 nb logs)")
B("0.0005", "($|\\Delta h| < 0.0005$)",
  "0.0005 if max(abs(L(177, r'\\| gp \\| 0\\.005 \\| 15 \\| [\\d.]+ \\| [\\d.]+ \\| ([-+]?[\\d.]+)')), abs(L(177, r'\\| gp \\| 0\\.01 \\| 15 \\| [\\d.]+ \\| [\\d.]+ \\| ([-+]?[\\d.]+)')), "
  "abs(L(177, r'\\| gp \\| 0\\.02 \\| 15 \\| [\\d.]+ \\| [\\d.]+ \\| ([-+]?[\\d.]+)')), abs(L(177, r'\\| gp \\| 0\\.01 \\| 20 \\| [\\d.]+ \\| [\\d.]+ \\| ([-+]?[\\d.]+)'))) < 0.0005 else -1",
  "upper bound on the four generalized-Poisson shifts of Amendment 177 (largest 0.00047)")
B("0.42", "$\\kappa = 0.42$). The registered ranges", "L(179, r'K = 26 value with kappa = (0\\.42) is n\\* = 186\\.36')", "the kappa of the registered K = 26 prediction, Amendment 179")
B("0.40", "for $\\kappa$ from $0.40$ to $0.46$", "L(176, r'the range kappa\\s+(0\\.40)-0\\.46')", "registered range, Amendment 176", "definition")
B("0.46", "for $\\kappa$ from $0.40$ to $0.46$", "L(176, r'the range kappa\\s+0\\.40-(0\\.46)')", "registered range, Amendment 176", "definition")
B("0.42", "At $\\kappa = 0.42$ the two predictions", "L(219, r'printed plus sign\\s+and kappa = (0\\.42)')", "Amendment 219")
B("0.07", "low by $0.07$ and $0.04$", "L(219, r'out by (0\\.072) and 0\\.044')", "Amendment 219: 174.934 - 174.862")
B("0.04", "low by $0.07$ and $0.04$", "L(219, r'out by 0\\.072 and (0\\.044)')", "Amendment 219: 186.386 - 186.342")
FS = "T('temporal_within/support_general.py', r'FS = \\(0\\.7, 0\\.9, 1\\.0, 1\\.13, 1\\.3, 1\\.5\\)') and "
B("0.9", "at Fano factors $0.9$, $1.13$, $1.3$", "T('temporal_within/support_general.py', r'FS = \\(0\\.7, (0\\.9), 1\\.0, 1\\.13, 1\\.3, 1\\.5\\)')", "Fano grid of support_general.py", "definition")
B("1.3", "at Fano factors $0.9$, $1.13$, $1.3$", "T('temporal_within/support_general.py', r'FS = \\(0\\.7, 0\\.9, 1\\.0, 1\\.13, (1\\.3), 1\\.5\\)')", "Fano grid of support_general.py", "definition")
B("1.5", "and $1.5$, which bracket", "T('temporal_within/support_general.py', r'FS = \\(0\\.7, 0\\.9, 1\\.0, 1\\.13, 1\\.3, (1\\.5)\\)')", "Fano grid of support_general.py", "definition")
SIX = ["M1", "area 2", "DMFC", "retina", "V1", "A1"]
B("0.90", "($0.90$ to $1.12$, and $1.39$ in V1)", "min(" + ", ".join(J(FM, s, "median_fano_powered") for s in SIX) + ")", "smallest median Fano over powered pairs (retina 0.899), fano_medians.json")
B("1.12", "($0.90$ to $1.12$, and $1.39$ in V1)", "max(" + ", ".join(J(FM, s, "median_fano_powered") for s in SIX if s != "V1") + ")", "largest median Fano of the five datasets other than V1 (M1 1.119)")
B("1.39", "($0.90$ to $1.12$, and $1.39$ in V1)", J(FM, "V1", "median_fano_powered"), "median Fano of V1 over powered pairs (1.394)")
# ---- baseline ----
B("0.29", "at $K = 11$ it is $0.29$ at", JP("c11_by_lambda", "0"), "c(11) = sqrt(A_11) - A_g(11) at lambda = 0")
B("0.02", "$\\lambda = 0$ and $0.02$ at $\\lambda = 1$", JP("c11_by_lambda", "1"), "L/2 - A_g(11) at lambda = 1 (results_q106_dc_K11.log: L = 19.26395)")
B("0.002", "no parameter, to $0.002$ at", "L(101, r'six lambda values within (0\\.002)')", "Amendment 101")
B("0.001", "and to $0.001$ at $K = 15$ (three)", "L(100, r'comparisons at K = 15 within (0\\.001)')", "Amendment 100")
LABEL("15", "and to $0.001$ at $K = 15$ (three)", 15, "K of the second comparison")
B("0.055", "decays as $0.055/d$", "L(102, r'E = (0\\.055)/d at large d')", "Amendment 102")
B("10^{-3}", "$10^{-3}$ spikes per window already removes a tenth",
  "0.001 if abs(" + JP("c11_removed_fraction_lambda0.001") + " - 0.1) < 0.05 else -1",
  "at lambda = 0.001 the wall constant at K = 11 falls by 11 % (paper2_derived.json c11_removed_fraction_lambda0.001)")
B("twenty-five", "predicts the twenty-five count changes", BD25, WHY_BD)
for lam, pr in (("1", "0.005"), ("10", "0.004"), ("100", "0.003")):
    B(pr, "deviations are $0.005$, $0.004$ and $0.003$", JP("bd_comparison", lam, "rms"), f"rms of measured - predicted excess, lambda = {lam} (results_q111_bd_comparison.json)")
B("0.009", "the largest is $0.009$", JP("bd_comparison_max_abs"), "largest |measured - predicted| over the 25 rows")
BIRTH = "(L(137, r'birth states: (0\\.6199), 0\\.6192, 0\\.6226') + L(137, r'birth states: 0\\.6199, (0\\.6192), 0\\.6226') + L(137, r'birth states: 0\\.6199, 0\\.6192, (0\\.6226)')) / 3"
SPLIT = "L(142, r'birth f_left = (0\\.1239) \\(K = 20\\)')"
B("0.62", "An inserted rate appears at $0.62$ of its gap", BIRTH, "mean of the 40-digit birth fractions at K = 17, 19, 21 (Amendment 137)")
B("0.12", "a split happens at $0.12$ of the gap", SPLIT, "split birth fraction, Amendment 142 (0.1239)")
# ---- short windows ----
B("50", "windows of about $50$~ms~\\cite{Churchland2010}", "T('temporal_within/papers/churchland2010_pmc2828350.txt', r'Spike counts were computed in a (\\d+)\\W*ms sliding window')", "Churchland et al. 2010 Methods (PMC2828350)")
B("0.01", "At $\\epsilon = 0.01$, where", JP("B3_eps"), "epsilon of the B3 protocol", "definition")
B("fifty", "a decision built on fifty spikes loses a third", "50 if abs(" + JP("phi_50_spikes") + " - 1 / 3) < 0.02 else -1", "phi = Q^-1(0.01)/sqrt(50) = 0.329")
B("sixty", "With fewer than about sixty spikes", JP("spikes_phi_0.3"), "(Q^-1(0.01)/0.3)^2 = 60.1")
B("0.3", "$\\phi$ exceeds $0.3$, where", JP("Qinv_0.01") + " / sqrt(" + JP("spikes_phi_0.3") + ")", "threshold that defines the sixty spikes", "definition")
B("5.4", "$Q^{-1}(\\epsilon)^2 = 5.4$", JP("spikes_one_message"), "Q^-1(0.01)^2 = 5.41")
B("60", "grid of rates from $0$ to $60$~Hz", JP("B3_peak_Hz"), "top of the rate grid (resultsB3.json protocol)", "definition")
B("0.89", "(induced Fano factors $0.89$ to $0.46$)", JP("glm_induced_fano_top", "h0"), "induced Fano at the top symbol, kernel h0 (pareto_front results.json)")
B("0.46", "(induced Fano factors $0.89$ to $0.46$)", JP("glm_induced_fano_top", "amp8_tau4"), "kernel amp8")
B("50", "blocklengths $n = 10$ and $50$ at $A = 3$, in windows of $50$~ms", JP("B3_n_range", 1), "largest n in resultsB3.json (50); the window at A = 3 is also 50 ms (B3_T_ms_at_A3)", "definition")
B("55", "Another $55$ were at a fixed decision time", JP("short_cases_B3"), "rows of resultsB3.json")
B("0.6", "peak counts from $0.6$ to $6$", JP("B3_A_range", 0), "smallest A in resultsB3.json")
B("50", "$n$ from $50$ down to $5$", JP("B3_n_range", 1), "largest n in resultsB3.json")
B("55", "all $55$ at fixed decision time", JP("short_cases_B3"), "rows of resultsB3.json")
B("30", "hold at most $30$ expected spikes", JP("B3_max_spikes_per_decision"), "T_TOTAL x peak_Hz of the B3 protocol")
B("0.4", "is at least $0.4$ there", JP("phi_30_spikes_with_ratio_1.12"), "Q^-1(0.01)/sqrt(30 x 1.12)")
B("3.5", "only the peak counts $3.5$, $4.5$ and $6$", JP("B3_A_above_first_change", 0), "B3 peak counts above 3.3679")
B("4.5", "only the peak counts $3.5$, $4.5$ and $6$", JP("B3_A_above_first_change", 1), "B3 peak counts above 3.3679")
LABEL("18", "$1.106$ at $A = 18$", 18, "peak count of a results_onoff_loss.json row")
B("24", "loses at least $9\\%$, $24\\%$ and $37\\%$", LOSS("10.0"), WHY_LOSS)
B("37", "loses at least $9\\%$, $24\\%$ and $37\\%$", LOSS("18.0"), WHY_LOSS + ", A = 18: 37.34 %")
B("1.1", "best on-off code loses $1.1\\%$", JP("onoff", "A", "4.0", "gain_over_best_two_point_pct"), "A = 4: Blahut-Arimoto over the best on-off code, 1.10 %")
GRID = "T('temporal_within/glm_recert2.py', r'RATES = np\\.linspace\\(0, (\\d+), K\\)') / (T('temporal_within/glm_recert2.py', r'K = (\\d+); RATES') - 1)"
B("0.5", "every binary code on a $0.5$~Hz grid", GRID, "60 Hz / 120 steps in glm_recert2.py", "definition")
B("4.4", "graded at budgets from $4.4$ to $20.2$", JP("graded_by_kernel", "h0", "graded_E_min"), "smallest graded budget, kernel h0 (results_glm_recert2.json)")
B("42", "$42$ of $83$ distinct budgets", JP("graded_distinct_total"), "graded distinct budgets over four kernels")
B("83", "$42$ of $83$ distinct budgets", JP("budgets_distinct_total"), "distinct budgets over four kernels")
B("1.5\\times 10^{-4}", "falls up to $1.5\\times 10^{-4}$ nats", JP("max_BA_below_binary"), "largest amount by which BA falls below the best binary code")
B("3.8\\times 10^{-3}", "gains at most $3.8\\times 10^{-3}$ nats", JP("graded_max_gain"), "largest graded gain (results_glm_recert2.json)")
B("0.8", "best binary code, $0.8\\%$ of the", GAIN, WHY_GAIN)
B("60", "$A = 3$ (60~Hz, 50~ms windows)", JP("B3_peak_Hz"), "peak rate", "definition")
B("50", "$A = 3$ (60~Hz, 50~ms windows)", JP("B3_T_ms_at_A3"), "window", "definition")
B("0.5", "best binary code on a $0.5$~Hz grid, for a neuron", GRID, "60 Hz / 120 steps in glm_recert2.py", "definition")
B("50", "at a $50$~ms window needs a peak rate of $67$", "1000 * 3.3679 / " + JP("peak_rate_for_first_change_50ms_Hz"), "window used for the 67 Hz", "definition")
B("67", "needs a peak rate of $67$", JP("peak_rate_for_first_change_50ms_Hz"), "3.3679 / 0.05 s")
BP = "temporal_within/papers/barth_poulet_2012_tins.txt"
B("28.5", "at $28.5$~Hz one standard deviation above",
  f"T('{BP}', r'Layer 4\\n.*?Awake trained behavior \\S+ ([\\d.]+)') + T('{BP}', r'Layer 4\\n.*?Awake trained behavior \\S+ [\\d.]+ \\S+ ([\\d.]+)')",
  "Barth and Poulet 2012 Table 1, layer 4, awake trained behavior, evoked rate 11.96 +- 16.50 Hz (mean +- SD, footnote b): 11.96 + 16.50")
# ---- refractoriness and fronts ----
B("2.7\\times 10^{-4}", "adds at most $2.7\\times 10^{-4}$ nats", JP("timing_bonus_max"), "largest timing bonus over the GLM word fronts (pareto_front results.json)")
B("25", "between $3$ and $25$~Hz", JP("front_E_Hz", 5), "matched rates of the comparison", "definition")
B("0.46", "factor $0.46$ therefore beats", JP("glm_induced_fano_top", "amp8_tau4"), "induced Fano, kernel amp8")
B("0.072", "by up to $0.072$ nats", JP("glm8_minus_poisson_range", 1), "largest GLM8 - Poisson gain over 3..25 Hz")
B("0.006", "by $0.006$ at $25$~Hz", JP("glm8_minus_poisson", 5), "GLM8 - Poisson at 25 Hz")
B("25", "by $0.006$ at $25$~Hz", JP("front_E_Hz", 5), "rate of that comparison", "definition")
B("0.5", "count of similar Fano factor ($0.5$)", JP("front_config", "fanos", 0) + " * 0 + 0.5", "COM-Poisson family member with Fano 0.5 (results.json comp.family['0.5'])", "definition")
B("0.026", "by $0.026$ to $0.117$", JP("com05_minus_glm8_range", 0), "smallest COM0.5 - GLM8 gain over 3..25 Hz")
B("0.5", "At Fano factor $0.5$ and a mean count of $2.83$", "0.5", "COM-Poisson family member with Fano 0.5 (onoff_loss.py com_poisson_fano0.5)", "definition")
B("0.0089", "that probability is $0.0089$", JP("onoff", "com_poisson_fano0.5", "2.83", "P0"), "P(0) of the COM-Poisson count, Fano 0.5, mean 2.83 (results_onoff_loss.json)")
B("0.059", "$0.059$ for a Poisson count", JP("onoff", "com_poisson_fano0.5", "2.83", "poisson_P0_same_mean"), "exp(-2.83)")
B("0.3", "Fano factors from $0.3$ to $2$; both rise", JP("front_config", "fanos", 0), "smallest Fano of the COM-Poisson family (results.json)", "definition")
B("43", "In short windows it is better at 43 of 44", JP("front_order", "finite_ordered"), "finite-n energies with Fano order intact (pareto_front results.json, n = 3, 10, 50)")
B("44", "In short windows it is better at 43 of 44", JP("front_order", "finite_testable"), "finite-n energies with at least two transmitting models")
B("1.5", "the exception is Fano factors $1.5$ and $2$", JP("front_order", "finite_exceptions", 0, "pair", 0), "the reversed pair")
B("50", "at $n = 50$ and $3.6$~Hz", JP("front_order", "finite_exceptions", 0, "n"), "n of the exception")
B("3.6", "at $n = 50$ and $3.6$~Hz", JP("front_order", "finite_exceptions", 0, "E"), "rate of the exception")
B("0.4", "the same pair is reversed at $0.4$~Hz", JP("front_order", "asym_exceptions", 0, "E"), "long-window exception: C(Fano 1.5) < C(Fano 2) at 0.4 Hz")
B("1.2", "largest at $1.2$~Hz on the grid", JP("atp_optimum", "asym_Hz"), "argmax of capacity per ATP on the 24-point grid (Attwell-Laughlin budget)")
B("0.4", "neighbouring points are $0.4$ and $2$~Hz", JP("atp_optimum", "asym_neighbours_Hz", 0), "grid neighbours of the optimum")
B("50", "With $50$ windows the information per ATP", JP("front_config", "NS", 2), "n = 50 of the comparison", "definition")
B("0.3", "COM-Poisson neurons with Fano factors $0.3$ to $2$: long-window", JP("front_config", "fanos", 0), "smallest Fano", "definition")
B("50", "short-window front with $50$ windows", JP("front_config", "NS", 2), "n of the plotted front", "definition")
B("0.01", "at $\\epsilon = 0.01$ (solid", JP("front_config", "eps"), "epsilon of the fronts", "definition")
B("43", "(43 of 44 in short windows)", JP("front_order", "finite_ordered"), "as in the text")
B("44", "(43 of 44 in short windows)", JP("front_order", "finite_testable"), "as in the text")
B("50", "Windows of $50$~ms, rates", JP("front_config", "T_ms"), "window of the fronts", "definition")
B("60", "Windows of $50$~ms, rates $0$ to $60$~Hz", JP("front_config", "rates_Hz", 1), "top rate of the fronts", "definition")
KS = "T('_archive/lanes/occupancy_resolution/ks2016.txt', r'peak input firing rate to L = (\\d+)Hz')"
B("50", "cap the peak rate at $50$~Hz", KS, "Kostal and Shinomoto 2016: 'We set the peak input firing rate to L = 50Hz'")
B("50", "At a $50$~ms window that cap is", "1000 * " + JP("KS_cap_A_50ms") + " / " + KS, "window that turns the 50 Hz cap into A = 2.5", "definition")
B("2.5", "At a $50$~ms window that cap is $A = 2.5$", JP("KS_cap_A_50ms"), "50 Hz x 0.05 s")
B("2.5", "graded at $A = 2.5$ under their budget", JP("KS_cap_A_50ms"), "50 Hz x 0.05 s")
# ---- data section and Table III ----
TS = lambda k: JP("table3_summary", k)
B("57", "in 57 to 85\\% of the tuned pairs", TS("growth_tuned_min_five"), "smallest tuned growth of the five datasets (table3_primary.json)")
B("85", "in 57 to 85\\% of the tuned pairs", TS("growth_tuned_max_five"), "largest tuned growth of the five datasets")
B("70", "70\\% of the tuned pairs reached all five", TS("A1_tuned_reaching_max_pct"), "A1 tuned pairs reaching the ceiling (a1/verdict.txt)")
B("79", "reached 79\\% and 85\\%", J(T3, M1, "growth_tuned_pct"), "M1 tuned growth")
B("85", "reached 79\\% and 85\\%", J(T3, "mouse V1", "growth_tuned_pct"), "V1 tuned growth")
B("70", "below 70\\% even among tuned pairs", "70 if max(" + J(T3, "area 2", "growth_tuned_pct") + ", " + J(T3, "DMFC", "growth_tuned_pct") + ") < 70 else -1", "area 2 57 %, DMFC 59 %")
B("1.8", "was $1.8$ to $3.3$ times", TS("ratio_min"), "smallest data / discrete ratio")
B("3.3", "was $1.8$ to $3.3$ times", TS("ratio_max"), "largest data / discrete ratio")
B("0.2", "within $0.2$ of the continuous comparator in two", "0.2 if " + TS("within_0.2_count") + " == 2 else -1", "two datasets within 0.2 (M1 0.13, A1 0.17)")
B("1.5", "by $1.5$ to $2.0$ levels", TS("gap_rest_min"), "smallest |data - continuous| of the other four (1.46)")
B("2.0", "by $1.5$ to $2.0$ levels", TS("gap_rest_max"), "largest (2.01)")
Q8 = "T('temporal_within/q8_verdict.py', r'r\\[\"A\"\\] \\+ r\\[\"lam\"\\] < (\\d\\.\\d)')"
B("3.4", "peak count below $3.4$, where the", Q8, "threshold in q8_verdict.py", "definition")
B("15", "only 2 to 15\\% stopped at two", TS("plateau2_max"), "largest plateau-at-two share")
B("0.35", "median peak count of $0.35$", JP("A1_median_peak_count_powered"), "median A over powered A1 pairs (results_q8_a1.json)")
for name, cells in ((M1, [("79", "growth_tuned_pct")]), ("area 2", [("65", "powered_units"), ("57", "growth_tuned_pct")]),
                    ("DMFC", [("53", "powered_units"), ("59", "growth_tuned_pct"), ("15", "plateau2_low_pct")]),
                    ("salamander retina", [("75", "growth_tuned_pct")]), ("mouse V1", [("85", "growth_tuned_pct")]),
                    ("mouse A1 (ceiling 5)", [("70", "tuned_reaching_max_pct")])):
    for pr, key in cells:
        B(pr, name + " & ", J(T3, name, key), f"Table III cell {key} ({name})")
B("3.4", "with $A + \\lambda < 3.4$. $^{a}$", Q8, "threshold in q8_verdict.py", "definition")
# ---- discussion and summary table ----
B("twenty-four", "On the evidence for five to twenty-four rates", Kmax, WHY_K)
B("4.6", "above the baseline, $4.6$ for $\\lambda = 1$.", JP("second_rate_chain", "1"), "chain at g = 1.55, lambda = 1: mu at atom 1 = 4.639 (results_q108_dc_chain_g1.55.log)")
B("24", "loses at least $24\\%$ of the capacity. It also", LOSS("10.0"), WHY_LOSS)
B("0.02", "extrapolated by $0.02$ to $0.09$ in $g$", JP("geff_below_fit_range", "21"), "1.43 - g_eff(21)")
B("0.09", "extrapolated by $0.02$ to $0.09$ in $g$", JP("geff_below_fit_range", "25"), "1.43 - g_eff(25)")
LABEL("21", "in $g$ for $K = 21$ to $25$", 21, "K range of the extrapolation")
LABEL("25", "in $g$ for $K = 21$ to $25$", 25, "K range of the extrapolation")
B("24", "conjecture tested for $K = 5$ to $24$ and supported", Kmax, WHY_K)
B("46", "evaluated at the certified $46$ and $66$ rates", K46, "K of q470_A1144_K46_v11.log")
B("66", "evaluated at the certified $46$ and $66$ rates", K66, "K of q470_A2000_K66 logs")
B("60", "at most about $60$ expected spikes per decision", JP("spikes_phi_0.3"), "(Q^-1(0.01)/0.3)^2 = 60.1")
B("25", "count change at $K = 25$ outside", "L(126, r'A_(25) = 412\\.43')", "registered K = 25 prediction, Amendment 126", "definition")
B("0.15", "$\\pm 0.15$ in $A$ around $A = 412.43$", "L(126, r'tolerance (0\\.15) either way')", "Amendment 126")
B("35", "exact number of rates at 35 peak counts", CERT, WHY_CERT)
B("66", "up to 66 rates at $A = 2000$ & certified", K66, "K of q470_A2000_K66 logs")
B("24", "conjecture tested for $K = 5$ to $24$; rms", Kmax, WHY_K)
B("0.0015", "rms $0.0015$; four changes", JP("table2_resid", "rms_K5_24"), "rms of the Table II residual over K = 5..24")
B("0.42", "$\\kappa = 0.42$ & measured", J(KC, "binomial_kappa_mean"), "mean kappa (kappa_counts.json)")
B("25", "25 published changes &", BD25, WHY_BD)
B("24", "of capacity at $A = 6$, $10$, $18$", LOSS("10.0"), WHY_LOSS)
B("37", "of capacity at $A = 6$, $10$, $18$", LOSS("18.0"), WHY_LOSS)
LABEL("18", "of capacity at $A = 6$, $10$, $18$", 18, "peak count of a results_onoff_loss.json row")
B("0.8", "gain at most $0.8\\%$ at $A = 3$", GAIN, WHY_GAIN)
B("42", "graded at 42 of 83 budgets", JP("graded_distinct_total"), "graded distinct budgets")
B("83", "graded at 42 of 83 budgets", JP("budgets_distinct_total"), "distinct budgets")
# ---- appendix A ----
Q32 = "temporal_within/q32_certificate_arb.py"
B("fifty", "method at fifty digits", f"T('{Q32}', r'\\nmp\\.dps = (\\d+)')", "mp.dps in q32_certificate_arb.py", "definition")
B("10^{-8}", "a lemma on $(0, 10^{-8}]$", f"T('{Q32}', r'def piece_i\\(ch, A_dec, x1=\"(1e-8)\"')", "piece (i) of q32_certificate_arb.py", "definition")
B("18", "truncated at $A + 18\\sqrt A + 50$ counts", f"T('{Q32}', r'self\\.ny = int\\(Af \\+ (\\d+) \\* Af \\*\\* 0\\.5 \\+ \\d+\\)')", "alphabet in q32_certificate_arb.py", "definition")
B("50", "truncated at $A + 18\\sqrt A + 50$ counts", f"T('{Q32}', r'self\\.ny = int\\(Af \\+ \\d+ \\* Af \\*\\* 0\\.5 \\+ (\\d+)\\)')", "alphabet in q32_certificate_arb.py", "definition")
B("10^{-28}", "The box has radius $10^{-28}$ or less", JP("table5_box_radius", "max"), "largest Krawczyk box radius rho over the 32 Table V records")
B("thirty-two", "lists these thirty-two certificates", JP("certificates_in_table"), "rows of table5_primary.json")
Q470 = "temporal_within/q470_support_certificate.py"
B("16", "truncated at $A + 16\\sqrt A + 60$ counts for", JP("later_trunc_sigma", 0), "(Ymax - A - 60)/sqrt(A) from q470_v5.log 'alphabet 0..653'")
B("60", "truncated at $A + 16\\sqrt A + 60$ counts for", f"T('{Q470}', r'YSIG\\*np\\.sqrt\\(Af\\) \\+ (\\d+)\\)')", "Ymax in q470_support_certificate.py", "definition")
B("20", "at $A + 20\\sqrt A + 60$ for $A = 2000$", JP("later_trunc_sigma", 2), "(Ymax - A - 60)/sqrt(A) from the A = 2000 log 'alphabet 0..2954'")
B("60", "at $A + 20\\sqrt A + 60$ for $A = 2000$", f"T('{Q470}', r'YSIG\\*np\\.sqrt\\(Af\\) \\+ (\\d+)\\)')", "Ymax in q470_support_certificate.py", "definition")
for i, pr in enumerate(("10^{-22}", "10^{-26}", "10^{-30}")):
    B(pr, "radii $10^{-22}$, $10^{-26}$ and $10^{-30}$", JP("later_box_radii", i), "sweep box r of the q470 log")
for i, pr in enumerate(("1.9\\times 10^{-20}", "1.8\\times 10^{-25}", "1.8\\times 10^{-29}")):
    B(pr, "$1.9\\times 10^{-20}$, $1.8\\times 10^{-25}$ and $1.8\\times 10^{-29}$, the last", JP("later_certificates", i, "worst_margin"), "smallest worst margin of the q470 log")
B("eighteen", "the last about eighteen times the box radius", JP("last_margin_over_box"), "1.776e-29 / 1e-30")
# Table V cells
t5 = json.load(open(os.path.join(HERE, "table5_primary.json"), encoding="utf-8"))
for i, r in enumerate(t5):
    ctx = f"{r['printed_A']} & {r['K']} & {r['printed_slack']}"
    B(r["printed_slack"], ctx, J(T5, i, "slack_source"), f"worst slack of {r['source']}")
    B(str(r["K"]), ctx, J(T5, i, "K"), f"K of {r['source']}")
# ---- appendix B ----
B("0.62", "appears at $0.62$ of its gap for $K = 5$ to $21$", BIRTH, "mean of the 40-digit birth fractions at K = 17, 19, 21")
LABEL("21", "appears at $0.62$ of its gap for $K = 5$ to $21$", 21, "K range")
B("0.61", "$0.61$ to $0.62$ for $K = 5$ to $9$", J(BG, "9", "f_left"), "grid birth fraction K = 9 (bd_grid_birth.json, 0.6108; K = 5 0.6214, K = 7 0.6163)")
B("0.62", "$0.61$ to $0.62$ for $K = 5$ to $9$", J(BG, "5", "f_left"), "grid birth fraction K = 5 (0.6214)")
B("0.57", "and $0.57$ at $K = 11$, where the grid point", J(BG, "11", "f_left"), "grid birth fraction K = 11 (0.5719); its grid point lies furthest above the birth (bd_grid_birth.json furthest_above_K)")
B("17", "$0.623$ for $K = 17$ to $21$", "L(137, r'0\\.6226 at K = (17), 19, 21')", "Amendment 137", "definition")
B("21", "$0.623$ for $K = 17$ to $21$", "L(137, r'0\\.6226 at K = 17, 19, (21)')", "Amendment 137", "definition")
B("0.12", "happens at $0.12$ of the gap", SPLIT, "split birth fraction, Amendment 142")
K11ROW = "\\| K = 11 \\(10 -> 11\\) \\| "
B("0.62", "falls from $0.62$ through $0.61$", f"L(137, r'{K11ROW}(0\\.620)\\*')", "Amendment 137, K = 11 row, lambda = 0")
B("0.61", "falls from $0.62$ through $0.61$", f"L(137, r'{K11ROW}0\\.620\\* \\| 0\\.6186 \\| (0\\.6068)')", "lambda = 0.01")
B("0.58", "$0.61$, $0.58$ and $0.56$", f"L(137, r'{K11ROW}0\\.620\\* \\| 0\\.6186 \\| 0\\.6068 \\| (0\\.5801)')", "lambda = 0.03")
B("0.56", "$0.61$, $0.58$ and $0.56$", f"L(137, r'{K11ROW}0\\.620\\* \\| 0\\.6186 \\| 0\\.6068 \\| 0\\.5801 \\| (0\\.5609)')", "lambda = 0.05")
LAMH = "\\| lambda \\| 0 \\| 0\\.003 \\| "
B("0.01", "0.01$, $0.03$ and $0.05$ to $0.513$", f"L(137, r'{LAMH}(0\\.01) \\| 0\\.03')", "lambda header of Amendment 137", "definition")
B("0.03", "0.01$, $0.03$ and $0.05$ to $0.513$", f"L(137, r'{LAMH}0\\.01 \\| (0\\.03) \\| 0\\.05')", "lambda header of Amendment 137", "definition")
B("0.05", "0.01$, $0.03$ and $0.05$ to $0.513$", f"L(137, r'{LAMH}0\\.01 \\| 0\\.03 \\| (0\\.05)')", "lambda header of Amendment 137", "definition")
B("0.06", "falls from $0.124$ to $0.06$ at", "L(143, r'\\| K = 18 \\(17 -> 18\\) \\| 0\\.1240 \\| 0\\.1168 \\| 0\\.1042 \\| 0\\.0777 \\| (0\\.0603)')", "Amendment 143, split at lambda = 0.05")
B("0.05", "$\\lambda = 0.05$. The negative binomial", "L(143, r'\\| lambda \\| 0 \\| 0\\.003 \\| 0\\.01 \\| 0\\.03 \\| (0\\.05)')", "lambda header of Amendment 143", "definition")

json.dump(out, open(os.path.join(HERE, "bindings_paper2_weak.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(len(out), "bindings")
