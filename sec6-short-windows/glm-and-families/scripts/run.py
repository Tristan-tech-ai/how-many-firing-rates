"""
Main computation: energy-vs-capacity Pareto fronts. Checkpointed to results.json.
x = mean firing rate (Hz, the A&L energy proxy: linear in rate). y = log M*(n,eps) in nats.
Run:  py run.py [comp|glm|all]
"""

import numpy as np, json, os, sys
from core import (
    asym_front,
    comp_P,
    front_point,
    Qinv,
    CV_from_P,
    fano_of_P,
    ba_front_parametric,
    reduce_outputs,
)
from glm import glm_word_P, glm_count_P, glm_induced_fano

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "results.json")
T = 0.05
K = 12
RATES = np.linspace(0, 60, K)
FANOS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
NS = [3, 10, 50]
EPS = 0.01
E_TARGETS = np.concatenate([np.linspace(0.4, 10, 13), np.linspace(11.5, 29, 11)])


def load():
    if os.path.exists(RES):
        try:
            return json.load(open(RES, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save(d):
    json.dump(d, open(RES, "w", encoding="utf-8"), indent=1)
    print(f"[saved] keys={list(d.keys())}")


def finite_front(P, rows, n, eps, cost=None):
    """Finite-N front at the energies the asymptotic sweep achieved. Returns raw (unclipped) second-order value
    and the achievable value max(0, raw). Also the induced Fano and support of the optimal code."""
    Qi = Qinv(eps)
    if cost is None:
        cost = RATES
    o = {"raw": [], "ach": [], "C": [], "V": [], "supp": [], "fano": [], "phi": []}
    for r in rows:
        f = front_point(
            P,
            cost,
            r["rate"],
            lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi,
            seed_qs=(r["q"],),
            allow_silence=False,
        )
        if f is None:
            for k in o:
                o[k].append(float("nan"))
            continue
        o["raw"].append(float(f["obj"]))
        o["ach"].append(float(max(0.0, f["obj"])))
        o["C"].append(float(f["C"]))
        o["V"].append(float(f["V"]))
        o["supp"].append(int((f["q"] > 1e-3).sum()))
        o["fano"].append(fano_of_P(P, f["q"]))
        o["phi"].append(float(np.sqrt(n * f["V"]) * Qi / (n * f["C"])) if f["C"] > 1e-12 else float("nan"))
    return o


def zero_crossing(E, raw):
    """Energy where the finite-N front crosses zero (below it, no reliable message at this n, eps)."""
    E = np.asarray(E, float)
    raw = np.asarray(raw, float)
    for i in range(len(E) - 1):
        if raw[i] < 0 <= raw[i + 1]:
            t = (0 - raw[i]) / (raw[i + 1] - raw[i])
            return float(E[i] + t * (E[i + 1] - E[i]))
    return float(E[0]) if raw[0] >= 0 else float("nan")


def efficiency_opt(E, y):
    """Energy maximising log M* per spike (y/E): where a ray from the origin is tangent to the front."""
    E = np.asarray(E, float)
    y = np.asarray(y, float)
    ok = (E > 1e-9) & np.isfinite(y)
    if not ok.any():
        return float("nan"), float("nan")
    eff = np.where(ok, y / np.where(E > 0, E, 1), -np.inf)
    i = int(np.nanargmax(eff))
    return float(E[i]), float(eff[i])


def run_comp(d):
    fam = {}
    for fano in FANOS:
        P = comp_P(RATES, fano, T)
        rows = asym_front(P, RATES, E_TARGETS)
        E = [r["rate"] for r in rows]
        rec = {
            "E": E,
            "C_asym": [r["C"] for r in rows],
            "V_asym": [r["V"] for r in rows],
            "supp_asym": [int((r["q"] > 1e-3).sum()) for r in rows],
            "fano_asym": [fano_of_P(P, r["q"]) for r in rows],
            "byN": {},
        }
        for n in NS:
            fn = finite_front(P, rows, n, EPS)
            asym = [n * c for c in rec["C_asym"]]
            rec["byN"][str(n)] = {
                "logM_asym": asym,
                "logM_finN_raw": fn["raw"],
                "logM_finN": fn["ach"],
                "C_finN": fn["C"],
                "V_finN": fn["V"],
                "supp_finN": fn["supp"],
                "fano_finN": fn["fano"],
                "phi": fn["phi"],
                "zero_crossing_Hz": zero_crossing(E, fn["raw"]),
                "eff_opt_asym_Hz": efficiency_opt(E, asym)[0],
                "eff_opt_finN_Hz": efficiency_opt(E, fn["ach"])[0],
            }
            zc = rec["byN"][str(n)]["zero_crossing_Hz"]
            rec["byN"][str(n)]["Nsp_at_crossing"] = float(n * zc * T) if np.isfinite(zc) else float("nan")
        fam[str(fano)] = rec
        print(
            f"Fano {fano}: asym sat {max(rec['C_asym']):.4f} nats/window; "
            + "  ".join(
                f"n={n}: cross {rec['byN'][str(n)]['zero_crossing_Hz']:.2f}Hz "
                f"(Nsp {rec['byN'][str(n)]['Nsp_at_crossing']:.2f})"
                for n in NS
            )
        )
    d["comp"] = {
        "rates_Hz": RATES.tolist(),
        "T": T,
        "eps": EPS,
        "fanos": FANOS,
        "NS": NS,
        "family": fam,
        "Qinv_sq": float(Qinv(EPS) ** 2),
    }
    return d


def run_glm(d):
    """GLM at matched readout, charged its REALISED mean firing rate (a refractory kernel suppresses spikes,
    so nominal 60 Hz may actually fire at 40 Hz; charging nominal would overstate its energy).
    counts: exact DP, fine bins. words: exact 2^K enumeration, coarser bins.
    Kernel h(tau) = -amp*exp(-tau/tau_ms); amp sweeps the induced dispersion."""
    from core import mean_rate_cost, word_counts

    out = {"count": {}, "word": {}, "kernels": [], "P0_mechanism": []}
    KB_COUNT, KB_WORD = 25, 14
    kernels = [None, {"amp": 2.0, "tau_ms": 4.0}, {"amp": 4.0, "tau_ms": 4.0}, {"amp": 8.0, "tau_ms": 4.0}]
    # sanity: for COM-Poisson the realised mean rate equals the nominal (so the fix changes nothing there)
    Pcp = comp_P(RATES, 0.5, T)
    out["comp_cost_check_max_abs_dev_Hz"] = float(np.abs(mean_rate_cost(Pcp, T) - RATES).max())
    for h in kernels:
        tag = "h0" if h is None else f"amp{h['amp']:g}_tau{h['tau_ms']:g}"
        out["kernels"].append({"tag": tag, "h": h})
        # ---- count readout (apples-to-apples with COM-Poisson) ----
        Pc = glm_count_P(RATES, h, T, KB_COUNT)
        cost = mean_rate_cost(Pc, T)
        rows = asym_front(Pc, cost, E_TARGETS[E_TARGETS <= cost.max()])
        E = [r["rate"] for r in rows]
        rec = {
            "E": E,
            "C_asym": [r["C"] for r in rows],
            "Kbins": KB_COUNT,
            "cost_realised_Hz": cost.tolist(),
            "nominal_Hz": RATES.tolist(),
            "induced_fano_at_opt": [fano_of_P(Pc, r["q"]) for r in rows],
            "byN": {},
        }
        for n in NS:
            fn = finite_front(Pc, rows, n, EPS, cost=cost)
            rec["byN"][str(n)] = {
                "logM_asym": [n * c for c in rec["C_asym"]],
                "logM_finN_raw": fn["raw"],
                "logM_finN": fn["ach"],
                "fano_finN": fn["fano"],
                "supp_finN": fn["supp"],
                "zero_crossing_Hz": zero_crossing(E, fn["raw"]),
            }
        out["count"][tag] = rec
        # the P(0) mechanism: a kernel acting only AFTER spikes cannot change P(no spike | rate)
        y0 = float(Pc[-1][0])
        out["P0_mechanism"].append(
            {
                "tag": tag,
                "P0_at_top_symbol": y0,
                "exp_minus_lamT": float(np.exp(-60 * T)),
                "realised_top_rate_Hz": float(cost[-1]),
                "induced_fano_top": float(fano_of_P(Pc, np.eye(K)[-1])),
            }
        )
        print(
            f"GLM counts {tag:>16}: C_sat={max(rec['C_asym']):.4f}  top symbol: nominal 60 Hz -> realised "
            f"{cost[-1]:.1f} Hz, Fano {out['P0_mechanism'][-1]['induced_fano_top']:.3f}, "
            f"P(0)={y0:.6f} (exp(-lamT)={np.exp(-3):.6f})"
        )
        # ---- word readout (same channel, richer readout: the TIMING BONUS) ----
        Pw_full = glm_word_P(RATES, h, T, KB_WORD)
        cost_w = mean_rate_cost(Pw_full, T, counts=word_counts(KB_WORD))
        Pw = reduce_outputs(Pw_full)
        Pc2 = glm_count_P(RATES, h, T, KB_WORD)
        cost_c2 = mean_rate_cost(Pc2, T)
        sw = ba_front_parametric(Pw, cost_w, np.concatenate([[0.0], np.geomspace(2e-3, 1.5, 18)]))
        Ew = [r["rate"] for r in sw]
        n = 10
        Qi = Qinv(EPS)
        obj = lambda C, V, rr: n * C - np.sqrt(max(n * V, 0)) * Qi
        wf = [front_point(Pw, cost_w, r["rate"], obj, seed_qs=(r["q"],), allow_silence=False) for r in sw]
        cr = asym_front(Pc2, cost_c2, np.array([e for e in Ew if e <= cost_c2.max()]))
        cf = [front_point(Pc2, cost_c2, r["rate"], obj, seed_qs=(r["q"],), allow_silence=False) for r in cr]
        g = lambda f: float(max(0.0, f["obj"])) if f is not None else 0.0  # no feasible 2-symbol code at E~0
        out["word"][tag] = {
            "Kbins": KB_WORD,
            "E_word": Ew,
            "C_word_asym": [r["C"] for r in sw],
            "logM_word_finN": [g(f) for f in wf],
            "E_count": [r["rate"] for r in cr],
            "C_count_asym": [r["C"] for r in cr],
            "logM_count_finN": [g(f) for f in cf],
            "induced_fano_count": glm_induced_fano(RATES, h, T, KB_WORD),
        }
        # timing bonus at MATCHED energy (interpolate the count front onto the word front's energies)
        bon = [
            float(cw - np.interp(e, cr and [r["rate"] for r in cr], [r["C"] for r in cr]))
            for e, cw in zip(Ew, [r["C"] for r in sw])
        ]
        out["word"][tag]["timing_bonus_vs_E"] = bon
        print(
            f"GLM words  {tag:>16}: max timing bonus over the front = {max(bon):+.4f} nats/window "
            f"(at E={Ew[int(np.argmax(bon))]:.1f} Hz); at saturation {bon[-1]:+.4f}"
        )
    d["glm"] = out
    return d


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = load()
    if what in ("comp", "all") and "comp" not in d:
        d = run_comp(d)
        save(d)
    if what in ("glm", "all") and "glm" not in d:
        d = run_glm(d)
        save(d)
    print("done:", list(d.keys()))
