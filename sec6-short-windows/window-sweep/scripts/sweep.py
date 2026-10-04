"""
Window sweep: is contribution #1 an artifact of the 50 ms window?

Control: hold TOTAL decision time T_total = n*T fixed, vary the window T (so n = T_total/T). This keeps the
total spike budget (hence the size of the dispersion penalty) roughly fixed, so any change in the
graded-vs-binary gap is attributable to code structure rather than penalty magnitude.

Run: py sweep.py [pins|sweep|eps|all]
Checkpointed to results.json.
"""

import numpy as np, json, os, sys
from scipy.stats import poisson
from core import comp_P, CV_from_P, ba_capacity_cost, asym_front, front_point, Qinv

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(DIR, "results.json")
PEAK = 60.0  # Hz, peak firing rate (same alphabet ceiling as all prior work)
T_TOTAL = 0.5  # s, total decision time held FIXED across the sweep
EPS = 0.01


def load():
    if os.path.exists(RES):
        try:
            return json.load(open(RES, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save(d):
    json.dump(d, open(RES, "w", encoding="utf-8"), indent=1)
    print(f"[saved] {list(d.keys())}")


def channel(T, K):
    rates = np.linspace(0.0, PEAK, K)
    return rates, comp_P(rates, 1.0, T)


def gap_at(T, K, E, n, eps=EPS, p_grid=61):
    """At window T and energy budget E (Hz): capacity-optimal input vs finite-blocklength optimal input.
    Returns the relative objective gain and both supports."""
    rates, P = channel(T, K)
    Qi = Qinv(eps)
    rows = asym_front(P, rates, np.array([E]))
    r = rows[0]
    qc = r["q"]
    Cc, Vc = CV_from_P(P, qc)
    obj = lambda C, V: n * C - np.sqrt(max(n * V, 0.0)) * Qi
    o_cap = obj(Cc, Vc)
    f = front_point(
        P, rates, r["rate"], lambda C, V, rr: obj(C, V), p_grid=p_grid, seed_qs=(qc,), allow_silence=False
    )
    if f is None:
        return None
    o_dag = f["obj"]
    denom = abs(n * Cc) if Cc > 1e-12 else float("nan")
    return {
        "T_ms": T * 1000,
        "K": K,
        "E_Hz": float(r["rate"]),
        "n": int(n),
        "C_cap": float(Cc),
        "V_cap": float(Vc),
        "obj_cap": float(o_cap),
        "obj_dag": float(o_dag),
        "gain_pct": float(100.0 * (o_dag - o_cap) / denom),
        "supp_cap": int((qc > 1e-3).sum()),
        "supp_dag": int((f["q"] > 1e-3).sum()),
        "nC_cap": float(n * Cc),
        "phi": float(np.sqrt(n * Vc) * Qi / (n * Cc)) if Cc > 0 else float("nan"),
    }


# ------------------------------------------------------------------ pins
def run_pins(d):
    out = {}
    # PIN 1: paper numbers at T=50 ms, K=12
    rates, P = channel(0.05, 12)
    q, C, e = ba_capacity_cost(P, rates, 0.0)
    V = CV_from_P(P, q)[1]
    out["pin1"] = {"C": C, "V": V, "ok": abs(C - 0.5944) < 2e-3 and abs(V - 0.2224) < 2e-3}
    print(f"PIN1 T=50ms K=12: C={C:.4f} (0.5944) V={V:.4f} (0.2224) -> {out['pin1']['ok']}")

    # PIN 2: continuous-time limit. n*C/T_total must -> A/e = 22.07 nats/s as T -> 0.
    target = PEAK / np.e
    rows = []
    for T in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        rates, P = channel(T, 25)
        q, C, e = ba_capacity_cost(P, rates, 0.0, iters=4000, tol=1e-12)
        rate_nats_per_s = C / T
        rows.append(
            {
                "T_ms": T * 1000,
                "C_per_window": float(C),
                "nats_per_s": float(rate_nats_per_s),
                "frac_of_continuous": float(rate_nats_per_s / target),
            }
        )
        print(
            f"PIN2 T={T*1000:>6.1f} ms  C={C:.5f} nats/window  -> {rate_nats_per_s:7.3f} nats/s "
            f"({100*rate_nats_per_s/target:5.1f}% of continuous-time {target:.3f})"
        )
    mono = all(rows[i + 1]["nats_per_s"] > rows[i]["nats_per_s"] - 1e-9 for i in range(len(rows) - 1))
    out["pin2"] = {
        "target_nats_per_s": float(target),
        "rows": rows,
        "monotone_increasing": bool(mono),
        "final_frac": rows[-1]["frac_of_continuous"],
        "ok": bool(mono and rows[-1]["frac_of_continuous"] > 0.90),
    }
    print(
        f"PIN2 monotone={mono}, reaches {100*rows[-1]['frac_of_continuous']:.1f}% of the continuous-time "
        f"capacity -> {out['pin2']['ok']}"
    )

    # PIN 3: falsifier
    Cdup, _ = CV_from_P(np.vstack([P[5], P[5]]), np.array([0.5, 0.5]))
    out["pin3"] = {"C_dup": float(Cdup), "fires": abs(Cdup) < 1e-12}
    print(f"PIN3 falsifier: C={Cdup:.2e} -> fires={out['pin3']['fires']}")

    # PIN 4: COM-Poisson nu=1 is Poisson
    from family import com_pmf

    p, m, f = com_pmf(3.0, 1.0, 40)
    dd = float(np.abs(p - poisson.pmf(np.arange(len(p)), 3.0)).max())
    out["pin4"] = {"max_abs_diff": dd, "ok": dd < 1e-12}
    print(f"PIN4 COM-Poisson nu=1 vs Poisson: {dd:.2e} -> {out['pin4']['ok']}")

    out["ALL_OK"] = all(
        out[k].get("ok", out[k].get("fires", False)) for k in ("pin1", "pin2", "pin3", "pin4")
    )
    print("ALL PINS PASS:", out["ALL_OK"])
    d["pins"] = out
    return d


# ------------------------------------------------------------------ the window sweep
def run_sweep(d):
    Ts = [0.002, 0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200]
    K = 21
    Es = [3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0]
    out = []
    print(f"\nWINDOW SWEEP (T_total={T_TOTAL*1000:.0f} ms fixed, n=T_total/T, K={K}, eps={EPS})")
    print(
        f"{'T(ms)':>7}{'n':>6}{'A=peak*T':>10} | {'best gain%':>11}{'at E(Hz)':>10}"
        f"{'supp_cap':>10}{'supp_dag':>10}{'phi':>7}"
    )
    for T in Ts:
        n = max(1, int(round(T_TOTAL / T)))
        rows = []
        for E in Es:
            g = gap_at(T, K, E, n)
            if g is not None and np.isfinite(g["gain_pct"]):
                rows.append(g)
        if not rows:
            continue
        best = max(rows, key=lambda r: r["gain_pct"])
        # also record the max support the capacity code ever uses at this T
        supp_max = max(r["supp_cap"] for r in rows)
        rec = {
            "T_ms": T * 1000,
            "n": n,
            "A_counts": PEAK * T,
            "rows": rows,
            "best_gain_pct": best["gain_pct"],
            "best_E_Hz": best["E_Hz"],
            "supp_cap_at_best": best["supp_cap"],
            "supp_dag_at_best": best["supp_dag"],
            "supp_cap_max": supp_max,
            "phi_at_best": best["phi"],
        }
        out.append(rec)
        print(
            f"{T*1000:>7.0f}{n:>6}{PEAK*T:>10.2f} | {best['gain_pct']:>11.2f}{best['E_Hz']:>10.1f}"
            f"{best['supp_cap']:>10}{best['supp_dag']:>10}{best['phi']:>7.2f}   (max supp_cap {supp_max})"
        )
    d["window_sweep"] = {"T_total_s": T_TOTAL, "K": K, "eps": EPS, "peak_Hz": PEAK, "rows": out}
    return d


# ------------------------------------------------------------------ alphabet-resolution control
def run_kcontrol(d):
    print("\nALPHABET CONTROL: does K (rate-level resolution) limit the support at large T?")
    out = []
    for T in (0.05, 0.10, 0.20):
        n = max(1, int(round(T_TOTAL / T)))
        for K in (12, 21, 31, 41):
            g = gap_at(T, K, 16.0, n)
            if g is None:
                continue
            out.append(g)
            print(
                f"  T={T*1000:>5.0f}ms K={K:>3}  gain={g['gain_pct']:>6.2f}%  "
                f"supp_cap={g['supp_cap']} supp_dag={g['supp_dag']}  C_cap={g['C_cap']:.4f}"
            )
    d["k_control"] = out
    return d


# ------------------------------------------------------------------ eps / n robustness
def run_eps(d):
    print("\nEPS / DECISION-TIME ROBUSTNESS (T=50 ms window, K=21, E=16 Hz)")
    T, K, E = 0.05, 21, 16.0
    out = []
    print(f"{'eps':>7}{'T_total(ms)':>13}{'n':>5}{'gain%':>9}{'supp_cap':>10}{'supp_dag':>10}{'phi':>7}")
    for eps in (0.001, 0.01, 0.05, 0.1, 0.2):
        for Ttot in (0.25, 0.5, 1.0, 2.5):
            n = max(1, int(round(Ttot / T)))
            g = gap_at(T, K, E, n, eps=eps)
            if g is None:
                continue
            g["eps"] = eps
            g["T_total_ms"] = Ttot * 1000
            out.append(g)
            print(
                f"{eps:>7}{Ttot*1000:>13.0f}{n:>5}{g['gain_pct']:>9.2f}{g['supp_cap']:>10}"
                f"{g['supp_dag']:>10}{g['phi']:>7.2f}"
            )
    d["eps_sweep"] = out
    return d


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    d = load()
    if what in ("pins", "all") and "pins" not in d:
        d = run_pins(d)
        save(d)
        if not d["pins"]["ALL_OK"]:
            raise SystemExit("PIN FAILURE - do not trust the sweep")
    if what in ("sweep", "all") and "window_sweep" not in d:
        d = run_sweep(d)
        save(d)
    if what in ("sweep", "all") and "k_control" not in d:
        d = run_kcontrol(d)
        save(d)
    if what in ("eps", "all") and "eps_sweep" not in d:
        d = run_eps(d)
        save(d)
    print("done:", list(d.keys()))
