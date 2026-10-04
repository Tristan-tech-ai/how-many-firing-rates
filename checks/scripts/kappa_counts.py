"""How many count changes the finite-K constant kappa of paper 2 (eq. heff, Section VI) rests on, counted from the solver logs.

One wall: the binomial count on [0, A] with A = rho N (q129 and q130 logs, lines "bin N K-1=k -> K=k+1: transition A = ... peak p").
A (K, N) pair is counted once, from the latest log that holds it (file time), and only if the solve converged (|peak| <= 1e-8;
the transitions of the runs sit at 1e-10 to 1e-14). kappa per point = (h_F - ref_K) / ((h_F - sqrt A) / K), h_F = sqrt(N)
asin(sqrt(A/N)), with the calibrated Poisson references of results_q138_kappa_points.json (same transition rule). A point is
informative when (h_F - sqrt A) / K >= 0.01 (near rho = 0 the ratio is 0/0).
Two walls: the symmetric binomial staircase n*(18..26) recorded in Amendment 181 (q135); h_F = sqrt(n) pi/2, excess over
A_g(K) + 2 c(K) from table2_primary.json where the table has the row (K <= 24).
Negative binomial: q106 nb logs, kappa_NB = Delta c / (metric part / K), metric part = h_F - sqrt(A_K) as the log prints it.
Writes kappa_counts.json next to this file. Reads only.
"""
import os, re, glob, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
RX = re.compile(r"^bin ([\d.e+]+) K-1=(\d+) -> K=(\d+): transition A = ([\d.]+) \((?:rho = [\d.]+; )?peak ([-+\d.e]+)\)", re.M)
RX_NB = re.compile(r"^nb (\d+): A_(\d+) = ([\d.]+) .*?Delta c = L/2 - sqrt\(A_P\) = ([-\d.]+); metric part ([-\d.]+)", re.M)


def amend_text(n):
    s = open(os.path.join(TW, "PREREG_Q32.md"), encoding="utf-8", errors="replace").read()
    return re.search(r"^## Amendment %d\b.*?(?=^## Amendment )" % n, s, re.M | re.S).group(0)


def main():
    q138 = json.load(open(os.path.join(TW, "results_q138_kappa_points.json"), encoding="utf-8"))
    refs = {int(k): v for k, v in q138["refs"].items()}
    pts, rejected = {}, []
    files = sorted(glob.glob(os.path.join(TW, "results_q129_*.log")) + glob.glob(os.path.join(TW, "results_q130_*.log")),
                   key=os.path.getmtime)
    for f in files:                                   # oldest first, so a later run overwrites an earlier one
        for m in RX.finditer(open(f, encoding="utf-8", errors="replace").read()):
            N, K, A, peak = float(m.group(1)), int(m.group(3)), float(m.group(4)), float(m.group(5))
            if abs(peak) > 1e-8:
                rejected.append({"K": K, "N": N, "A": A, "peak": peak, "log": os.path.basename(f), "why": "not converged"})
                continue
            pts[(K, N)] = (A, os.path.basename(f))
    rows = []
    for (K, N), (A, f) in sorted(pts.items()):
        hF = math.sqrt(N) * math.asin(math.sqrt(A / N))
        x = (hF - math.sqrt(A)) / K
        kap = (hF - refs[K]) / x if x > 0 else None
        rows.append({"K": K, "N": N, "A": A, "rho": A / N, "hF": hF, "V_over_K": x, "kappa": kap, "informative": x >= 0.01, "log": f})
    by_K = {}
    for r in rows:
        b = by_K.setdefault(str(r["K"]), {"all": 0, "informative": 0})
        b["all"] += 1
        b["informative"] += int(r["informative"])
    one = [r["kappa"] for r in rows if r["informative"]]
    out = {"one_wall_by_K": by_K, "one_wall_total": len(rows), "one_wall_informative": len(one),
           "one_wall_kappa_mean": float(np.mean(one)), "one_wall_kappa_sd": float(np.std(one)),
           "one_wall_kappa_min": float(min(one)), "one_wall_kappa_max": float(max(one)),
           "one_wall_rho_range_informative": [min(r["rho"] for r in rows if r["informative"]), max(r["rho"] for r in rows if r["informative"])],
           "rejected": rejected}
    # two walls
    t181 = amend_text(181)
    ns = [float(v) for v in re.search(r"n\*\(18\.\.26\) = ([\d., ]+)\.", t181).group(1).replace(" ", "").split(",") if v]
    t2 = json.load(open(os.path.join(HERE, "table2_primary.json"), encoding="utf-8"))
    two = []
    for K, n in zip(range(18, 27), ns):
        hF = math.sqrt(n) * math.pi / 2
        vk = math.sqrt(n) * (math.pi / 2 - 1) / K
        row = {"K": K, "n": n, "V_over_K": vk}
        if str(K) in t2:
            row["excess"] = hF - (t2[str(K)]["A_g"] + 2 * t2[str(K)]["c"])
            row["kappa"] = row["excess"] / vk
        two.append(row)
    k2 = [r["kappa"] for r in two if "kappa" in r]
    out.update({"two_wall_points": two, "two_wall_total": len(two), "two_wall_with_table_row": len(k2),
                "two_wall_kappa_min": float(min(k2)), "two_wall_kappa_max": float(max(k2)), "two_wall_kappa_mean": float(np.mean(k2))})
    # negative binomial
    nb = {}
    for f in sorted(glob.glob(os.path.join(TW, "results_q106_nb_*.log")), key=os.path.getmtime):
        for m in RX_NB.finditer(open(f, encoding="utf-8", errors="replace").read()):
            r, K, A, dc, met = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
            nb[(r, K)] = {"r": r, "K": K, "A": A, "delta_c": dc, "metric": met, "kappa_nb": dc / (met / K), "log": os.path.basename(f)}
    knb = [v["kappa_nb"] for v in nb.values()]
    out.update({"nb_points": list(nb.values()), "nb_total": len(nb), "nb_kappa_mean": float(np.mean(knb)),
                "nb_kappa_min": float(min(knb)), "nb_kappa_max": float(max(knb))})
    allk = one + k2
    out.update({"binomial_kappa_mean": float(np.mean(allk)), "binomial_kappa_sd": float(np.std(allk)), "binomial_points": len(allk)})
    for K, b in sorted(by_K.items(), key=lambda kv: int(kv[0])):
        print(f"one wall K = {K:>2s}: {b['all']:3d} converged transitions, {b['informative']:3d} informative")
    print(f"one wall: {len(rows)} transitions, {len(one)} informative, kappa {out['one_wall_kappa_mean']:.4f} +- {out['one_wall_kappa_sd']:.4f}"
          f" (range {out['one_wall_kappa_min']:.3f}-{out['one_wall_kappa_max']:.3f}), rho {out['one_wall_rho_range_informative']}")
    print(f"rejected (not converged): {[(r['K'], r['N'], r['peak']) for r in rejected]}")
    print(f"two walls: {len(two)} transitions (K 18-26), kappa with a Table II row (K <= 24): "
          f"{', '.join(f'{v:.3f}' for v in k2)}")
    print(f"negative binomial: {len(nb)} points, kappa_NB {out['nb_kappa_mean']:.3f} (range {out['nb_kappa_min']:.3f}-{out['nb_kappa_max']:.3f})")
    print(f"binomial, one and two walls: {len(allk)} points, kappa {out['binomial_kappa_mean']:.4f} +- {out['binomial_kappa_sd']:.4f}")
    json.dump(out, open(os.path.join(HERE, "kappa_counts.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
