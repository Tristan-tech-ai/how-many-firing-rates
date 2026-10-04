"""Build Table III of paper 2 (the level-count test on six datasets) from the printed outputs of q8_verdict.py only.

Sources (outputs of q8_verdict.py on each dataset's results file):
  M1      gate_truth/q8_rerun/results_q8_rep_verdict_out.txt  (q8_verdict.py results_q8_rep.json, rerun 4 Oct; same JSON verdict as
          the stored results_q8_rep_verdict.json)
  area 2  area2/area2_verdict.txt      DMFC  area2/dmfc_verdict.txt      retina  retina/verdict.txt
  V1      v1/verdict.txt               A1    a1/verdict.txt
Columns: powered units; registered verdict; mean N_cv(all) over powered pairs for data / discrete simulation / continuous simulation;
growth among tuned powered pairs (N_cv(all) >= 2); plateau at two levels among powered pairs with A + lambda < 3.4.
Writes table3_primary.json and table3_rows.tex.
"""
import os, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
SRC = [("M1 (MC\\_Maze)", "gate_truth/q8_rerun/results_q8_rep_verdict_out.txt"), ("area 2", "area2/area2_verdict.txt"),
       ("DMFC", "area2/dmfc_verdict.txt"), ("salamander retina", "retina/verdict.txt"), ("mouse V1", "v1/verdict.txt"),
       ("mouse A1 (ceiling 5)", "a1/verdict.txt")]


def parse(path):
    t = open(os.path.join(TW, path), encoding="utf-8", errors="replace").read()
    g = lambda rx: re.search(rx, t)
    m = g(r"powered \(K2 sep >= 0\.8\): (\d+) pairs over (\d+) distinct units")
    mean = g(r"mean N_cv\(all\) powered: data ([\d.]+)\s+vs their discrete sims ([\d.]+)\s+vs their smooth sims ([\d.]+)")
    tun = g(r"DIAGNOSTIC among TUNED powered pairs \(N_cv\(all\) >= 2, n=(\d+)\): growth (\d+)%")
    p2 = g(r"low A\+lam \(<3\.4\): n=(\d+)\s+plateau-at-2: (\d+)%")
    ver = g(r"VERDICT: Q8-(\w+)")
    dist = g(r"N_cv\(all\) distribution among powered: (\{.*\})")
    d = {int(k): int(v) for k, v in re.findall(r"np\.int64\((\d+)\): np\.int64\((\d+)\)", dist.group(1))} if dist else {}
    return dict(source=path, powered_pairs=int(m.group(1)), powered_units=int(m.group(2)), data=float(mean.group(1)),
                disc=float(mean.group(2)), cont=float(mean.group(3)), tuned_pairs=int(tun.group(1)), growth_tuned_pct=int(tun.group(2)),
                low_pairs=int(p2.group(1)), plateau2_low_pct=int(p2.group(2)), verdict=ver.group(1).lower(), Ncv_distribution=d)


def main():
    out, rows = {}, []
    for name, path in SRC:
        r = parse(path)
        r["ratio_data_disc"] = r["data"] / r["disc"]
        r["gap_data_cont"] = r["data"] - r["cont"]
        if r["Ncv_distribution"]:
            mx = max(r["Ncv_distribution"])
            r["tuned_reaching_max_pct"] = 100 * r["Ncv_distribution"][mx] / max(1, sum(v for k, v in r["Ncv_distribution"].items() if k >= 2))
        out[name] = r
        grow = (f"{r['tuned_reaching_max_pct']:.0f}" + r"\%$^{a}$") if "A1" in name else (f"{r['growth_tuned_pct']}" + r"\%")
        rows.append(f"{name} & {r['verdict']} & {r['powered_units']} & {r['data']:.2f} / {r['disc']:.2f} / {r['cont']:.2f} & "
                    + grow + f" & {r['plateau2_low_pct']}" + r"\%\\")
    rat = [r["ratio_data_disc"] for r in out.values()]
    gaps = sorted(abs(r["gap_data_cont"]) for r in out.values())
    out["summary"] = dict(ratio_min=min(rat), ratio_max=max(rat), gaps_sorted=gaps,
                          growth_range=[min(r["growth_tuned_pct"] for k, r in out.items() if k != "summary" and "A1" not in k),
                                        max(r["growth_tuned_pct"] for k, r in out.items() if k != "summary" and "A1" not in k)],
                          plateau2_range=[min(r["plateau2_low_pct"] for k, r in out.items() if k != "summary"),
                                          max(r["plateau2_low_pct"] for k, r in out.items() if k != "summary")])
    import statistics
    a1 = json.load(open(os.path.join(TW, "results_q8_a1.json"), encoding="utf-8"))
    pw = [u["W"]["0.05"] for u in a1["units"] if u["W"].get("0.05", {}).get("powered")]
    out["summary"]["A1_median_A_powered"] = statistics.median(r["A"] for r in pw)
    out["summary"]["A1_Nstar_values_powered"] = sorted({r["Nstar"] for r in pw})
    json.dump(out, open(os.path.join(HERE, "table3_primary.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(HERE, "table3_rows.tex"), "w", newline="\n").write("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(json.dumps(out["summary"]))
    a1 = out["mouse A1 (ceiling 5)"]
    print("A1 tuned pairs", a1["tuned_pairs"], "reaching five:", a1["Ncv_distribution"].get(5), f"= {a1.get('tuned_reaching_max_pct', 0):.1f}%")


if __name__ == "__main__":
    main()
