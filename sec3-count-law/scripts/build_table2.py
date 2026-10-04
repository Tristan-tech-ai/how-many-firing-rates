"""Build Table II of paper 2 (count changes and wall constant) from primary output files only.

Inputs (all machine outputs; no number is typed here except the chain-law coefficients, which are checked by the number gate):
  A_g(K)  Gaussian count changes, 128-bit full-chain relocation (c60_cap_fullchain.py):
          centre_scripts/c60_gate_K3..K6, K13.json; c60_small_K7..K12.json; c60_K14..K25.json   (file K_b gives A_g(K_b + 1))
  A_K     Poisson count changes by the newborn-weight -> 0 unfolding (q80_AK_by_unfolding.py):
          results_q80_A_transition_<tag>.json for K = 5..24
Output: table2_primary.json and table2_rows.tex next to this file; prints the rms of the closure.
"""
import os, json, glob, math, re

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
CS = os.path.join(TW, "centre_scripts")
E0, E1 = 0.9840, 0.2347          # chain law E(g) = E0 - E1 g, fitted to the chain (paper 2, Sec. III-A)


def ag_table():
    ag, src = {}, {}
    pats = [("c60_gate_K*.json", r"c60_gate_K(\d+)\.json"), ("c60_small_K*.json", r"c60_small_K(\d+)\.json"),
            ("c60_K*.json", r"c60_K(\d+)\.json")]
    for g, rx in pats:
        for f in glob.glob(os.path.join(CS, g)):
            m = re.fullmatch(rx, os.path.basename(f))
            if not m:
                continue
            kb = int(m.group(1))
            d = json.load(open(f))
            if "A_t" in d and (kb + 1) not in ag:
                ag[kb + 1] = float(d["A_t"]); src[kb + 1] = os.path.basename(f)
    return ag, src


def ak_table():
    ak, src = {}, {}
    for f in glob.glob(os.path.join(TW, "results_q80_A_transition_*.json")):
        d = json.load(open(f))
        tag = os.path.basename(f)[len("results_q80_A_transition_"):-5]
        m = re.match(r"(?:gate)?K(\d+)", tag)
        if not m:
            continue
        K = int(m.group(1))
        A = float(d["A_transition"])
        if K in ak and abs(ak[K] - A) > 1e-6:
            src[K] += f" | {tag} {A:.6f}"
            continue
        ak[K] = A; src[K] = tag
    return ak, src


def main():
    ag, ags = ag_table()
    ak, aks = ak_table()
    rows, out = [], {}
    for K in range(5, 25):
        if K not in ak or K not in ag or (K + 1) not in ag or (K - 1) not in ag:
            out[K] = {"missing": [x for x in (K, K - 1, K + 1) if x not in ag] + ([] if K in ak else ["A_K"])}
            continue
        c = math.sqrt(ak[K]) - ag[K]
        g = ag[K + 1] - ag[K - 1]
        cl = (E0 - E1 * g) / 2
        out[K] = dict(A_g=ag[K], A_K=ak[K], c=c, g_eff=g, closure=cl, diff=c - cl, A_g_src=ags[K], A_K_src=aks[K])
        rows.append(f"{K} & {ag[K]:.5f} & {ak[K]:.4f} & {c:.4f} & {cl:.4f} & ${c - cl:+.4f}$\\\\")
    def rms(lo, hi):
        d = [out[k]["diff"] for k in range(lo, hi + 1) if "diff" in out[k]]
        return math.sqrt(sum(x * x for x in d) / len(d)), len(d)
    out["rms_5_20"], out["rms_5_24"] = rms(5, 20), rms(5, 24)
    json.dump(out, open(os.path.join(HERE, "table2_primary.json"), "w"), indent=1)
    open(os.path.join(HERE, "table2_rows.tex"), "w", newline="\n").write("\n".join(rows) + "\n")
    for K in range(5, 25):
        print(K, out[K] if "missing" in out[K] else
              f"A_g {out[K]['A_g']:.6f}  A_K {out[K]['A_K']:.5f}  c {out[K]['c']:.5f}  closure {out[K]['closure']:.5f}  "
              f"diff {out[K]['diff']:+.5f}   [{out[K]['A_g_src']}; {out[K]['A_K_src']}]")
    print("rms K=5..20: %.5f (n=%d);  K=5..24: %.5f (n=%d)" % (*out["rms_5_20"], *out["rms_5_24"]))


if __name__ == "__main__":
    main()
