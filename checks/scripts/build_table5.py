"""Build Table V of paper 2 (certified counts N(A) = K with the worst chord slack) from the certificate output files only, and
compare it row by row with the table printed in paper2.tex.

Inputs: every results_*.json in temporal_within whose records carry "ok", "K", "A" and "worst_slack" (the certificate scripts
q32_recertify_v4.py, q53b and the q47/q49 runs, and the arb replication). Output: table5_primary.json; prints every printed row
with its source and any disagreement.
"""
import os, re, json, glob, math

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)


def records(d, src):
    if isinstance(d, dict):
        if {"K", "A", "worst_slack"} <= set(d) and isinstance(d.get("A"), (int, float)):
            yield dict(A=float(d["A"]), K=int(d["K"]), slack=float(d["worst_slack"]), ok=bool(d.get("ok", d.get("proved", False))), src=src)
        for v in d.values():
            yield from records(v, src)
    elif isinstance(d, list):
        for v in d:
            yield from records(v, src)


def main():
    recs = []
    for f in sorted(glob.glob(os.path.join(TW, "results_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        recs += list(records(d, os.path.basename(f)))
    tex = open(os.path.join(TW, "paper_neuron", "paper2.tex"), encoding="utf-8").read()
    body = tex[tex.find("\\label{tab:cert}"):]
    body = body[:body.find("\\end{tabular}")]
    printed = []
    for line in body.split("\n"):
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        for j in range(0, len(cells) - 2, 3):
            if re.fullmatch(r"\d+\.\d+", cells[j] or "") and re.fullmatch(r"\d+", cells[j + 1] or ""):
                printed.append((float(cells[j]), int(cells[j + 1]), cells[j + 2], cells[j]))
    out, bad = [], 0
    for A, K, s_txt, a_txt in printed:
        dec = len(a_txt.split(".")[1])
        cand = [r for r in recs if r["K"] == K and abs(r["A"] - A) <= 0.5 * 10 ** -dec + 1e-12 and r["ok"]]
        mant, ex = s_txt.lower().split("e")
        sd = len(mant.split(".")[1]) if "." in mant else 0
        tol = 0.5 * 10 ** (int(ex) - sd)
        match = [r for r in cand if abs(r["slack"] - float(s_txt)) <= tol * (1 + 1e-9)]
        status = "OK" if match else ("SLACK-DIFFERS" if cand else "NO-CERTIFICATE")
        bad += status != "OK"
        best = (match or cand or [None])[0]
        out.append(dict(printed_A=a_txt, K=K, printed_slack=s_txt, status=status,
                        source=best and best["src"], A_source=best and best["A"], slack_source=best and best["slack"],
                        all_sources=sorted({r["src"] for r in cand})))
        print(f"{a_txt:>9} K={K:<3} {s_txt:>8}  {status:15} {best and best['src']}  A={best and best['A']}  slack={best and best['slack']:.3g}"
              if best else f"{a_txt:>9} K={K:<3} {s_txt:>8}  {status}")
    json.dump(out, open(os.path.join(HERE, "table5_primary.json"), "w", encoding="utf-8"), indent=1)
    print(f"{len(printed)} printed rows, {bad} without a matching certificate record")


if __name__ == "__main__":
    main()
