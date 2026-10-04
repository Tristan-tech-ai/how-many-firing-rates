"""Aggregate the Q26 contrast test across all completed Allen functional-connectivity sessions.
Per session: per-orientation contrast maps (9 contrasts x 15 repeats) and the 36-cell map, at the 0.5 s window.
Reports the pre-registered quantities: stair/graded split, growth 8 -> 15, comparator means, nearer-continuous.
"""

import json, os, glob, sys, numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
W = "0.5"
files = sorted(glob.glob(os.path.join(HERE, "results_q26_[0-9]*.json"))) + [
    os.path.join(HERE, "results_q26.json"),
    os.path.join(HERE, "results_q26_s2.json"),
]
rows = []
for f in files:
    if not os.path.exists(f):
        continue
    sid = os.path.basename(f)[len("results_q26_") : -len(".json")]
    R = json.load(open(f))
    ents = [
        e
        for u in R["units"]
        for e in u["W"][W]["per_orientation"]
        if "Ncv_by_m" in e and e["Ncv_by_m"][-1] >= 2 and e.get("K1_shuffle_Ncv", 1) == 1
    ]
    m36 = [
        u["W"][W]["map36"]
        for u in R["units"]
        if u["W"][W]["map36"] and u["W"][W]["map36"]["Ncv_by_m"][-1] >= 2
    ]
    if not ents:
        continue
    nall = np.array([e["Ncv_by_m"][-1] for e in ents])
    n8 = np.array([e["Ncv_by_m"][0] for e in ents])
    ns = np.array([e["Nstar"] for e in ents])
    cont = np.array([e["comp"]["shrunk"]["cont"][-1] for e in ents])
    disc = np.array([e["comp"]["shrunk"]["disc"][-1] for e in ents])
    near = float(
        np.mean(np.abs(cont - nall) < np.abs(disc - nall))
        + 0.5 * np.mean(np.abs(cont - nall) == np.abs(disc - nall))
    )
    r = {
        "session": sid,
        "n_units": len(R["units"]),
        "n_tuned_maps": len(ents),
        "Ncv_all": float(nall.mean()),
        "cont": float(cont.mean()),
        "disc": float(disc.mean()),
        "stair": float(np.mean(nall <= ns + 1)),
        "graded": float(np.mean(nall > ns + 1)),
        "growth": float(np.mean(nall > n8)),
        "near_cont": near,
    }
    if m36:
        n36 = np.array([e["Ncv_by_m"][-1] for e in m36])
        c36 = np.array([e["comp"]["shrunk"]["cont"][-1] for e in m36])
        d36 = np.array([e["comp"]["shrunk"]["disc"][-1] for e in m36])
        r.update(
            {
                "n36": len(m36),
                "Ncv36": float(n36.mean()),
                "cont36": float(c36.mean()),
                "disc36": float(d36.mean()),
                "near36": float(
                    np.mean(np.abs(c36 - n36) < np.abs(d36 - n36))
                    + 0.5 * np.mean(np.abs(c36 - n36) == np.abs(d36 - n36))
                ),
            }
        )
    rows.append(r)
    print(
        f"{sid:>12s}: maps {len(ents):4d}  N_cv {r['Ncv_all']:.2f}  cont {r['cont']:.2f}  disc {r['disc']:.2f}  stair {100*r['stair']:.0f}%  graded {100*r['graded']:.0f}%  growth {100*r['growth']:.0f}%  nearer-cont {100*near:.0f}%"
        + (
            f" | 36-cell n={r['n36']:3d} data {r['Ncv36']:.2f} cont {r['cont36']:.2f} disc {r['disc36']:.2f} nearer {100*r['near36']:.0f}%"
            if m36
            else ""
        ),
        flush=True,
    )
print(f"\n=== {len(rows)} sessions pooled ===")
for k, lab in (
    ("Ncv_all", "N_cv(15) per-orientation"),
    ("cont", "continuous-own"),
    ("disc", "discrete-own"),
    ("stair", "fraction <= N*+1"),
    ("graded", "fraction > N*+1"),
    ("growth", "growth 8->15"),
    ("near_cont", "nearer continuous"),
):
    v = np.array([r[k] for r in rows])
    print(f"  {lab:26s}: mean {v.mean():.3f}  range {v.min():.3f}-{v.max():.3f}")
m = [r for r in rows if "Ncv36" in r]
for k, lab in (
    ("Ncv36", "36-cell data"),
    ("cont36", "36-cell continuous-own"),
    ("disc36", "36-cell discrete-own"),
    ("near36", "36-cell nearer continuous"),
):
    v = np.array([r[k] for r in m])
    print(f"  {lab:26s}: mean {v.mean():.3f}  range {v.min():.3f}-{v.max():.3f}")
tot = sum(r["n_tuned_maps"] for r in rows)
print(f"  total tuned contrast maps: {tot}; total 36-cell tuned units: {sum(r.get('n36', 0) for r in rows)}")
json.dump(rows, open(os.path.join(HERE, "results_q26_aggregate.json"), "w"), indent=1)
print("[saved] results_q26_aggregate.json")
