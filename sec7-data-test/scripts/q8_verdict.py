"""
Mechanical verdict for Q8 per PREREG_Q8.md. Written BEFORE results_q8.json existed.
Definitions fixed here (the PREREG left 'plateau' verbal):
  growth  := N_cv(m=all) > N_cv(m=32)          (the last doubling of trials still adds a level)
  plateau := not growth
  match   := |N_cv(m=all) - N*(A, lam)| <= 1
Run: py q8_verdict.py
"""

import numpy as np, json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "results_q8.json")))
WS = [str(w) for w in R["config"]["W"]]
print(
    f"K1 shuffle control: N_cv == 1 in {R['K1']['pass']}/{R['K1']['total']} "
    f"({100*R['K1']['pass']/max(R['K1']['total'],1):.1f}%)  -> {'PASS' if R['K1']['pass'] >= 0.95*R['K1']['total'] else '*** FAIL: statistic finds levels in noise ***'}\n"
)
rows = []
for u in R["units"]:
    for W in WS:
        r = u["W"].get(W, {})
        if "Ncv_by_m" not in r:
            continue
        rows.append(
            {
                "unit": u["unit"],
                "W": W,
                "A": r["A"],
                "lam": r["lam"],
                "fano": r["fano"],
                "Nstar": r["Nstar"],
                "powered": r["powered"],
                "sep": r["K2_sep_frac"],
                "Ncv": r["Ncv_by_m"],
                "sim_disc": r["sim_disc_Ncv_by_m"],
                "sim_smooth": r["sim_smooth_Ncv_by_m"],
            }
        )
n_all = len(rows)
pw = [r for r in rows if r["powered"]]
units_pw = len(set(r["unit"] for r in pw))
print(
    f"(unit, W) pairs analysed: {n_all};  powered (K2 sep >= 0.8): {len(pw)} pairs over {units_pw} distinct units"
)
print(
    "power by window: "
    + "  ".join(
        f"W{int(float(W)*1000)}: {sum(1 for r in pw if r['W']==W)}/{sum(1 for r in rows if r['W']==W)}"
        for W in WS
    )
)
# comparator behaviour of the statistic on the simulations (all pairs)
d_g = np.mean([r["sim_disc"][-1] > r["sim_disc"][-2] for r in rows])
s_g = np.mean([r["sim_smooth"][-1] > r["sim_smooth"][-2] for r in rows])
print(
    f"comparator (simulations, mean N_cv): discrete sims show growth in {100*d_g:.0f}% of pairs; smooth sims in {100*s_g:.0f}%"
)
print(
    f"                 mean N_cv(all): discrete sims {np.mean([r['sim_disc'][-1] for r in rows]):.2f}, smooth sims {np.mean([r['sim_smooth'][-1] for r in rows]):.2f}, data {np.mean([r['Ncv'][-1] for r in rows]):.2f}\n"
)
if units_pw < 20:
    print(
        f"VERDICT: Q8-UNDERPOWERED ({units_pw} powered units < 20). Power table above; the bar is not lowered."
    )
    sys.exit()
growth = np.array([r["Ncv"][-1] > r["Ncv"][-2] for r in pw])
plateau = ~growth
match = np.array([abs(r["Ncv"][-1] - r["Nstar"]) <= 1 for r in pw])
low = np.array([r["A"] + r["lam"] < 3.4 for r in pw])
print("POWERED PAIRS:")
print(
    f"  plateau: {100*plateau.mean():.0f}%   growth: {100*growth.mean():.0f}%   plateau AND |N_cv - N*|<=1: {100*(plateau & match).mean():.0f}%"
)
print(
    f"  low A+lam (<3.4): n={low.sum()}  plateau-at-2: {100*np.mean([(r['Ncv'][-1]==2 and not g) for r,g in zip(pw,growth) if r['A']+r['lam']<3.4]) if low.any() else float('nan'):.0f}%"
)
print(
    f"  high A+lam (>=3.4): n={(~low).sum()}  growth: {100*growth[~low].mean() if (~low).any() else float('nan'):.0f}%"
)
print(
    f"  N_cv(all) distribution among powered: {dict(zip(*np.unique([r['Ncv'][-1] for r in pw], return_counts=True)))}"
)
print(
    f"  N* distribution among powered:        {dict(zip(*np.unique([r['Nstar'] for r in pw], return_counts=True)))}"
)
print(
    f"  mean N_cv(all) powered: data {np.mean([r['Ncv'][-1] for r in pw]):.2f}  vs their discrete sims {np.mean([r['sim_disc'][-1] for r in pw]):.2f}  vs their smooth sims {np.mean([r['sim_smooth'][-1] for r in pw]):.2f}"
)
# DIAGNOSTIC (added before results existed; does not change the rule): the +-1 match admits untuned pairs (N_cv = 1)
one = np.array([r["Ncv"][-1] == 1 for r in pw])
print(
    f"  DIAGNOSTIC: powered pairs with N_cv(all) = 1 (no resolvable tuning): {one.sum()} ({100*one.mean():.0f}%);"
    f" plateau&match with N_cv(all) >= 2: {(plateau & match & ~one).sum()} ({100*(plateau & match & ~one).mean():.0f}% of powered)"
)
tuned = pw and (~one).any()
if tuned:
    g2 = growth[~one]
    pm2 = (plateau & match)[~one]
    print(
        f"  DIAGNOSTIC among TUNED powered pairs (N_cv(all) >= 2, n={(~one).sum()}): growth {100*g2.mean():.0f}%   plateau&match {100*pm2.mean():.0f}%"
    )
if (plateau & match).mean() >= 0.7:
    v = "Q8-DISCRETE — plateau at N*(A, lam) in >= 70% of powered pairs"
elif (
    low.any()
    and (~low).any()
    and np.mean([(r["Ncv"][-1] == 2 and not g) for r, g in zip(pw, growth) if r["A"] + r["lam"] < 3.4]) >= 0.7
    and growth[~low].mean() >= 0.7
):
    v = "Q8-BINARY-ONLY — Bethge's version"
elif growth.mean() >= 0.7:
    v = "Q8-SMOOTH — growth without plateau in >= 70% of powered pairs; the discrete-levels prediction fails here"
else:
    v = "Q8-MIXED — no pre-registered outcome reached 70%"
print(f"\nVERDICT: {v}")
json.dump(
    {
        "verdict": v,
        "n_pairs": n_all,
        "n_powered": len(pw),
        "units_powered": units_pw,
        "plateau_frac": float(plateau.mean()),
        "growth_frac": float(growth.mean()),
        "plateau_match_frac": float((plateau & match).mean()),
    },
    open(
        os.path.join(
            HERE, (sys.argv[1].replace(".json", "") if len(sys.argv) > 1 else "results_q8") + "_verdict.json"
        ),
        "w",
    ),
    indent=1,
)
