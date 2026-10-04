"""PREREG_Q10.md Amendment 1: shape test on the retina's full PSTH (all bins). Seed 14."""

import numpy as np, json, os, sys, time, scipy.io as sio

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(14)
NSIM = 4
src = open(os.path.join(HERE, "q10_shape.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(HERE, "q10_shape.py")}
exec(src.split("HYPS = [")[0], ns)  # reuse transforms/ks/classify/sim_means
ns["rng"] = rng
classify, sim_means = ns["classify"], ns["sim_means"]
HYPS = ["sqrt", "rate", "log", "two"]
M = sio.loadmat(os.path.join(HERE, "retina", "movieBinnedSpiking.mat"))
B = M["binned"]
nreps = M["nreps"].ravel().astype(int)
NBIN = B.shape[1]
rows = []
t0 = time.time()
for cell in range(93):
    for mov in range(5):
        nr = nreps[mov]
        X = B[:nr, :, cell, mov].astype(float)
        for W, w in (("0.0167", 1), ("0.05", 3), ("0.1", 6)):
            C = np.stack([X[:, b : b + w].sum(1) for b in range(0, NBIN - w + 1, w)], 1)
            m = C.mean(0)
            keep = m >= 0.02
            excl = 1 - keep.mean()
            m = m[keep]
            if len(m) < 100 or m.max() - m.min() < 0.05:
                continue
            fano = (
                float(
                    np.mean(
                        [
                            C[:, i].var(ddof=1) / C[:, i].mean()
                            for i in np.where(keep)[0]
                            if C[:, i].mean() > 0.05
                        ]
                    )
                )
                if keep.any()
                else 1.0
            )
            win, d = classify(m)
            lo, hi = float(m.min()), float(m.max())
            conf = {h: 0 for h in HYPS}
            for h in HYPS:
                for _ in range(NSIM):
                    conf[h] += classify(sim_means(h, lo, hi, len(m), [nr] * len(m), fano, w == 1))[0] == h
            rec = {h: conf[h] / NSIM for h in HYPS}
            powered = min(rec.values()) >= 0.6
            rows.append(
                {
                    "cell": cell,
                    "movie": mov,
                    "W": W,
                    "n_bins": int(len(m)),
                    "excluded_frac": float(excl),
                    "win": win,
                    "ks": d,
                    "recovery": rec,
                    "powered": bool(powered),
                }
            )
    if cell % 15 == 0:
        print(f"cell {cell} [{time.time()-t0:.0f}s] rows {len(rows)}", flush=True)
for W in ("0.0167", "0.05", "0.1"):
    t = [r for r in rows if r["W"] == W]
    pw = [r for r in t if r["powered"]]
    print(
        f"W={W}: pairs {len(t)}, powered {len(pw)} ({100*len(pw)/max(len(t),1):.0f}%); mean bins {np.mean([r['n_bins'] for r in t]):.0f}, excluded {100*np.mean([r['excluded_frac'] for r in t]):.0f}%; WINS (powered): "
        + "  ".join(
            f"{h} {100*np.mean([r['win']==h for r in pw]) if pw else float('nan'):.0f}%" for h in HYPS
        )
        + "; mean KS: "
        + "  ".join(f"{h} {np.mean([r['ks'][h] for r in pw]) if pw else float('nan'):.3f}" for h in HYPS)
    )
json.dump(rows, open(os.path.join(HERE, "results_q10_retina_full.json"), "w"), indent=1)
print("[saved]")
