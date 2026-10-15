"""q1951_range_readings.py (15 Oct 2026, task P2WEAKAUDIT step 1c): does the tested PH range cover the fibre's full rate range?
q1926 takes lam = min of the PH level means and A = their range, with the silent condition excluded; the PH levels start at 20 dB.
Readings only (no K and no D is computed): per fibre of the 265, the silent count (PH_silent, 0-50 ms) against lam, and the largest RLF
level mean (0-50 ms, levels 0-69 dB) against the largest PH level mean."""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1926_ph_steps as Q  # noqa: E402

r26 = {r["path"]: r for r in json.load(open(os.path.join(HERE, "paper_neuron", "q1926_rows.json")))}
W = 0.050
rows = []
for rec in json.load(open(Q.CENSUS)):
    if rec.get("path") not in r26:
        continue
    lev, counts, lam, A = Q.fibre(rec)
    sp = rec["spikes"]
    cnt = lambda k: np.array([sum(1 for t in tr if 0.0 <= t < W) for tr in sp[k]], float)
    sil = cnt("PH_silent").mean() if "PH_silent" in sp and len(sp["PH_silent"]) else np.nan
    rk = [k for k in sp if k.startswith("RLF_ABI")]
    rmax = max(cnt(k).mean() for k in rk) if rk else np.nan
    rmin_lev = min(float(re.sub(r"[^0-9.\-]", "", k[7:])) for k in rk) if rk else np.nan
    rows.append(dict(lam=lam, top=lam + A, sil=sil, rlfmax=rmax, ph_lo=lev[0], n_sil=len(sp.get("PH_silent", []))))
lam = np.array([r["lam"] for r in rows]); sil = np.array([r["sil"] for r in rows]); top = np.array([r["top"] for r in rows])
rmax = np.array([r["rlfmax"] for r in rows])
ok = ~np.isnan(sil)
print("Reading: fibres %d, with a PH silent condition %d (median %d trials); lowest PH level %s dB" % (
    len(rows), int(ok.sum()), int(np.median([r["n_sil"] for r in rows])), sorted(set(r["ph_lo"] for r in rows))))
d = lam[ok] - sil[ok]
print("Reading: lam (lowest PH level mean) minus silent count: quartiles %s; share with lam above silent by > 1 count %.3f, by > 0.5 %.3f"
      % (np.round(np.quantile(d, [0.25, 0.5, 0.75]), 3), float(np.mean(d > 1)), float(np.mean(d > 0.5))))
okr = ~np.isnan(rmax)
e = rmax[okr] - top[okr]
print("Reading: with an RLF run %d; RLF max level mean minus PH max: quartiles %s; share with RLF max above PH max by > 1 count %.3f"
      % (int(okr.sum()), np.round(np.quantile(e, [0.25, 0.5, 0.75]), 3), float(np.mean(e > 1))))
full = (np.maximum(top, np.where(okr, rmax, top)) - np.where(ok, np.minimum(lam, sil), lam)) / (top - lam)
print("Reading: full range (silent to larger of PH and RLF max) over the PH range: quartiles %s" % np.round(np.quantile(full, [0.25, 0.5, 0.75]), 3))
