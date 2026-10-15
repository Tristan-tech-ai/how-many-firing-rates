"""q1957_recompute.py: recompute every number of amendment 2304 from paper_neuron/q1957_rows.json (bootstrap as registered, seed 2303)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import q1925_rlf_shape as Q  # noqa: E402

rows = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1957_rows.json")))
d = np.array([r["d"] for r in rows]); g = np.array([r["d_gom"] for r in rows if r["d_gom"] is not None])
sp = np.array([r["spont"] for r in rows])
print("n %d; d mean %.4f median %.4f frac>0 %.3f LB %.4f" % (len(d), d.mean(), np.median(d), np.mean(d > 0), Q.boot_lower(d, 2000, 2303)))
print("gompertz n %d; mean %.4f median %.4f frac>0 %.3f LB %.4f" % (len(g), g.mean(), np.median(g), np.mean(g > 0), Q.boot_lower(g, 2000, 2303)))
for nm, m in (("low", sp < 0.5), ("high", sp >= 0.5)):
    gm = np.array([r["d_gom"] for r, k in zip(rows, m) if k and r["d_gom"] is not None])
    print("%s spont n %d: d mean %.4f frac %.3f; gompertz mean %.4f frac>0 %.3f" % (nm, m.sum(), d[m].mean(), np.mean(d[m] > 0), gm.mean(), np.mean(gm > 0)))
