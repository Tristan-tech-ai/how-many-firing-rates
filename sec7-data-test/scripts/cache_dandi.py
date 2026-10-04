"""
Build Q8-style caches for two more NLB datasets by streaming NWB from DANDI (remfile + h5py, no browser).
Cache fields match what q8_levels_*.py expect: spikes (concatenated), sidx (START offsets), trial_type (condition
code), move_onset_time (alignment time). Conditions and alignment fixed in PREREG_Q8_area2.md.
Run: py cache_dandi.py
"""

import remfile, h5py, numpy as np, time, os

HERE = os.path.dirname(os.path.abspath(__file__))


def stream(asset):
    return h5py.File(remfile.File(f"https://api.dandiarchive.org/api/assets/{asset}/download/"), "r")


def units_of(f):
    u = f["units"]
    sp = u["spike_times"][:].astype(np.float64)
    end = u["spike_times_index"][:].astype(np.int64)
    start = np.concatenate([[0], end[:-1]])
    return sp, start


def s(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


t0 = time.time()
# ---- DMFC_RSG (000130): condition = (is_short, ts, is_eye, theta); align = set_time; exclude outliers/'none'
f = stream("c90cbccc-31a5-4815-88e6-822d8c5ca68c")
tr = f["intervals/trials"]
sp, start = units_of(f)
keep = (~tr["is_outlier"][:]) & (s(tr["split"][:]) != "none") & np.isfinite(tr["set_time"][:])
key = np.stack([tr["is_short"][:].astype(int), tr["ts"][:], tr["is_eye"][:].astype(int), tr["theta"][:]], 1)[
    keep
]
_, code = np.unique(key, axis=0, return_inverse=True)
np.savez(
    os.path.join(HERE, "area2", "dmfc_rsg_cache.npz"),
    spikes=sp,
    sidx=start,
    trial_type=code.astype(np.int64),
    move_onset_time=tr["set_time"][:][keep].astype(np.float64),
)
u, c = np.unique(code, return_counts=True)
print(
    f"DMFC_RSG: units {len(start)}, trials kept {keep.sum()}/{len(keep)}, conditions {len(u)}, trials/cond min/med/max {c.min()}/{int(np.median(c))}/{c.max()}  [{time.time()-t0:.0f}s]"
)
# ---- Area2_Bump (000127): condition = (cond_dir, ctr_hold_bump); align = move_onset_time; rewarded trials only
f = stream("ded26b6c-418d-43f5-8a37-dfd072c2dbd4")
tr = f["intervals/trials"]
sp, start = units_of(f)
keep = (s(tr["result"][:]) == "R") & np.isfinite(tr["move_onset_time"][:])
key = np.stack([tr["cond_dir"][:], tr["ctr_hold_bump"][:].astype(int)], 1)[keep]
_, code = np.unique(key, axis=0, return_inverse=True)
np.savez(
    os.path.join(HERE, "area2", "area2_bump_cache.npz"),
    spikes=sp,
    sidx=start,
    trial_type=code.astype(np.int64),
    move_onset_time=tr["move_onset_time"][:][keep].astype(np.float64),
)
u, c = np.unique(code, return_counts=True)
print(
    f"Area2_Bump: units {len(start)}, trials kept {keep.sum()}/{len(keep)}, conditions {len(u)}, trials/cond min/med/max {c.min()}/{int(np.median(c))}/{c.max()}  [{time.time()-t0:.0f}s]"
)
