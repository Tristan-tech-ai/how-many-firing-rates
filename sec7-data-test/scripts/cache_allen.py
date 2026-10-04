import h5py, numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))
f = h5py.File(os.path.join(HERE, "ses-715093703.nwb"), "r")


def s(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


u = f["units"]
sp = u["spike_times"][:].astype(np.float64)
end = u["spike_times_index"][:].astype(np.int64)
start = np.concatenate([[0], end[:-1]])
area = np.load(os.path.join(HERE, "unit_area.npy"), allow_pickle=True)
qual = s(u["quality"][:])
keep_u = np.where((area == "VISp") & (qual == "good"))[0]
print("VISp good units:", len(keep_u))
# re-pack spikes for kept units only
spikes = np.concatenate([sp[start[k] : end[k]] for k in keep_u])
sidx = np.concatenate([[0], np.cumsum([end[k] - start[k] for k in keep_u])[:-1]])
t = f["intervals/static_gratings_presentations"]
ori = t["orientation"][:].astype(float)
sf = t["spatial_frequency"][:].astype(float)
ph = s(t["phase"][:])
st = t["start_time"][:].astype(float)
valid = ph != "N/A"
key = np.stack([ori, sf, np.array([float(p) for p in np.where(valid, ph, "0")])], 1)[valid]
uk, code = np.unique(key, axis=0, return_inverse=True)
print("conditions:", len(uk), "presentations:", valid.sum())
rng = np.random.default_rng(12)
sub = np.sort(rng.choice(len(uk), 36, replace=False))
m = np.isin(code, sub)
code36 = np.searchsorted(sub, code[m])
st36 = st[valid][m]
u36, c36 = np.unique(code36, return_counts=True)
print("36-condition subset: repeats min/med/max", c36.min(), int(np.median(c36)), c36.max())
np.savez(
    os.path.join(HERE, "v1_static36_cache.npz"),
    spikes=spikes,
    sidx=sidx,
    trial_type=code36.astype(np.int64),
    move_onset_time=st36,
)
np.savez(
    os.path.join(HERE, "v1_static_all_cache.npz"),
    spikes=spikes,
    sidx=sidx,
    trial_type=code.astype(np.int64),
    move_onset_time=st[valid],
)
print("caches written")
