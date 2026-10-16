"""q1895 (15 Oct 2026): census of DANDI 001262 (Heeringa 2024, Sci. Data 11:411; gerbil auditory-nerve single fibres) for
paper 2 direction B (task P2DATA). For every NWB asset it streams only units/tag, units/spike_times_index and
units/spike_times (no raw waveforms) and records: recording type (tag prefix), conditions, repetitions per condition,
and the spike count per trial. Output data_ext/dandi001262/census.json (one entry per asset); resumable: assets
already in the file are skipped. Facts only; no test is run here.
Run: py q1895_dandi001262_census.py   (the data are read as HDF5 only)
"""
import collections
import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import remfile

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_ext", "dandi001262")
OUT = os.path.join(D, "census.json")


def real_url(aid):
    r = urllib.request.urlopen("https://api.dandiarchive.org/api/assets/%s/download/" % aid)
    u = r.geturl(); r.close(); return u


def one(a):
    try:
        f = h5py.File(remfile.File(real_url(a["id"])), "r")
        u = f["units"]
        tags = [t.decode() if isinstance(t, bytes) else str(t) for t in u["tag"][:]]
        idx = u["spike_times_index"][:].astype(np.int64); st = u["spike_times"][:]
        starts = np.concatenate([[0], idx[:-1]]).astype(np.int64)
        counts = [int(np.sum(~np.isnan(st[s:e]))) for s, e in zip(starts, idx)]
        cond = collections.defaultdict(list); spikes = collections.defaultdict(list)
        for t, c, s, e in zip(tags, counts, starts, idx):
            m = re.match(r"(.*)_rep(\d+)$", t)
            key = m.group(1) if m else t
            cond[key].append(c)
            if key.split("_")[0] in ("RLF", "PH"):           # spike times kept for the intensity recordings only
                v = st[s:e]; spikes[key].append([round(float(x), 6) for x in v[~np.isnan(v)]])
        types = collections.Counter(k.split("_")[0] for k in cond)
        return {"path": a["path"], "id": a["id"], "types": dict(types), "n_cond": len(cond),
                "reps": {k: len(v) for k, v in cond.items()}, "counts": {k: v for k, v in cond.items()}, "spikes": dict(spikes)}
    except Exception as e:
        return {"path": a["path"], "id": a["id"], "error": repr(e)[:200]}


def main():
    assets = json.load(open(os.path.join(D, "assets.json")))
    done = json.load(open(OUT)) if os.path.exists(OUT) else []
    seen = {d["id"] for d in done if "error" not in d}
    todo = [a for a in assets if a["id"] not in seen]
    done = [d for d in done if "error" not in d]
    print("todo", len(todo), flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            done.append(r)
            if i % 20 == 0:
                json.dump(done, open(OUT, "w")); print("written", len(done), flush=True)
    json.dump(done, open(OUT, "w")); print("finished", len(done), "errors", sum("error" in d for d in done), flush=True)


if __name__ == "__main__":
    sys.exit(main())
