"""PREREG_Q27.md: plateau units = cell class or SNR floor? Logistic regression, CV AUC, bootstrap. Seed 1001."""

import numpy as np, json, os, sys, csv, h5py

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = r"scratch"
rng = np.random.default_rng(1001)
U = {r["id"]: r for r in csv.DictReader(open(os.path.join(SCR, "units.csv"), encoding="utf-8"))}
CH = {r["id"]: r for r in csv.DictReader(open(os.path.join(SCR, "channels.csv"), encoding="utf-8"))}


def feats(uid):
    r = U.get(str(uid))
    if r is None:
        return None
    ch = CH.get(r.get("ecephys_channel_id", ""), {})
    try:
        return dict(
            duration=float(r["duration"]),
            pt=float(r["PT_ratio"]),
            fr=float(r["firing_rate"]),
            depth=float(ch.get("probe_vertical_position", "nan")),
        )
    except Exception:
        return None


rows = []
# (i) V1 static session 715093703: recover unit ids with the cache_allen.py filter
f = h5py.File(os.path.join(HERE, "v1", "ses-715093703.nwb"), "r")
ids = f["units"]["id"][:]
qual = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in f["units"]["quality"][:]])
area = np.load(os.path.join(HERE, "v1", "unit_area.npy"), allow_pickle=True)
keep_u = np.where((area == "VISp") & (qual == "good"))[0]
R = json.load(open(os.path.join(HERE, "results_q8_v1.json")))
for u in R["units"]:
    r = u["W"].get("0.25", {})
    if not ("Ncv_by_m" in r and r["Ncv_by_m"][-1] >= 2):
        continue
    uid = int(ids[keep_u[u["unit"]]])
    fe = feats(uid)
    if fe is None:
        continue
    rows.append(
        dict(
            session="static",
            plateau=int(r["Ncv_by_m"][-1] <= 2),
            A=r["A"],
            fano=r["fano"],
            peak=r["lam"] + r["A"],
            **fe,
        )
    )
# (ii) contrast sessions: 36-cell map at 0.5 s; Fano recomputed from the caches
for tag, cache, res in (
    ("c1", "v1_fc_contrast_cache.npz", "results_q26.json"),
    ("c2", "v1_fc_contrast_cache_s2.npz", "results_q26_s2.json"),
):
    Z = np.load(os.path.join(HERE, "v1", cache))
    sp, sidx, code, st, uids = Z["spikes"], Z["sidx"], Z["trial_type"], Z["move_onset_time"], Z["unit_ids"]
    ends = np.append(sidx[1:], len(sp))
    R = json.load(open(os.path.join(HERE, res)))
    for u in R["units"]:
        m = u["W"]["0.5"]["map36"]
        if not (m and m["Ncv_by_m"][-1] >= 2):
            continue
        k = u["unit"]
        s = sp[sidx[k] : ends[k]]
        c = (np.searchsorted(s, st + 0.5) - np.searchsorted(s, st)).astype(float)
        g = [c[code == j] for j in range(36)]
        mu = np.array([x.mean() for x in g])
        v = np.array([x.var(ddof=1) for x in g])
        ok = mu > 0.1
        fano = float(np.mean(v[ok] / mu[ok])) if ok.sum() >= 3 else 1.0
        fe = feats(int(uids[k]))
        if fe is None:
            continue
        rows.append(
            dict(
                session=tag,
                plateau=int(m["Ncv_by_m"][-1] <= 2),
                A=m["A"],
                fano=fano,
                peak=m["lam"] + m["A"],
                **fe,
            )
        )
print(
    "rows:",
    len(rows),
    {
        s: (sum(1 for r in rows if r["session"] == s), sum(r["plateau"] for r in rows if r["session"] == s))
        for s in ("static", "c1", "c2")
    },
)
BASE = ["logA", "logFano", "logPeak"]
CLASS = ["duration", "pt", "depth", "logFR"]


def design(rs, cols, sess_ind=True):
    X = []
    for r in rs:
        d = dict(
            logA=np.log(max(r["A"], 1e-3)),
            logFano=np.log(max(r["fano"], 1e-3)),
            logPeak=np.log(max(r["peak"], 1e-3)),
            duration=r["duration"],
            pt=r["pt"],
            depth=r["depth"],
            logFR=np.log(max(r["fr"], 1e-3)),
        )
        x = [d[c] for c in cols]
        if sess_ind:
            x += [1.0 if r["session"] == "c1" else 0.0, 1.0 if r["session"] == "c2" else 0.0]
        X.append(x)
    X = np.array(X, float)
    X = np.where(np.isfinite(X), X, np.nanmean(np.where(np.isfinite(X), X, np.nan), 0))
    return X


def fit(X, y, l2=1e-2):
    Xs = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xs.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xs @ w))
        Wd = p * (1 - p) + 1e-9
        g = Xs.T @ (y - p) - l2 * w
        H = (Xs * Wd[:, None]).T @ Xs + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def auc(y, s):
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])))


def cv_auc(X, y, reps=20, k=5):
    out = []
    for _ in range(reps):
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, k)
        s = np.zeros(len(y))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            mu = X[tr].mean(0)
            sd = X[tr].std(0) + 1e-9
            w = fit((X[tr] - mu) / sd, y[tr])
            s[f] = predict(w, (X[f] - mu) / sd)
        out.append(auc(y, s))
    return float(np.nanmean(out)), float(np.nanstd(out))


y = np.array([r["plateau"] for r in rows], float)
Xb = design(rows, BASE)
Xf = design(rows, BASE + CLASS)
ab, sb = cv_auc(Xb, y)
af, sf = cv_auc(Xf, y)
print(
    f"pooled: plateau rate {y.mean():.2f}; CV AUC baseline (A, Fano, peak) {ab:.3f} +/- {sb:.3f}; full (+waveform duration, PT ratio, depth, firing rate) {af:.3f} +/- {sf:.3f}; delta {af-ab:+.3f}"
)
res = {"n": len(rows), "auc_base": ab, "auc_full": af, "delta": af - ab, "coef": {}}
mu = Xf.mean(0)
sd = Xf.std(0) + 1e-9
w = fit((Xf - mu) / sd, y)
names = BASE + CLASS + ["sess_c1", "sess_c2"]
boots = []
for _ in range(500):
    b = rng.integers(0, len(y), len(y))
    boots.append(fit((Xf[b] - mu) / sd, y[b]))
boots = np.array(boots)
lo = np.percentile(boots, 2.5, 0)
hi = np.percentile(boots, 97.5, 0)
for i, nm in enumerate(names):
    print(
        f"  coef {nm:9s} {w[i+1]:+.2f}  [{lo[i+1]:+.2f}, {hi[i+1]:+.2f}]"
        + ("  <- excludes 0" if lo[i + 1] > 0 or hi[i + 1] < 0 else "")
    )
    res["coef"][nm] = [float(w[i + 1]), float(lo[i + 1]), float(hi[i + 1])]
print("\nper session (class coefficients, standardised, bootstrap 95%):")
for s in ("static", "c1", "c2"):
    rs = [r for r in rows if r["session"] == s]
    ys = np.array([r["plateau"] for r in rs], float)
    if ys.sum() < 5 or (1 - ys).sum() < 5:
        print(f"  {s}: too few plateau or non-plateau units ({int(ys.sum())} / {int((1-ys).sum())})")
        continue
    Xs = design(rs, BASE + CLASS, sess_ind=False)
    m_ = Xs.mean(0)
    sd_ = Xs.std(0) + 1e-9
    ws = fit((Xs - m_) / sd_, ys)
    bb = np.array(
        [fit((Xs[b] - m_) / sd_, ys[b]) for b in (rng.integers(0, len(ys), len(ys)) for _ in range(300))]
    )
    l_ = np.percentile(bb, 2.5, 0)
    h_ = np.percentile(bb, 97.5, 0)
    ab_, _ = cv_auc(design(rs, BASE, sess_ind=False), ys, reps=10)
    af_, _ = cv_auc(Xs, ys, reps=10)
    print(
        f"  {s} (n={len(rs)}, plateau {int(ys.sum())}): AUC base {ab_:.3f} full {af_:.3f} delta {af_-ab_:+.3f}; "
        + ", ".join(
            f"{nm} {ws[i+1]:+.2f} [{l_[i+1]:+.2f},{h_[i+1]:+.2f}]"
            for i, nm in enumerate(BASE + CLASS)
            if nm in CLASS
        )
    )
    res[s] = {"n": len(rs), "plateau": int(ys.sum()), "auc_base": ab_, "auc_full": af_}
json.dump(res, open(os.path.join(HERE, "results_q27.json"), "w"), indent=1)
print("[saved] results_q27.json")
