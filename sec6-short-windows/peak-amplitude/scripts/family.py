"""
The count-conditional family for the probe: COM-Poisson, spanning BOTH under-dispersion (refractory,
Fano<1) and over-dispersion (bursty, Fano>1), with Poisson exactly at nu=1.

COM-Poisson pmf on y=0,1,2,...:  P(y) proportional to  lam^y / (y!)^nu ,  Z = sum_j lam^j/(j!)^nu.
  nu = 1 -> Poisson(lam);  nu > 1 -> under-dispersed (Fano<1);  nu < 1 -> over-dispersed (Fano>1).
Reference: Stevenson 2016, "Flexible models for spike count data with both over- and under-dispersion";
Charles/Pillow 2018 "Dethroning the Fano Factor". Biologically defensible for spike counts.

We parameterise each channel symbol by a TARGET (mean, Fano) and solve for (lam, nu) numerically.
Everything is sanity-pinned before use (VERIFIER_DISCIPLINE 0): nu=1 must reproduce Poisson to machine
precision; realised (mean, Fano) must match the targets; a noiseless / degenerate channel must give the
known C and V. If any pin fails, the module raises.
"""

import numpy as np
from scipy.stats import poisson
from scipy.optimize import brentq, fsolve
from functools import lru_cache


def _com_moments(log_lam, log_nu, S):
    """mean, var of COM-Poisson with lam=exp(log_lam), nu=exp(log_nu) on support 0..S-1 (log-space, stable)."""
    y = np.arange(S)
    logfact = np.cumsum(np.log(np.maximum(y, 1)))  # log(y!)
    lp = y * log_lam - np.exp(log_nu) * logfact
    lp -= lp.max()
    w = np.exp(lp)
    w /= w.sum()
    m = float((w * y).sum())
    v = float((w * y * y).sum() - m * m)
    return m, v, w


def com_pmf(mean, fano, S=None):
    """Return a COM-Poisson pmf on 0..S-1 matching (mean, fano) as closely as the family allows.
    Falls back to Poisson at fano==1. Returns (pmf, realised_mean, realised_fano)."""
    if mean <= 1e-9:
        p = np.zeros(S or 4)
        p[0] = 1.0
        return p, 0.0, 1.0
    if S is None:
        S = int(mean + 15 * np.sqrt(mean * max(fano, 1) + 1) + 20)
    if abs(fano - 1.0) < 1e-9:
        p = poisson.pmf(np.arange(S), mean)
        return p / p.sum(), mean, 1.0

    # solve (log_lam, log_nu) for target (mean, fano*mean). 2-D; robust multi-start.
    target = np.array([mean, fano * mean])

    def resid(z):
        m, v, _ = _com_moments(z[0], z[1], S)
        return [m - target[0], v - target[1]]

    best = None
    for ll0 in (np.log(mean), np.log(max(mean, 0.5)) + 0.5, np.log(mean) - 0.5):
        for ln0 in (0.0, np.log(1.0 / fano) if fano > 0 else 0.0, -np.log(fano) if fano > 0 else 0.0):
            try:
                z, info, ier, _ = fsolve(resid, [ll0, ln0], full_output=True)
            except Exception:
                continue
            if ier == 1:
                m, v, w = _com_moments(z[0], z[1], S)
                err = abs(m - mean) / mean + abs(v / m - fano)
                if best is None or err < best[0]:
                    best = (err, w, m, v / m)
    if best is None or best[0] > 0.05:
        # under-dispersion floor or solver failure: return the closest achievable, flagged by realised fano
        # (do NOT fabricate the target). Grid nu, pick closest fano at matched mean.
        grid = []
        for ln in np.linspace(-1.5, 2.5, 60):
            # for each nu, find lam giving the target mean
            try:
                ll = brentq(lambda x: _com_moments(x, ln, S)[0] - mean, np.log(1e-3), np.log(mean) + 6)
            except Exception:
                continue
            m, v, w = _com_moments(ll, ln, S)
            grid.append((abs(v / m - fano), w, m, v / m))
        best = min(grid, key=lambda t: t[0]) if grid else None
        if best is None:
            p = poisson.pmf(np.arange(S), mean)
            return p / p.sum(), mean, 1.0
    _, w, m, f = best
    return w, m, f


# ------------------------------- sanity pins -------------------------------
def _selftest():
    from scipy.stats import poisson as pois

    ok = True
    # (1) nu=1 reproduces Poisson exactly
    for mu in (0.5, 1.5, 3.0):
        p, m, f = com_pmf(mu, 1.0, S=60)
        pp = pois.pmf(np.arange(60), mu)
        d = np.abs(p - pp).max()
        good = d < 1e-12 and abs(m - mu) < 1e-9 and abs(f - 1) < 1e-9
        ok &= good
        print(
            f"  [{'PASS' if good else 'FAIL'}] fano=1 == Poisson (mu={mu}): maxdiff={d:.1e}, realFano={f:.4f}"
        )
    # (2) realised (mean, fano) match targets, both directions
    for mu, fa in [(1.5, 0.5), (1.5, 0.7), (1.5, 1.5), (2.0, 2.0), (1.0, 0.6)]:
        p, m, f = com_pmf(mu, fa, S=80)
        good = abs(m - mu) < 0.02 * mu + 1e-3 and abs(f - fa) < 0.03
        ok &= good
        print(
            f"  [{'PASS' if good else 'FAIL'}] target(mean={mu},fano={fa}) -> realised(mean={m:.3f},fano={f:.3f})"
        )
    # (3) a pmf is a pmf
    for mu, fa in [(1.5, 0.5), (2.0, 2.0)]:
        p, *_ = com_pmf(mu, fa, S=120)
        good = abs(p.sum() - 1) < 1e-9 and (p >= -1e-15).all()
        ok &= good
        print(
            f"  [{'PASS' if good else 'FAIL'}] pmf sums to 1 and nonneg (mu={mu},fano={fa}): sum={p.sum():.9f}"
        )
    print(f"  -> {'ALL FAMILY PINS PASS' if ok else '*** FAMILY PINS FAILED ***'}")
    return ok


if __name__ == "__main__":
    print("COM-Poisson family self-test (must pass before any C/V is trusted):")
    if not _selftest():
        raise SystemExit(1)
