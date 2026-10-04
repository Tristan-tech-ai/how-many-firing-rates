"""
GLM spike-train channel, EXACT (no Monte Carlo, so no error bars are needed).

Model. Window T split into Kbins bins of width dt. Symbol x sets a baseline rate lam_x (Hz). Conditional
intensity in bin t with spike history y:
        lam_t = lam_x * exp( sum_{j>=1} h_j * y_{t-j} ),      p_t = 1 - exp(-lam_t*dt)
h is a refractory kernel specified in MILLISECONDS, h(tau) = -amp * exp(-tau/tau_ms), re-discretised for each
bin width (so a change of Kbins is a genuine sensitivity check, not a change of neuron). History is reset at
window onset (windows are independent channel uses) -- flagged as a modelling limit in the readout.

Two INDEPENDENT exact computations, which must agree (that agreement is a check that can fail):
  glm_word_P  : enumerate all 2^Kbins binary words, exact product of per-bin conditionals -> P[x, word]
  glm_count_P : dynamic programming over (last-J-bins state, running count)                -> P[x, count]
Marginalising the word matrix to counts must reproduce the DP matrix.
"""

import numpy as np

J_CAP = 10  # history bins retained (kernel truncated at ~3 tau); keeps the DP state space at 2^J


def _kernel(h_ms, dt, Kbins):
    """Discretise h(tau) = -amp*exp(-tau/tau_ms) onto bin lags. Returns h[0..J-1] for lags 1..J bins."""
    if h_ms is None:
        return np.zeros(0)
    amp = float(h_ms["amp"])
    tau = float(h_ms["tau_ms"])
    J = int(min(J_CAP, Kbins, max(1, np.ceil(3.0 * tau / (dt * 1000.0)))))
    lags_ms = np.arange(1, J + 1) * dt * 1000.0
    return -amp * np.exp(-lags_ms / tau)


def glm_word_P(rates_hz, h_ms=None, T=0.05, Kbins=14):
    """Exact P[x, word] over all 2^Kbins words. Rows sum to 1."""
    rates_hz = np.asarray(rates_hz, float)
    dt = T / Kbins
    h = _kernel(h_ms, dt, Kbins)
    J = len(h)
    W = 1 << Kbins
    idx = np.arange(W, dtype=np.int64)
    Y = ((idx[:, None] >> np.arange(Kbins)[None, :]) & 1).astype(np.float64)  # Y[w, t] = bit t of word w
    S = np.zeros((W, Kbins))
    for j in range(1, J + 1):  # history sum, reset at onset
        S[:, j:] += h[j - 1] * Y[:, : Kbins - j]
    ES = np.exp(S)
    rows = []
    for lam in rates_hz:
        if lam <= 1e-12:  # silence: only the empty word
            r = np.zeros(W)
            r[0] = 1.0
            rows.append(r)
            continue
        p = 1.0 - np.exp(-lam * ES * dt)
        p = np.clip(p, 1e-300, 1.0 - 1e-16)
        logP = (Y * np.log(p) + (1.0 - Y) * np.log1p(-p)).sum(axis=1)
        rows.append(np.exp(logP))
    P = np.vstack(rows)
    return P / P.sum(axis=1, keepdims=True)


def glm_count_P(rates_hz, h_ms=None, T=0.05, Kbins=14):
    """Exact P[x, count] by DP over (last-J-bins state, running count). Exact for any Kbins."""
    rates_hz = np.asarray(rates_hz, float)
    dt = T / Kbins
    h = _kernel(h_ms, dt, Kbins)
    J = len(h)
    nS = 1 << J if J > 0 else 1
    mask = nS - 1
    # history sum for each state: state bit b (b=0 -> most recent bin, lag 1)
    Ssum = np.zeros(nS)
    for s in range(nS):
        tot = 0.0
        for b in range(J):
            if (s >> b) & 1:
                tot += h[b]
        Ssum[s] = tot
    ES = np.exp(Ssum)
    rows = []
    for lam in rates_hz:
        if lam <= 1e-12:
            r = np.zeros(Kbins + 1)
            r[0] = 1.0
            rows.append(r)
            continue
        p = 1.0 - np.exp(-lam * ES * dt)
        p = np.clip(p, 0.0, 1.0)
        dp = np.zeros((nS, Kbins + 1))
        dp[0, 0] = 1.0  # history reset at onset
        for _ in range(Kbins):
            nxt = np.zeros_like(dp)
            for s in range(nS):
                col = dp[s]
                if not col.any():
                    continue
                ps = p[s]
                s0 = ((s << 1) & mask) if J > 0 else 0  # no spike
                s1 = (((s << 1) | 1) & mask) if J > 0 else 0  # spike
                nxt[s0] += col * (1.0 - ps)
                nxt[s1, 1:] += col[:-1] * ps
            dp = nxt
        r = dp.sum(axis=0)
        rows.append(r)
    P = np.vstack(rows)
    return P / P.sum(axis=1, keepdims=True)


def words_to_counts(Pw, Kbins):
    """Marginalise a word matrix to counts (for the two-implementation cross-check)."""
    idx = np.arange(1 << Kbins, dtype=np.int64)
    popc = np.zeros(1 << Kbins, dtype=np.int64)
    for b in range(Kbins):
        popc += (idx >> b) & 1
    out = np.zeros((Pw.shape[0], Kbins + 1))
    for c in range(Kbins + 1):
        m = popc == c
        if m.any():
            out[:, c] = Pw[:, m].sum(axis=1)
    return out


def glm_induced_fano(rates_hz, h_ms=None, T=0.05, Kbins=14, q=None):
    """Induced count Fano (variance/mean of the count distribution), averaged over used symbols under q.
    This is what places the GLM against the COM-Poisson family at MATCHED dispersion."""
    P = glm_count_P(rates_hz, h_ms, T, Kbins)
    y = np.arange(P.shape[1])
    m = P @ y
    v = P @ (y * y) - m**2
    if q is None:
        q = np.ones(P.shape[0]) / P.shape[0]
    q = np.asarray(q, float)
    q = q / q.sum()
    use = (q > 1e-6) & (m > 1e-9)
    if not use.any():
        return float("nan")
    return float((q[use] * (v[use] / m[use])).sum() / q[use].sum())
