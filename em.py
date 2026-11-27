"""
EM for the 36-region LDS with a matrix-normal connectome prior
(Protocol and Model Spec, sections 5, 9 and 10; work-plan phase 3).

E-step: Kalman filter + RTS smoother (kalman.py) on every segment; expected sufficient
statistics pooled over all transitions t -> t+1 inside segments (N in total):
    S00 = sum (x_{t|T} x_{t|T}^T + P_{t|T})          t = 1..T-1
    S11 = sum (x_{t+1|T} x_{t+1|T}^T + P_{t+1|T})
    S10 = sum (x_{t+1|T} x_{t|T}^T + P_{t+1,t|T})
M-step (closed form; lambda = kappa N):
    A   = (S10 + lambda A0)(S00 + lambda I)^-1
    Q   = [S11 - A S10^T - S10 A^T + A S00 A^T + lambda (A - A0)(A - A0)^T] / (N + n)
    r_i = (1/N_i) sum_{t: i visible} [(y_ti - x_ti|T)^2 + (P_t|T)_ii]
Floors: eigenvalues of Q at 1e-6, every r_i at 1e-3. Sigma0 is fixed (not learned) and the
first-step term is excluded from the A and Q updates.
Objective (log posterior), maximised and checked for monotonicity every iteration:
    L = log p(y | theta) - (n/2) log|Q| - (lambda/2) tr[Q^-1 (A - A0)(A - A0)^T]
Stop when the gain in L per observed entry is below 1e-6 for 5 consecutive iterations, or
at max_iter. A decrease of more than 1e-8 (relative) raises EMDecreaseError.
Hidden regions (R3) have no r_i; their rows/columns of A and Q are learned only through
the prior and their coupling to visible regions.

Optional connectome prior on Q (tier B amendment 1, approved Nov 26): Q ~ IW(nu Q_target),
nu = kappa_Q N, conjugate with the matrix-normal prior on A. The Q update becomes
    Q = [S11 - A S10^T - S10 A^T + A S00 A^T + lambda (A - A0)(A - A0)^T + nu Q_target] / (N + n + nu)
and L gains - (nu/2) log|Q| - (nu/2) tr(Q^-1 Q_target). kappa_Q = 0 (default) reproduces the
A-only EM exactly. Q_target comes from prior.q_target().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

import kalman as K

Q_FLOOR, R_FLOOR = 1e-6, 1e-3


class EMDecreaseError(RuntimeError):
    pass


@dataclass
class Params:
    A: np.ndarray
    Q: np.ndarray
    r: np.ndarray
    Sigma0: np.ndarray

    def copy(self):
        return Params(self.A.copy(), self.Q.copy(), self.r.copy(), self.Sigma0.copy())


@dataclass
class Stats:
    S00: np.ndarray
    S11: np.ndarray
    S10: np.ndarray
    N: int                    # transitions
    r_num: np.ndarray         # per-region residual sums over visible frames
    N_i: np.ndarray           # per-region visible-frame counts
    loglik: float
    n_obs: int                # observed entries


@dataclass
class FitResult:
    params: Params
    A0: np.ndarray
    kappa: float
    logpost: float
    history: List[float]
    n_iter: int
    converged: bool
    flags: List[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
def _sym(M):
    return 0.5 * (M + M.T)


def _floor_eig(M, floor):
    w, V = np.linalg.eigh(_sym(M))
    return _sym((V * np.maximum(w, floor)) @ V.T)


def _masks(segments, obs):
    """Per-segment (n,) or (T, n) masks; non-finite entries are always missing."""
    out = []
    for k, Y in enumerate(segments):
        m = obs[k] if isinstance(obs, (list, tuple)) else obs
        m = np.ones(Y.shape[1], bool) if m is None else np.asarray(m, bool)
        out.append(m)
    return out


def e_step(segments: Sequence[np.ndarray], obs, p: Params, steady_state: bool = True,
           keep: bool = False):
    """Expected sufficient statistics over all segments. keep=True also returns the list of
    (FilterResult, SmootherResult) per segment (for scoring)."""
    n = p.A.shape[0]
    S00 = np.zeros((n, n)); S11 = np.zeros((n, n)); S10 = np.zeros((n, n))
    r_num = np.zeros(n); N_i = np.zeros(n, dtype=np.int64)
    N = 0; ll = 0.0; n_obs = 0
    kept = []
    for Y, m in zip(segments, _masks(segments, obs)):
        Y = np.asarray(Y, np.float64)
        fr, sr = K.filter_smooth(Y, m, p.A, p.Q, p.r, p.Sigma0, steady_state=steady_state)
        xs, Ps = sr.x_smooth, sr.P_smooth
        S00 += xs[:-1].T @ xs[:-1] + Ps[:-1].sum(0)
        S11 += xs[1:].T @ xs[1:] + Ps[1:].sum(0)
        S10 += xs[1:].T @ xs[:-1] + sr.P_lag.sum(0)
        N += Y.shape[0] - 1
        M = (np.broadcast_to(m, Y.shape) if m.ndim == 1 else m) & np.isfinite(Y)
        res = np.where(M, (np.where(M, Y, 0.0) - xs) ** 2 + np.einsum('tii->ti', Ps), 0.0)
        r_num += res.sum(0); N_i += M.sum(0)
        ll += fr.loglik; n_obs += fr.n_obs
        if keep:
            kept.append((fr, sr))
    st = Stats(S00, S11, S10, N, r_num, N_i, ll, n_obs)
    return (st, kept) if keep else st


def m_step(st: Stats, p: Params, A0: np.ndarray, kappa: float, Q_target=None, kappa_Q: float = 0.0) -> Params:
    n = A0.shape[0]
    lam = kappa * st.N
    nu = kappa_Q * st.N if Q_target is not None else 0.0
    I = np.eye(n)
    A = np.linalg.solve((st.S00 + lam * I).T, (st.S10 + lam * A0).T).T
    D = A - A0
    R = st.S11 - A @ st.S10.T - st.S10 @ A.T + A @ st.S00 @ A.T + lam * D @ D.T
    Q = (R + nu * Q_target) / (st.N + n + nu) if nu > 0 else R / (st.N + n)
    Q = _floor_eig(Q, Q_FLOOR)
    r = p.r.copy()
    vis = st.N_i > 0
    r[vis] = np.maximum(st.r_num[vis] / st.N_i[vis], R_FLOOR)
    return Params(A, Q, r, p.Sigma0)


def log_posterior(st: Stats, p: Params, A0: np.ndarray, kappa: float, Q_target=None,
                  kappa_Q: float = 0.0) -> float:
    n = A0.shape[0]
    lam = kappa * st.N
    nu = kappa_Q * st.N if Q_target is not None else 0.0
    sign, logdet = np.linalg.slogdet(p.Q)
    L = st.loglik - 0.5 * n * logdet
    if lam > 0:
        D = p.A - A0
        L -= 0.5 * lam * np.trace(np.linalg.solve(p.Q, D @ D.T))
    if nu > 0:
        L -= 0.5 * nu * logdet + 0.5 * nu * np.trace(np.linalg.solve(p.Q, Q_target))
    return float(L)


# --------------------------------------------------------------------------------------
# Initialisation (spec section 10)
# --------------------------------------------------------------------------------------
def visible_regions(segments, obs):
    """Regions observed in at least one frame of the data."""
    n = segments[0].shape[1]
    vis = np.zeros(n, bool)
    for Y, m in zip(segments, _masks(segments, obs)):
        M = (np.broadcast_to(m, Y.shape) if m.ndim == 1 else m) & np.isfinite(Y)
        vis |= M.any(0)
    return vis


def hf_floor(segments, vis, fs=1.2, band=(0.45, 0.59), nperseg=128):
    """Flat high-frequency power (Welch, 128-frame windows) as a fraction of variance,
    per visible region, averaged over segments weighted by length."""
    from scipy.signal import welch
    num = np.zeros(int(vis.sum())); den = 0.0
    for Y in segments:
        X = Y[:, vis]
        if len(X) < 8:
            continue
        f, P = welch(X, fs=fs, nperseg=min(nperseg, len(X)), axis=0)
        b = (f >= band[0]) & (f <= band[1])
        if not b.any():
            continue
        num += len(X) * P[b].mean(0) * (fs / 2) / X.var(0)
        den += len(X)
    return num / den


def init_params(segments, obs, A0, ridge_lambda=100.0, jitter_rng=None, fs=1.2) -> Params:
    """Section-10 initialisation, identical for all variants.
    A: ridge VAR(1) on the visible block; entries touching hidden regions = A0.
    r: high-frequency floor fraction, floored at 0.05 (hidden regions: mean visible value, unused).
    Sigma_x = Cov(y) - diag(r) (hidden block: mean visible variance on the diagonal), eig floor 1e-4.
    Q = Sigma_x - A Sigma_x A^T, projected to PSD, floored at 1e-4. Sigma0 = Sigma_x (fixed).
    jitter_rng: 10% multiplicative jitter on A (restart 3 of the phase-4 sanity fits)."""
    segs = [np.asarray(Y, np.float64) for Y in segments]
    n = segs[0].shape[1]
    vis = visible_regions(segs, obs)
    v = np.flatnonzero(vis)
    for Y in segs:
        if not np.all(np.isfinite(Y[:, v])):
            raise ValueError("init_params expects visible regions fully observed (R1/R2/R3 design)")
    Xv = [Y[:, v] for Y in segs]
    S00 = sum(X[:-1].T @ X[:-1] for X in Xv); S10 = sum(X[1:].T @ X[:-1] for X in Xv)
    Av = np.linalg.solve((S00 + ridge_lambda * np.eye(len(v))).T, S10.T).T
    A = np.array(A0, np.float64, copy=True)
    A[np.ix_(v, v)] = Av
    if jitter_rng is not None:
        A = A * (1.0 + 0.1 * jitter_rng.standard_normal(A.shape))
    r = np.full(n, np.nan)
    r[v] = np.maximum(hf_floor(segs, vis, fs=fs), 0.05)
    r[~vis] = np.mean(r[v])
    C = np.cov(np.concatenate(Xv).T)
    Sx = np.zeros((n, n))
    Sx[np.ix_(v, v)] = C - np.diag(r[v])
    h = np.flatnonzero(~vis)
    Sx[h, h] = np.mean(np.diag(Sx)[v])
    Sx = _floor_eig(Sx, 1e-4)
    Q = _floor_eig(Sx - A @ Sx @ A.T, 1e-4)
    return Params(A, Q, r, Sx)


def warm_start(p: Params, A0, vis) -> Params:
    """Warm start from another fit (e.g. M0): keep its parameters but reset A's entries
    that touch hidden regions to the new model's A0 (spec section 10 init rule)."""
    q = p.copy()
    h = ~np.asarray(vis, bool)
    q.A[h, :] = A0[h, :]; q.A[:, h] = A0[:, h]
    return q


# --------------------------------------------------------------------------------------
def fit(segments, obs, A0, kappa, init: Optional[Params] = None, max_iter=300, tol=1e-6,
        patience=5, steady_state=True, rho_shrink=True, verbose=False, Q_target=None,
        kappa_Q: float = 0.0) -> FitResult:
    """Fit (A, Q, r) by EM with prior mean A0 and relative strength kappa (lambda = kappa N);
    optionally with the inverse-Wishart prior on Q (Q_target, kappa_Q; nu = kappa_Q N)."""
    A0 = np.asarray(A0, np.float64)
    p = init.copy() if init is not None else init_params(segments, obs, A0)
    qp = dict(Q_target=Q_target, kappa_Q=kappa_Q)
    st = e_step(segments, obs, p, steady_state)
    L = log_posterior(st, p, A0, kappa, **qp)
    hist = [L]; small = 0; converged = False; it = 0
    for it in range(1, max_iter + 1):
        p = m_step(st, p, A0, kappa, **qp)
        st = e_step(segments, obs, p, steady_state)
        Lnew = log_posterior(st, p, A0, kappa, **qp)
        if Lnew < L - 1e-8 * abs(L):
            raise EMDecreaseError(f"log posterior decreased at iteration {it}: {L:.10g} -> {Lnew:.10g}")
        gain = (Lnew - L) / st.n_obs
        hist.append(Lnew); L = Lnew
        if verbose:
            print(f"  iter {it:3d}  L = {L:.6f}  gain/obs = {gain:.2e}")
        small = small + 1 if gain < tol else 0
        if small >= patience:
            converged = True
            break
    flags = []
    if not converged:
        flags.append('max_iter')
    rho = float(np.max(np.abs(np.linalg.eigvals(p.A))))
    if rho >= 0.98:
        flags.append('rho>=0.98')
    if rho_shrink and rho >= 1.0:
        p.A = p.A * (0.99 / rho)
        flags.append('rho_shrunk')
    lam = kappa * st.N
    diag = {
        'n_iter': it, 'logpost': L, 'rho_A': rho,
        'cond_S00_lam': float(np.linalg.cond(st.S00 + lam * np.eye(len(A0)))),
        'min_eig_Q': float(np.linalg.eigvalsh(p.Q).min()),
        'r_max': float(np.nanmax(p.r[st.N_i > 0])), 'r_min': float(np.nanmin(p.r[st.N_i > 0])),
        'frac_neg_A0_offdiag': float(np.mean(A0[~np.eye(len(A0), dtype=bool)] < 0)),
        'N': st.N, 'n_obs': st.n_obs, 'kappa': kappa, 'kappa_Q': kappa_Q if Q_target is not None else 0.0,
        'flags': flags,
    }
    return FitResult(p, A0, kappa, L, hist, it, converged, flags, diag)


# --------------------------------------------------------------------------------------
# Scores (spec section 11)
# --------------------------------------------------------------------------------------
def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.sqrt(a @ a), np.sqrt(b @ b)
    if da < 1e-12 * max(1.0, db) or db == 0:
        return 0.0                      # constant estimate (decoupled hidden region): chance level
    return float(a @ b / (da * db))


def fisher_mean(rs):
    rs = np.clip(np.asarray(rs, float), -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def hidden_r(segments_truth, results, hidden, which='smooth'):
    """T2: Pearson r between estimated and observed hidden series, per hidden region, pooled
    over a fly's segments, then Fisher-z averaged. segments_truth: that fly's segments with
    the hidden columns intact; results: matching (FilterResult, SmootherResult) list."""
    h = np.flatnonzero(hidden)
    est = np.concatenate([(sr.x_smooth if which == 'smooth' else fr.x_filt)[:, h] for fr, sr in results])
    tru = np.concatenate([Y[:, h] for Y in segments_truth])
    per = [pearson(est[:, j], tru[:, j]) for j in range(len(h))]
    return fisher_mean(per), per


def one_step_r2(Y, fr, vis, t_from=0):
    """T1: 1 - SSE/SS over visible entries, predictions C x_{t+1|t} vs y_{t+1}, t+1 >= t_from."""
    v = np.flatnonzero(vis)
    idx = np.arange(max(1, t_from), Y.shape[0])
    e = Y[np.ix_(idx, v)] - fr.x_pred[np.ix_(idx, v)]
    return float(1.0 - np.sum(e * e) / np.sum(Y[np.ix_(idx, v)] ** 2))


def fit_record(res: FitResult, config: dict) -> dict:
    """One JSON-able record per fit (spec section 15 deliverables)."""
    return {'config': config, **{k: (v if not isinstance(v, np.generic) else v.item())
                                 for k, v in res.diagnostics.items()}}
