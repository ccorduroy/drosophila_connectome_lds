"""
Kalman filter and Rauch-Tung-Striebel smoother for the 36-region LDS
(Protocol and Model Spec, sections 5 and 8; work-plan phase 2, gate G1).

Model, per segment (segments restart; never joined across a gap):
    x_1 ~ N(0, Sigma0)
    x_{t+1} = A x_t + w_t,          w_t ~ N(0, Q)            Q full, PD
    y_t     = C_t x_t + v_t,        v_t ~ N(0, R_{V_t V_t})  R = diag(r)
C_t is the selection of the visible regions V_t at frame t, so hiding a region only
removes rows of C_t; x, A, Q, Sigma0 always stay n x n (n = 36).

Conventions
    Y    : (T, n) float64, one segment, frames x regions. (The npz stores 36 x T; use
           load_standardised() below, which transposes.)
    obs  : visibility mask, either (n,) bool (same visible set at every frame, the
           R1/R3 case) or (T, n) bool (time-varying). Non-finite entries of Y are
           also treated as missing. Values of Y at missing entries are never read.
    r    : (n,) observation-noise variances (diagonal of R). A 2-D R is accepted and
           its diagonal used.

Numerics (spec section 8): double precision; S_t handled by Cholesky solves, never
explicit inverses; Joseph-form P_{t|t}; every covariance symmetrised after each step;
an all-missing frame skips the update. Optional steady-state mode (mean recursions
compiled with numba when installed, NumPy otherwise) freezes the gain and
covariances once the Riccati recursion has converged (only when the mask is constant
within the segment); it must agree with the exact recursion to < 1e-4 relative
(test U9) before it is relied on.

Data access goes only through build_dataset.load(), as the spec requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

LOG2PI = np.log(2.0 * np.pi)

# ---- compiled mean recursions for the steady-state block (numba if available) ----------
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                   # pure-NumPy fallback, same results
    HAVE_NUMBA = False
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _ss_filter_means(A, Kg, Yv, v, xf0, x_pred, x_filt, E, t0):
    """Frames t0..T-1 with a frozen gain: x_pred = A x_filt[t-1]; x_filt = x_pred + K e."""
    xf = xf0.copy()
    for s in range(t0, x_pred.shape[0]):
        xp = A @ xf
        e = Yv[s] - xp[v]
        xf = xp + Kg @ e
        x_pred[s] = xp; x_filt[s] = xf; E[s - t0] = e


@njit(cache=True)
def _ss_smoother_means(J, x_filt, x_pred, xs, t_hi, t_lo):
    """Frames t = t_hi down to t_lo (inclusive) with a constant J."""
    for t in range(t_hi, t_lo - 1, -1):
        xs[t] = x_filt[t] + J @ (xs[t + 1] - x_pred[t + 1])


def _sym(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + P.T)


def _rel_change(P_new: np.ndarray, P_old: np.ndarray) -> float:
    return float(np.max(np.abs(P_new - P_old)) / max(np.max(np.abs(P_new)), 1e-300))


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
@dataclass
class FilterResult:
    x_pred: np.ndarray        # (T, n)    x_{t|t-1};  x_pred[0] = 0
    P_pred: np.ndarray        # (T, n, n) P_{t|t-1};  P_pred[0] = Sigma0
    x_filt: np.ndarray        # (T, n)    x_{t|t}
    P_filt: np.ndarray        # (T, n, n) P_{t|t}
    loglik: float             # log p(y_{1:T}) of this segment
    n_obs: int                # number of observed entries (sum of m_t)
    steady_from: Optional[int]  # first frame whose covariances were frozen (None = exact throughout)

    @property
    def y_pred(self) -> np.ndarray:
        """One-step predictions of every region, x_{t|t-1} (rows = frames). T1 uses
        y_pred[t+1] (= C x_{t+1|t}) against y[t+1] on visible regions."""
        return self.x_pred


@dataclass
class SmootherResult:
    x_smooth: np.ndarray      # (T, n)      x_{t|T}
    P_smooth: np.ndarray      # (T, n, n)   P_{t|T}
    P_lag: np.ndarray         # (T-1, n, n) P_lag[t] = Cov(x_{t+1}, x_t | y_{1:T}) = P_{t+1|T} J_t^T
    steady_range: Optional[tuple]  # (lo, hi): frames t in [lo, hi) whose smoother covs were frozen


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------
def _prepare(Y, obs, r, A, Q, Sigma0):
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("Y must be (T, n)")
    T, n = Y.shape
    A = np.asarray(A, np.float64); Q = np.asarray(Q, np.float64); Sigma0 = np.asarray(Sigma0, np.float64)
    r = np.asarray(r, np.float64)
    if r.ndim == 2:
        r = np.diag(r).copy()
    for name, M in (("A", A), ("Q", Q), ("Sigma0", Sigma0)):
        if M.shape != (n, n):
            raise ValueError(f"{name} must be ({n}, {n}), got {M.shape}")
    if r.shape != (n,):
        raise ValueError(f"r must be ({n},), got {r.shape}")
    if obs is None:
        obs = np.ones(n, bool)
    obs = np.asarray(obs, bool)
    finite = np.isfinite(Y)
    if obs.shape == (n,):
        mask = np.broadcast_to(obs, (T, n)) & finite
    elif obs.shape == (T, n):
        mask = obs & finite
    else:
        raise ValueError(f"obs must be ({n},) or ({T}, {n}), got {obs.shape}")
    if np.any(r[np.any(mask, axis=0)] <= 0):
        raise ValueError("observation-noise variances must be positive for observed regions")
    constant_mask = bool(np.all(mask == mask[0]))
    return Y, mask, r, A, Q, _sym(Sigma0), constant_mask


# --------------------------------------------------------------------------------------
# Filter
# --------------------------------------------------------------------------------------
def kalman_filter(Y, obs, A, Q, r, Sigma0, steady_state: bool = False,
                  ss_tol: float = 1e-12, ss_min_burn: int = 10) -> FilterResult:
    """Exact time-varying Kalman filter for one segment (spec section 8).

    steady_state=True freezes K_t, P_{t|t}, P_{t+1|t} and S_t once the prediction
    covariance changes by less than ss_tol (relative max-abs) between frames, after at
    least ss_min_burn frames. Only used when the visible set is constant in the segment;
    otherwise the exact recursion runs regardless.
    """
    Y, mask, r, A, Q, Sigma0, constant_mask = _prepare(Y, obs, r, A, Q, Sigma0)
    T, n = Y.shape
    I = np.eye(n)
    x_pred = np.empty((T, n)); P_pred = np.empty((T, n, n))
    x_filt = np.empty((T, n)); P_filt = np.empty((T, n, n))
    ll = 0.0
    n_obs = int(mask.sum())
    steady_from = None
    use_ss = steady_state and constant_mask and mask[0].any()

    xp = np.zeros(n); Pp = Sigma0.copy()
    t = 0
    while t < T:
        if t > 0:
            xp = A @ x_filt[t - 1]
            Pp = _sym(A @ P_filt[t - 1] @ A.T + Q)
        x_pred[t] = xp; P_pred[t] = Pp
        v = np.flatnonzero(mask[t])
        if v.size == 0:                          # all missing: skip the update
            x_filt[t] = xp; P_filt[t] = Pp
            t += 1
            continue
        PCt = Pp[:, v]                           # P C^T          (n, m)
        S = _sym(Pp[np.ix_(v, v)] + np.diag(r[v]))   # C P C^T + R_t  (m, m)
        L = np.linalg.cholesky(S)                # lower
        K = cho_solve((L, True), PCt.T).T        # P C^T S^{-1}   (n, m)
        e = Y[t, v] - xp[v]
        x_filt[t] = xp + K @ e
        IKC = I.copy(); IKC[:, v] -= K           # I - K C
        P_filt[t] = _sym(IKC @ Pp @ IKC.T + (K * r[v]) @ K.T)   # Joseph form
        alpha = solve_triangular(L, e, lower=True)
        ll -= 0.5 * (v.size * LOG2PI + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)

        if use_ss and t >= ss_min_burn and _rel_change(Pp, P_pred[t - 1]) < ss_tol:
            # ---- steady state from frame t+1 on: covariances, gain and S fixed ----
            steady_from = t + 1
            if t + 1 < T:
                Pp_ss = _sym(A @ P_filt[t] @ A.T + Q)
                # Gain and S recomputed at the frozen P_pred so the frozen block is self-consistent
                PCt = Pp_ss[:, v]
                S = _sym(Pp_ss[np.ix_(v, v)] + np.diag(r[v]))
                L = np.linalg.cholesky(S)
                K = cho_solve((L, True), PCt.T).T
                IKC = I.copy(); IKC[:, v] -= K
                Pf_ss = _sym(IKC @ Pp_ss @ IKC.T + (K * r[v]) @ K.T)
                P_pred[t + 1:] = Pp_ss; P_filt[t + 1:] = Pf_ss
                logdetS = 2.0 * np.sum(np.log(np.diag(L)))
                E = np.empty((T - t - 1, v.size))
                # mean recursion only: O(n m) per frame
                _ss_filter_means(A, np.ascontiguousarray(K), np.ascontiguousarray(Y[:, v]), v,
                                 x_filt[t], x_pred, x_filt, E, t + 1)
                Al = solve_triangular(L, E.T, lower=True)
                ll -= 0.5 * (E.shape[0] * (v.size * LOG2PI + logdetS) + np.sum(Al * Al))
            break
        t += 1

    return FilterResult(x_pred, P_pred, x_filt, P_filt, float(ll), n_obs, steady_from)


# --------------------------------------------------------------------------------------
# RTS smoother
# --------------------------------------------------------------------------------------
def rts_smoother(fr: FilterResult, A, steady_state: bool = False, ss_tol: float = 1e-12,
                 ss_min_burn: int = 10) -> SmootherResult:
    """Rauch-Tung-Striebel backward pass (spec section 8), including the lag-one
    cross-covariance P_{t+1,t|T} = P_{t+1|T} J_t^T needed by the M-step.

        J_t     = P_{t|t} A^T P_{t+1|t}^{-1}          (Cholesky solve, no inverse)
        x_{t|T} = x_{t|t} + J_t (x_{t+1|T} - x_{t+1|t})
        P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t^T

    With steady_state=True and a steady filter (fr.steady_from = f0), J_t is constant for
    t >= f0; the backward covariance recursion runs exactly from the end of the segment
    until P_{t|T} converges, is then frozen, and only the means are propagated down to f0.
    Frames before f0 always use the exact recursion.
    """
    A = np.asarray(A, np.float64)
    T, n = fr.x_filt.shape
    xs = np.empty((T, n)); Ps = np.empty((T, n, n)); Plag = np.empty((max(T - 1, 0), n, n))
    xs[-1] = fr.x_filt[-1]; Ps[-1] = fr.P_filt[-1]
    f0 = fr.steady_from if (steady_state and fr.steady_from is not None) else None

    def gain(t):   # J^T = P_{t+1|t}^{-1} A P_{t|t}  (both symmetric)
        return cho_solve(cho_factor(fr.P_pred[t + 1]), A @ fr.P_filt[t]).T

    def step(t, J):
        xs[t] = fr.x_filt[t] + J @ (xs[t + 1] - fr.x_pred[t + 1])
        Ps[t] = _sym(fr.P_filt[t] + J @ (Ps[t + 1] - fr.P_pred[t + 1]) @ J.T)
        Plag[t] = Ps[t + 1] @ J.T

    t = T - 2
    steady_range = None
    if f0 is not None and f0 <= T - 2:
        J = gain(T - 2)                          # constant over t >= f0
        while t >= f0:
            step(t, J)
            if (T - 2 - t) >= ss_min_burn and t < T - 2 and _rel_change(Ps[t], Ps[t + 1]) < ss_tol:
                break
            t -= 1
        if t >= f0:                              # converged at t: freeze covs on [f0, t)
            hi = t
            if hi > f0:
                Ps[f0:hi] = Ps[hi]; Plag[f0:hi] = Plag[hi]
                _ss_smoother_means(J, fr.x_filt, fr.x_pred, xs, hi - 1, f0)
                steady_range = (f0, hi)
        t = f0 - 1
    while t >= 0:                                # exact recursion
        step(t, gain(t))
        t -= 1
    return SmootherResult(xs, Ps, Plag, steady_range)


def filter_smooth(Y, obs, A, Q, r, Sigma0, steady_state: bool = False, smooth: bool = True):
    """Filter (and optionally smooth) one segment. Returns (FilterResult, SmootherResult|None)."""
    fr = kalman_filter(Y, obs, A, Q, r, Sigma0, steady_state=steady_state)
    sr = rts_smoother(fr, A, steady_state=steady_state) if smooth else None
    return fr, sr


def run_segments(segments: Sequence[np.ndarray], obs, A, Q, r, Sigma0,
                 steady_state: bool = False, smooth: bool = True):
    """Run every segment of a fly (or of a whole data set) with a fresh filter from
    N(0, Sigma0) per segment (spec section 5, 'Segments restart').

    segments : list of (T_k, n) arrays.  obs : one mask for all segments ((n,)) or a list
    of per-segment masks. Returns (list of (FilterResult, SmootherResult|None), total loglik).
    """
    out, ll = [], 0.0
    for k, Yk in enumerate(segments):
        ok = obs[k] if isinstance(obs, (list, tuple)) else obs
        fr, sr = filter_smooth(Yk, ok, A, Q, r, Sigma0, steady_state=steady_state, smooth=smooth)
        out.append((fr, sr)); ll += fr.loglik
    return out, ll


# --------------------------------------------------------------------------------------
# Data helpers (spec section 3, step 5; section 5, initial state)
# --------------------------------------------------------------------------------------
def standardise_fly(segs: Sequence[np.ndarray]):
    """Zero mean, unit variance per region over all of a fly's kept frames (pooling its
    segments). segs are (n, T_k) as stored in the npz. Returns list of (T_k, n) float64,
    plus (mean, std). Hidden-region masking is the harness's job: in R3 the model never
    sees hidden columns, whatever their standardisation."""
    X = [np.asarray(s, np.float64).T for s in segs]
    allX = np.concatenate(X, axis=0)
    mu = allX.mean(0); sd = allX.std(0)
    if np.any(sd == 0):
        raise ValueError("a region has zero variance in this fly")
    return [(x - mu) / sd for x in X], mu, sd


def load_standardised(npz_path: str):
    """-> ({fly_id: [ (T_k, 36) standardised arrays ]}, region names, structural, meta),
    via build_dataset.load (the only reader the model code may use)."""
    from build_dataset import load
    flies, names, S, meta = load(npz_path)
    return {fid: standardise_fly(segs)[0] for fid, segs in flies.items()}, names, S, meta


def make_sigma0(P_data: np.ndarray, r0: np.ndarray, visible: Optional[np.ndarray] = None,
                floor: float = 1e-4) -> np.ndarray:
    """Sigma0 = P_data - diag(r0), eigenvalues floored to stay PD (spec section 5).
    Hidden regions (visible[i] False; R3) get the mean visible variance of
    (P_data - diag r0) on the diagonal and zero off-diagonals (spec section 9)."""
    S0 = _sym(np.asarray(P_data, np.float64) - np.diag(r0))
    if visible is not None:
        visible = np.asarray(visible, bool)
        h = ~visible
        if h.any():
            fill = float(np.mean(np.diag(S0)[visible]))
            S0[h, :] = 0.0; S0[:, h] = 0.0
            S0[h, h] = fill
    w, V = np.linalg.eigh(S0)
    return _sym((V * np.maximum(w, floor)) @ V.T)