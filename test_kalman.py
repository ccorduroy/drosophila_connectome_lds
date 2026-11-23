"""
Gate G1 tests for kalman.py (Protocol and Model Spec, section 13; work-plan phase 2):
U1, U2, U3, U4, U6, U9.

    python test_kalman.py                                  # synthetic tests (all of G1 except real-data U9)
    python test_kalman.py --npz data_processed/fly_ito36.npz   # + U9 on three real flies
    pytest -q test_kalman.py                               # same synthetic tests under pytest

U9 (section-13 table, added Nov 23): the steady-state filter/smoother agrees with the
exact time-varying recursion to a maximum relative difference below 1e-4, on synthetic
data and on three real flies.

Running this file also writes seeds.json (first run) or verifies it (every later run);
see ensure_seeds().
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import kalman as K

TOL_U1 = 1e-9; TOL_U2 = 1e-8; TOL_U4 = 1e-9; TOL_U6 = -1e-10; TOL_U9 = 1e-4


# --------------------------------------------------------------------------------------
# seeds.json (spec sections 10, 11, 15): every seed and random draw, derived from the
# global seed 675 by fixed rules, written once and then only verified. Commit it (and its
# SHA-256) before any phase-3 simulation or any real-data fit.
# --------------------------------------------------------------------------------------
GLOBAL_SEED = 675
N_REGIONS = 36
SEEDS_PATH = os.environ.get('SEEDS_JSON', 'seeds.json')


def unit_seed(test_no, case=0):
    """Unit test Uk, case c -> [675, 1000 + k, c]. Offset by 1000 so no unit test reuses
    a protocol seed (the old [675, 9, 0] for U9 was shuffle permutation 0)."""
    return [GLOBAL_SEED, 1000 + test_no, case]


def build_seeds():
    hidden = {}
    for p in (0.25, 0.50, 0.75):
        k = round(p * N_REGIONS)
        hidden[str(int(100 * p))] = [
            sorted(int(i) for i in np.random.default_rng([GLOBAL_SEED, int(100 * p), j])
                   .choice(N_REGIONS, k, replace=False))
            for j in range(10)]
    shuffles = [[int(i) for i in np.random.default_rng([GLOBAL_SEED, 9, s]).permutation(N_REGIONS)]
                for s in range(20)]
    return {
        'global_seed': GLOBAL_SEED,
        'n_regions': N_REGIONS,
        'region_order': "indices into the 'regions' array of fly_ito36.npz (= structural-matrix row order)",
        'rules': {
            'hidden_sets': 'set j of fraction p: sorted(default_rng([675, int(100*p), j]).choice(36, round(36*p), replace=False)), j = 0..9 (spec section 11)',
            'shuffles': 'permutation s: default_rng([675, 9, s]).permutation(36), s = 0..19; applied as W[pi][:, pi] after row normalisation (spec section 6)',
            'restart_jitter': 'R1 fold f (held-out fly index in sorted fly order): default_rng([675, 10, f]), 10% multiplicative jitter on A (spec section 10)',
            'simulations': 'simulation Sk, replicate r: default_rng([675, 100 + k, r]), k = 0..4 (spec section 13)',
            'unit_tests': 'unit test Uk, case c: default_rng([675, 1000 + k, c])',
        },
        'hidden_sets': hidden,
        'shuffles': shuffles,
        'tiers': {
            'A': {'hidden_sets': [0, 1, 2, 3, 4], 'shuffles': list(range(10)),
                  'models': ['M0', 'M1', 'M2', 'M3', 'M4'], 'regimes': ['R3'],
                  'random_search_configs': 16, 'tuning_em_iters': 30, 'final_em_iters_max': 150},
            'B': {'hidden_sets': list(range(10)), 'shuffles': list(range(20)),
                  'models': ['M0', 'M1', 'M2', 'M3', 'M4', 'M5'], 'regimes': ['R3', 'R1', 'R2'],
                  'random_search_configs': 16, 'tuning_em_iters': 30, 'final_em_iters_max': 150},
        },
        'reductions': [],   # any pre-declared cut (spec section 15 cut lines) is appended here BEFORE results are seen
    }


def ensure_seeds(path=SEEDS_PATH):
    """Write seeds.json if absent; otherwise verify it matches the fixed rules exactly
    (apart from 'created' and any recorded 'reductions'). Returns the seeds dict."""
    want = build_seeds()
    if not os.path.exists(path):
        out = copy.deepcopy(want); out['created'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"seeds.json written to {os.path.abspath(path)}")
        return out
    with open(path) as f:
        have = json.load(f)
    a = {k: v for k, v in have.items() if k not in ('created', 'reductions')}
    b = {k: v for k, v in want.items() if k not in ('created', 'reductions')}
    if a != b:
        raise RuntimeError(f"{path} does not match the frozen seed rules; it was edited or the rules changed. "
                           "Do not overwrite it: restore the committed copy.")
    return have


# --------------------------------------------------------------------------------------
# Random systems and brute-force joint Gaussian
# --------------------------------------------------------------------------------------
def random_system(n, rng, rho=0.9):
    A = rng.standard_normal((n, n))
    A *= rho / max(abs(np.linalg.eigvals(A)))
    B = rng.standard_normal((n, n)); Q = B @ B.T / n + 0.1 * np.eye(n)
    r = rng.uniform(0.1, 1.0, n)
    B = rng.standard_normal((n, n)); S0 = B @ B.T / n + 0.5 * np.eye(n)
    return A, Q, r, S0


def simulate(A, Q, r, S0, T, rng):
    n = A.shape[0]
    Lq = np.linalg.cholesky(Q); x = np.linalg.cholesky(S0) @ rng.standard_normal(n)
    X = np.empty((T, n))
    for t in range(T):
        if t > 0:
            x = A @ x + Lq @ rng.standard_normal(n)
        X[t] = x
    return X, X + np.sqrt(r) * rng.standard_normal((T, n))


def joint_state_cov(A, Q, S0, T):
    """Cov of the stacked state (x_1..x_T), (T n) x (T n)."""
    n = A.shape[0]
    Sig = [S0]
    for _ in range(1, T):
        Sig.append(A @ Sig[-1] @ A.T + Q)
    Apow = [np.eye(n)]
    for _ in range(1, T):
        Apow.append(A @ Apow[-1])
    Kxx = np.empty((T * n, T * n))
    for t in range(T):
        for s in range(t + 1):
            B = Apow[t - s] @ Sig[s]                       # Cov(x_t, x_s), t >= s
            Kxx[t*n:(t+1)*n, s*n:(s+1)*n] = B
            Kxx[s*n:(s+1)*n, t*n:(t+1)*n] = B.T
    return Kxx


def brute_condition(Kxx, Y, mask, r, upto):
    """E[x | y_{1..upto}] and Cov, where only masked entries of frames < upto are observed."""
    T, n = Y.shape
    m = mask.copy(); m[upto:] = False
    idx = np.flatnonzero(m.ravel())                       # positions in the stacked vector
    if idx.size == 0:
        return np.zeros(T * n), Kxx.copy()
    Kyy = Kxx[np.ix_(idx, idx)] + np.diag(np.tile(r, T)[idx])
    Kxy = Kxx[:, idx]
    c = cho_factor(Kyy)
    mean = Kxy @ cho_solve(c, Y.ravel()[idx])
    cov = Kxx - Kxy @ cho_solve(c, Kxy.T)
    return mean, cov


def brute_loglik(Kxx, Y, mask, r):
    T, n = Y.shape
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return 0.0
    Kyy = Kxx[np.ix_(idx, idx)] + np.diag(np.tile(r, T)[idx])
    L = np.linalg.cholesky(Kyy); a = np.linalg.solve(L, Y.ravel()[idx])
    return float(-0.5 * (idx.size * np.log(2*np.pi) + 2*np.sum(np.log(np.diag(L))) + a @ a))


def blk(M, t, s, n):
    return M[t*n:(t+1)*n, s*n:(s+1)*n]


def compare_to_brute(A, Q, r, S0, Y, mask):
    """Max abs errors (filt mean, filt cov, smooth mean, smooth cov, lag cov, loglik)."""
    T, n = Y.shape
    fr, sr = K.filter_smooth(Y, mask, A, Q, r, S0)
    Kxx = joint_state_cov(A, Q, S0, T)
    ef = efc = 0.0
    for t in range(T):
        mu, C = brute_condition(Kxx, Y, mask, r, t + 1)
        ef = max(ef, np.max(np.abs(fr.x_filt[t] - mu[t*n:(t+1)*n])))
        efc = max(efc, np.max(np.abs(fr.P_filt[t] - blk(C, t, t, n))))
    mu, C = brute_condition(Kxx, Y, mask, r, T)
    es = max(np.max(np.abs(sr.x_smooth[t] - mu[t*n:(t+1)*n])) for t in range(T))
    esc = max(np.max(np.abs(sr.P_smooth[t] - blk(C, t, t, n))) for t in range(T))
    el = max((np.max(np.abs(sr.P_lag[t] - blk(C, t + 1, t, n))) for t in range(T - 1)), default=0.0)
    ell = abs(fr.loglik - brute_loglik(Kxx, Y, mask, r))
    return ef, efc, es, esc, el, ell


# --------------------------------------------------------------------------------------
# U1-U4: n = 3, T = 10 to 200
# --------------------------------------------------------------------------------------
CASES = [(0, 10), (1, 50), (2, 200)]


def test_U1_filter_smoother_full_obs():
    for seed, T in CASES:
        rng = np.random.default_rng(unit_seed(1, seed))
        A, Q, r, S0 = random_system(3, rng)
        _, Y = simulate(A, Q, r, S0, T, rng)
        ef, efc, es, esc, _, _ = compare_to_brute(A, Q, r, S0, Y, np.ones((T, 3), bool))
        assert max(ef, efc, es, esc) < TOL_U1, (seed, T, ef, efc, es, esc)


def test_U2_loglik():
    for seed, T in CASES:
        rng = np.random.default_rng(unit_seed(2, seed))
        A, Q, r, S0 = random_system(3, rng)
        _, Y = simulate(A, Q, r, S0, T, rng)
        fr = K.kalman_filter(Y, None, A, Q, r, S0)
        d = abs(fr.loglik - brute_loglik(joint_state_cov(A, Q, S0, T), Y, np.ones((T, 3), bool), r))
        assert d < TOL_U2, (seed, T, d)


def _random_mask(T, n, rng, p_miss=0.35, p_allmiss=0.1):
    mask = rng.random((T, n)) > p_miss
    mask[rng.random(T) < p_allmiss] = False
    mask[0] = False                    # all-missing first frame
    mask[-1] = False                   # all-missing last frame
    if T > 4:
        mask[2:4] = False              # consecutive all-missing frames
    return mask


def test_U3_missing():
    for seed, T in CASES:
        rng = np.random.default_rng(unit_seed(3, seed))
        A, Q, r, S0 = random_system(3, rng)
        _, Y = simulate(A, Q, r, S0, T, rng)
        mask = _random_mask(T, 3, rng)
        Y = Y.copy(); Y[~mask] = rng.standard_normal((~mask).sum()) * 1e6   # must never be read
        ef, efc, es, esc, el, ell = compare_to_brute(A, Q, r, S0, Y, mask)
        assert max(ef, efc, es, esc) < TOL_U1, (seed, T, ef, efc, es, esc)
        assert ell < TOL_U2, (seed, T, ell)
        # NaN-as-missing gives the identical answer
        Yn = Y.copy(); Yn[~mask] = np.nan
        a = K.kalman_filter(Y, mask, A, Q, r, S0); b = K.kalman_filter(Yn, None, A, Q, r, S0)
        assert np.array_equal(a.x_filt, b.x_filt) and a.loglik == b.loglik


def test_U4_lag_one_cross_cov():
    for seed, T in CASES:
        rng = np.random.default_rng(unit_seed(4, seed))
        A, Q, r, S0 = random_system(3, rng)
        _, Y = simulate(A, Q, r, S0, T, rng)
        for mask in (np.ones((T, 3), bool), _random_mask(T, 3, rng)):
            el = compare_to_brute(A, Q, r, S0, Y, mask)[4]
            assert el < TOL_U4, (seed, T, el)


# --------------------------------------------------------------------------------------
# U6: Joseph-form covariances symmetric and PSD over 40,000 steps at n = 36
# --------------------------------------------------------------------------------------
def test_U6_psd_long_run(T=40_000, n=36):
    rng = np.random.default_rng(unit_seed(6))
    A, Q, r, S0 = random_system(n, rng, rho=0.97)
    Q = Q * 1e-3; r = r * 1e-4         # stress: tiny noise -> ill-conditioned covariances
    _, Y = simulate(A, Q, r, S0, T, rng)
    mask = rng.random((T, n)) > 0.3    # time-varying, so no steady state is ever reached
    mask[rng.random(T) < 0.02] = False
    fr, sr = K.filter_smooth(Y, mask, A, Q, r, S0)
    worst = np.inf
    for P in (fr.P_pred, fr.P_filt, sr.P_smooth):
        for i in range(0, T, 5000):
            B = P[i:i + 5000]
            assert np.array_equal(B, B.transpose(0, 2, 1)), "covariance not exactly symmetric"
            worst = min(worst, np.linalg.eigvalsh(B).min())
    assert worst >= TOL_U6, worst
    return worst


# --------------------------------------------------------------------------------------
# U9: steady-state path agrees with the exact recursion (< 1e-4 relative)
# --------------------------------------------------------------------------------------
def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def steady_vs_exact(Y, obs, A, Q, r, S0):
    ex_f, ex_s = K.filter_smooth(Y, obs, A, Q, r, S0, steady_state=False)
    ss_f, ss_s = K.filter_smooth(Y, obs, A, Q, r, S0, steady_state=True)
    d = {'x_filt': _rel(ss_f.x_filt, ex_f.x_filt), 'P_filt': _rel(ss_f.P_filt, ex_f.P_filt),
         'x_pred': _rel(ss_f.x_pred, ex_f.x_pred), 'P_pred': _rel(ss_f.P_pred, ex_f.P_pred),
         'x_smooth': _rel(ss_s.x_smooth, ex_s.x_smooth), 'P_smooth': _rel(ss_s.P_smooth, ex_s.P_smooth),
         'P_lag': _rel(ss_s.P_lag, ex_s.P_lag),
         'loglik': abs(ss_f.loglik - ex_f.loglik) / abs(ex_f.loglik)}
    return d, ss_f.steady_from, ss_s.steady_range


def test_U9_steady_state_synthetic():
    rng = np.random.default_rng(unit_seed(9))
    n, T = 36, 1900
    A, Q, r, S0 = random_system(n, rng, rho=0.9)
    _, Y = simulate(A, Q, r, S0, T, rng)
    for obs in (np.ones(n, bool), rng.permutation(n) >= 18):   # C = I, and 50% hidden
        d, f0, rng_s = steady_vs_exact(Y, obs, A, Q, r, S0)
        assert f0 is not None and f0 < 200, f0                  # steady state actually used
        assert max(d.values()) < TOL_U9, d


# --------------------------------------------------------------------------------------
# U9 on three real flies (spec section 8: "verify against the exact version on three flies")
# Parameters: the section-10 initialisation, pooled over all 20 flies, C = I and one
# 50%-hidden set. These are realistic parameter values, not a fit.
# --------------------------------------------------------------------------------------
def section10_init(flies, lam=100.0, fs=1.2):
    from scipy.signal import welch
    segs = [s for v in flies.values() for s in v]
    n = segs[0].shape[1]
    S00 = sum(s[:-1].T @ s[:-1] for s in segs); S10 = sum(s[1:].T @ s[:-1] for s in segs)
    A = np.linalg.solve((S00 + lam * np.eye(n)).T, S10.T).T          # ridge VAR(1)
    fr = []
    for s in segs:
        f, p = welch(s, fs=fs, nperseg=min(128, len(s)), axis=0)
        band = (f >= 0.45) & (f <= 0.59)
        fr.append(p[band].mean(0) * (fs / 2) / s.var(0))           # flat floor x full band / variance
    r0 = np.maximum(np.mean(fr, 0), 0.05)
    Y = np.concatenate(segs); P = np.cov(Y.T)
    w, V = np.linalg.eigh(P - np.diag(r0)); Sx = (V * np.maximum(w, 1e-4)) @ V.T
    Q0 = Sx - A @ Sx @ A.T
    w, V = np.linalg.eigh(K._sym(Q0)); Q0 = (V * np.maximum(w, 1e-4)) @ V.T
    return A, K._sym(Q0), r0, P


def u9_real(npz_path, fly_ids=('2017-10-26_1', '2017-11-08_1', '2018-10-20_1')):
    flies, names, _, _ = K.load_standardised(npz_path)
    A, Q, r0, P = section10_init(flies)
    n = len(names)
    hidden = np.zeros(n, bool)
    hidden[ensure_seeds()['hidden_sets']['50'][0]] = True      # committed 50% set 0
    print(f"  section-10 init: rho(A) = {max(abs(np.linalg.eigvals(A))):.3f}, "
          f"mean r0 = {r0.mean():.3f}")
    worst = 0.0
    for fid in fly_ids:
        for label, vis in (("C=I", np.ones(n, bool)), ("50% hidden", ~hidden)):
            S0 = K.make_sigma0(P, r0, visible=vis)
            for k, Yk in enumerate(flies[fid]):
                d, f0, _ = steady_vs_exact(Yk, vis, A, Q, r0, S0)
                m = max(d.values()); worst = max(worst, m)
                print(f"  {fid} seg {k} {label:10s} T={len(Yk):5d} steady from {f0}: "
                      f"max rel diff {m:.2e} ({max(d, key=d.get)})")
    print(f"  U9 real: worst {worst:.2e} -> {'PASS' if worst < TOL_U9 else 'FAIL'}")
    return worst < TOL_U9


def time_e_step(npz_path):
    """Rough timing of one E-step (filter + smoother over all 20 flies), exact and steady."""
    flies, names, _, _ = K.load_standardised(npz_path)
    A, Q, r0, P = section10_init(flies)
    S0 = K.make_sigma0(P, r0)
    segs = [s for v in flies.values() for s in v]
    for ss in (False, True):
        t0 = time.perf_counter(); K.run_segments(segs, None, A, Q, r0, S0, steady_state=ss)
        print(f"  E-step over {sum(len(s) for s in segs)} frames, steady_state={ss}: "
              f"{time.perf_counter() - t0:.2f} s")


# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', help='fly_ito36.npz, for U9 on three real flies and E-step timing', default='data_processed/fly_ito36.npz')
    a = ap.parse_args()

    seeds = ensure_seeds()          # before any simulated data is created

    tests = [test_U1_filter_smoother_full_obs, test_U2_loglik, test_U3_missing,
             test_U4_lag_one_cross_cov, test_U6_psd_long_run, test_U9_steady_state_synthetic]
    
    ok = True
    for f in tests:
        t0 = time.perf_counter()
        try:
            f(); res = 'PASS'
        except AssertionError as e:
            res = f'FAIL {e}'; ok = False

        print(f"{f.__name__:40s} {res}  ({time.perf_counter() - t0:.1f} s)")

    if a.npz:
        print("U9 on real flies:")
        ok &= u9_real(a.npz)
        time_e_step(a.npz)

    print("G1:", "all checks pass" if ok else "FAILED")

    sys.exit(0 if ok else 1)