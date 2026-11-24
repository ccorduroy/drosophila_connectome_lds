"""
Phase-3 unit tests (Protocol and Model Spec, section 13): U5, U7, U8, plus prior-recipe checks.

    python test_em.py          (or: pytest -q test_em.py)

U5  EM monotonicity: L non-decreasing every iteration, kappa = 0 and kappa > 0, with and
    without missing data (random masks, all-missing frames, and a never-observed region).
U7  Standardisation invariance: rescaling a region's data by a constant c (and the starting
    parameters by the matching change of units) rescales that state and leaves per-region
    one-step R^2 and hidden-region r unchanged. Tested from a transformed start because the
    ridge initialisation is not itself scale-equivariant; this isolates the EM code.
U8  Reduction: r = 1e-6, C = I, one EM iteration equals the closed-form ridge-with-prior fit
    fit_prior() in analysis_demo.py (A = (X1 X0^T + lambda A0)(X0 X0^T + lambda I)^-1). The
    function is taken from that file's source (the file runs its whole analysis on import);
    without the file the test falls back to the same formula written out here.
"""
import sys
import time

import numpy as np

import em
import kalman as K
import prior as P
from integrity import unit_seed, ensure_seeds
from test_kalman import random_system, simulate

TOL_U5 = 1e-8; TOL_U7 = 1e-6; TOL_U8 = 1e-5


def _data(n, Ts, rng, rho=0.85):
    A, Q, r, S0 = random_system(n, rng, rho=rho)
    return [simulate(A, Q, r, S0, T, rng)[1] for T in Ts], (A, Q, r, S0)


def _check_monotone(hist):
    h = np.asarray(hist)
    drops = (h[:-1] - h[1:]) / np.abs(h[:-1])
    return float(drops.max())


def test_U5_monotonic():
    worst = -np.inf
    for case, (kappa, missing) in enumerate([(0.0, False), (0.0, True), (0.1, False), (0.1, True)]):
        rng = np.random.default_rng(unit_seed(5, case))
        n = 5
        segs, _ = _data(n, (150, 120), rng)
        A0 = 0.4 * np.eye(n) + 0.1 * P.uniform(n)
        if missing:
            obs = []
            for Y in segs:
                m = rng.random(Y.shape) > 0.3
                m[rng.random(len(Y)) < 0.1] = False
                m[:, 4] = False                       # region 4 never observed (R3-style)
                obs.append(m)
            segs = [np.where(m, Y, 0.0) for Y, m in zip(segs, obs)]
            init = em.Params(A0.copy() + 0.05 * rng.standard_normal((n, n)), np.eye(n) * 0.5,
                             np.full(n, 0.3), np.eye(n))
        else:
            obs, init = None, None
        res = em.fit(segs, obs, A0, kappa, init=init, max_iter=60, tol=-np.inf)
        d = _check_monotone(res.history)
        worst = max(worst, d)
        assert d <= TOL_U5, (case, d)
    return worst


def _transform(p: em.Params, D):
    Di = np.diag(1.0 / np.diag(D))
    return em.Params(D @ p.A @ Di, D @ p.Q @ D, p.r * np.diag(D) ** 2, D @ p.Sigma0 @ D)


def test_U7_scale_invariance():
    rng = np.random.default_rng(unit_seed(7))
    n = 5
    segs, _ = _data(n, (300, 250), rng)
    hidden = np.zeros(n, bool); hidden[4] = True
    vis = ~hidden
    c = np.ones(n); c[1] = 3.0; c[4] = 0.25            # rescale a visible and the hidden region
    D = np.diag(c)
    segs2 = [Y * c for Y in segs]
    A0 = np.zeros((n, n))
    p0 = em.init_params(segs, vis, 0.3 * np.eye(n) + 0.05 * P.uniform(n))
    r1 = em.fit(segs, vis, A0, 0.0, init=p0, max_iter=25, tol=-np.inf)
    r2 = em.fit(segs2, vis, A0, 0.0, init=_transform(p0, D), max_iter=25, tol=-np.inf)
    A_back = np.diag(1 / c) @ r2.params.A @ D
    dA = np.max(np.abs(A_back - r1.params.A)) / np.max(np.abs(r1.params.A))
    _, k1 = em.e_step(segs, vis, r1.params, keep=True)
    _, k2 = em.e_step(segs2, vis, r2.params, keep=True)
    dR2 = max(abs(em.one_step_r2(Y[:, [i]], _Sub(fr, i), [True]) -
                  em.one_step_r2(Y2[:, [i]], _Sub(fr2, i), [True]))
              for (Y, (fr, _)), (Y2, (fr2, _)) in zip(zip(segs, k1), zip(segs2, k2))
              for i in np.flatnonzero(vis))
    hr1 = em.hidden_r(segs, k1, hidden)[0]; hr2 = em.hidden_r(segs2, k2, hidden)[0]
    dr = abs(hr1 - hr2)
    assert max(dA, dR2, dr) < TOL_U7, (dA, dR2, dr)
    return dA, dR2, dr


class _Sub:
    """FilterResult restricted to one region (for per-region R^2)."""
    def __init__(self, fr, i):
        self.x_pred = fr.x_pred[:, [i]]


def _fit_prior_from_analysis_demo(n, path='analysis_demo.py'):
    """Extract fit_prior from analysis_demo.py without running the script."""
    import ast
    import os
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding='utf-8').read())
    fn = next((f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == 'fit_prior'), None)
    if fn is None:
        return None
    ns = {'np': np, 'n': n}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)
    return ns['fit_prior']


def test_U8_ridge_reduction():
    rng = np.random.default_rng(unit_seed(8))
    n = 36
    segs, (A, Q, _, S0) = _data(n, (800, 600, 700), rng, rho=0.8)
    Wt = P.wiring(np.exp(rng.standard_normal((n, n)) * 2))
    A0 = P.prior_mean('M3', Wt=Wt, d=0.5, a_u=0.25, a_w=1.0)
    for kappa in (0.0, 0.01, 1.0):
        p = em.Params(A.copy(), Q.copy(), np.full(n, 1e-6), S0.copy())
        st = em.e_step(segs, None, p)
        A_em = em.m_step(st, p, A0, kappa).A
        N = sum(len(Y) - 1 for Y in segs); lam = kappa * N
        fp = _fit_prior_from_analysis_demo(n)
        if fp is not None:
            x0 = np.concatenate([Y[:-1].T for Y in segs], 1); x1 = np.concatenate([Y[1:].T for Y in segs], 1)
            A_ridge = fp(x0, x1, A0, lam)
        else:
            Syy0 = sum(Y[:-1].T @ Y[:-1] for Y in segs); Syy1 = sum(Y[1:].T @ Y[:-1] for Y in segs)
            A_ridge = np.linalg.solve((Syy0 + lam * np.eye(n)).T, (Syy1 + lam * A0).T).T
        d = np.max(np.abs(A_em - A_ridge))
        assert d < TOL_U8, (kappa, d)
    return d, ('analysis_demo.fit_prior' if _fit_prior_from_analysis_demo(n) is not None else 'formula')


def test_prior_recipe():
    rng = np.random.default_rng(unit_seed(20))
    n = 36
    S = np.exp(rng.standard_normal((n, n)) * 2)
    Wt = P.wiring(S)
    assert np.allclose(Wt.sum(1), 1) and np.all(np.diag(Wt) == 0)
    A0 = P.prior_mean('M3', Wt=Wt, d=0.5, a_u=0.25, a_w=1.0)
    assert np.allclose(A0.sum(1), 0.75)                       # row sums d + a_u
    perm = ensure_seeds()['shuffles'][0]
    A4 = P.prior_mean('M4', Wt=Wt, d=0.5, a_u=0.25, a_w=1.0, perm=perm)
    assert np.allclose(A4.sum(1), 0.75) and np.allclose(np.diag(A4), 0.5)
    assert np.allclose(np.sort(A4.ravel()), np.sort(A0.ravel()))   # same weight distribution
    for m in ('M0', 'M1', 'M2', 'M3'):
        for h in P.grid_points(m, Wt):
            assert h['d'] + h['a_u'] <= 0.9
            assert P.spectral_radius(P.prior_mean(m, n=n, Wt=Wt, d=h['d'], a_u=h['a_u'], a_w=h['a_w'])) < 0.98
    assert np.allclose(P.prior_mean('M2', n=n, d=0.5, a_u=0.25), P.prior_mean('M3', Wt=Wt, d=0.5, a_u=0.25, a_w=0))


if __name__ == '__main__':
    ensure_seeds()
    ok = True
    for f in (test_prior_recipe, test_U5_monotonic, test_U7_scale_invariance, test_U8_ridge_reduction):
        t0 = time.perf_counter()
        try:
            out = f(); res = 'PASS' + (f'  ({out})' if out is not None else '')
        except AssertionError as e:
            res = f'FAIL {e}'; ok = False
        print(f"{f.__name__:28s} {res}  [{time.perf_counter() - t0:.1f} s]")
    print("phase-3 unit tests:", "all pass" if ok else "FAILED")
    sys.exit(0 if ok else 1)
