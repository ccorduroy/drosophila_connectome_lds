"""
Phase 4: first real fits -- M0 in R1 on 20 flies, baselines B1-B3, timing, gate G3
(Protocol and Model Spec, sections 10-13 and 15; work plan phase 4).

    python run_r1.py --npz data_processed/fly_ito36.npz --workers 8
    python run_r1.py ... --fixed_kappa 0          # skip the inner kappa search (faster; not the spec setting)
    python run_r1.py ... --smoke                  # 2 folds, few iterations: checks the code runs, no result

R1 = leave-one-fly-out. For each of the 20 held-out flies:
  * Baselines on the SAME folds and metric (spec section 12 asks for B1-B8 to be recomputed with
    the final harness before any Kalman model is scored; B1-B3 are the T1 baselines, B4-B8 are
    T2 and are recomputed with the R3 harness in phase 5):
      B1 persistence (next frame = this frame); B2 pooled diagonal AR(1), ridge 100;
      B3 ridge VAR(1), lambda = 100, on the 19 training flies.
  * M0 (flat prior, A0 = 0). kappa from the coarse grid (0, 0.01, 1) by inner 3-fold
    cross-validation over the 19 training flies: inner fold i = training flies [i::3] in sorted
    order (sizes 7, 6, 6; deterministic, no seed), 30 EM iterations per tuning fit, scored by T1
    on the inner held-out flies. Then the final fit at the chosen kappa with three restarts
    (section 10): ridge, A0 (A = 0, Q = Sigma_x), and ridge with 10% multiplicative jitter on A
    (default_rng([675, 10, fold]) from seeds.json); at most 150 EM iterations; the restart with
    the highest final log posterior is kept. Restart spread on the test fly is reported.
  * T1 on the held-out fly: filtered one-step prediction x_{t+1|t} vs y_{t+1}, all regions
    visible, pooled over segments exactly as analysis_demo.r2: 1 - SSE / sum(y^2) over all
    within-segment transitions.
  * Diagnostics: rho(A) before any shrink, EM iterations and convergence, standardised
    innovations (lag-1 autocorrelation, excess kurtosis) for rule N1.
Timing: one E-step over 19 training flies at the fitted parameters (steady-state path), and a
projection of the tier A budget.

Gate G3 (work plan): M0 inside the guard rails (section 12: expected near 0.34-0.40; a median
above about 0.53 means leakage, below about 0.30 means a bug) and rho(A) < 1 in all 20 folds.
Also reported, not gated: M0 >= B3 (section 13), rule N1 (section 14).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import kurtosis

import em
import kalman as K
from integrity import ensure_seeds, check_hashes, GLOBAL_SEED

KAPPAS = (0.0, 0.01, 1.0)              # coarse grid for R1 (spec section 15)
RIDGE = 100.0
GUARD = {'expected': (0.34, 0.40), 'bug_below': 0.30, 'leak_above': 0.53}
B_REF = {'B1': -0.019, 'B2': 0.274, 'B3': 0.340}       # feasibility medians (section 12)
TIER_A_ITERS = 200_000                                  # section 15 upper bound (Nov 23)

_G = {}


def _init(payload):
    _G.update(payload)


# --------------------------------------------------------------------------------------
# Metric and baselines (identical to analysis_demo.py)
# --------------------------------------------------------------------------------------
def _pairs(segs):
    """(T, n) segments -> stacked within-segment transitions X0, X1 as (n, N)."""
    return (np.concatenate([Y[:-1].T for Y in segs], 1), np.concatenate([Y[1:].T for Y in segs], 1))


def r2(pred, X1):
    return float(1.0 - np.sum((X1 - pred) ** 2) / np.sum(X1 ** 2))


def baselines(flies, order):
    n = flies[order[0]][0].shape[1]
    P = {f: _pairs(flies[f]) for f in order}
    out = {'B1': {}, 'B2': {}, 'B3': {}}
    for h in order:
        tr = [P[f] for f in order if f != h]
        G = sum(x0 @ x0.T for x0, _ in tr); H = sum(x1 @ x0.T for x0, x1 in tr)
        x0, x1 = P[h]
        out['B1'][h] = r2(x0, x1)
        a = np.diag(H) / (np.diag(G) + RIDGE)
        out['B2'][h] = r2(a[:, None] * x0, x1)
        A = H @ np.linalg.inv(G + RIDGE * np.eye(n))
        out['B3'][h] = r2(A @ x0, x1)
    return out


# --------------------------------------------------------------------------------------
# Kalman scoring
# --------------------------------------------------------------------------------------
def t1_score(segs, p: em.Params):
    """Filtered one-step R^2 pooled over segments (first frame of each segment not scored),
    plus standardised-innovation statistics."""
    sse = ss = 0.0
    ac, inn = [], []
    for Y in segs:
        fr = K.kalman_filter(Y, None, p.A, p.Q, p.r, p.Sigma0, steady_state=True)
        e = Y[1:] - fr.x_pred[1:]
        sse += float(np.sum(e * e)); ss += float(np.sum(Y[1:] ** 2))
        s = np.sqrt(np.einsum('tii->ti', fr.P_pred[1:]) + p.r)
        z = e / s
        inn.append(z)
        if len(z) > 2:
            ac.append((len(z), [np.corrcoef(z[:-1, i], z[1:, i])[0, 1] for i in range(z.shape[1])]))
    Z = np.concatenate(inn)
    w = np.array([a for a, _ in ac], float)
    lag1 = np.average(np.array([b for _, b in ac]), axis=0, weights=w)       # per region
    return 1.0 - sse / ss, {'innov_lag1_median_abs': float(np.median(np.abs(lag1))),
                            'innov_excess_kurtosis_median': float(np.median(kurtosis(Z, axis=0)))}


def _segments(fids):
    return [Y for f in fids for Y in _G['flies'][f]]


def _start(segs, kind, fold):
    n = segs[0].shape[1]
    A0 = np.zeros((n, n))
    p = em.init_params(segs, None, A0, ridge_lambda=RIDGE)
    if kind == 'A0':
        p.A = A0.copy()
        p.Q = em._floor_eig(p.Sigma0.copy(), 1e-4)            # Q = Sigma_x - A0 Sigma_x A0^T with A0 = 0
    elif kind == 'jitter':
        rng = np.random.default_rng([GLOBAL_SEED, 10, fold])  # seeds.json rules['restart_jitter']
        p.A = p.A * (1.0 + 0.1 * rng.standard_normal(p.A.shape))
    return p


def tune_job(job):
    """(fold, kappa, inner) -> T1 on the inner held-out flies."""
    fold, kappa, inner, tune_iter = job
    order = _G['order']
    train = [f for i, f in enumerate(order) if i != fold]
    val = train[inner::3]
    fit_on = [f for f in train if f not in val]
    segs = _segments(fit_on)
    res = em.fit(segs, None, np.zeros((segs[0].shape[1],) * 2), kappa,
                 init=_start(segs, 'ridge', fold), max_iter=tune_iter)
    return fold, kappa, inner, t1_score(_segments(val), res.params)[0]


def final_job(job):
    """(fold, kappa, restart) -> fit on 19 flies, score on the held-out fly."""
    fold, kappa, kind, max_iter = job
    order = _G['order']
    test = order[fold]
    segs = _segments([f for f in order if f != test])
    t0 = time.perf_counter()
    res = em.fit(segs, None, np.zeros((segs[0].shape[1],) * 2), kappa,
                 init=_start(segs, kind, fold), max_iter=max_iter)
    fit_s = time.perf_counter() - t0
    score, inn = t1_score(_G['flies'][test], res.params)
    t0 = time.perf_counter(); em.e_step(segs, None, res.params); estep_s = time.perf_counter() - t0
    return {'fold': fold, 'fly': test, 'restart': kind, 'kappa': kappa, 'logpost': res.logpost,
            'T1': score, **inn, 'rho_A': res.diagnostics['rho_A'], 'n_iter': res.n_iter,
            'converged': res.converged, 'flags': res.flags, 'fit_s': fit_s, 'estep_s': estep_s,
            'n_frames_train': int(sum(len(Y) for Y in segs)), 'r_mean': float(np.mean(res.params.r))}


def _map(pool, fn, jobs):
    return list(map(fn, jobs)) if pool is None else list(pool.map(fn, jobs))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--fixed_kappa', type=float, default=None)
    ap.add_argument('--max_iter', type=int, default=150)
    ap.add_argument('--tune_iter', type=int, default=30)
    ap.add_argument('--smoke', action='store_true', help='2 folds, few iterations: code check only')
    ap.add_argument('--no_hash_check', action='store_true', help='never for the gate run')
    ap.add_argument('--out', default='results/phase4_r1.json')
    cfg = ap.parse_args()
    if cfg.smoke:
        cfg.max_iter, cfg.tune_iter = 4, 2

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/')])
    ensure_seeds()
    flies, names, _, _ = K.load_standardised(cfg.npz)
    order = sorted(flies)
    folds = list(range(len(order)))[: 2 if cfg.smoke else None]
    print(f"R1 leave-one-fly-out: {len(order)} flies, {len(names)} regions; folds {len(folds)}; "
          f"workers {cfg.workers}{'; SMOKE (not a result)' if cfg.smoke else ''}")

    # ---- baselines first (section 12) ----
    B = baselines(flies, order)
    print("Baselines (T1, same folds and metric), median over flies [feasibility reference]:")
    for k in ('B1', 'B2', 'B3'):
        v = np.array(list(B[k].values()))
        print(f"   {k}: {np.median(v):+.3f}  (range {v.min():+.3f} to {v.max():+.3f})  [{B_REF[k]:+.3f}]")

    pool = None if cfg.workers <= 1 else ProcessPoolExecutor(cfg.workers, initializer=_init,
                                                             initargs=({'flies': flies, 'order': order},))
    if pool is None:
        _init({'flies': flies, 'order': order})
    t_start = time.perf_counter()
    try:
        # ---- kappa by inner 3-fold CV ----
        if cfg.fixed_kappa is not None:
            chosen = {f: cfg.fixed_kappa for f in folds}
            inner = {}
        else:
            jobs = [(f, k, i, cfg.tune_iter) for f in folds for k in KAPPAS for i in range(3)]
            res = _map(pool, tune_job, jobs)
            inner = {}
            for f, k, i, s in res:
                inner.setdefault(f, {}).setdefault(k, []).append(s)
            chosen = {f: max(KAPPAS, key=lambda k: np.mean(inner[f][k])) for f in folds}
        # ---- final fits, three restarts ----
        jobs = [(f, chosen[f], kind, cfg.max_iter) for f in folds for kind in ('ridge', 'A0', 'jitter')]
        fits = _map(pool, final_job, jobs)
    finally:
        if pool is not None:
            pool.shutdown()
    wall = time.perf_counter() - t_start

    # ---- per fold: keep the restart with the highest log posterior ----
    per = {}
    for r in fits:
        per.setdefault(r['fold'], []).append(r)
    rows = []
    for f in folds:
        rs = per[f]
        best = max(rs, key=lambda r: r['logpost'])
        spread = max(r['T1'] for r in rs) - min(r['T1'] for r in rs)
        rows.append({**best, 'restart_T1': {r['restart']: r['T1'] for r in rs},
                     'restart_T1_spread': spread, 'B3': B['B3'][order[f]],
                     'inner_cv': {str(k): float(np.mean(v)) for k, v in inner.get(f, {}).items()}})

    m0 = np.array([r['T1'] for r in rows]); b3 = np.array([r['B3'] for r in rows])
    rho = np.array([r['rho_A'] for r in rows])
    med = float(np.median(m0))
    n_below_b3 = int(np.sum(m0 < b3 - 0.01))
    lag1 = float(np.median([r['innov_lag1_median_abs'] for r in rows]))
    kurt = float(np.median([r['innov_excess_kurtosis_median'] for r in rows]))
    estep = float(np.median([r['estep_s'] for r in rows]))
    print("\nM0 in R1 (T1, filtered one-step R^2 on the held-out fly):")
    for r in rows:
        print(f"   {r['fly']}: M0 {r['T1']:.3f}  B3 {r['B3']:.3f}  kappa {r['kappa']:g}  rho(A) {r['rho_A']:.3f}  "
              f"restart {r['restart']} (spread {r['restart_T1_spread']:.4f})  iters {r['n_iter']}"
              f"{'' if r['converged'] else ' (cap)'}")
    print(f"   median M0 {med:.3f} (range {m0.min():.3f} to {m0.max():.3f}); median B3 {np.median(b3):.3f}; "
          f"median M0 - B3 {np.median(m0 - b3):+.4f}")
    in_rails = GUARD['bug_below'] <= med <= GUARD['leak_above']
    in_expected = GUARD['expected'][0] <= med <= GUARD['expected'][1]
    rho_ok = bool(np.all(rho < 1.0))
    n1 = (n_below_b3 > len(rows) / 2) or lag1 > 0.1 or kurt > 3
    print("\nChecks:")
    print(f"   guard rails: median {med:.3f} {'inside' if in_rails else 'OUTSIDE'} [{GUARD['bug_below']}, "
          f"{GUARD['leak_above']}]; expected band {GUARD['expected']}: {'yes' if in_expected else 'no'}")
    print(f"   rho(A) < 1 in all folds: {'yes' if rho_ok else 'NO'} (max {rho.max():.3f})")
    print(f"   M0 >= B3 (section 13): median M0 - B3 {np.median(m0 - b3):+.4f} -> {'yes' if med >= np.median(b3) else 'no'}")
    print(f"   rule N1: {n_below_b3}/{len(rows)} flies more than 0.01 below B3; innovations median |lag-1| "
          f"{lag1:.3f} (> 0.1 fires), median excess kurtosis {kurt:.2f} (> 3 fires) -> {'FIRES' if n1 else 'does not fire'}")
    print(f"   restarts: max spread in held-out T1 across the three starts {max(r['restart_T1_spread'] for r in rows):.4f}; "
          f"winners {dict(zip(*np.unique([r['restart'] for r in rows], return_counts=True)))}")
    n_cap = sum(not r['converged'] for r in rows)
    print(f"   EM hit the {cfg.max_iter}-iteration cap in {n_cap}/{len(rows)} kept fits")
    print(f"\nTiming: one E-step over 19 training flies (~{rows[0]['n_frames_train']} frames): median {estep:.2f} s; "
          f"this run {wall / 60:.1f} min on {cfg.workers} workers")
    proj = TIER_A_ITERS * estep / max(cfg.workers, 1) / 3600
    print(f"   tier A projection: {TIER_A_ITERS:,} EM iterations (upper bound) x {estep:.2f} s / {cfg.workers} workers "
          f"= {proj:.1f} h")
    g3 = in_rails and rho_ok
    print(f"\nG3: {'PASS' if g3 else 'FAIL'} (M0 inside guard rails; rho(A) < 1 in all folds)"
          + (" -- SMOKE, not a gate result" if cfg.smoke else ""))

    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': vars(cfg), 'baselines': B, 'folds': rows,
                   'summary': {'median_M0': med, 'median_B3': float(np.median(b3)),
                               'median_M0_minus_B3': float(np.median(m0 - b3)), 'guard_rails_ok': in_rails,
                               'in_expected_band': in_expected, 'rho_all_below_1': rho_ok,
                               'n1_fires': bool(n1), 'n_flies_below_B3': n_below_b3,
                               'innov_lag1_median_abs': lag1, 'innov_excess_kurtosis_median': kurt,
                               'estep_s_median': estep, 'wall_s': wall, 'tier_A_projection_h': proj,
                               'G3': bool(g3)}}, f, indent=1, default=str)
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
