"""
Simulation checks S0-S3 (Protocol and Model Spec, section 13; work-plan phase 3, gate G2).

    python simulate.py --npz data_processed/fly_ito36.npz                 # S0-S3 (gate G2)
    python simulate.py --npz ... --only S1 --workers 8
    python simulate.py --npz ... --quick                                  # small sizes, smoke test

n = 36 with structure from the real Connectivity matrix (W~, section 6). 20 simulated flies
of 1,900 frames, shared (A, Q, R), noise floor 0.47 of variance as in the real data, each
fly standardised per region like the real pipeline (S0-S3). Seeds: seeds.json rule
[675, 100 + k, replicate] for Sk; hidden sets and shuffles from seeds.json.

Simulation-only choices (fixed here, identical for every model):
  * d and a_u are held at the generating values (0.5, 0.25); kappa (0.01, 0.1, 1) and, for
    wiring models, a_w (0, 0.5, 1, 2) are tuned. R3: nested hiding (spec section 10) over the
    committed 3-fold split of the visible regions (seeds.json), tuning fits of 30 EM
    iterations warm-started from M0, refit of at most 100 at the chosen setting. R2: tuned on
    5 separate simulated tuning flies cut the same way.
  * M4 = mean over 5 shuffled priors (seeds.json shuffles 0-4), tuned on the mean over the
    same 5 (tuning on a subset and scoring on more let M4 pick a_w from lucky draws: S2, Nov 23).
  * S1 at 25/50/75% hidden; S2, S3 at 50% hidden; 3 hidden sets each.
Pass criteria (G2), using the spec's decision rule ("supported" = paired Wilcoxon p < 0.05
over flies, median difference >= 0.02 in r, 95% bootstrap interval excluding zero):
S0 gauge confirmed; S1: M3 - M4 and M3 - M2 supported in R3 at every fraction; S2: neither
supported (no false positive for H2); S3: the matching-permutation prior beats M3
significantly and M3 - M2 is not supported. Significant-but-below-SESOI differences are
printed so they are not hidden.
S4 (recovery) was dropped on Nov 23 (spec section 13): its 5% criterion cannot be met at
the 47% noise floor, and EM from the section-10 initialisation converges slowly.

S0 (spec section 13): (a) rescaling or sign-flipping a never-observed region leaves the
log-likelihood unchanged (relative change < 1e-8); (b) M0 and M1 leave hidden regions
decoupled, so their hidden-region r is at chance (|r| < 0.05 for every fly).
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')          # one BLAS thread per worker (spec section 15)

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import wilcoxon

import em
import kalman as K
import prior as P
from integrity import ensure_seeds, sim_seed, check_hashes

N_FLIES, T_FLY, FLOOR = 20, 1900, 0.47
TRUE_H = {'d': 0.5, 'a_u': 0.25, 'a_w': 1.0}


# --------------------------------------------------------------------------------------
# Ground truth and data
# --------------------------------------------------------------------------------------
def make_truth(base_A0, rng, noise=0.2, rho_max=0.95):
    """A_true = base + noise of 20% of its Frobenius norm; Q random PD, scaled so the state
    holds 1 - FLOOR of each region's variance; r_i = FLOOR / (1 - FLOOR) * Var(x_i)."""
    n = base_A0.shape[0]
    E = rng.standard_normal((n, n))
    A = base_A0 + noise * np.linalg.norm(base_A0) / np.linalg.norm(E) * E
    rho = P.spectral_radius(A)
    if rho >= rho_max:
        A *= rho_max / rho
    B = rng.standard_normal((n, n))
    Q = B @ B.T / n + 0.5 * np.eye(n)
    Sx = solve_discrete_lyapunov(A, Q)
    s = (1 - FLOOR) / np.mean(np.diag(Sx))
    Q *= s; Sx *= s
    r = FLOOR / (1 - FLOOR) * np.diag(Sx)
    return A, Q, r, Sx


def simulate_flies(A, Q, r, Sx, rng, n_flies=N_FLIES, T=T_FLY, standardise=True):
    n = A.shape[0]
    Lq, L0 = np.linalg.cholesky(Q), np.linalg.cholesky(Sx)
    flies = []
    for _ in range(n_flies):
        X = np.empty((T, n)); x = L0 @ rng.standard_normal(n)
        W = rng.standard_normal((T, n)) @ Lq.T
        for t in range(T):
            if t:
                x = A @ x + W[t]
            X[t] = x
        Y = X + np.sqrt(r) * rng.standard_normal((T, n))
        if standardise:
            Y = (Y - Y.mean(0)) / Y.std(0)
        flies.append([Y])
    return flies


def random_wiring(n, rng):
    """A row-normalised wiring unrelated to the connectome (S2)."""
    S = np.exp(2.0 * rng.standard_normal((n, n)))
    return P.wiring(S)


# --------------------------------------------------------------------------------------
# Statistics (spec section 11)
# --------------------------------------------------------------------------------------
def hodges_lehmann(d):
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def paired(a, b, n_boot=10_000, seed=0):
    """Paired difference a - b over flies: Wilcoxon p, HL estimate, 95% bootstrap CI."""
    d = np.asarray(a) - np.asarray(b)
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    rng = np.random.default_rng(seed)
    boots = [hodges_lehmann(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'p': p, 'hl': hodges_lehmann(d), 'ci': [float(lo), float(hi)], 'median': float(np.median(d))}


SESOI_R = 0.02        # smallest effect of interest in Pearson r (spec section 2 decision rule)


def supported(s, sesoi=SESOI_R):
    """Spec decision rule for one comparison: p < 0.05, median paired difference >= SESOI,
    95% bootstrap interval excludes zero (Holm across the 9 primary tests applies to real data)."""
    return s['p'] < 0.05 and s['median'] >= sesoi and s['ci'][0] > 0


def significant(s):
    """Detection without the SESOI: p < 0.05 and the bootstrap interval above zero."""
    return s['p'] < 0.05 and s['ci'][0] > 0


def fmt(s):
    return (f"median {s['median']:+.4f}, HL {s['hl']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}], "
            f"p = {s['p']:.3g}{'  -> supported by the spec rule' if supported(s) else ''}")


# --------------------------------------------------------------------------------------
# R3 and R2 fits (run in worker processes)
# --------------------------------------------------------------------------------------
_G = {}


def _init_worker(payload):
    _G.update(payload)


def _r3_job(job):
    """job: dict(fit_hidden, score, A0, kappa, warm, max_iter, keep_params)
    -> (per-fly hidden r on `score` regions, smoothed and filtered (F, 2); params or None; diagnostics)."""
    flies = _G['flies']
    n = flies[0][0].shape[1]
    vis = np.ones(n, bool); vis[list(job['fit_hidden'])] = False
    segs = [np.where(vis, Y, np.nan) for f in flies for Y in f]       # hidden never enters the fit
    init = em.warm_start(job['warm'], job['A0'], vis) if job['warm'] is not None else None
    res = em.fit(segs, vis, job['A0'], job['kappa'], init=init, max_iter=job['max_iter'])
    _, kept = em.e_step(segs, vis, res.params, keep=True)
    sc = np.zeros(n, bool); sc[list(job['score'])] = True
    out, k = [], 0
    for f in flies:
        kk = kept[k:k + len(f)]; k += len(f)
        out.append((em.hidden_r(f, kk, sc, 'smooth')[0], em.hidden_r(f, kk, sc, 'filt')[0]))
    return np.array(out), (res.params if job['keep_params'] else None), res.diagnostics


def _r2_job(job):
    """job: (which, fly, m, A0, kappa, max_iter) -> one-step R^2 on that fly's last 30%."""
    which, fly, m, A0, kappa, max_iter = job
    Y = _G[which][fly][0]
    res = em.fit([Y[:m + 1]], None, A0, kappa, max_iter=max_iter)
    fr = K.kalman_filter(Y, None, res.params.A, res.params.Q, res.params.r, res.params.Sigma0,
                         steady_state=True)
    return em.one_step_r2(Y, fr, np.ones(Y.shape[1], bool), t_from=int(0.7 * len(Y)))


def _pool(workers, payload):
    if workers <= 1:
        _init_worker(payload)
        return None
    return ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(payload,))


def _map(pool, fn, jobs):
    return list(map(fn, jobs)) if pool is None else list(pool.map(fn, jobs))


def _fz(arr, axis=0):
    return np.tanh(np.mean(np.arctanh(np.clip(arr, -0.999999, 0.999999)), axis=axis))


def _configs(model, cfg):
    """Simulation tuning grid: kappa for every prior model, a_w for wiring models; d and a_u at
    the generating values (a reduced version of the section-6 grid, same for every model)."""
    ks = cfg.tune_kappa
    if model in ('M1', 'M2'):
        return [{'kappa': k} for k in ks]
    return [{'kappa': k, 'a_w': a} for k in ks for a in cfg.tune_aw]


def _A0(model, Wt, h, perm=None):
    n = Wt.shape[0]
    if model == 'M0':
        return np.zeros((n, n))
    if model == 'M1':
        return P.prior_mean('M1', n=n, d=TRUE_H['d'])
    if model == 'M2':
        return P.prior_mean('M2', n=n, d=TRUE_H['d'], a_u=TRUE_H['a_u'])
    hh = {**TRUE_H, 'a_w': h.get('a_w', TRUE_H['a_w'])}
    if model == 'M4':
        return P.prior_mean('M4', Wt=Wt, perm=perm, **hh)
    return P.prior_mean('M3', Wt=h.get('Wt', Wt), **hh)


def r3_scores(flies, Wt, models, fractions, n_sets, perms, cfg, extra_wirings=None, tag=0):
    """R3 per-fly hidden r for each model, averaged over hidden sets (Fisher z); M4 averaged
    over perms. Hyperparameters chosen by nested hiding (spec section 10) with the committed
    3-fold split of each hidden set's visible regions (seeds.json 'validation_folds'): each
    fold is hidden in turn as well as the test set, each grid configuration is fitted
    (cfg.tune_iter EM iterations, warm-started from M0 fitted with the same mask) and scored on
    that fold; the best mean over folds is refitted on the test mask (cfg.max_iter iterations,
    warm-started from M0). M4 is tuned on its permutation average.
    extra_wirings: {name: Wt'} models built like M3 on another wiring (S3's matching permutation)."""
    seeds = ensure_seeds()
    n = Wt.shape[0]
    extra_wirings = extra_wirings or {}
    out = {}
    pool = _pool(cfg.workers, {'flies': flies})
    try:
        for frac in fractions:
            sets = seeds['hidden_sets'][str(int(100 * frac))][:n_sets]
            key = str(int(100 * frac))
            vals = [seeds['validation_folds'][key][si][:cfg.n_val] for si in range(len(sets))]
            # stage A: M0 on the test mask and on every test+validation mask
            jobsA, keyA = [], []
            for si, h in enumerate(sets):
                jobsA.append(dict(fit_hidden=h, score=h, A0=np.zeros((n, n)), kappa=0.0, warm=None,
                                  max_iter=cfg.max_iter_m0, keep_params=True)); keyA.append((si, None))
                for j, v in enumerate(vals[si]):
                    jobsA.append(dict(fit_hidden=h + v, score=v, A0=np.zeros((n, n)), kappa=0.0, warm=None,
                                      max_iter=cfg.max_iter_m0, keep_params=True)); keyA.append((si, j))
            resA = dict(zip(keyA, _map(pool, _r3_job, jobsA)))
            # model list: (name, kind, Wt for the model)
            mods = [(m, m, Wt) for m in models if m != 'M0'] + [(nm, 'M3', W2) for nm, W2 in extra_wirings.items()]
            # stage B: tuning fits on validation masks
            jobsB, keyB = [], []
            for si, h in enumerate(sets):
                for name, kind, W in mods:
                    for ci, c in enumerate(_configs(kind, cfg)):
                        for j, v in enumerate(vals[si]):
                            for pi_i, pi in enumerate(perms[:cfg.tune_perms] if kind == 'M4' else [None]):
                                A0 = _A0(kind, W, {**c, 'Wt': W}, pi)
                                jobsB.append(dict(fit_hidden=h + v, score=v, A0=A0, kappa=c['kappa'],
                                                  warm=resA[(si, j)][1], max_iter=cfg.tune_iter, keep_params=False))
                                keyB.append((si, name, ci))
            resB = _map(pool, _r3_job, jobsB)
            val_score = {}
            for key, (sc, _, _) in zip(keyB, resB):
                val_score.setdefault(key, []).append(_fz(sc[:, 0]))      # mean over flies, one per val set / perm
            chosen = {}
            for si in range(len(sets)):
                for name, kind, W in mods:
                    cs = _configs(kind, cfg)
                    best = max(range(len(cs)), key=lambda ci: np.mean(val_score[(si, name, ci)]))
                    chosen[(si, name)] = cs[best]
            # stage C: refit at the chosen configuration on the test mask, score test hidden regions
            jobsC, keyC = [], []
            for si, h in enumerate(sets):
                for name, kind, W in mods:
                    c = chosen[(si, name)]
                    for pi in (perms if kind == 'M4' else [None]):
                        jobsC.append(dict(fit_hidden=h, score=h, A0=_A0(kind, W, {**c, 'Wt': W}, pi),
                                          kappa=c['kappa'], warm=resA[(si, None)][1], max_iter=cfg.max_iter,
                                          keep_params=False)); keyC.append((si, name))
            resC = _map(pool, _r3_job, jobsC)
            per = {'M0': {si: [resA[(si, None)][0]] for si in range(len(sets))}}
            for (si, name), (sc, _, _) in zip(keyC, resC):
                per.setdefault(name, {}).setdefault(si, []).append(sc)
            out[frac] = {m: _fz(np.stack([_fz(np.stack(v)) for v in d.values()])) for m, d in per.items()}
            out[frac]['_chosen'] = {f"{name} set {si}": c for (si, name), c in chosen.items()}
            out[frac]['_diag'] = [r[2] for r in list(resA.values()) + resC]
    finally:
        if pool is not None:
            pool.shutdown()
    return out                      # out[frac][model] -> (F, 2): smoothed, filtered


def r2_scores(flies, tune_flies, Wt, ms, perms, cfg):
    """R2 per-fly one-step R^2 for M0, M2, M3, M4 at each m. Hyperparameters per (model, m)
    chosen on separate tuning flies cut the same way (the spec tunes on the other 19 flies)."""
    mods = ['M2', 'M3', 'M4']
    pool = _pool(cfg.workers, {'flies': flies, 'tune': tune_flies})
    try:
        jobs, keys = [], []
        for m in ms:
            for mod in mods:
                for ci, c in enumerate(_configs(mod, cfg)):
                    for pi in (perms[:cfg.tune_perms] if mod == 'M4' else [None]):
                        for f in range(len(tune_flies)):
                            jobs.append(('tune', f, m, _A0(mod, Wt, c, pi), c['kappa'], cfg.max_iter))
                            keys.append((m, mod, ci))
        sc = {}
        for k, v in zip(keys, _map(pool, _r2_job, jobs)):
            sc.setdefault(k, []).append(v)
        chosen = {(m, mod): _configs(mod, cfg)[max(range(len(_configs(mod, cfg))),
                                                    key=lambda ci: np.mean(sc[(m, mod, ci)]))]
                  for m in ms for mod in mods}
        jobs, keys = [], []
        for m in ms:
            for f in range(len(flies)):
                jobs.append(('flies', f, m, np.zeros_like(Wt), 0.0, cfg.max_iter)); keys.append((m, 'M0', f))
                for mod in mods:
                    c = chosen[(m, mod)]
                    for pi in (perms if mod == 'M4' else [None]):
                        jobs.append(('flies', f, m, _A0(mod, Wt, c, pi), c['kappa'], cfg.max_iter))
                        keys.append((m, mod, f))
        acc = {}
        for (m, mod, f), v in zip(keys, _map(pool, _r2_job, jobs)):
            acc.setdefault((m, mod), {}).setdefault(f, []).append(v)
    finally:
        if pool is not None:
            pool.shutdown()
    out = {k: np.array([np.mean(d[f]) for f in sorted(d)]) for k, d in acc.items()}
    return out, chosen


# --------------------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------------------
def S0(Wt, cfg):
    rng = np.random.default_rng(sim_seed(0))
    n = Wt.shape[0]
    A, Q, r, Sx = make_truth(P.prior_mean('M3', Wt=Wt, **TRUE_H), rng)
    flies = simulate_flies(A, Q, r, Sx, rng, n_flies=cfg.n_flies)
    hidden = ensure_seeds()['hidden_sets']['50'][0]
    vis = np.ones(n, bool); vis[hidden] = False
    # (a) gauge: rescale / flip hidden coordinates -> identical likelihood
    c = np.ones(n); c[hidden] = rng.choice([-3.0, -0.5, 0.4, 2.5], len(hidden))
    D, Di = np.diag(c), np.diag(1 / c)
    segs = [np.where(vis, f[0], np.nan) for f in flies[:3]]
    ll1 = sum(K.kalman_filter(Y, vis, A, Q, r, Sx).loglik for Y in segs)
    ll2 = sum(K.kalman_filter(Y, vis, D @ A @ Di, D @ Q @ D, r, D @ Sx @ D).loglik for Y in segs)
    gauge = abs(ll1 - ll2) / abs(ll1)
    # (b) M0 and M1 leave hidden regions decoupled -> chance-level r; M3 shown for contrast
    sc = r3_scores(flies, Wt, ['M0', 'M1', 'M3'], [0.5], 1, [], cfg, tag=0)[0.5]
    worst = max(np.max(np.abs(sc['M0'][:, 0])), np.max(np.abs(sc['M1'][:, 0])))
    ok = gauge < 1e-8 and worst < 0.05
    print(f"S0 gauge: rel. loglik change under hidden rescaling/sign flip = {gauge:.1e} (< 1e-8)")
    print(f"   hidden r, 50% hidden, median over flies: M0 {np.median(sc['M0'][:, 0]):+.3f}, "
          f"M1 {np.median(sc['M1'][:, 0]):+.3f}, M3 {np.median(sc['M3'][:, 0]):+.3f}; "
          f"max |r| M0/M1 = {worst:.3f} (< 0.05)")
    print(f"   S0: {'PASS' if ok else 'FAIL'}")
    return ok, {'gauge_rel': gauge, 'max_abs_r_M0_M1': worst,
                'median_r': {m: float(np.median(sc[m][:, 0])) for m in ('M0', 'M1', 'M3')}}


def _r3_report(name, sc, frac, a, b):
    s = paired(sc[frac][a][:, 0], sc[frac][b][:, 0])
    print(f"   {name} {int(frac*100)}% hidden, {a} - {b}: {fmt(s)}")
    return s


def S1(Wt, cfg, perms):
    rng = np.random.default_rng(sim_seed(1))
    A, Q, r, Sx = make_truth(P.prior_mean('M3', Wt=Wt, **TRUE_H), rng)
    flies = simulate_flies(A, Q, r, Sx, rng, n_flies=cfg.n_flies)
    sc = r3_scores(flies, Wt, ['M0', 'M2', 'M3', 'M4'], cfg.fractions, cfg.n_sets, perms, cfg, tag=1)
    print("S1 (true A aligned with the connectome), R3:")
    det, rec = True, {'R3': {}, 'R2': {}}
    for fr in cfg.fractions:
        a = _r3_report('S1', sc, fr, 'M3', 'M4'); b = _r3_report('S1', sc, fr, 'M3', 'M2')
        rec['R3'][fr] = {'M3-M4': a, 'M3-M2': b, 'chosen': sc[fr]['_chosen'],
                         'median_r': {m: float(np.median(sc[fr][m][:, 0])) for m in ('M0', 'M2', 'M3', 'M4')}}
        print("      median hidden r: " + ", ".join(f"{m} {v:.3f}" for m, v in rec['R3'][fr]['median_r'].items()))
        det &= supported(a) and supported(b)
    if 0.25 in cfg.fractions and 0.75 in cfg.fractions:
        g = lambda fr: sc[fr]['M3'][:, 0] - sc[fr]['M4'][:, 0]
        h3 = paired(g(0.75), g(0.25))
        rec['H3_like'] = h3
        print(f"   H3 quantity, (M3-M4 at 75%) - (M3-M4 at 25%): {fmt(h3)}")
    ok_r2 = None
    if cfg.r2_m:
        print("S1, R2 (one-step R^2 on the last 30%, fit on the first m frames):")
        tune = simulate_flies(A, Q, r, Sx, rng, n_flies=cfg.n_tune_flies)
        s2, chosen = r2_scores(flies[:cfg.n_flies_r2], tune, Wt, cfg.r2_m, perms, cfg)
        gaps = []
        for m in cfg.r2_m:
            a = paired(s2[(m, 'M3')], s2[(m, 'M4')]); b = paired(s2[(m, 'M3')], s2[(m, 'M2')])
            med = {k: float(np.median(s2[(m, k)])) for k in ('M0', 'M2', 'M3', 'M4')}
            print(f"   m = {m:4d}: median R^2 " + ", ".join(f"{k} {v:.3f}" for k, v in med.items())
                  + f";  M3-M4 {fmt(a)}")
            gaps.append(a['hl'])
            rec['R2'][m] = {'M3-M4': a, 'M3-M2': b, 'median_R2': med,
                            'chosen': {k: chosen[(m, k)] for k in ('M2', 'M3', 'M4')}}
        a0 = rec['R2'][cfg.r2_m[0]]
        ok_r2 = supported(a0['M3-M4'], 0.01) and supported(a0['M3-M2'], 0.01) and gaps[-1] < gaps[0]
        print(f"   R2: M3 beats M4 and M2 at m = {cfg.r2_m[0]} and the gap shrinks: {'yes' if ok_r2 else 'no'}")
    print(f"   S1: planted effect supported in R3 at every fraction (spec rule): {'PASS' if det else 'FAIL'}")
    rec['detected_R3'] = det; rec['R2_ok'] = ok_r2
    return det, rec


def S2(Wt, cfg, perms):
    rng = np.random.default_rng(sim_seed(2))
    n = Wt.shape[0]
    Wr = random_wiring(n, rng)
    A, Q, r, Sx = make_truth(P.prior_mean('M3', Wt=Wr, **TRUE_H), rng)
    flies = simulate_flies(A, Q, r, Sx, rng, n_flies=cfg.n_flies)
    sc = r3_scores(flies, Wt, ['M0', 'M2', 'M3', 'M4'], cfg.fractions_s23, cfg.n_sets, perms, cfg, tag=2)
    print("S2 (true A unrelated to the connectome), R3:")
    ok, rec = True, {}
    for fr in cfg.fractions_s23:
        a = _r3_report('S2', sc, fr, 'M3', 'M4'); b = _r3_report('S2', sc, fr, 'M3', 'M2')
        ok &= not supported(a) and not supported(b)           # a false positive = supported by the spec rule
        if significant(a) or significant(b):
            print("      note: a difference is significant but below the 0.02 smallest effect of interest")
        rec[fr] = {'M3-M4': a, 'M3-M2': b, 'chosen': sc[fr]['_chosen'],
                   'within_bootstrap_error': {k: s['ci'][0] <= 0 <= s['ci'][1] for k, s in (('M3-M4', a), ('M3-M2', b))}}
    print(f"   S2: no false positive for H2 (spec rule): {'PASS' if ok else 'FAIL'}")
    return ok, rec


def S3(Wt, cfg, perms):
    rng = np.random.default_rng(sim_seed(3))
    n = Wt.shape[0]
    pi_true = rng.permutation(n)                     # not one of the committed shuffles
    A, Q, r, Sx = make_truth(P.prior_mean('M4', Wt=Wt, perm=pi_true, **TRUE_H), rng)
    flies = simulate_flies(A, Q, r, Sx, rng, n_flies=cfg.n_flies)
    sc = r3_scores(flies, Wt, ['M0', 'M2', 'M3'], cfg.fractions_s23, cfg.n_sets, perms, cfg,
                   extra_wirings={'M4match': P.permute(Wt, pi_true)}, tag=3)
    print("S3 (true A built from a permuted connectome), R3:")
    ok, rec = True, {}
    for fr in cfg.fractions_s23:
        a = _r3_report('S3', sc, fr, 'M4match', 'M3'); b = _r3_report('S3', sc, fr, 'M3', 'M2')
        ok &= significant(a) and not supported(b)
        rec[fr] = {'M4match-M3': a, 'M3-M2': b, 'chosen': sc[fr]['_chosen']}
    print(f"   S3: matching permutation wins (significant), M3 not supported over M2: {'PASS' if ok else 'FAIL'}")
    return ok, rec


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--matrix', default='Connectivity')
    ap.add_argument('--only', nargs='*', default=['S0', 'S1', 'S2', 'S3'], choices=['S0', 'S1', 'S2', 'S3'])
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--quick', action='store_true', help='small sizes: smoke test only, not a gate result')
    ap.add_argument('--no_hash_check', action='store_true', help='skip hashes.sha256 (never for gate runs)')
    ap.add_argument('--out', default='results/simulations.json')
    cfg = ap.parse_args()
    # Sizes (full run = what G2 is judged on)
    cfg.n_flies = 20; cfg.n_sets = 3; cfg.n_shuffles = 5
    cfg.fractions = [0.25, 0.5, 0.75]; cfg.fractions_s23 = [0.5]
    cfg.n_val = 3; cfg.tune_iter = 30; cfg.tune_perms = 5
    cfg.tune_kappa = [0.01, 0.1, 1.0]; cfg.tune_aw = [0.0, 0.5, 1.0, 2.0]
    cfg.max_iter_m0 = 150; cfg.max_iter = 100
    cfg.r2_m = [30, 120, 500, 1000]; cfg.n_flies_r2 = 20; cfg.n_tune_flies = 5
    if cfg.quick:
        cfg.n_flies = 8; cfg.n_sets = 1; cfg.n_shuffles = 2; cfg.fractions = [0.5]
        cfg.n_val = 1; cfg.tune_iter = 5; cfg.tune_perms = 2; cfg.tune_kappa = [0.1, 1.0]; cfg.tune_aw = [0.0, 1.0]
        cfg.max_iter_m0 = 15; cfg.max_iter = 10; cfg.r2_m = [60, 500]; cfg.n_flies_r2 = 6; cfg.n_tune_flies = 3

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/')])
    seeds = ensure_seeds()
    from build_dataset import load
    _, names, S, _ = load(cfg.npz)
    Wt = P.wiring(S[cfg.matrix].values)
    perms = seeds['shuffles'][:cfg.n_shuffles]
    print(f"structure: {cfg.matrix}, n = {len(names)}; {cfg.n_flies} flies x {T_FLY} frames; "
          f"fractions {cfg.fractions}; {cfg.n_sets} hidden sets; {cfg.n_shuffles} shuffles; "
          f"workers {cfg.workers}{'; QUICK MODE' if cfg.quick else ''}")

    results, verdict = {}, {}
    fns = {'S0': lambda: S0(Wt, cfg), 'S1': lambda: S1(Wt, cfg, perms), 'S2': lambda: S2(Wt, cfg, perms),
           'S3': lambda: S3(Wt, cfg, perms)}
    for k in cfg.only:
        t0 = time.perf_counter()
        verdict[k], results[k] = fns[k]()
        print(f"   [{k}: {time.perf_counter() - t0:.0f} s]\n")
    g2 = [k for k in ('S0', 'S1', 'S2', 'S3')]
    if all(k in verdict for k in g2):
        print("G2:", "PASS" if all(verdict[k] for k in g2) else "FAIL",
              "(S0 gauge, S1 detects, S2 and S3 no false positive)" + (" -- QUICK MODE, not a gate result" if cfg.quick else ""))
    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': {k: v for k, v in vars(cfg).items()}, 'verdict': verdict, 'results': results},
                  f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
