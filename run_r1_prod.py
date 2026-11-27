"""
Tier B R1 production run: population regime, leave-one-fly-out (Protocol and Model Spec,
sections 2 (H5, H6), 10, 11 and 15). Secondary regime, expected null.

    python run_r1_prod.py --npz data_processed/fly_ito36.npz --workers 6
    python run_r1_prod.py ... --by day          # sensitivity: leave-one-recording-day-out, cluster bootstrap over days
    python run_r1_prod.py ... --smoke           # 2 held-out flies, tiny grid and iterations: code check only
    python run_r1_prod.py ... --report_only     # re-print from the cache

Design (everything that defines the run comes from seeds.json through integrity.tier_settings,
tier B, protocol v2):
  Folds     each held-out fly (or recording day with --by day); training = the other 19 flies,
            all 36 regions observed (section 11).
  Models    M0-M5 on the coarse grid (tierb.py); configurations with rho(A0) >= 0.98 dropped.
  Tuning    inner 3-fold cross-validation over the training flies: inner fold i = training flies
            [i::3] in sorted order (sizes 7, 6, 6; the phase-4 rule), tuning fits of
            tier['tuning_em_iters'] EM iterations from the ridge initialisation (section 10).
            Two objectives, as section 10 lists: T1 = mean one-step R^2 on the inner held-out flies;
            T2 at fraction p = Fisher-z mean hidden-region r on those flies with the tier's hidden
            sets of fraction p hidden at test. M4 is tuned on the mean over all tier shuffles.
  Final     refit on the 19 training flies at each chosen configuration (at most
            tier['final_em_iters_max'] iterations, single ridge initialisation as in the R3
            production fits), scored on the held-out fly.
  Scores    T1: filtered one-step R^2 over all regions, pooled over segments (run_r1.t1_score);
            secondary: held-out log-likelihood per observed entry, R^2 / noise ceiling (H6; ceiling
            = 1 - the fly's mean high-frequency floor, em.hf_floor). T2: for every hidden set of the
            tier at 25/50/75%, the hidden regions are masked in the held-out fly only; Pearson r
            between the smoothed estimate and the observed series per hidden region, Fisher-z over
            regions, then over sets (filtered r and hidden-region R^2 secondary). M4 = mean over
            shuffles (Fisher-z for r).
  Baselines B1-B3 (T1; run_r1.baselines) and B4-B8 (T2; run_r3.baselines), same folds and sets.
  Tests     secondary and uncorrected (section 11): M3 against M0, M1, M2, M4 and M5 for T1 and for
            T2 at each fraction, paired Wilcoxon over flies, HL estimate and 95% bootstrap interval,
            flagged when they meet the decision rule (SESOI 0.01 in R^2, 0.02 in r). H5 expects no effect.
Hidden data never enter a fit: training flies are fully observed, the held-out fly enters no fit
or tuning step, and its hidden regions are masked only when it is scored.
Checkpointing: every finished fit is appended to <cache>; a rerun skips finished fits.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import em
import kalman as K
import prior as P
import tierb as T
from build_distances import load_distances
from integrity import ensure_seeds, check_hashes, tier_settings

FRACTIONS = (0.25, 0.5, 0.75)
_G = {}


def _init(payload):
    _G.update(payload)


def day_of(fly):
    return fly.split('_')[0]


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def score_t1(segs, p):
    """Filtered one-step R^2 pooled over segments (first frame of each segment unscored) and
    held-out log-likelihood per observed entry."""
    sse = ss = ll = 0.0; nobs = 0
    for Y in segs:
        fr = K.kalman_filter(Y, None, p.A, p.Q, p.r, p.Sigma0, steady_state=True)
        e = Y[1:] - fr.x_pred[1:]
        sse += float(np.sum(e * e)); ss += float(np.sum(Y[1:] ** 2)); ll += fr.loglik; nobs += fr.n_obs
    return 1.0 - sse / ss, ll / nobs


def score_t2(segs, p, hidden):
    """Hidden regions masked in this fly only: (r smoothed, r filtered, hidden R^2 smoothed)."""
    n = p.A.shape[0]
    hid = np.zeros(n, bool); hid[list(hidden)] = True
    res = [K.filter_smooth(Y, ~hid, p.A, p.Q, p.r, p.Sigma0, steady_state=True) for Y in segs]
    rs = em.hidden_r(segs, res, hid, 'smooth')[0]
    rf = em.hidden_r(segs, res, hid, 'filt')[0]
    est = np.concatenate([sr.x_smooth[:, hid] for _, sr in res]); tru = np.concatenate([Y[:, hid] for Y in segs])
    return rs, rf, float(1.0 - np.sum((tru - est) ** 2) / np.sum(tru ** 2))


def _segs(flies):
    return [Y for f in flies for Y in _G['flies'][f]]


def _fit(train, key, max_iter):
    segs = _segs(train)
    wir = _G['wir']
    return em.fit(segs, None, wir.a0(key), wir.kappa(key), max_iter=max_iter)


def tune_job(job):
    train = _G['folds'][job['fold']]['train']
    val = train[job['inner']::3]
    res = _fit([f for f in train if f not in val], job['eff'], job['iters'])
    p = res.params
    out = {'key': job['key'], 'T1': {f: score_t1(_G['flies'][f], p)[0] for f in val}, 'T2': {}}
    for fr, sets in _G['sets'].items():
        out['T2'][str(fr)] = {f: float(T.fz([score_t2(_G['flies'][f], p, h)[0] for h in sets])) for f in val}
    out['n_iter'] = res.n_iter; out['rho_A'] = res.diagnostics['rho_A']
    return out


def final_job(job):
    fold = _G['folds'][job['fold']]
    t0 = time.perf_counter()
    res = _fit(fold['train'], job['eff'], job['iters'])
    fit_s = time.perf_counter() - t0
    p = res.params
    per_fly = {}
    for f in fold['test']:
        segs = _G['flies'][f]
        t1, llo = score_t1(segs, p)
        t2 = {str(fr): [score_t2(segs, p, h) for h in sets] for fr, sets in _G['sets'].items()}
        per_fly[f] = {'T1': t1, 'loglik_per_obs': llo,
                      'T2_smooth': {k: [v[0] for v in vs] for k, vs in t2.items()},
                      'T2_filt': {k: [v[1] for v in vs] for k, vs in t2.items()},
                      'T2_R2': {k: [v[2] for v in vs] for k, vs in t2.items()}}
    d = res.diagnostics
    return {'key': job['key'], 'per_fly': per_fly, 'logpost': res.logpost, 'n_iter': res.n_iter,
            'converged': res.converged, 'flags': res.flags, 'rho_A': d['rho_A'], 'fit_s': fit_s,
            'min_eig_Q': d['min_eig_Q'], 'r_min': d['r_min'], 'r_max': d['r_max'], 'cond_S00_lam': d['cond_S00_lam']}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--centroids', default='data_processed/centroids_ito36.npz')
    ap.add_argument('--matrix', default='Connectivity')
    ap.add_argument('--protocol', default='v2', choices=['v1', 'v2'])
    ap.add_argument('--by', default='fly', choices=['fly', 'day'], help='day = leave-one-recording-day-out sensitivity run')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--smoke', action='store_true', help='2 held-out units, 2 configs per model, 2 shuffles, few iterations')
    ap.add_argument('--report_only', action='store_true')
    ap.add_argument('--no_hash_check', action='store_true', help='never for a real run')
    cfg = ap.parse_args()
    tag = f"r1_tierB_{cfg.protocol}" + ('_day' if cfg.by == 'day' else '') + ('_smoke' if cfg.smoke else '')
    cfg.cache = cfg.cache or f"results/{tag}/fits.jsonl"
    cfg.out = cfg.out or f"results/{tag}.json"

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/'), cfg.centroids.replace('\\', '/')])
    seeds = ensure_seeds()
    tier = tier_settings(seeds, 'B', cfg.protocol)
    if 'R1' not in tier['regimes']:
        raise RuntimeError("tier B settings do not include R1 (seeds.json reductions)")
    flies, names, S, _ = K.load_standardised(cfg.npz)
    order = sorted(flies)
    Wt = P.wiring(S[cfg.matrix].values)
    D, _ = load_distances(cfg.centroids, names)
    shuffle_ids = list(tier['shuffles'])
    set_idx = list(tier['hidden_sets'])
    tune_iter, final_iter = tier['tuning_em_iters'], tier['final_em_iters_max']
    if cfg.smoke:
        shuffle_ids, set_idx, tune_iter, final_iter = shuffle_ids[:2], set_idx[:2], 2, 3
    wir = T.Wirings(Wt, P.distance_wiring(D), [(s, seeds['shuffles'][s]) for s in shuffle_ids])
    table, dropped = T.model_table(wir, shuffle_ids)
    if cfg.smoke:
        table = {m: v[-2:] for m, v in table.items()}
    sets = {fr: [seeds['hidden_sets'][str(int(100 * fr))][j] for j in set_idx] for fr in FRACTIONS}

    if cfg.by == 'fly':
        units = [[f] for f in order]
    else:
        days = sorted({day_of(f) for f in order})
        units = [[f for f in order if day_of(f) == d] for d in days]
    if cfg.smoke:
        units = units[:2]
    folds = [{'test': u, 'train': [f for f in order if f not in u]} for u in units]
    n_eff = len({k for m in table for _, ks in table[m] for k in ks})
    print(f"R1 tier B, protocol {cfg.protocol}, leave-one-{cfg.by}-out: {len(folds)} folds; models "
          + ", ".join(f"{m} {len(table[m])}" for m in T.MODELS)
          + f" configs ({n_eff} distinct fits per data set; dropped for rho >= 0.98: {sum(len(v) for v in dropped.values())}); "
          f"shuffles {len(shuffle_ids)}; hidden sets {len(set_idx)} per fraction; tuning {tune_iter} it, final <= {final_iter} it; "
          f"workers {cfg.workers}" + ('; SMOKE (not a result)' if cfg.smoke else ''), flush=True)

    # ---- baselines (cheap) ----
    from run_r1 import baselines as b_t1
    from run_r3 import baselines as b_t2
    B1 = b_t1(flies, order)
    B2 = b_t2(flies, order, Wt, sets, [seeds['shuffles'][s] for s in shuffle_ids])
    test_flies = [f for fo in folds for f in fo['test']]

    payload = {'flies': flies, 'wir': wir, 'folds': folds, 'sets': sets}
    cache = T.Cache(cfg.cache)
    pool = None
    if not cfg.report_only and cfg.workers > 1:
        pool = ProcessPoolExecutor(cfg.workers, initializer=_init, initargs=(payload,))
    else:
        _init(payload)
    t_run = time.perf_counter()
    try:
        effs = sorted({k for m in table for _, ks in table[m] for k in ks})
        jobs = [{'key': f"T|{fi}|{i}|{e}", 'fold': fi, 'inner': i, 'eff': e, 'iters': tune_iter}
                for fi in range(len(folds)) for i in range(3) for e in effs]
        if not cfg.report_only:
            T.run_jobs(pool, cache, jobs, tune_job, 'tuning (inner 3-fold CV)')
        # ---- selection: per fold, model and objective ----
        def inner_score(fi, keys, obj, fr=None):
            vals = []
            for k in keys:                                   # M4: one key per shuffle
                per_inner = []
                for i in range(3):
                    r = cache.d.get(f"T|{fi}|{i}|{k}")
                    if r is None:
                        return -np.inf
                    v = list(r['T1'].values()) if obj == 'T1' else list(r['T2'][str(fr)].values())
                    per_inner.append(np.mean(v) if obj == 'T1' else T.fz(v))
                vals.append(np.mean(per_inner))
            return float(np.mean(vals))
        chosen = {}
        for fi in range(len(folds)):
            for m in T.MODELS:
                cs = table[m]
                chosen[(fi, m, 'T1')] = int(np.argmax([inner_score(fi, ks, 'T1') for _, ks in cs]))
                for fr in FRACTIONS:
                    chosen[(fi, m, fr)] = int(np.argmax([inner_score(fi, ks, 'T2', fr) for _, ks in cs]))
        need = {(fi, k) for (fi, m, _), ci in chosen.items() for k in table[m][ci][1]}
        jobs = [{'key': f"F|{fi}|{e}", 'fold': fi, 'eff': e, 'iters': final_iter} for fi, e in sorted(need)]
        if not cfg.report_only:
            T.run_jobs(pool, cache, jobs, final_job, 'final fits')
    finally:
        if pool is not None:
            pool.shutdown()
    wall = time.perf_counter() - t_run

    # ---- aggregate per fly ----
    fold_of = {f: fi for fi, fo in enumerate(folds) for f in fo['test']}
    score = {'T1': {}, 'loglik_per_obs': {}, **{f'T2_{k}|{fr}': {} for k in ('smooth', 'filt', 'R2') for fr in FRACTIONS}}
    complete = True
    for m in T.MODELS:
        for name in score:
            score[name][m] = {}
        for f in test_flies:
            fi = fold_of[f]
            def recs(obj):
                ks = table[m][chosen[(fi, m, obj)]][1]
                return [cache.d.get(f"F|{fi}|{k}") for k in ks]
            r1 = recs('T1')
            if any(r is None for r in r1):
                complete = False; continue
            score['T1'][m][f] = float(np.mean([r['per_fly'][f]['T1'] for r in r1]))
            score['loglik_per_obs'][m][f] = float(np.mean([r['per_fly'][f]['loglik_per_obs'] for r in r1]))
            for fr in FRACTIONS:
                rr = recs(fr)
                if any(r is None for r in rr):
                    complete = False; continue
                score[f'T2_smooth|{fr}'][m][f] = float(T.fz([T.fz(r['per_fly'][f]['T2_smooth'][str(fr)]) for r in rr]))
                score[f'T2_filt|{fr}'][m][f] = float(T.fz([T.fz(r['per_fly'][f]['T2_filt'][str(fr)]) for r in rr]))
                score[f'T2_R2|{fr}'][m][f] = float(np.mean([np.mean(r['per_fly'][f]['T2_R2'][str(fr)]) for r in rr]))
    ceiling = {f: float(1.0 - np.mean(em.hf_floor(flies[f], np.ones(len(names), bool)))) for f in test_flies}

    med = lambda d: float(np.median(list(d.values()))) if d else float('nan')
    print(f"\nR1 results ({'complete' if complete else 'INCOMPLETE: rerun to resume'}), median over {len(test_flies)} flies:")
    print("   T1 one-step R^2: " + ", ".join(f"{m} {med(score['T1'][m]):.4f}" for m in T.MODELS)
          + "; baselines " + ", ".join(f"{b} {np.median([B1[b][f] for f in test_flies]):.4f}" for b in ('B1', 'B2', 'B3')))
    print("   held-out log-likelihood per entry: " + ", ".join(f"{m} {med(score['loglik_per_obs'][m]):.4f}" for m in T.MODELS))
    if score['T1']['M3']:
        print(f"   H6: T1 / noise ceiling (M3) median {np.median([score['T1']['M3'][f] / ceiling[f] for f in score['T1']['M3']]):.3f}; "
              f"ceiling median {np.median(list(ceiling.values())):.3f}")
    for fr in FRACTIONS:
        print(f"   T2 hidden r, {int(fr * 100)}% hidden at test: " + ", ".join(f"{m} {med(score[f'T2_smooth|{fr}'][m]):.4f}" for m in T.MODELS)
              + "; baselines " + ", ".join(f"{b} {np.median([B2[fr][b][f] for f in test_flies]):.4f}" for b in ('B4', 'B5', 'B6', 'B7', 'B8')))
    tests = {}
    if complete and len(test_flies) >= 6:
        print("\nSecondary tests (uncorrected; H5 expects no prior effect), M3 minus each model:")
        for name, metric in [('T1', 'R2')] + [(f'T2_smooth|{fr}', 'r') for fr in FRACTIONS]:
            tests[name] = {}
            for b in ('M0', 'M1', 'M2', 'M4', 'M5'):
                a_ = [score[name]['M3'][f] for f in test_flies]; b_ = [score[name][b][f] for f in test_flies]
                s = T.compare(a_, b_, metric)
                if cfg.by == 'day':
                    s['cluster'] = T.cluster_ci(a_, b_, [day_of(f) for f in test_flies])
                tests[name][f'M3-{b}'] = s
                print(f"   {name:16s} M3-{b}: {T.fmt(s, metric)}"
                      + (f"; day-cluster CI [{s['cluster']['ci_days'][0]:+.4f}, {s['cluster']['ci_days'][1]:+.4f}]" if 'cluster' in s else ''))
    fins = [cache.d[k] for k in (f"F|{fi}|{e}" for fi, e in need) if k in cache.d]
    if fins:
        rho = np.array([r['rho_A'] for r in fins])
        print(f"\n   final fits {len(fins)}: {sum(not r['converged'] for r in fins)} hit the {final_iter}-iteration cap; "
              f"rho(A) max {rho.max():.3f} ({int(np.sum(rho >= 1))} >= 1, shrunk)")
    print(f"wall time this invocation {wall / 60:.1f} min")

    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': vars(cfg), 'tier': tier, 'complete': complete,
                   'configs': {m: [h for h, _ in table[m]] for m in T.MODELS}, 'dropped_configs': dropped,
                   'chosen': {f"{fi}|{m}|{obj}": table[m][ci][0] for (fi, m, obj), ci in chosen.items()},
                   'folds': folds, 'scores': score, 'noise_ceiling': ceiling,
                   'baselines_T1': B1, 'baselines_T2': {str(k): v for k, v in B2.items()}, 'tests': tests,
                   'diagnostics': [{k: v for k, v in r.items() if k != 'per_fly'} for r in fins]},
                  f, indent=1, default=str)
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
