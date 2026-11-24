"""
Phase 5: R3 production run (tier A by default) -- hidden-region reconstruction with M0-M4,
baselines B4-B8 recomputed as Pearson r, and the preliminary H1-H3 tests
(Protocol and Model Spec, sections 2, 6, 10-12, 15; work plan phase 5, gate G4).

    python run_r3.py --npz data_processed/fly_ito36.npz --workers 16          # tier A
    python run_r3.py ... --smoke                                               # code check, not a result
    python run_r3.py ... --report_only                                         # re-print from the cache

Everything that defines the run comes from seeds.json through integrity.tier_settings():
hidden sets (tier A: 0-4 per fraction), shuffles (tier A: 0-4), validation folds (3 per
hidden set), random-search configurations (M0: all 5 kappa; M1, M2, M3: 16 each; M4 uses
M3's configurations), tuning EM iterations (30) and the final-fit cap (150).

Procedure for every fraction p in (25, 50, 75%) and hidden set j (spec section 10):
  Stage A  M0 (kappa = 0) fitted on the test mask and on each test + validation-fold mask
           (at most 150 EM iterations): the warm starts.
  Stage B  every configuration of every model fitted on each test + fold mask for 30 EM
           iterations, warm-started from the Stage A fit with the same mask (entries of A
           touching hidden regions reset to the model's A0), scored by hidden-region r on that
           fold's regions; M4 on each of its shuffles.
  Select   per model, the configuration with the highest mean validation score (mean over
           the 3 folds, and over the shuffles for M4).
  Stage C  refit at the chosen configuration on the test mask (at most 150 iterations, warm
           from the Stage A test fit; M4 once per shuffle), then score the test hidden regions.
Hidden-region data never enter any fit, tuning step or model standardisation (hidden columns
are NaN in every fit; masks are from seeds.json).

Scores (T2, spec section 11): per fly, Pearson r between the smoothed estimate and the
standardised observed series of each hidden region, Fisher-z averaged over hidden regions,
then over hidden sets; M4 per fly = Fisher-z mean over its shuffles. The filtered estimate is
reported as secondary. Baselines B4-B8 use the same hidden sets and metric.

Tests (preliminary for tier A; confirmatory on tier B): H1 M3 - M1 per fraction; H2 M3 - M4
and M3 - M2 per fraction, combined as an intersection-union test (p = max of the two); H3
(M3 - M4 at 75%) - (M3 - M4 at 25%). Holm over the primary family of 7 tests (H1 and H2 at
each of 3 fractions, H3 once; section 2, corrected Nov 24); a hypothesis is supported only if
Holm p < 0.05, median paired difference >= 0.02 and the 95% bootstrap interval excludes 0.
For H2 the median and interval rule must hold for both comparisons.

Checkpointing: every finished fit is appended to <cache>/fits.jsonl (Stage A parameters as
.npz); a rerun skips finished fits, so a crash loses at most the fits in flight.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import em
import kalman as K
import prior as P
from integrity import ensure_seeds, check_hashes, tier_settings
from simulate import paired, hodges_lehmann   # same statistics as the simulation checks

FRACTIONS = (0.25, 0.5, 0.75)
MODELS = ('M0', 'M1', 'M2', 'M3', 'M4')
SESOI_R = 0.02
B_REF_R2 = {'B4': (0.413, 0.397, 0.313), 'B5': (0.428, 0.421, 0.329), 'B6': (0.285, 0.295, 0.243)}

_G = {}


def _init(payload):
    _G.update(payload)


def _fz(x, axis=0):
    return np.tanh(np.mean(np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999)), axis=axis))


# --------------------------------------------------------------------------------------
# Model specification
# --------------------------------------------------------------------------------------
def a0_for(model, cfg, perm=None):
    Wt, n = _G['Wt'], _G['n']
    h = {'d': cfg.get('d', 0.0), 'a_u': cfg.get('a_u', 0.0), 'a_w': cfg.get('a_w', 0.0)}
    if model == 'M0':
        return np.zeros((n, n))
    if model == 'M4':
        return P.prior_mean('M4', Wt=Wt, perm=perm, **h)
    return P.prior_mean(model, n=n, Wt=Wt, **h)


def configs_for(model, seeds, Wt, perms):
    """Committed random-search configurations, minus any with rho(A0) >= 0.98 (spec section 6)."""
    src = seeds['random_search']['M3' if model in ('M3', 'M4') else model]
    keep, dropped = [], []
    for c in src:
        h = {'d': c.get('d', 0.0), 'a_u': c.get('a_u', 0.0), 'a_w': c.get('a_w', 0.0)}
        if model == 'M0':
            ok = True
        elif model == 'M4':
            ok = all(P.spectral_radius(P.prior_mean('M4', Wt=Wt, perm=p, **h)) < 0.98 for p in perms)
        else:
            ok = P.spectral_radius(P.prior_mean(model, n=Wt.shape[0], Wt=Wt, **h)) < 0.98
        (keep if ok else dropped).append(c)
    return keep, dropped


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------
def _fit_and_score(job):
    """job keys: key, stage, fit_hidden, score, model, cfg, perm, warm (Params|None), max_iter."""
    flies, order, n = _G['flies'], _G['order'], _G['n']
    vis = np.ones(n, bool); vis[list(job['fit_hidden'])] = False
    segs = [np.where(vis, Y, np.nan) for f in order for Y in flies[f]]
    A0 = a0_for(job['model'], job['cfg'], job['perm'])
    init = em.warm_start(job['warm'], A0, vis) if job['warm'] is not None else None
    t0 = time.perf_counter()
    res = em.fit(segs, vis, A0, job['cfg'].get('kappa', 0.0), init=init, max_iter=job['max_iter'])
    fit_s = time.perf_counter() - t0
    _, kept = em.e_step(segs, vis, res.params, keep=True)
    sc = np.zeros(n, bool); sc[list(job['score'])] = True
    per_fly, k = {}, 0
    for f in order:
        kk = kept[k:k + len(flies[f])]; k += len(flies[f])
        rs, per_region = em.hidden_r(flies[f], kk, sc, 'smooth')
        rf, _ = em.hidden_r(flies[f], kk, sc, 'filt')
        per_fly[f] = {'r_smooth': rs, 'r_filt': rf, 'per_region': per_region}
    p = res.params
    out = {'key': job['key'], 'stage': job['stage'], 'per_fly': per_fly, 'logpost': res.logpost,
           'n_iter': res.n_iter, 'converged': res.converged, 'flags': res.flags,
           'rho_A': res.diagnostics['rho_A'], 'fit_s': fit_s}
    if job['stage'] == 'A':
        out['params'] = {'A': p.A.tolist(), 'Q': p.Q.tolist(), 'r': p.r.tolist(), 'Sigma0': p.Sigma0.tolist()}
    return out


# --------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------
class Cache:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.d = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line); self.d[r['key']] = r

    def add(self, r):
        self.d[r['key']] = r
        with open(self.path, 'a') as f:
            f.write(json.dumps(r) + '\n')

    def params(self, key):
        p = self.d[key]['params']
        return em.Params(np.array(p['A']), np.array(p['Q']), np.array(p['r']), np.array(p['Sigma0']))


def run_stage(pool, cache, jobs, label):
    todo = [j for j in jobs if j['key'] not in cache.d]
    print(f"  stage {label}: {len(jobs)} fits, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    t0 = time.perf_counter(); done = 0
    if pool is None:
        it = map(_fit_and_score, todo)
    else:
        futs = [pool.submit(_fit_and_score, j) for j in todo]
        it = (f.result() for f in as_completed(futs))
    for r in it:
        cache.add(r); done += 1
        if done % max(1, len(todo) // 20) == 0 or done == len(todo):
            el = time.perf_counter() - t0
            print(f"    {done}/{len(todo)} done, {el / 60:.1f} min elapsed, "
                  f"~{el / done * (len(todo) - done) / 60:.0f} min left", flush=True)


# --------------------------------------------------------------------------------------
# Baselines B4-B8 as Pearson r (same hidden sets and metric)
# --------------------------------------------------------------------------------------
def _r_rows(est, tru):
    return [em.pearson(est[i], tru[i]) for i in range(len(tru))]


def baselines(flies, order, Wt, sets_by_frac, perms, K_CTX=2):
    """B4 static Gaussian conditioning (covariance from the other 19 flies); B5 B4 plus +/-2 frames of
    visible context, ridge 100; B6 wiring-weighted average of visible regions; B7 same with shuffled
    wiring (mean over shuffles); B8 uniform average. Pearson r per hidden region, Fisher-z over
    regions, then over hidden sets. r does not depend on a gain, so B6-B8 need no data from hidden
    regions at all and are strict R3 baselines; B4 and B5 use other flies' hidden data (R1-like)."""
    n = Wt.shape[0]
    Z = {f: [Y.T for Y in flies[f]] for f in order}                      # (n, T) segments
    allZ = {f: np.concatenate(Z[f], 1) for f in order}
    Sxx = {f: allZ[f] @ allZ[f].T for f in order}; Nf = {f: allZ[f].shape[1] for f in order}
    Stot, Ntot = sum(Sxx.values()), sum(Nf.values())
    U = P.uniform(n)
    out = {}
    for frac, sets in sets_by_frac.items():
        acc = {b: {f: [] for f in order} for b in ('B4', 'B5', 'B6', 'B7', 'B8')}
        for Hh in sets:
            Hh = np.array(Hh); V = np.setdiff1d(np.arange(n), Hh)

            def ctx(segs):
                X, Y = [], []
                for a in segs:
                    T = a.shape[1]
                    if T <= 2 * K_CTX:
                        continue
                    X.append(np.concatenate([a[V][:, K_CTX + o:T - K_CTX + o] for o in range(-K_CTX, K_CTX + 1)], 0))
                    Y.append(a[:, K_CTX:T - K_CTX])
                return np.concatenate(X, 1), np.concatenate(Y, 1)
            C = {f: ctx(Z[f]) for f in order}
            Gx = {f: C[f][0] @ C[f][0].T for f in order}; Hx = {f: C[f][1][Hh] @ C[f][0].T for f in order}
            Gt, Ht = sum(Gx.values()), sum(Hx.values())

            def wavg(Wm, X):
                w = Wm[np.ix_(Hh, V)]
                return (w / w.sum(1, keepdims=True)) @ X
            for h in order:
                Ctr = (Stot - Sxx[h]) / (Ntot - Nf[h])
                B = Ctr[np.ix_(Hh, V)] @ np.linalg.inv(Ctr[np.ix_(V, V)] + 1e-3 * np.eye(len(V)))
                Y = allZ[h][Hh]
                acc['B4'][h].append(_fz(_r_rows(B @ allZ[h][V], Y)))
                Bc = (Ht - Hx[h]) @ np.linalg.inv(Gt - Gx[h] + 100 * np.eye(Gt.shape[0]))
                acc['B5'][h].append(_fz(_r_rows(Bc @ C[h][0], C[h][1][Hh])))
                acc['B6'][h].append(_fz(_r_rows(wavg(Wt, allZ[h][V]), Y)))
                acc['B7'][h].append(_fz([_fz(_r_rows(wavg(P.permute(Wt, p), allZ[h][V]), Y)) for p in perms]))
                acc['B8'][h].append(_fz(_r_rows(wavg(U, allZ[h][V]), Y)))
        out[frac] = {b: {f: float(_fz(v)) for f, v in d.items()} for b, d in acc.items()}
    return out


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------
def holm(pvals, m=None):
    """Holm step-down adjusted p-values; m = family size (>= len(pvals); extra slots count as p = 1)."""
    p = np.asarray(pvals, float); k = len(p); m = m or k
    order = np.argsort(p); adj = np.empty(k); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def tests(score, order):
    """score[frac][model] -> {fly: r}. Returns the primary-family table."""
    v = lambda fr, m: np.array([score[fr][m][f] for f in order])
    rows = []
    for fr in FRACTIONS:
        rows.append({'test': f'H1 {int(fr * 100)}%', 'parts': {'M3-M1': paired(v(fr, 'M3'), v(fr, 'M1'))}})
        rows.append({'test': f'H2 {int(fr * 100)}%', 'parts': {'M3-M4': paired(v(fr, 'M3'), v(fr, 'M4')),
                                                                'M3-M2': paired(v(fr, 'M3'), v(fr, 'M2'))}})
    g = lambda fr: v(fr, 'M3') - v(fr, 'M4')
    rows.append({'test': 'H3 (75% - 25%)', 'parts': {'gap75-gap25': paired(g(0.75), g(0.25))}})
    raw = [max(s['p'] for s in r['parts'].values()) for r in rows]          # IUT for H2
    adj = holm(raw)                                                            # family of 7
    for r, pr, pa in zip(rows, raw, adj):
        r['p_raw'], r['p_holm'] = float(pr), float(pa)
        r['supported'] = bool(pa < 0.05 and all(s['median'] >= SESOI_R and s['ci'][0] > 0 for s in r['parts'].values()))
    return rows


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--tier', default='A', choices=['A', 'B'])
    ap.add_argument('--matrix', default='Connectivity')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--cache', default=None, help='default results/r3_tier<A|B>/fits.jsonl')
    ap.add_argument('--out', default=None, help='default results/r3_tier<A|B>.json')
    ap.add_argument('--smoke', action='store_true', help='1 fraction, 1 set, 2 configs, 1 fold, 2 shuffles, few iterations')
    ap.add_argument('--report_only', action='store_true', help='no fitting: report from the cache')
    ap.add_argument('--no_hash_check', action='store_true', help='never for a real run')
    cfg = ap.parse_args()
    tag = f"r3_tier{cfg.tier}" + ('_smoke' if cfg.smoke else '')
    cfg.cache = cfg.cache or f"results/{tag}/fits.jsonl"
    cfg.out = cfg.out or f"results/{tag}.json"

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/')])
    seeds = ensure_seeds()
    if 'random_search' not in seeds:
        raise RuntimeError("seeds.json has no random-search configurations: run `python integrity.py --amend`, "
                           "re-hash and commit first")
    tier = tier_settings(seeds, cfg.tier)
    if cfg.tier == 'B':
        raise NotImplementedError("tier B adds M5, R1 and R2; run tier A first (spec section 15)")
    flies, names, S, _ = K.load_standardised(cfg.npz)
    order = sorted(flies)
    n = len(names)
    Wt = P.wiring(S[cfg.matrix].values)
    perms = [seeds['shuffles'][s] for s in tier['shuffles']]
    set_idx = tier['hidden_sets']
    fracs = FRACTIONS
    tune_iter, final_iter = tier['tuning_em_iters'], tier['final_em_iters_max']
    n_folds = 3
    if cfg.smoke:
        fracs, set_idx, perms, n_folds, tune_iter, final_iter = (0.5,), set_idx[:1], perms[:2], 1, 2, 3

    cfgs, dropped = {}, {}
    for m in MODELS:
        cfgs[m], dropped[m] = configs_for(m, seeds, Wt, perms)
        if cfg.smoke:
            cfgs[m] = cfgs[m][:2]
    print(f"R3 tier {cfg.tier}: {len(order)} flies, fractions {fracs}, hidden sets {set_idx}, shuffles "
          f"{tier['shuffles'] if not cfg.smoke else 'first 2'}, {n_folds} validation folds; configs "
          + ", ".join(f"{m} {len(cfgs[m])}" for m in MODELS)
          + f" (dropped for rho >= 0.98: {sum(len(v) for v in dropped.values())}); tuning {tune_iter} it, "
          f"final <= {final_iter} it; workers {cfg.workers}" + ('; SMOKE (not a result)' if cfg.smoke else ''))

    sets = {fr: {j: seeds['hidden_sets'][str(int(100 * fr))][j] for j in set_idx} for fr in fracs}
    folds = {fr: {j: seeds['validation_folds'][str(int(100 * fr))][j][:n_folds] for j in set_idx} for fr in fracs}

    # ---- baselines B4-B8 (cheap; recomputed every run) ----
    t0 = time.perf_counter()
    B = baselines(flies, order, Wt, {fr: [sets[fr][j] for j in set_idx] for fr in fracs}, perms)
    print(f"Baselines B4-B8, hidden-region Pearson r, median over flies ({time.perf_counter() - t0:.0f} s):")
    for fr in fracs:
        print(f"   {int(fr * 100)}% hidden: " + ", ".join(f"{b} {np.median(list(B[fr][b].values())):.3f}"
                                                         for b in ('B4', 'B5', 'B6', 'B7', 'B8')))

    cache = Cache(cfg.cache)
    pool = None
    if not cfg.report_only:
        payload = {'flies': flies, 'order': order, 'n': n, 'Wt': Wt}
        if cfg.workers > 1:
            pool = ProcessPoolExecutor(cfg.workers, initializer=_init, initargs=(payload,))
        else:
            _init(payload)
    else:
        _init({'flies': flies, 'order': order, 'n': n, 'Wt': Wt})
    t_run = time.perf_counter()
    try:
        # Stage A
        jobsA = []
        for fr in fracs:
            for j in set_idx:
                h = sets[fr][j]
                jobsA.append(dict(key=f"A|{fr}|{j}|test", stage='A', fit_hidden=h, score=h, model='M0',
                                  cfg={'kappa': 0.0}, perm=None, warm=None, max_iter=final_iter))
                for q, v in enumerate(folds[fr][j]):
                    jobsA.append(dict(key=f"A|{fr}|{j}|fold{q}", stage='A', fit_hidden=h + v, score=v, model='M0',
                                      cfg={'kappa': 0.0}, perm=None, warm=None, max_iter=final_iter))
        if not cfg.report_only:
            run_stage(pool, cache, jobsA, 'A (M0 warm starts)')
        # Stage B
        jobsB = []
        for fr in fracs:
            for j in set_idx:
                h = sets[fr][j]
                for q, v in enumerate(folds[fr][j]):
                    warm = cache.params(f"A|{fr}|{j}|fold{q}") if f"A|{fr}|{j}|fold{q}" in cache.d else None
                    for m in MODELS:
                        for ci, c in enumerate(cfgs[m]):
                            for si, pi in (enumerate(perms) if m == 'M4' else [(None, None)]):
                                jobsB.append(dict(key=f"B|{fr}|{j}|fold{q}|{m}|c{ci}|s{si}", stage='B',
                                                  fit_hidden=h + v, score=v, model=m, cfg=c, perm=pi,
                                                  warm=warm, max_iter=tune_iter))
        if not cfg.report_only:
            run_stage(pool, cache, jobsB, 'B (tuning)')
        # Select
        chosen = {}
        for fr in fracs:
            for j in set_idx:
                for m in MODELS:
                    def val(ci):
                        ks = [k for k in cache.d if k.startswith(f"B|{fr}|{j}|") and f"|{m}|c{ci}|" in k]
                        return np.mean([_fz(list(r['r_smooth'] for r in cache.d[k]['per_fly'].values())) for k in ks]) \
                            if ks else -np.inf
                    scores = [val(ci) for ci in range(len(cfgs[m]))]
                    chosen[(fr, j, m)] = int(np.argmax(scores))
        # Stage C
        jobsC = []
        for fr in fracs:
            for j in set_idx:
                h = sets[fr][j]
                warm = cache.params(f"A|{fr}|{j}|test") if f"A|{fr}|{j}|test" in cache.d else None
                for m in MODELS:
                    ci = chosen[(fr, j, m)]
                    for si, pi in (enumerate(perms) if m == 'M4' else [(None, None)]):
                        jobsC.append(dict(key=f"C|{fr}|{j}|{m}|c{ci}|s{si}", stage='C', fit_hidden=h, score=h,
                                          model=m, cfg=cfgs[m][ci], perm=pi, warm=warm, max_iter=final_iter))
        if not cfg.report_only:
            run_stage(pool, cache, jobsC, 'C (final fits)')
    finally:
        if pool is not None:
            pool.shutdown()
    wall = time.perf_counter() - t_run

    # ---- aggregate: per fly, Fisher-z over shuffles (M4), then over hidden sets ----
    score, score_f, diag = {}, {}, []
    for fr in fracs:
        score[fr], score_f[fr] = {}, {}
        for m in MODELS:
            per_set, per_set_f = [], []
            for j in set_idx:
                ci = chosen[(fr, j, m)]
                ks = [f"C|{fr}|{j}|{m}|c{ci}|s{si}" for si in (range(len(perms)) if m == 'M4' else [None])]
                rs = [cache.d[k] for k in ks if k in cache.d]
                if not rs:
                    continue
                per_set.append([_fz([r['per_fly'][f]['r_smooth'] for r in rs]) for f in order])
                per_set_f.append([_fz([r['per_fly'][f]['r_filt'] for r in rs]) for f in order])
                diag += [{'frac': fr, 'set': j, 'model': m, 'config': cfgs[m][ci], 'rho_A': r['rho_A'],
                          'n_iter': r['n_iter'], 'converged': r['converged'], 'flags': r['flags']} for r in rs]
            if per_set:
                score[fr][m] = dict(zip(order, (float(x) for x in _fz(np.array(per_set), axis=0))))
                score_f[fr][m] = dict(zip(order, (float(x) for x in _fz(np.array(per_set_f), axis=0))))

    complete = all(m in score[fr] for fr in fracs for m in MODELS)
    print(f"\nHidden-region r (smoothed), median over flies [filtered in brackets]; tier {cfg.tier} = preliminary:")
    for fr in fracs:
        if all(m in score[fr] for m in MODELS):
            print(f"   {int(fr * 100)}% hidden: " + ", ".join(
                f"{m} {np.median(list(score[fr][m].values())):.3f} [{np.median(list(score_f[fr][m].values())):.3f}]"
                for m in MODELS))
    if diag:
        rho = np.array([d['rho_A'] for d in diag])
        cap = sum(not d['converged'] for d in diag)
        print(f"   final fits: {len(diag)}; rho(A) max {rho.max():.3f} ({int(np.sum(rho >= 1))} >= 1, shrunk); "
              f"{cap} hit the {final_iter}-iteration cap")
        print("   chosen configurations: " + "; ".join(
            f"{int(fr*100)}%/set {j}/{m}: {cfgs[m][chosen[(fr, j, m)]]}" for fr in fracs for j in set_idx[:1] for m in MODELS[1:]))

    rows = []
    if complete and not cfg.smoke and set(fracs) == set(FRACTIONS):
        rows = tests(score, order)
        print("\nPrimary family (Holm over 7; supported = Holm p < 0.05, median >= 0.02, 95% CI > 0):")
        for r in rows:
            parts = "; ".join(f"{k}: median {s['median']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]"
                              for k, s in r['parts'].items())
            print(f"   {r['test']:16s} p {r['p_raw']:.2g}, Holm {r['p_holm']:.2g} -> "
                  f"{'SUPPORTED' if r['supported'] else 'not supported'}   ({parts})")
        sims = "S1 found the opposite sign (-0.032) under a planted connectome effect; S2 showed M4 tuning noise can favour H2 by about 0.003"
        print(f"   notes: {sims}.")
        # rank of the real wiring among the shuffles, per fly (M3 vs each shuffle's final fit)
        for fr in fracs:
            ranks = []
            for f in order:
                m4 = []
                for j in set_idx:
                    ci = chosen[(fr, j, 'M4')]
                    m4.append([cache.d[f"C|{fr}|{j}|M4|c{ci}|s{si}"]['per_fly'][f]['r_smooth'] for si in range(len(perms))])
                per_shuffle = _fz(np.array(m4), axis=0)
                ranks.append(int(np.sum(per_shuffle >= score[fr]['M3'][f])) + 1)
            print(f"   {int(fr * 100)}%: rank of real wiring among {len(perms) + 1} (1 = best), median over flies "
                  f"{np.median(ranks):.0f}; best in {sum(r == 1 for r in ranks)}/{len(order)} flies")
        print("   Kalman vs baselines (median over flies of M3 - B): " + "; ".join(
            f"{int(fr*100)}%: " + ", ".join(f"{b} {np.median([score[fr]['M3'][f] - B[fr][b][f] for f in order]):+.3f}"
                                           for b in ('B6', 'B7', 'B8')) for fr in fracs))
    elif not complete:
        print("\n(run incomplete: rerun the same command to resume from the cache)")
    print(f"\nwall time this invocation {wall / 60:.1f} min")

    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': vars(cfg), 'tier': tier, 'n_configs': {m: len(cfgs[m]) for m in MODELS},
                   'dropped_configs': dropped, 'baselines_r': {str(k): v for k, v in B.items()},
                   'chosen': {f"{fr}|{j}|{m}": cfgs[m][ci] for (fr, j, m), ci in chosen.items()},
                   'score_smooth': {str(k): v for k, v in score.items()},
                   'score_filt': {str(k): v for k, v in score_f.items()},
                   'tests': rows, 'diagnostics': diag, 'complete': complete}, f, indent=1, default=str)
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
