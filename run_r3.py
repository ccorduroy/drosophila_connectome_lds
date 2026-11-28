"""
Phase 5: R3 production run (tier A by default) -- hidden-region reconstruction with M0-M4,
baselines B4-B8 recomputed as Pearson r, and the preliminary H1-H3 tests
(Protocol and Model Spec, sections 2, 6, 10-12, 15; work plan phase 5, gate G4).

    python run_r3.py --npz data_processed/fly_ito36.npz --workers 16          # tier A', protocol v2
    python run_r3.py ... --protocol v1 --report_only                           # the original tier A report
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

Protocol v2 (default; tier B amendments approved Nov 26, recorded in seeds.json 'protocol_v2'):
connectome inverse-Wishart prior on Q with kappa_Q = 1 (target from prior.q_target: visible block
of the Stage A warm start, hidden blocks from the model's own wiring -- uniform M2, real M3,
shuffled M4, none for M0/M1; Sigma0's hidden blocks by the same rule); configurations from
random_search_v2 (kappa >= 0.01 for M1-M4); final-fit cap 300. Stage A warm starts are fitted
without the Q prior, as before. Results go to results/r3_tierA_v2.json (cache
results/r3_tierA_v2/) so the protocol-v1 results stay intact. Also reported: whether M3 reaches
B6 before H2 is interpreted, and H1 against B8.

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
    if model == 'M5':                       # distance prior: M3's recipe on exp(-D / l) (spec section 6)
        return P.prior_mean('M3', Wt=_G['Wd'], **h)
    return P.prior_mean(model, n=n, Wt=Wt, **h)


def configs_for(model, seeds, Wt, perms, source='random_search', Wd=None):
    """Committed random-search configurations, minus any with rho(A0) >= 0.98 (spec section 6).
    M4 and M5 use M3's configurations (seeds.json rules); M5's check uses the distance wiring Wd."""
    src = seeds[source]['M3' if model in ('M3', 'M4', 'M5') else model]
    keep, dropped = [], []
    for c in src:
        h = {'d': c.get('d', 0.0), 'a_u': c.get('a_u', 0.0), 'a_w': c.get('a_w', 0.0)}
        if model == 'M0':
            ok = True
        elif model == 'M4':
            ok = all(P.spectral_radius(P.prior_mean('M4', Wt=Wt, perm=p, **h)) < 0.98 for p in perms)
        elif model == 'M5':
            ok = P.spectral_radius(P.prior_mean('M3', Wt=Wd, **h)) < 0.98
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
    qkw = {}
    if job.get('kappa_Q', 0.0) > 0 and init is not None:
        W = P.wiring_for(job['model'], Wt=_G['Wt'], perm=job['perm'], Wdist=_G.get('Wd'), n=n)
        Qt = P.q_target(job['warm'].Q, vis, W, job.get('q_frac', 0.5))
        init.Q = Qt.copy()
        init.Sigma0 = P.q_target(job['warm'].Sigma0, vis, W, job.get('q_frac', 0.5))
        qkw = dict(Q_target=Qt, kappa_Q=job['kappa_Q'])
    t0 = time.perf_counter()
    res = em.fit(segs, vis, A0, job['cfg'].get('kappa', 0.0), init=init, max_iter=job['max_iter'], **qkw)
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


def reuse_v2_fits(cache, seeds, Wt, perms, cfgs, v2_path, kQ):
    """Protocol v3 nests v2: every model's v2 configurations are the first entries of its v3 list, in the
    same order, so a v2 job and the v3 job with the same key are the same computation (stage A: M0
    warm starts; stages B and C: same mask, model, configuration index, shuffle, Q prior and iteration
    cap). Checks that nesting, then copies those finished fits into the v3 cache."""
    v2_tier = tier_settings(seeds, 'A', 'v2')
    for k in ('hidden_sets', 'shuffles', 'tuning_em_iters', 'final_em_iters_max', 'kappa_Q', 'q_structured_frac'):
        if v2_tier[k] != tier_settings(seeds, 'A', 'v3')[k]:
            raise RuntimeError(f"v2 and v3 differ in {k}: no reuse")
    n_v2 = {}
    for m in MODELS:
        c2, _ = configs_for(m, seeds, Wt, perms, 'random_search_v2')
        if cfgs[m][:len(c2)] != c2:
            raise RuntimeError(f"{m}: v3 configurations do not start with the v2 ones: no reuse")
        n_v2[m] = len(c2)
    if not os.path.exists(v2_path):
        raise RuntimeError(f"{v2_path} not found")
    v2 = Cache(v2_path)
    added = 0
    for key, r in v2.d.items():
        if key in cache.d:
            continue
        parts = key.split('|')
        if parts[0] in ('B', 'C'):
            m, ci = (parts[4], parts[5]) if parts[0] == 'B' else (parts[3], parts[4])
            if int(ci[1:]) >= n_v2[m]:
                continue
        cache.add(r); added += 1
    print(f"  reused {added} finished protocol-v2 fits from {v2_path} (nesting checked)", flush=True)


def reuse_fits(cache, seeds, Wt, Wd, src_tier, src_protocol, src_path, tier, cfgs, n_folds):
    """Copy finished fits of another R3 run (e.g. tier A-prime v3 into tier B v3) whose jobs are the same
    computation as a job of this run. A job is defined by its stage, fraction, hidden set (seeds.json index),
    validation fold, model, configuration (matched by value, not index) and shuffle (matched by shuffle
    id), plus the iteration caps, the Q prior and the warm start (stage A of the same mask, itself reused only
    when identical). Settings that would change any fit must agree, else nothing is reused."""
    src = tier_settings(seeds, src_tier, src_protocol)
    for k in ('tuning_em_iters', 'final_em_iters_max', 'q_prior', 'kappa_Q', 'q_structured_frac', 'random_search'):
        if src[k] != tier[k]:
            raise RuntimeError(f"--reuse_from: {k} differs ({src[k]} vs {tier[k]}): no reuse")
    if n_folds != 3:
        raise RuntimeError("--reuse_from: validation folds differ: no reuse")
    src_perms = [seeds['shuffles'][s] for s in src['shuffles']]
    src_cfgs = {m: configs_for(m, seeds, Wt, src_perms, src['random_search'], Wd)[0] for m in src['models']}
    dst_sid = {s: i for i, s in enumerate(tier['shuffles'])}
    sets_dst = set(tier['hidden_sets'])
    if not os.path.exists(src_path):
        raise RuntimeError(f"{src_path} not found")
    srcc = Cache(src_path)
    added = skipped = 0
    for key, r in srcc.d.items():
        p = key.split('|')
        if int(p[2]) not in sets_dst:
            skipped += 1; continue
        if p[0] == 'A':
            new = key
        else:
            m, ci, si = (p[4], p[5], p[6]) if p[0] == 'B' else (p[3], p[4], p[5])
            if m not in cfgs:
                skipped += 1; continue
            c = src_cfgs[m][int(ci[1:])]
            if c not in cfgs[m]:
                skipped += 1; continue
            nci = f"c{cfgs[m].index(c)}"
            if m == 'M4':
                sid = src['shuffles'][int(si[1:])]
                if sid not in dst_sid:
                    skipped += 1; continue
                nsi = f"s{dst_sid[sid]}"
            else:
                nsi = si
            new = '|'.join(p[:4] + [m, nci, nsi]) if p[0] == 'B' else '|'.join(p[:3] + [m, nci, nsi])
        if new in cache.d:
            continue
        cache.add({**r, 'key': new}); added += 1
    print(f"  reused {added} finished fits from {src_path} (tier {src_tier}, protocol {src_protocol}; matched by "
          f"configuration value and shuffle id; {skipped} not applicable)", flush=True)


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
    if 'M5' in score[FRACTIONS[0]]:              # secondary, outside the Holm family: wiring vs proximity
        for fr in FRACTIONS:
            s5 = paired(v(fr, 'M3'), v(fr, 'M5'))
            rows.append({'test': f'secondary: M3 - M5 {int(fr * 100)}%', 'parts': {'M3-M5': s5}, 'p_raw': s5['p'],
                         'meets_rule_uncorrected': bool(s5['p'] < 0.05 and s5['median'] >= SESOI_R and s5['ci'][0] > 0)})
    return rows


def sim_note(protocol):
    """Simulation context for the H2 and H3 lines, read from the G2 results of the same protocol
    (results/simulations_v2.json for v2, results/simulations.json for v1): the H3 quantity under
    S1's planted connectome effect, and S2's M3 - M4 when there is no connectome effect."""
    path = 'results/simulations_v2.json' if protocol in ('v2', 'v3') else 'results/simulations.json'   # v3 changes the grid only
    try:
        with open(path) as f:
            res = json.load(f)['results']
        h3 = res['S1']['H3_like']['median']
        s2 = max(v['M3-M4']['median'] for v in res['S2'].values())
        return (f"G2 simulations ({protocol}, {path}): S1 found the H3 quantity {h3:+.3f} under a planted "
                f"connectome effect ({'opposite sign to H3' if h3 < 0 else 'same sign as H3'}); S2 showed M4 tuning "
                f"noise can favour H2 by about {s2:.3f} with no connectome effect")
    except (OSError, KeyError, ValueError) as e:
        return f"G2 simulation results for protocol {protocol} not available ({path}: {e.__class__.__name__})"


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--tier', default='A', choices=['A', 'B'])
    ap.add_argument('--protocol', default='v2', choices=['v1', 'v2', 'v3'],
                    help='v3 (Nov 27) = v2 with the grid-edge extension (seeds.json random_search_v3)')
    ap.add_argument('--reuse_v2', action='store_true',
                    help='v3 only: copy finished v2 fits whose job is identical (same stage, mask, model, '
                         'configuration and shuffle; checked) into the v3 cache instead of refitting them')
    ap.add_argument('--matrix', default='Connectivity')
    ap.add_argument('--centroids', default='data_processed/centroids_ito36.npz', help='frozen distances for M5 (tier B)')
    ap.add_argument('--reuse_from', default=None,
                    help='TIER,PROTOCOL,FITS.jsonl: copy finished fits of that run whose jobs are identical to jobs of '
                         'this run (same stage, mask, model, configuration and shuffle, matched by value; checked)')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--cache', default=None, help='default results/r3_tier<A|B>/fits.jsonl')
    ap.add_argument('--out', default=None, help='default results/r3_tier<A|B>.json')
    ap.add_argument('--smoke', action='store_true', help='1 fraction, 1 set, 2 configs, 1 fold, 2 shuffles, few iterations')
    ap.add_argument('--report_only', action='store_true', help='no fitting: report from the cache')
    ap.add_argument('--no_hash_check', action='store_true', help='never for a real run')
    cfg = ap.parse_args()
    tag = f"r3_tier{cfg.tier}" + ({'v2': '_v2', 'v3': '_v3'}.get(cfg.protocol, '')) + ('_smoke' if cfg.smoke else '')
    cfg.cache = cfg.cache or f"results/{tag}/fits.jsonl"
    cfg.out = cfg.out or f"results/{tag}.json"

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/')])
    seeds = ensure_seeds()
    if 'random_search' not in seeds:
        raise RuntimeError("seeds.json has no random-search configurations: run `python integrity.py --amend`, "
                           "re-hash and commit first")
    tier = tier_settings(seeds, cfg.tier, cfg.protocol)
    kQ = tier['kappa_Q'] if tier['q_prior'] else 0.0
    global MODELS
    MODELS = tuple(tier['models'])
    if 'R3' not in tier['regimes']:
        raise RuntimeError(f"tier {cfg.tier} settings do not include R3")
    flies, names, S, _ = K.load_standardised(cfg.npz)
    order = sorted(flies)
    n = len(names)
    Wt = P.wiring(S[cfg.matrix].values)
    Wd = None
    if 'M5' in MODELS:
        from build_distances import load_distances
        if not cfg.no_hash_check:
            check_hashes(required=[cfg.centroids.replace('\\', '/')])
        Wd = P.distance_wiring(load_distances(cfg.centroids, names)[0])
    perms = [seeds['shuffles'][s] for s in tier['shuffles']]
    set_idx = tier['hidden_sets']
    fracs = FRACTIONS
    tune_iter, final_iter = tier['tuning_em_iters'], tier['final_em_iters_max']
    n_folds = 3
    if cfg.smoke:
        fracs, set_idx, perms, n_folds, tune_iter, final_iter = (0.5,), set_idx[:1], perms[:2], 1, 2, 3

    cfgs, dropped = {}, {}
    for m in MODELS:
        cfgs[m], dropped[m] = configs_for(m, seeds, Wt, perms, tier['random_search'], Wd)
        if cfg.smoke:
            cfgs[m] = cfgs[m][:2]
    print(f"R3 tier {cfg.tier}, protocol {cfg.protocol}" + (f" (Q prior, kappa_Q = {kQ:g})" if kQ else "")
          + f": {len(order)} flies, fractions {fracs}, hidden sets {set_idx}, shuffles "
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
    if cfg.reuse_v2 and not cfg.report_only:
        if cfg.protocol != 'v3' or cfg.smoke:
            raise ValueError("--reuse_v2 is for the full protocol-v3 run only")
        reuse_v2_fits(cache, seeds, Wt, perms, cfgs, f"results/r3_tier{cfg.tier}_v2/fits.jsonl", kQ)
    if cfg.reuse_from and not cfg.report_only:
        src_tier, src_protocol, src_path = cfg.reuse_from.split(',')
        reuse_fits(cache, seeds, Wt, Wd, src_tier, src_protocol, src_path, tier, cfgs, n_folds)
    pool = None
    if not cfg.report_only:
        payload = {'flies': flies, 'order': order, 'n': n, 'Wt': Wt, 'Wd': Wd}
        if cfg.workers > 1:
            pool = ProcessPoolExecutor(cfg.workers, initializer=_init, initargs=(payload,))
        else:
            _init(payload)
    else:
        _init({'flies': flies, 'order': order, 'n': n, 'Wt': Wt, 'Wd': Wd})
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
                                                  warm=warm, max_iter=tune_iter, kappa_Q=kQ,
                                                  q_frac=tier['q_structured_frac']))
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
                                          model=m, cfg=cfgs[m][ci], perm=pi, warm=warm, max_iter=final_iter,
                                          kappa_Q=kQ, q_frac=tier['q_structured_frac']))
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
    label = (f"tier B (protocol {cfg.protocol}) = confirmatory" if cfg.tier == 'B' else
             f"tier {cfg.tier}" + {"v2": "'", "v3": "' (grid-edge extension)"}.get(cfg.protocol, '') + f" (protocol {cfg.protocol}) = preliminary")
    print(f"\nHidden-region r (smoothed), median over flies [filtered in brackets]; {label}:")
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

    # ---- hidden-set-level summary (added Nov 26): flies share each set's mask, tuning and joint fit,
    #      so the hidden sets are the units that vary independently ----
    set_level = {}
    if complete:
        from scipy.stats import binomtest
        print("\nHidden-set summary (median over flies of the paired difference, per set; sign count over sets):")
        for fr in fracs:
            per = {}
            for j in set_idx:
                fly_r = {}
                for m in MODELS:
                    ci = chosen[(fr, j, m)]
                    ks = [f"C|{fr}|{j}|{m}|c{ci}|s{si}" for si in (range(len(perms)) if m == 'M4' else [None])]
                    fly_r[m] = np.array([_fz([cache.d[k]['per_fly'][f]['r_smooth'] for k in ks]) for f in order])
                per[j] = {'M3-M1': float(np.median(fly_r['M3'] - fly_r['M1'])),
                          'M3-M4': float(np.median(fly_r['M3'] - fly_r['M4'])),
                          'M3-M2': float(np.median(fly_r['M3'] - fly_r['M2'])),
                          **({'M3-M5': float(np.median(fly_r['M3'] - fly_r['M5']))} if 'M5' in fly_r else {}),
                          'kappa0': [m for m in ('M2', 'M3', 'M4') if cfgs[m][chosen[(fr, j, m)]].get('kappa', 0) == 0]}
            summ = {}
            for comp in [c for c in ('M3-M1', 'M3-M4', 'M3-M2', 'M3-M5') if c in per[set_idx[0]]]:
                d = np.array([per[j][comp] for j in set_idx]); pos = int(np.sum(d > 0)); nz = int(np.sum(d != 0))
                summ[comp] = {'positive': pos, 'nonzero': nz, 'median': float(np.median(d)),
                              'sign_p': float(binomtest(pos, nz).pvalue) if nz else 1.0}
            set_level[fr] = {'per_set': per, 'summary': summ}
            print(f"   {int(fr * 100)}% hidden:")
            for j in set_idx:
                q = per[j]
                print(f"      set {j}: M3-M1 {q['M3-M1']:+.3f}  M3-M4 {q['M3-M4']:+.3f}  M3-M2 {q['M3-M2']:+.3f}"
                      + (f"  M3-M5 {q['M3-M5']:+.3f}" if 'M3-M5' in q else "")
                      + (f"   (kappa = 0 chosen for {', '.join(q['kappa0'])})" if q['kappa0'] else ''))
            print("      over sets: " + "; ".join(f"{c} positive in {v['positive']}/{v['nonzero']} (median {v['median']:+.3f}, "
                                                  f"sign test p {v['sign_p']:.2g})" for c, v in summ.items()))
        print(f"      (with {len(set_idx)} sets the smallest possible two-sided sign-test p is {2 * 0.5 ** len(set_idx):.4g})")

    rows = []
    if complete and not cfg.smoke and set(fracs) == set(FRACTIONS):
        rows = tests(score, order)
        print("\nPrimary family (Holm over 7; supported = Holm p < 0.05, median >= 0.02, 95% CI > 0):")
        for r in rows:
            parts = "; ".join(f"{k}: median {s['median']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]"
                              for k, s in r['parts'].items())
            if 'p_holm' not in r:
                ok = ' -> meets the rule' if r['meets_rule_uncorrected'] else ""
                print(f"   {r['test']} (uncorrected): p {r['p_raw']:.2g}{ok}   ({parts})")
                continue
            print(f"   {r['test']:16s} p {r['p_raw']:.2g}, Holm {r['p_holm']:.2g} -> "
                  f"{'SUPPORTED' if r['supported'] else 'not supported'}   ({parts})")
        print(f"   notes: {sim_note(cfg.protocol)}.")
        # rank of the real wiring among the shuffles, per fly (M3 vs each shuffle's final fit)
        for fr in fracs:
            ranks, ties = [], 0
            for f in order:
                m4 = []
                for j in set_idx:
                    ci = chosen[(fr, j, 'M4')]
                    m4.append([cache.d[f"C|{fr}|{j}|M4|c{ci}|s{si}"]['per_fly'][f]['r_smooth'] for si in range(len(perms))])
                per_shuffle = _fz(np.array(m4), axis=0)
                d = per_shuffle - score[fr]['M3'][f]
                ranks.append(int(np.sum(d > 1e-12)) + 1); ties += int(np.sum(np.abs(d) <= 1e-12))
            print(f"   {int(fr * 100)}%: rank of real wiring among {len(perms) + 1} (1 = best; ties counted in its favour), "
                  f"median over flies {np.median(ranks):.0f}; best in {sum(r == 1 for r in ranks)}/{len(order)} flies"
                  + (f"; {ties} fly-shuffle ties (identical models)" if ties else ''))
        print("   Kalman vs baselines (median over flies of M3 - B): " + "; ".join(
            f"{int(fr*100)}%: " + ", ".join(f"{b} {np.median([score[fr]['M3'][f] - B[fr][b][f] for f in order]):+.3f}"
                                           for b in ('B6', 'B7', 'B8')) for fr in fracs))
        b6_ok = {fr: bool(np.median(list(score[fr]['M3'].values())) >= np.median(list(B[fr]['B6'].values())))
                 for fr in fracs}
        print("   amendment 4 -- H2 is interpreted only if M3 reaches B6: " + "; ".join(
            f"{int(fr*100)}%: {'yes' if ok else 'NO, report this first'}" for fr, ok in b6_ok.items()))
        h1b8 = {fr: paired(np.array([score[fr]['M3'][f] for f in order]), np.array([B[fr]['B8'][f] for f in order]))
                for fr in fracs}
        print("   H1 against B8 (uniform average): " + "; ".join(
            f"{int(fr*100)}%: median {s_['median']:+.3f} [{s_['ci'][0]:+.3f}, {s_['ci'][1]:+.3f}]" for fr, s_ in h1b8.items()))
        rows.append({'test': 'secondary: M3 reaches B6', 'by_fraction': {str(k): v for k, v in b6_ok.items()}})
        rows.append({'test': 'secondary: H1 against B8', 'by_fraction': {str(k): v for k, v in h1b8.items()}})
    elif not complete:
        print("\n(run incomplete: rerun the same command to resume from the cache)")
    print(f"\nwall time this invocation {wall / 60:.1f} min")

    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': vars(cfg), 'tier': tier, 'protocol': cfg.protocol, 'n_configs': {m: len(cfgs[m]) for m in MODELS},
                   'dropped_configs': dropped, 'baselines_r': {str(k): v for k, v in B.items()},
                   'chosen': {f"{fr}|{j}|{m}": cfgs[m][ci] for (fr, j, m), ci in chosen.items()},
                   'score_smooth': {str(k): v for k, v in score.items()},
                   'score_filt': {str(k): v for k, v in score_f.items()},
                   'tests': rows, 'set_level': {str(k): v for k, v in set_level.items()},
                   'diagnostics': diag, 'complete': complete}, f, indent=1, default=str)
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
