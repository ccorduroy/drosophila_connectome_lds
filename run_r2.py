"""
Tier B R2 run: data-limited regime, learning curve over m (Protocol and Model Spec, sections 2
(H4), 10, 11 and 15). Secondary.

    python run_r2.py --npz data_processed/fly_ito36.npz --workers 6
    python run_r2.py ... --smoke           # 4 flies, 2 values of m, tiny grid: code check only
    python run_r2.py ... --report_only     # re-print from the cache

Design (settings from seeds.json through integrity.tier_settings, tier B, protocol v2):
  Split     per fly, the within-segment transitions in time order (segments are never joined);
            the last 30% of them are the test transitions, and the model is fitted on the first m
            transitions alone (m = 30, 60, 120, 250, 500, 1000; section 11). A (fly, m) pair is
            scored only if those m transitions end before the test part starts (2018-10-20_1 has
            629 such transitions, so it has no m = 1000 score); missing pairs are reported.
  Models    M0-M5 on the coarse grid of section 15 (tierb.py; section 10's table says "full grid"
            for R2, section 15 and prior.COARSE_GRID say coarse: coarse is used here).
  Fit       EM on the fly's first m transitions only, ridge initialisation, at most
            tier['final_em_iters_max'] iterations; every fit is used both for tuning and scoring.
  Score     filtered one-step R^2 over the test transitions (filter run over each whole segment, so
            the state at the test start is a causal estimate), 1 - SSE / sum(y^2).
  Tuning    for target fly f at a given m, each model's configuration is the one with the highest
            mean R^2 over the other 19 flies cut the same way (section 10); M4 on the mean over all
            tier shuffles. The fit on fly g never sees fly f, so no extra fits are needed.
  Baseline  B3m: ridge VAR(1) (lambda = 100) on the same m transitions, same test transitions.
  Tests     per m, secondary and uncorrected: H4 is M3 - M0; also M3 against M1, M2, M4 and M5
            (paired Wilcoxon over flies, HL estimate, 95% bootstrap interval; SESOI 0.01 in R^2).
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

M_VALUES = (30, 60, 120, 250, 500, 1000)
TEST_FRAC = 0.3
RIDGE = 100.0
_G = {}


def _init(payload):
    _G.update(payload)


# --------------------------------------------------------------------------------------
def split(segs, test_frac=TEST_FRAC):
    """Transitions as (segment, t) with t -> t+1 inside the segment, in time order; returns the
    list, the index where the test part starts, and the number of training transitions available."""
    tr = [(k, t) for k, Y in enumerate(segs) for t in range(len(Y) - 1)]
    n_test = int(round(test_frac * len(tr)))
    start = len(tr) - n_test
    return tr, start


def train_segments(segs, m):
    """The fly's first m transitions as segment pieces (a piece keeps its own frames; never joined)."""
    out, left = [], m
    for Y in segs:
        if left <= 0:
            break
        k = min(left, len(Y) - 1)
        if k >= 1:
            out.append(Y[:k + 1]); left -= k
    return out


def available(segs, m):
    _, start = split(segs)
    return m <= start


def test_r2(segs, pred_fn):
    """1 - SSE / sum(y^2) over the test transitions; pred_fn(k, Y) -> (T_k, n) one-step predictions
    (row t+1 predicts frame t+1)."""
    tr, start = split(segs)
    test = tr[start:]
    preds = {}
    sse = ss = 0.0
    for k, t in test:
        if k not in preds:
            preds[k] = pred_fn(k, segs[k])
        y = segs[k][t + 1]
        e = y - preds[k][t + 1]
        sse += float(e @ e); ss += float(y @ y)
    return 1.0 - sse / ss


def job_fn(job):
    segs = _G['flies'][job['fly']]
    wir = _G['wir']
    tr_segs = train_segments(segs, job['m'])
    t0 = time.perf_counter()
    res = em.fit(tr_segs, None, wir.a0(job['eff']), wir.kappa(job['eff']), max_iter=job['iters'])
    p = res.params
    r2 = test_r2(segs, lambda k, Y: K.kalman_filter(Y, None, p.A, p.Q, p.r, p.Sigma0, steady_state=True).x_pred)
    return {'key': job['key'], 'R2': r2, 'n_iter': res.n_iter, 'converged': res.converged, 'flags': res.flags,
            'rho_A': res.diagnostics['rho_A'], 'fit_s': time.perf_counter() - t0}


def ridge_b3m(segs, m, n):
    pieces = train_segments(segs, m)
    X0 = np.concatenate([Y[:-1] for Y in pieces]); X1 = np.concatenate([Y[1:] for Y in pieces])
    A = np.linalg.solve(X0.T @ X0 + RIDGE * np.eye(n), X0.T @ X1).T

    def pred(k, Y):
        out = np.zeros_like(Y); out[1:] = Y[:-1] @ A.T
        return out
    return test_r2(segs, pred)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_processed/fly_ito36.npz')
    ap.add_argument('--centroids', default='data_processed/centroids_ito36.npz')
    ap.add_argument('--matrix', default='Connectivity')
    ap.add_argument('--protocol', default='v3', choices=['v1', 'v2', 'v3'])
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--cache', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--report_only', action='store_true')
    ap.add_argument('--no_hash_check', action='store_true', help='never for a real run')
    cfg = ap.parse_args()
    tag = f"r2_tierB_{cfg.protocol}" + ('_smoke' if cfg.smoke else '')
    cfg.cache = cfg.cache or f"results/{tag}/fits.jsonl"
    cfg.out = cfg.out or f"results/{tag}.json"

    if not cfg.no_hash_check:
        check_hashes(required=[cfg.npz.replace('\\', '/'), cfg.centroids.replace('\\', '/')])
    seeds = ensure_seeds()
    tier = tier_settings(seeds, 'B', cfg.protocol)
    if 'R2' not in tier['regimes']:
        raise RuntimeError("tier B settings do not include R2 (seeds.json reductions)")
    flies, names, S, _ = K.load_standardised(cfg.npz)
    order = sorted(flies)
    n = len(names)
    Wt = P.wiring(S[cfg.matrix].values)
    D, _ = load_distances(cfg.centroids, names)
    shuffle_ids = list(tier['shuffles'])
    ms, iters = list(M_VALUES), tier['final_em_iters_max']
    if cfg.smoke:
        order, ms, shuffle_ids, iters = order[:4], [30, 250], shuffle_ids[:2], 3
    wir = T.Wirings(Wt, P.distance_wiring(D), [(s, seeds['shuffles'][s]) for s in shuffle_ids])
    table, dropped = T.model_table(wir, shuffle_ids)
    if cfg.smoke:
        table = {m_: v[-2:] for m_, v in table.items()}
    effs = sorted({k for m_ in table for _, ks in table[m_] for k in ks})
    avail = {(f, m): available(flies[f], m) for f in order for m in ms}
    missing = [f"{f} m={m}" for (f, m), ok in avail.items() if not ok]
    print(f"R2 tier B, protocol {cfg.protocol}: {len(order)} flies, m = {ms}, test = last {int(TEST_FRAC * 100)}% of "
          f"transitions; models " + ", ".join(f"{m_} {len(table[m_])}" for m_ in T.MODELS)
          + f" configs ({len(effs)} distinct fits per fly and m); shuffles {len(shuffle_ids)}; EM <= {iters} it; "
          f"workers {cfg.workers}; unscored (m reaches the test part): {missing or 'none'}"
          + ('; SMOKE (not a result)' if cfg.smoke else ''), flush=True)

    B3m = {m: {f: ridge_b3m(flies[f], m, n) for f in order if avail[(f, m)]} for m in ms}
    cache = T.Cache(cfg.cache)
    payload = {'flies': flies, 'wir': wir}
    pool = None
    if not cfg.report_only and cfg.workers > 1:
        pool = ProcessPoolExecutor(cfg.workers, initializer=_init, initargs=(payload,))
    else:
        _init(payload)
    t_run = time.perf_counter()
    try:
        # largest m first: those fits take longest, so the pool stays busy at the end
        jobs = [{'key': f"{f}|{m}|{e}", 'fly': f, 'm': m, 'eff': e, 'iters': iters}
                for m in sorted(ms, reverse=True) for f in order if avail[(f, m)] for e in effs]
        if not cfg.report_only:
            T.run_jobs(pool, cache, jobs, job_fn, 'fits')
    finally:
        if pool is not None:
            pool.shutdown()
    wall = time.perf_counter() - t_run

    # ---- per fly, model and config: R^2 (M4 = mean over shuffles) ----
    def val(f, m, keys):
        rs = [cache.d.get(f"{f}|{m}|{k}") for k in keys]
        return float(np.mean([r['R2'] for r in rs])) if all(r is not None for r in rs) else np.nan
    score, chosen, complete = {}, {}, True
    for m in ms:
        fl = [f for f in order if avail[(f, m)]]
        for mod in T.MODELS:
            V = np.array([[val(f, m, ks) for _, ks in table[mod]] for f in fl])      # (flies, configs)
            if np.isnan(V).any():
                complete = False
            score[(m, mod)] = {}
            for i, f in enumerate(fl):
                others = np.delete(V, i, axis=0)
                ci = int(np.nanargmax(np.nanmean(others, axis=0))) if np.isfinite(others).any() else 0
                chosen[(m, mod, f)] = table[mod][ci][0]
                score[(m, mod)][f] = float(V[i, ci])

    print(f"\nR2 learning curve ({'complete' if complete else 'INCOMPLETE: rerun to resume'}), median one-step R^2 over flies:")
    print("      m  " + "  ".join(f"{x:>7s}" for x in T.MODELS + ('B3m',)) + "   n")
    for m in ms:
        print(f"   {m:4d}  " + "  ".join(f"{np.median(list(score[(m, x)].values())):7.4f}" for x in T.MODELS)
              + f"  {np.median(list(B3m[m].values())):7.4f}   {len(B3m[m])}")
    tests = {}
    if complete and len(order) >= 6:
        print("\nSecondary tests per m (uncorrected; H4 = M3 - M0):")
        for m in ms:
            fl = sorted(score[(m, 'M3')])
            tests[m] = {}
            for b in ('M0', 'M1', 'M2', 'M4', 'M5'):
                s = T.compare([score[(m, 'M3')][f] for f in fl], [score[(m, b)][f] for f in fl], 'R2')
                tests[m][f'M3-{b}'] = s
                print(f"   m = {m:4d}  M3-{b}: {T.fmt(s, 'R2')}")
    recs = [r for r in cache.d.values()]
    if recs:
        rho = np.array([r['rho_A'] for r in recs])
        print(f"\n   fits {len(recs)}: {sum(not r['converged'] for r in recs)} hit the {iters}-iteration cap; "
              f"rho(A) max {rho.max():.3f} ({int(np.sum(rho >= 1))} >= 1, shrunk)")
    print(f"wall time this invocation {wall / 60:.1f} min")

    os.makedirs(os.path.dirname(cfg.out) or '.', exist_ok=True)
    with open(cfg.out, 'w') as f:
        json.dump({'config': vars(cfg), 'tier': tier, 'complete': complete, 'm_values': ms, 'unscored': missing,
                   'configs': {x: [h for h, _ in table[x]] for x in T.MODELS}, 'dropped_configs': dropped,
                   'scores': {f"{m}|{x}": v for (m, x), v in score.items()},
                   'chosen': {f"{m}|{x}|{fl}": c for (m, x, fl), c in chosen.items()},
                   'baseline_B3m': {str(m): v for m, v in B3m.items()}, 'tests': {str(m): v for m, v in tests.items()},
                   'diagnostics_summary': {'n_fits': len(recs), 'n_capped': sum(not r['converged'] for r in recs)}},
                  f, indent=1, default=str)
    print(f"results -> {cfg.out}")


if __name__ == '__main__':
    main()
