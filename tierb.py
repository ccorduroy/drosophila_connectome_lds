"""
Shared pieces for the tier B harnesses R1 (run_r1_prod.py) and R2 (run_r2.py)
(Protocol and Model Spec, sections 6, 10, 11 and 15).

Models M0-M5 on the coarse grid of section 15 (prior.COARSE_GRID: kappa 0, 0.01, 1; d 0, 0.5;
a_u 0, 0.25; a_w 0, 1), every variant with the same budget. M6 is cut (decision 6). In R1 and
R2 every region is observed during fitting, so:
  * the initialisation does not depend on A0 (em.init_params only uses A0 for hidden entries),
    and a fit is fully determined by (kappa, A0). Configurations that give the same (kappa, A0)
    are fitted once and shared ("effective key"): kappa = 0 is the flat fit for every model; a_w = 0
    reduces M3, M4 and M5 to the M2 prior with the same (d, a_u); M0 with kappa > 0 is M2 at
    d = a_u = 0. Each model is still scored and tuned on all of its own configurations.
  * the protocol-v2 prior on Q is not used (it acts on hidden blocks, and there are none in
    training; same choice as simulate.py for R2), and the kappa >= 0.01 amendment is R3-only.
M4 uses every shuffle of the tier (seeds.json through integrity.tier_settings), is tuned on the
mean over exactly those shuffles and scored as the mean over them (section 11).
"""
import json
import os

import numpy as np

import prior as P

MODELS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5')
SESOI = {'r': 0.02, 'R2': 0.01}


def fz(x, axis=0):
    return np.tanh(np.mean(np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999)), axis=axis))


# --------------------------------------------------------------------------------------
# Configurations and effective keys
# --------------------------------------------------------------------------------------
def coarse_configs(model, grid=P.COARSE_GRID):
    """All coarse-grid settings of a model's free hyperparameters (others at 0), d + a_u <= 0.9."""
    import itertools
    names = P.FREE[model]
    out = []
    for vals in itertools.product(*(grid[k] for k in names)):
        h = {'kappa': 0.0, 'd': 0.0, 'a_u': 0.0, 'a_w': 0.0, **{k: float(v) for k, v in zip(names, vals)}}
        if h['d'] + h['a_u'] <= 0.9:
            out.append(h)
    return out


def eff_key(model, h, shuffle=None):
    """Identifier of the fit a (model, config, shuffle) needs when every region is observed."""
    k, d, au, aw = h['kappa'], h['d'], h['a_u'], h['a_w']
    if k == 0:
        return 'flat'
    if model in ('M0', 'M1', 'M2') or aw == 0:
        return f"U|k{k:g}|d{d:g}|au{au:g}"
    w = {'M3': 'conn', 'M5': 'dist', 'M4': f"shuf{shuffle}"}[model]
    return f"{w}|k{k:g}|d{d:g}|au{au:g}|aw{aw:g}"


class Wirings:
    def __init__(self, Wt, Wd, shuffles):
        self.Wt, self.Wd, self.shuffles = Wt, Wd, {s: p for s, p in shuffles}   # shuffle id -> permutation
        self.n = Wt.shape[0]

    def a0(self, key):
        """A0 for an effective key."""
        n = self.n
        if key == 'flat':
            return np.zeros((n, n))
        parts = key.split('|'); kind = parts[0]
        v = {p[:2] if p[:2] in ('aw', 'au') else p[0]: float(p[2:] if p[:2] in ('aw', 'au') else p[1:]) for p in parts[1:]}
        d, au, aw = v['d'], v['au'], v.get('aw', 0.0)
        if kind == 'U':
            return P.prior_mean('M2', n=n, d=d, a_u=au)
        if kind == 'conn':
            return P.prior_mean('M3', Wt=self.Wt, d=d, a_u=au, a_w=aw)
        if kind == 'dist':
            return P.prior_mean('M3', Wt=self.Wd, d=d, a_u=au, a_w=aw)
        if kind.startswith('shuf'):
            return P.prior_mean('M4', Wt=self.Wt, perm=self.shuffles[int(kind[4:])], d=d, a_u=au, a_w=aw)
        raise ValueError(key)

    @staticmethod
    def kappa(key):
        return 0.0 if key == 'flat' else float(key.split('|')[1][1:])


def model_table(wir: Wirings, shuffle_ids):
    """{model: [(config, [effective keys])]} with configurations whose A0 has rho >= 0.98 dropped
    (section 6; for M4, if any shuffle fails). Returns (table, dropped)."""
    table, dropped = {}, {}
    for m in MODELS:
        keep, drop = [], []
        for h in coarse_configs(m):
            keys = [eff_key(m, h, s) for s in shuffle_ids] if m == 'M4' else [eff_key(m, h)]
            ok = all(P.spectral_radius(wir.a0(k)) < 0.98 for k in keys)
            (keep if ok else drop).append((h, keys) if ok else h)
        table[m], dropped[m] = keep, drop
    return table, dropped


# --------------------------------------------------------------------------------------
# Checkpoint cache (one JSON line per finished job), as in run_r3.py
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


def run_jobs(pool, cache, jobs, fn, label):
    import time
    from concurrent.futures import as_completed
    todo = [j for j in jobs if j['key'] not in cache.d]
    print(f"  {label}: {len(jobs)} fits, {len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    t0 = time.perf_counter(); done = 0
    it = map(fn, todo) if pool is None else (f.result() for f in as_completed([pool.submit(fn, j) for j in todo]))
    for r in it:
        cache.add(r); done += 1
        if done % max(1, len(todo) // 20) == 0 or done == len(todo):
            el = time.perf_counter() - t0
            print(f"    {done}/{len(todo)} done, {el / 60:.1f} min elapsed, ~{el / done * (len(todo) - done) / 60:.0f} min left",
                  flush=True)


# --------------------------------------------------------------------------------------
# Statistics (section 11; same functions as the simulation checks)
# --------------------------------------------------------------------------------------
def compare(a, b, metric):
    """Paired a - b over flies with the spec decision rule (uncorrected: R1 and R2 are secondary)."""
    from simulate import paired
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    s = paired(a[ok], b[ok])
    s['n'] = int(ok.sum())
    s['meets_rule'] = bool(s['p'] < 0.05 and s['median'] >= SESOI[metric] and s['ci'][0] > 0)
    return s


def cluster_ci(a, b, clusters, n_boot=10_000, seed=0):
    """Cluster bootstrap over recording days of the median paired difference (R1 sensitivity run)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    cl = np.asarray(clusters); u = np.unique(cl)
    groups = [d[cl == c] for c in u]
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(u), len(u))])) for _ in range(n_boot)]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return {'median': float(np.median(d)), 'ci_days': [float(lo), float(hi)], 'n_days': int(len(u))}


def fmt(s, metric):
    return (f"median {s['median']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}], p {s['p']:.2g}, n {s['n']}"
            + ('  (meets rule)' if s['meets_rule'] else ''))
