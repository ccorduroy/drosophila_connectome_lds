"""
Run integrity for the connectome-prior LDS project (Protocol and Model Spec, sections 3, 10, 11, 15).

  ensure_seeds()   seeds.json: written once from fixed rules, verified on every later run
  check_hashes()   hashes.sha256: SHA-256 of every frozen file, asserted at the start of every run
  sim_seed(k, r)   seed for simulation Sk, replicate r  (rule recorded in seeds.json)

hashes.sha256 is the sha256sum format ("<hex>  <path>", paths relative to the file's folder,
forward slashes), as written by the PowerShell or sha256sum commands.
"""
import copy
import hashlib
import json
import os
import time

import numpy as np


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
    folds = {key: [validation_folds(sets[j], key, j) for j in range(len(sets))] for key, sets in hidden.items()}
    search = random_search_configs()
    search_v2 = random_search_configs(kappas=(0.01, 0.1, 1.0), stream=401)
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
            'random_search': 'R3 tuning configurations (spec section 10: 16 per variant): M0 uses all 5 kappa values; for M1, M2 and M3 (M3 shared by M4 and M5), the grid of section 6 restricted to d + a_u <= 0.9, in itertools.product order of (kappa, d, a_u, a_w), sampled without replacement by default_rng([675, 400, k]).choice(len(grid), 16), k = 1, 2, 3; configurations with rho(A0) >= 0.98 for the real wiring or any shuffle are dropped at run time and reported (adopted Nov 24)',
            'random_search_v2': 'protocol v2 (tier B amendment 2, approved Nov 26): as random_search but with kappa restricted to (0.01, 0.1, 1) for M1-M4 and default_rng([675, 401, k]); M0 unchanged',
            'protocol_v2': 'tier B amendments approved Nov 26 after the tier A run (spec section 12): connectome inverse-Wishart prior on Q with kappa_Q = 1 (prior.q_target, structured fraction 0.5); random_search_v2; final-fit cap 300; hidden-set summary, B6 check before H2 and H1 against B8 reported. Tier A results (protocol v1) stay reported as run',
            'validation_folds': 'hidden set j of fraction p: visible = sorted complement; order = default_rng([675, 300, int(100*p), j]).permutation(len(visible)); folds = array_split(visible[order], 3), each sorted. R3 tuning hides each fold in turn and scores the mean over the 3 (spec sections 10, 11; adopted Nov 23)',
        },
        'hidden_sets': hidden,
        'shuffles': shuffles,
        'validation_folds': folds,
        'random_search': search,
        'random_search_v2': search_v2,
        'protocol_v2': {'q_prior': True, 'kappa_Q': 1.0, 'q_structured_frac': 0.5,
                        'random_search': 'random_search_v2', 'final_em_iters_max': 300,
                        'approved': '2026-11-26'},
        'tiers': {
            'A': {'hidden_sets': [0, 1, 2, 3, 4], 'shuffles': list(range(10)),
                  'models': ['M0', 'M1', 'M2', 'M3', 'M4'], 'regimes': ['R3'],
                  'random_search_configs': 16, 'tuning_em_iters': 30, 'final_em_iters_max': 150},
            'B': {'hidden_sets': list(range(10)), 'shuffles': list(range(20)),
                  'models': ['M0', 'M1', 'M2', 'M3', 'M4', 'M5'], 'regimes': ['R3', 'R1', 'R2'],
                  'random_search_configs': 16, 'tuning_em_iters': 30, 'final_em_iters_max': 150},
        },
        'validation_fallback': 'if tier A runs short: folds 0 and 1 only, declared in reductions before any run',
        'reductions': [],   # any pre-declared cut (spec section 15 cut lines) is appended here BEFORE results are seen
    }


SEARCH_GRID = {'kappa': (0.0, 0.001, 0.01, 0.1, 1.0), 'd': (0.0, 0.25, 0.5, 0.75),
               'a_u': (0.0, 0.25, 0.5), 'a_w': (0.0, 0.5, 1.0, 2.0)}       # spec section 6 (= prior.GRID)
SEARCH_FREE = {'M1': ('kappa', 'd'), 'M2': ('kappa', 'd', 'a_u'), 'M3': ('kappa', 'd', 'a_u', 'a_w')}


def random_search_configs(k=16, kappas=None, stream=400):
    import itertools
    grid = dict(SEARCH_GRID)
    if kappas is not None:
        grid['kappa'] = tuple(kappas)
    out = {'M0': [{'kappa': v} for v in SEARCH_GRID['kappa']]}        # M0 keeps all 5 (amendment 2 is M1-M4)
    for idx, m in enumerate(('M1', 'M2', 'M3'), start=1):
        names = SEARCH_FREE[m]
        pts = [dict(zip(names, v)) for v in itertools.product(*(grid[x] for x in names))
               if v[names.index('d')] + (v[names.index('a_u')] if 'a_u' in names else 0.0) <= 0.9]
        pick = np.random.default_rng([GLOBAL_SEED, stream, idx]).choice(len(pts), min(k, len(pts)), replace=False)
        out[m] = [{x: float(y) for x, y in pts[i].items()} for i in sorted(int(j) for j in pick)]
    return out


def validation_folds(hidden_set, p_key, j, n=N_REGIONS, k=3):
    visible = np.setdiff1d(np.arange(n), hidden_set)
    order = np.random.default_rng([GLOBAL_SEED, 300, int(p_key), j]).permutation(len(visible))
    return [sorted(int(i) for i in f) for f in np.array_split(visible[order], k)]


META = ('created', 'reductions', 'amendments')


def ensure_seeds(path=SEEDS_PATH):
    """Write seeds.json if absent; otherwise verify it matches the fixed rules exactly
    (apart from 'created', 'reductions' and 'amendments'). Returns the seeds dict.
    A seeds.json committed before a rule was added fails here; run amend_seeds() once."""
    want = build_seeds()
    if not os.path.exists(path):
        out = copy.deepcopy(want); out['created'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1)
        print(f"seeds.json written to {os.path.abspath(path)} -- commit it before any further run")
        return out
    with open(path) as f:
        have = json.load(f)
    a = {k: v for k, v in have.items() if k not in META}
    b = {k: v for k, v in want.items() if k not in META}
    if a != b:
        hint = (" New seed rules are not in the file yet: run `python integrity.py --amend` once, "
                "then re-hash and commit." if _compatible(a, b) else
                " Do not overwrite it: restore the committed copy.")
        raise RuntimeError(f"{path} does not match the frozen seed rules.{hint}")
    return have


def _compatible(have, want):
    """True if `want` only adds keys (top level, or inside dicts) to `have`."""
    for k, v in have.items():
        if k not in want:
            return False
        if isinstance(v, dict) and isinstance(want[k], dict):
            if any(x not in want[k] or want[k][x] != y for x, y in v.items()):
                return False
        elif v != want[k]:
            return False
    return True


def amend_seeds(reason, path=SEEDS_PATH):
    """Add entries for newly adopted rules to an existing seeds.json, keeping every existing
    draw, and record the amendment. Refuses if any existing entry would change (a changed
    draw is not an amendment)."""
    want = build_seeds()
    with open(path) as f:
        have = json.load(f)
    if not _compatible({k: v for k, v in have.items() if k not in META},
                       {k: v for k, v in want.items() if k not in META}):
        raise RuntimeError("refusing to amend: an existing draw or rule would change (that is not an amendment)")
    added = [k for k in want if k not in META and (k not in have or have[k] != want[k])]
    if not added:
        print("seeds.json already up to date"); return have
    for k in added:
        have[k] = want[k]
    have.setdefault('amendments', []).append({'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                              'added_or_extended': added, 'reason': reason})
    with open(path, 'w') as f:
        json.dump(have, f, indent=1)
    print(f"seeds.json amended ({', '.join(added)}). Re-hash it and commit seeds.json + hashes.sha256 "
          "before any further run.")
    return have


def tier_settings(seeds, tier, protocol='v1'):
    """Effective settings of tier 'A' or 'B': seeds['tiers'][tier] with every recorded
    reduction for that tier applied in order, then (protocol 'v2') the Nov 26 amendments.
    Harness code reads tiers only through this."""
    t = copy.deepcopy(seeds['tiers'][tier])
    for red in seeds.get('reductions', []):
        if red.get('tier') == tier:
            t[red['setting']] = red['value']
    t.update({'q_prior': False, 'kappa_Q': 0.0, 'q_structured_frac': 0.5, 'random_search': 'random_search'})
    if protocol == 'v2':
        if 'protocol_v2' not in seeds:
            raise RuntimeError("seeds.json has no protocol_v2 entry: run `python integrity.py --amend`, re-hash and commit")
        t.update({k: v for k, v in seeds['protocol_v2'].items() if k != 'approved'})
    elif protocol != 'v1':
        raise ValueError(protocol)
    t['protocol'] = protocol
    return t


def add_reduction(tier, setting, value, reason, path=SEEDS_PATH):
    """Append a pre-declared reduction (spec section 15) to seeds.json. Existing draws and
    rules are untouched; ensure_seeds() ignores 'reductions', so verification still passes.
    Re-hash seeds.json and commit it before any further run."""
    seeds = ensure_seeds(path)
    if setting not in seeds['tiers'][tier]:
        raise ValueError(f"tier {tier} has no setting {setting!r}")
    for red in seeds.get('reductions', []):
        if red.get('tier') == tier and red['setting'] == setting and red['value'] == value:
            print("reduction already recorded"); return seeds
    seeds.setdefault('reductions', []).append({
        'date': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'tier': tier, 'setting': setting,
        'from': tier_settings(seeds, tier)[setting], 'value': value, 'reason': reason})
    with open(path, 'w') as f:
        json.dump(seeds, f, indent=1)
    print(f"reduction recorded: tier {tier} {setting} -> {value}. Re-hash seeds.json and commit it "
          "with hashes.sha256 before any further run.")
    return seeds


def sim_seed(sim_no, replicate=0):
    """Simulation Sk, replicate r -> [675, 100 + k, r] (rules['simulations'] in seeds.json)."""
    return [GLOBAL_SEED, 100 + sim_no, replicate]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def check_hashes(path=os.environ.get('HASHES_FILE', 'hashes.sha256'), required=()):
    """Assert every file listed in hashes.sha256 still has its recorded SHA-256.
    required: paths (as written in the file) that must be listed, e.g. the npz a run reads.
    Raises RuntimeError on a missing hash file, a missing or changed file, or a required
    file that is not listed. Returns {path: hex}."""
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found: create and commit it before any run (spec section 3)")
    root = os.path.dirname(os.path.abspath(path))
    listed, bad = {}, []
    with open(path, encoding='ascii') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hx, rel = line.split(None, 1)
            rel = rel.lstrip('*').replace('\\', '/')
            listed[rel] = hx.lower()
            full = os.path.join(root, rel)
            if not os.path.exists(full):
                bad.append(f"missing: {rel}")
            elif sha256(full) != hx.lower():
                bad.append(f"changed: {rel}")
    for r in required:
        if r.replace('\\', '/') not in listed:
            bad.append(f"not listed in {path}: {r}")
    if bad:
        raise RuntimeError("frozen-file check failed:\n  " + "\n  ".join(bad))
    return listed


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='seeds.json and hash checks')
    ap.add_argument('--amend', action='store_true', help='add newly adopted seed rules to an existing seeds.json')
    ap.add_argument('--reason', default='protocol v2: tier B amendments approved Nov 26 (Q prior, kappa >= 0.01 grid, cap 300; spec section 12)')
    ap.add_argument('--check', action='store_true', help='verify seeds.json and hashes.sha256')
    ap.add_argument('--tier_a_5_shuffles', action='store_true',
                    help='record the Nov 23 reduction: tier A uses shuffles 0-4, M4 tuned and scored on the same 5')
    a = ap.parse_args()
    if a.amend:
        amend_seeds(a.reason)
    if a.tier_a_5_shuffles:
        add_reduction('A', 'shuffles', [0, 1, 2, 3, 4],
                      'Nov 23: M4 is tuned on the mean over exactly the shuffles it is scored on (S2 '
                      'diagnosis); tuning on all 10 would cost about 10x M3, so tier A uses shuffles 0-4, '
                      'tuned and scored on the same 5 (spec sections 6, 11, 15)')
    if a.check or not (a.amend or a.tier_a_5_shuffles):
        ensure_seeds(); print('seeds.json OK')
        check_hashes(); print('hashes.sha256 OK')
