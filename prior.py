"""
Connectome prior construction (Protocol and Model Spec, sections 6 and 7).

Recipe for one structural matrix S (36 x 36, row/column order = fly_ito36.npz 'regions'):
    M  = (S + S^T) / 2                         symmetrise
    W  = log10(1 + M), diagonal set to 0       compress
    W~ = W / row sums                          row-normalise (every row sums to 1)
    U  = (11^T - I) / (n - 1)                  uniform reference, row sums 1
    A0 = d I + a_u U + a_w (W~ - U)            prior mean; row sums d + a_u

Models (prior mean A0; free hyperparameters):
    M0 flat        0                              kappa
    M1 persistence d I                            kappa, d
    M2 uniform     d I + a_u U                    kappa, d, a_u
    M3 connectome  d I + a_u U + a_w (W~ - U)     kappa, d, a_u, a_w
    M4 shuffled    as M3 with W~[pi][:, pi]       same as M3 (pi from seeds.json, applied after row normalisation)
    M5 distance    as M3 with W = exp(-D / l)     same as M3 (D = centroid distances, l = median off-diagonal)
    M6 FC ceiling  cut (spec decision 6)

Prior distribution: A | Q ~ MN(A0, Q, (lambda I)^-1), lambda = kappa * N (N = transitions in the fit).
"""
import itertools

import numpy as np

MODELS = ('M0', 'M1', 'M2', 'M3', 'M4', 'M5')

# Hyperparameter grid, identical for every variant (spec section 6)
GRID = {'kappa': (0.0, 0.001, 0.01, 0.1, 1.0),
        'd': (0.0, 0.25, 0.5, 0.75),
        'a_u': (0.0, 0.25, 0.5),
        'a_w': (0.0, 0.5, 1.0, 2.0)}
# Coarse grid for R1 and R2 (spec section 15)
COARSE_GRID = {'kappa': (0.0, 0.01, 1.0), 'd': (0.0, 0.5), 'a_u': (0.0, 0.25), 'a_w': (0.0, 1.0)}
FREE = {'M0': ('kappa',), 'M1': ('kappa', 'd'), 'M2': ('kappa', 'd', 'a_u'),
        'M3': ('kappa', 'd', 'a_u', 'a_w'), 'M4': ('kappa', 'd', 'a_u', 'a_w'),
        'M5': ('kappa', 'd', 'a_u', 'a_w')}


def _row_normalise(W):
    W = np.array(W, dtype=np.float64)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(1)
    if np.any(rs <= 0):
        raise ValueError(f"rows with no off-diagonal weight: {np.flatnonzero(rs <= 0).tolist()}")
    return W / rs[:, None]


def wiring(S):
    """W~ from a structural matrix S (symmetrise, log10(1+.), zero diagonal, row-normalise)."""
    S = np.asarray(S, np.float64)
    if S.shape[0] != S.shape[1] or np.any(S < 0):
        raise ValueError("S must be square and non-negative")
    M = 0.5 * (S + S.T)
    return _row_normalise(np.log10(1.0 + M))


def distance_wiring(D):
    """W~ for M5 from a centroid distance matrix D: exp(-D / l), l = median off-diagonal distance."""
    D = np.asarray(D, np.float64)
    off = D[~np.eye(len(D), dtype=bool)]
    return _row_normalise(np.exp(-D / np.median(off)))


def uniform(n):
    return (np.ones((n, n)) - np.eye(n)) / (n - 1)


def permute(Wt, perm):
    """Shuffled control: the same relabelling on rows and columns of the row-normalised W~."""
    p = np.asarray(perm)
    return Wt[np.ix_(p, p)]


def prior_mean(model, n=None, Wt=None, d=0.0, a_u=0.0, a_w=0.0, perm=None):
    """A0 for a model. Wt: row-normalised wiring (required for M3-M5); perm: for M4."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model}")
    if n is None:
        n = Wt.shape[0]
    I = np.eye(n)
    if model == 'M0':
        return np.zeros((n, n))
    if model == 'M1':
        return d * I
    U = uniform(n)
    if model == 'M2':
        return d * I + a_u * U
    if Wt is None:
        raise ValueError(f"{model} needs Wt")
    if model == 'M4':
        if perm is None:
            raise ValueError("M4 needs a permutation (seeds.json 'shuffles')")
        Wt = permute(Wt, perm)
    return d * I + a_u * U + a_w * (Wt - U)


def spectral_radius(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def frac_negative_offdiag(A0):
    off = ~np.eye(A0.shape[0], dtype=bool)
    return float(np.mean(A0[off] < 0))


def admissible(model, Wt, h, perm=None):
    """Spec section 6 discard rule: d + a_u > 0.9 or rho(A0) >= 0.98 is discarded."""
    if h.get('d', 0.0) + h.get('a_u', 0.0) > 0.9:
        return False
    A0 = prior_mean(model, n=None if Wt is not None else 36, Wt=Wt, d=h.get('d', 0.0),
                    a_u=h.get('a_u', 0.0), a_w=h.get('a_w', 0.0), perm=perm)
    return spectral_radius(A0) < 0.98


def grid_points(model, Wt=None, grid=GRID, perm=None):
    """All admissible hyperparameter settings for a model (non-free ones held at 0).
    For M4 pass any one permutation for the radius check, or None to skip that check (the
    spec tunes M4 on a permutation average; check every permutation in the harness)."""
    names = FREE[model]
    pts = []
    for vals in itertools.product(*(grid[k] for k in names)):
        h = {'kappa': 0.0, 'd': 0.0, 'a_u': 0.0, 'a_w': 0.0, **dict(zip(names, vals))}
        if model == 'M4' and perm is None:
            if h['d'] + h['a_u'] > 0.9:
                continue
        elif not admissible(model, Wt, h, perm):
            continue
        pts.append(h)
    return pts


def sample_configs(model, k, rng, Wt=None, grid=GRID, perm=None):
    """Random search (spec section 10, R3): k distinct admissible grid points."""
    pts = grid_points(model, Wt, grid, perm)
    idx = rng.choice(len(pts), size=min(k, len(pts)), replace=False)
    return [pts[i] for i in sorted(idx)]


def q_target(base, vis, W=None, structured_frac=0.5, floor=1e-6):
    """Scale matrix (prior mean) for the connectome prior on Q, and the same rule for Sigma0
    (spec section 12, tier B amendment 1, approved Nov 26).

    base : n x n matrix whose visible block is kept (the M0 warm start's Q or Sigma0; it never
           contains hidden-region data). vis : (n,) bool. W : the model's wiring (uniform for M2,
           real W~ for M3, shuffled W~ for M4, distance wiring for M5) or None for M0/M1.
    Hidden blocks: G = W[hidden, visible] row-normalised; Q_hv = g G Q_vv; Q_hh = g^2 G Q_vv G^T +
    (1 - f) qbar I, with g chosen so the structured part carries a fraction f (= 0.5) of the mean
    visible variance qbar. W = None: Q_hv = 0, Q_hh = qbar I (hidden regions decoupled)."""
    base = 0.5 * (np.asarray(base, np.float64) + np.asarray(base, np.float64).T)
    vis = np.asarray(vis, bool)
    v, h = np.flatnonzero(vis), np.flatnonzero(~vis)
    T = base.copy()
    if h.size == 0:
        return T
    Bvv = base[np.ix_(v, v)]
    qbar = float(np.mean(np.diag(Bvv)))
    T[h, :] = 0.0; T[:, h] = 0.0
    if W is None:
        T[np.ix_(h, h)] = qbar * np.eye(len(h))
    else:
        G = np.asarray(W, np.float64)[np.ix_(h, v)]
        rs = G.sum(1, keepdims=True)
        if np.any(rs <= 0):
            raise ValueError("a hidden region has no wiring weight onto visible regions")
        G = G / rs
        C = G @ Bvv @ G.T
        g = np.sqrt(structured_frac * qbar / np.mean(np.diag(C)))
        T[np.ix_(h, v)] = g * G @ Bvv
        T[np.ix_(v, h)] = T[np.ix_(h, v)].T
        T[np.ix_(h, h)] = g * g * C + (1.0 - structured_frac) * qbar * np.eye(len(h))
    w, V = np.linalg.eigh(0.5 * (T + T.T))
    return (V * np.maximum(w, floor)) @ V.T


def wiring_for(model, Wt=None, perm=None, Wdist=None, n=None):
    """The wiring matrix a model's Q target uses (None = decoupled)."""
    if model in ('M0', 'M1'):
        return None
    if model == 'M2':
        return uniform(n if n is not None else Wt.shape[0])
    if model == 'M3':
        return Wt
    if model == 'M4':
        return permute(Wt, perm)
    if model == 'M5':
        return Wdist
    raise ValueError(model)
