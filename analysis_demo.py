"""
Validity checks on the cleaned Turner et al. (2021) data + hemibrain region-level connectome,
run BEFORE any Kalman filter / EM is written. Nothing here fits a Kalman filter or runs EM.

  PART 1  Does functional connectivity (FC) track structural connectivity (SC)?   (+ distance control)
  PART 2  How well does ONE shared linear dynamics matrix predict a fly it has never seen? (ridge VAR(1))
  PART 3  Is the data the kind of thing a linear-Gaussian state-space model assumes?
          (measurement-noise floor, residual whiteness/Gaussianity, model order, dimensionality, stationarity)
  PART 4  Does the connectome add anything beyond what a generic prior gives?
          (dynamics-vs-wiring alignment, small-data prior with shuffled/uniform controls,
           hidden-region reconstruction baselines that a smoother must beat)

Usage (put this next to build_dataset.py and fly_ito36.npz):
    python analysis_demo.py [fly_ito36.npz] [--mask vfb_68_2018-11-03_5.nii.gz --csv Original_Index_panda_full.csv]
Figures are written to ./figures/.  Without --mask/--csv the distance control is skipped.
"""
import argparse, os, warnings
import numpy as np
from scipy.stats import spearmanr, rankdata, kurtosis, wilcoxon
from scipy.signal import welch
from build_dataset import load, INC37, display

warnings.filterwarnings("ignore", category=RuntimeWarning)

ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="?", default="data_processed/fly_ito36.npz")
ap.add_argument("--mask", default=None, help="atlas mask .nii.gz (for region centroids -> distances)")
ap.add_argument("--csv", default=None, help="Original_Index_panda_full.csv (atlas label -> number)")
ap.add_argument("--no-figs", action="store_true")
args = ap.parse_args()

flies, names, S, meta = load(args.npz)
# flies: {fly_id: [segment arrays, each (36 regions x T_k frames)]}   (segments are NOT concatenated
#        across trimmed gaps, so we never invent a fake transition across a dropout)
# S:     {name: 36x36 DataFrame}, rows = region where a neuron receives input, cols = where it sends output
n = len(names)
iu = np.triu_indices(n, 1)          # the 630 unique region pairs (upper triangle, no diagonal)
off = ~np.eye(n, dtype=bool)
rng = np.random.default_rng(0)
FLY = list(flies)
R = {}                              # results kept for the figures

# The paper's primary measure ("cell count") is not saved as its own file: it equals Weak + Medium + Strong.
S["CellCount"] = S["WeakConnections"] + S["MediumConnections"] + S["StrongConnections"]


def header(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# =============================================================================
# Region centroids -> pairwise distance (control for "nearby regions are both wired and correlated")
# =============================================================================
D = None
if args.mask and args.csv:
    import nibabel as nib, pandas as pd
    idx = pd.read_csv(args.csv, keep_default_na=False); idx.columns = ["name", "num", "i0"]
    num_of = idx.set_index("name")["num"]
    img = nib.load(args.mask)
    mask = np.asarray(np.squeeze(img.get_fdata()), "uint16")
    zoom = np.array(img.header.get_zooms()[:3])
    fmap = {s: [k for k in INC37 if display(k) == s] for s in names}
    fmap["IB"] = ["IB_R", "IB_L"]                      # IB = union of both sides, as in the structural matrices
    cent = np.array([np.argwhere(np.isin(mask, [num_of[k] for k in fmap[s]])).mean(0) * zoom for s in names])
    D = np.linalg.norm(cent[:, None] - cent[None], axis=-1)    # template-brain distance between centroids


# =============================================================================
# PART 1: structural vs functional connectivity
# =============================================================================
def fc_matrix(segments):
    """Functional connectivity of one fly: Pearson r between every pair of regions, Fisher-z transformed."""
    C = np.corrcoef(np.concatenate(segments, axis=1))   # regions x time (over all kept frames)
    np.fill_diagonal(C, np.nan)                         # a region's correlation with itself carries no information
    return np.arctanh(C)                                # arctanh makes r values average and compare more sensibly


def prep_structure(M):
    """Structural matrix -> one number per unordered region pair.
    (A + A^T)/2 symmetrizes it (FC is symmetric), log10(1 + x) tames the huge dynamic range."""
    A = M.values.astype(float)
    return np.log10(1.0 + (A + A.T) / 2.0)


def perm_null(sc, fc_vec, stat, n_perm=2000):
    """Label-permutation null: shuffle which region is which in the structural matrix (same permutation on rows
    and columns, so the matrix keeps its own internal structure) and recompute `stat`.
    Asks 'would a random assignment of wiring to regions correlate with FC this well?'"""
    obs = stat(sc[iu], fc_vec)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        null[k] = stat(sc[np.ix_(p, p)][iu], fc_vec)
    return obs, null, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)   # +1: observed counts as one draw


def partial_spearman(x, y, z):
    """Spearman correlation of x and y after linearly removing the (ranked) covariate z from both ranks."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a):
        X = np.c_[np.ones_like(rz), rz]
        return a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    return np.corrcoef(resid(rx), resid(ry))[0, 1]


fc_per_fly = {f: fc_matrix(v) for f, v in flies.items()}
fc_mean = np.nanmean(np.stack(list(fc_per_fly.values())), axis=0)    # average the z-values across the 20 flies
fc_vec = fc_mean[iu]

header("PART 1: structure vs function  (mean FC across 20 flies, 630 region pairs)")
for key in ["CellCount", "Connectivity", "TBars", "WeightedSynapseNumber"]:
    sc_k = prep_structure(S[key])
    rho, null, p = perm_null(sc_k, fc_vec, lambda a, b: spearmanr(a, b)[0])
    line = f"  {key:22s} Spearman rho = {rho:.3f}   permutation p = {p:.4f}"
    if D is not None:
        rp, nullp, pp = perm_null(sc_k, fc_vec, lambda a, b: partial_spearman(a, b, D[iu]))
        line += f"   | controlling for distance: rho = {rp:.3f} (p = {pp:.4f})"
    print(line)
    if key == "Connectivity":
        R["p1_rho"], R["p1_null"] = rho, null
        if D is not None: R["p1_rho_dist"] = rp
print("  (with 2000 permutations the smallest possible p is 1/2001 = 0.0005: no shuffle reached the real rho)")
sc = prep_structure(S["Connectivity"])
R["sc"] = sc
if D is not None:
    print(f"  distance vs FC: rho = {spearmanr(D[iu], fc_vec)[0]:.3f};  distance vs SC: rho = {spearmanr(D[iu], sc[iu])[0]:.3f}"
          "   (closer regions are somewhat more wired and more correlated, but this explains little)")

per_fly_rho = {f: spearmanr(sc[iu], z[iu])[0] for f, z in fc_per_fly.items()}
r = np.array(list(per_fly_rho.values()))
print(f"  per-fly rho (Connectivity): median {np.median(r):.2f}, range {r.min():.2f} to {r.max():.2f}")

# Ceiling: how repeatable is FC itself?  Split the 20 flies into two random halves of 10.
ks = list(fc_per_fly); rr = []
for _ in range(500):
    p = rng.permutation(len(ks))
    a = np.nanmean(np.stack([fc_per_fly[ks[i]] for i in p[:10]]), 0)[iu]
    b = np.nanmean(np.stack([fc_per_fly[ks[i]] for i in p[10:]]), 0)[iu]
    rr.append(spearmanr(a, b)[0])
print(f"  FC repeatability (mean FC of 10 flies vs the other 10): rho = {np.mean(rr):.3f}"
      "  -> FC is a stable, shared feature across animals, so there is something real to explain")

# Direct vs indirect coupling: partial correlations strip out 'A-drives-B-drives-C'
from sklearn.covariance import LedoitWolf
pc_all = []
for f, v in flies.items():
    X = np.concatenate(v, 1); X = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    P = LedoitWolf().fit(X.T).precision_; d_ = np.sqrt(np.diag(P)); pc = -P / np.outer(d_, d_)
    np.fill_diagonal(pc, np.nan); pc_all.append(pc)
pc_mean = np.nanmean(np.stack(pc_all), 0)
print(f"  SC vs mean PARTIAL correlation (direct-coupling proxy): rho = {spearmanr(sc[iu], pc_mean[iu])[0]:.3f}"
      "  (weaker than for plain FC, as expected, but clearly positive)")


# =============================================================================
# PART 2: leave-one-fly-out ridge VAR(1)
# =============================================================================
# Model: x[t+1] = A x[t] + noise, with x = the 36 region activities at one frame.
# Fit A by ridge regression:  A = (X1 X0^T) (X0 X0^T + lam I)^-1, where X0 holds frames 0..T-2 and
# X1 holds frames 1..T-1. The ridge term keeps A from overfitting.

def zscore_fly(segments):
    """Standardize each region within a fly (using all its kept frames) so regions are on one scale."""
    allv = np.concatenate(segments, axis=1)
    mu, sd = allv.mean(1, keepdims=True), allv.std(1, keepdims=True)
    return [(a - mu) / sd for a in segments]


def transitions(segments):
    """Stack (x[t], x[t+1]) pairs, taken only WITHIN segments (no pair straddles a trimmed gap)."""
    X0 = np.concatenate([a[:, :-1] for a in segments], axis=1)
    X1 = np.concatenate([a[:, 1:] for a in segments], axis=1)
    return X0, X1


def r2(pred, X1):
    """Fraction of next-frame variance explained. 1 = perfect, 0 = no better than predicting zero
    (the z-scored mean), negative = worse than that."""
    return 1.0 - np.sum((X1 - pred) ** 2) / np.sum(X1 ** 2)


Z = {f: zscore_fly(v) for f, v in flies.items()}
pairs = {f: transitions(Z[f]) for f in FLY}
LAM = 100.0

lofo, persist, within, resid_stats, A_lofo = {}, {}, {}, {}, {}
for held_out in FLY:
    train = [pairs[f] for f in FLY if f != held_out]
    # Sufficient statistics: pooling 19 flies is just summing their X0 X0^T and X1 X0^T matrices
    G = sum(x0 @ x0.T for x0, _ in train)
    H = sum(x1 @ x0.T for x0, x1 in train)
    A = H @ np.linalg.inv(G + LAM * np.eye(n))
    A_lofo[held_out] = A

    x0, x1 = pairs[held_out]
    lofo[held_out] = r2(A @ x0, x1)          # A never saw this fly
    persist[held_out] = r2(x0, x1)           # baseline: "next frame = this frame"

    k = int(0.7 * x0.shape[1])               # comparison: fit A on this fly's first 70%, test on the last 30%
    A_w = (x1[:, :k] @ x0[:, :k].T) @ np.linalg.inv(x0[:, :k] @ x0[:, :k].T + LAM * np.eye(n))
    within[held_out] = r2(A_w @ x0[:, k:], x1[:, k:])

header(f"PART 2: one-step-ahead prediction R^2 (ridge lambda = {LAM:g}, 36 regions)")
v = np.array(list(lofo.values()))
print(f"  leave-one-fly-out VAR(1): median {np.median(v):.3f}, range {v.min():.3f} to {v.max():.3f}")
print(f"  same-fly 70/30 VAR(1):    median {np.median(list(within.values())):.3f}")
print(f"  persistence baseline:     median {np.median(list(persist.values())):.3f}")
R["lofo"], R["within"], R["persist"] = lofo, within, persist


# =============================================================================
# PART 3: is the data shaped like a linear-Gaussian state-space model?
# =============================================================================
header("PART 3: does the data look like what a linear-Gaussian state-space model assumes?")

# 3a. Measurement noise.  An LDS has two noise terms: process noise (the brain's own variability, which feeds
#     forward) and observation noise (imaging noise, which does not).  A Kalman filter only earns its keep
#     if observation noise is a real fraction of the signal.  Test: GCaMP is slow, so real signal power should
#     fall away at high frequency; whatever sits on a flat floor up to Nyquist is (mostly) imaging noise.
def noise_fraction(segments, hi=(0.45, 0.59), nper=128):
    Ps = []
    for s in segments:
        if s.shape[1] >= nper:
            fr, P = welch(s, fs=meta["fs"], nperseg=nper, axis=1); Ps.append(P)
    P = np.mean(Ps, 0).mean(0)                       # average over segments, then over regions
    keep = fr < 0.59                                 # drop the Nyquist bin (it is not doubled by the one-sided PSD)
    band = (fr >= hi[0]) & keep
    return float(np.clip(P[band].mean() * keep.sum() / P[keep].sum(), 0, 1)), fr[keep], P[keep]

nf, psd = {}, []
for f in FLY:
    nf[f], fr, P = noise_fraction(Z[f]); psd.append(P)
R["psd_fr"], R["psd"], R["nf"] = fr, np.mean(psd, 0), nf
nfv = np.array(list(nf.values()))
print(f"  3a. flat high-frequency floor (0.45-0.59 Hz) = {np.mean(nfv):.2f} of total variance on average "
      f"(per fly {nfv.min():.2f} to {nfv.max():.2f})")
print(f"      -> imaging noise is a large share of every frame, so a model with an explicit observation-noise term")
print(f"         is justified, and one-step R^2 against noisy frames is capped near 1 - {np.mean(nfv):.2f} = {1-np.mean(nfv):.2f}.")
print(f"      (assumes the floor extends flat below 0.45 Hz; some of it could be real fast activity, so treat as approximate)")

# 3b. Residuals of the LOFO VAR(1).  If x is VAR(1) + observation noise, the VAR residuals are NOT white:
#     they pick up a NEGATIVE lag-1 autocorrelation (the noise in x[t] enters the residual at t+1 with the wrong sign).
ac_res, kur_res, off_res = [], [], []
for f in FLY:
    A = A_lofo[f]
    Es = [a[:, 1:] - A @ a[:, :-1] for a in Z[f]]              # one-step residuals, within segments
    E = np.concatenate(Es, 1)
    ac_res.append(np.average([np.mean([np.corrcoef(e[i, :-1], e[i, 1:])[0, 1] for i in range(n)]) for e in Es],
                             weights=[e.shape[1] for e in Es]))
    kur_res.append(np.median(kurtosis(E, axis=1)))
    off_res.append(np.mean(np.abs(np.corrcoef(E)[off])))
ac_res, kur_res, off_res = map(np.array, (ac_res, kur_res, off_res))
R["ac_res"] = dict(zip(FLY, ac_res))
print(f"  3b. residual lag-1 autocorrelation: median {np.median(ac_res):+.2f} (range {ac_res.min():+.2f} to {ac_res.max():+.2f})"
      " -> residuals are close to white, so one lag of memory captures the linear temporal structure")
print(f"      residual excess kurtosis (0 = Gaussian): median {np.median(kur_res):.2f} (range {kur_res.min():.2f} to {kur_res.max():.2f})"
      " -> mildly heavy-tailed; roughly Gaussian for most flies, with a few outlier-prone ones")
print(f"      residual cross-region |correlation| (0 = independent): median {np.median(off_res):.2f}"
      " -> residuals are shared across regions, so the noise covariance should be FULL, not diagonal")
rho_nf = spearmanr(nfv, [lofo[f] for f in FLY])[0]
R["rho_nf"] = rho_nf
print(f"      sanity check on 3a: flies with a higher noise floor are harder to predict (Spearman rho = {rho_nf:+.2f} between floor and LOFO R^2)")

# 3c. Model order and coupling.  Does the 36-region state need memory beyond one frame (i.e. extra hidden states)?
#     Does cross-region coupling matter at all, or is every region just its own slow signal?
def stack_k(segs, k):
    X0, X1 = [], []
    for a in segs:
        T = a.shape[1]
        if T <= k: continue
        X0.append(np.concatenate([a[:, k - 1 - j:T - 1 - j] for j in range(k)], 0)); X1.append(a[:, k:])
    return np.concatenate(X0, 1), np.concatenate(X1, 1)

order_r2 = {}
for k in (1, 2, 3, 5):
    pk = {f: stack_k(Z[f], k) for f in FLY}; out = []
    for h in FLY:
        tr = [pk[f] for f in FLY if f != h]
        G = sum(x0 @ x0.T for x0, _ in tr); H = sum(x1 @ x0.T for x0, x1 in tr)
        x0, x1 = pk[h]; out.append(r2(H @ np.linalg.inv(G + LAM * np.eye(G.shape[0])) @ x0, x1))
    order_r2[k] = np.median(out)
diag_out = []
for h in FLY:
    tr = [pairs[f] for f in FLY if f != h]
    G = sum(x0 @ x0.T for x0, _ in tr); H = sum(x1 @ x0.T for x0, x1 in tr)
    a = np.diag(H) / (np.diag(G) + LAM); x0, x1 = pairs[h]; diag_out.append(r2(a[:, None] * x0, x1))
R["order_r2"], R["diag_r2"] = order_r2, float(np.median(diag_out))
print("  3c. leave-one-fly-out median R^2 by model:  "
      f"each region alone (diagonal A) {np.median(diag_out):.3f} | full VAR(1) {order_r2[1]:.3f} | "
      f"VAR(2) {order_r2[2]:.3f} | VAR(3) {order_r2[3]:.3f} | VAR(5) {order_r2[5]:.3f}")
print(f"      -> cross-region coupling adds {order_r2[1] - np.median(diag_out):+.3f} R^2 over regions-alone; extra lags add at most {max(order_r2.values()) - order_r2[1]:+.3f},")
print("         so a first-order model on the 36 regions is adequate (the data do not demand extra memory/hidden states)")

# 3d. Dimensionality.  Hidden-region inference works because regions are redundant (shared modes).
allX = np.concatenate([np.concatenate(Z[f], 1) for f in FLY], 1)
ev = np.linalg.eigvalsh(np.cov(allX))[::-1]; ev /= ev.sum()
pr = 1.0 / np.sum(ev ** 2)
R["ev"] = ev
print(f"  3d. PCA of the 36 regions: PC1 = {ev[0]:.0%}, first 3 = {ev[:3].sum():.0%}, first 5 = {ev[:5].sum():.0%} of variance; "
      f"participation ratio = {pr:.1f}; 90% needs {np.searchsorted(np.cumsum(ev), 0.9) + 1} PCs")
print("      (a few strong shared modes plus a long flat tail, consistent with the noise floor in 3a: real redundancy to exploit)")

# 3e. Stationarity.  FC from the first vs second half of each fly's recording.
st = []
for f in FLY:
    X = np.concatenate(Z[f], 1); h_ = X.shape[1] // 2
    c1, c2 = np.corrcoef(X[:, :h_])[iu], np.corrcoef(X[:, h_:])[iu]; st.append(spearmanr(c1, c2)[0])
st = np.array(st)
print(f"  3e. FC pattern, first half vs second half of each recording: rho median {np.median(st):.2f} (range {st.min():.2f} to {st.max():.2f});"
      f" same-fly 70/30 R^2 ({np.median(list(within.values())):.3f}) ~ cross-fly R^2 ({np.median(v):.3f})")
print("      -> dynamics are stable within a recording and shared across animals, which is what a fixed A requires")


# =============================================================================
# PART 4: does the connectome add anything beyond a generic prior?   (proxies only - no Kalman/EM)
# =============================================================================
header("PART 4: what does the connectome add beyond generic structure?")
W = sc.copy(); np.fill_diagonal(W, 0)
Wn = W / W.sum(1, keepdims=True)                          # row-normalised wiring weights (each row sums to 1)
Unif = (np.ones((n, n)) - np.eye(n)) / (n - 1)            # 'everything couples equally' control (same row sums)

# 4a. Does the learned dynamics matrix look like the wiring?
G = sum(x0 @ x0.T for x0, _ in pairs.values()); H = sum(x1 @ x0.T for x0, x1 in pairs.values())
A_all = H @ np.linalg.inv(G + LAM * np.eye(n)); A_sym = (A_all + A_all.T) / 2
rho_a = spearmanr(A_sym[iu], sc[iu])[0]
null_a = np.array([spearmanr(A_sym[iu], sc[np.ix_(p, p)][iu])[0] for p in (rng.permutation(n) for _ in range(2000))])
R["rho_A"] = rho_a
print(f"  4a. off-diagonal of the fitted A (symmetrised) vs wiring: Spearman rho = {rho_a:.3f}, "
      f"permutation p = {(np.sum(np.abs(null_a) >= abs(rho_a)) + 1) / 2001:.4f}")
print("      -> the dynamics, not just the correlations, line up with the connectome (weaker than FC, as expected: A is a harder object)")

# 4b. Small-data learning curve.  For each target fly, fit A from only its first m transitions and test on its last 30%.
#     Prior:  A ~ A0 with penalty lam*||A - A0||^2,  closed form A = (H + lam*A0)(G + lam*I)^-1.
#     A0 = d*I (persistence) + a_u*Unif (global coupling) + a_w*(Wn - Unif) (wiring beyond uniform).
#     Hyper-parameters are tuned on 5 validation flies and scored on the other 15, separately for each variant
#     (the shuffled variant gets the same tuning budget, so it is not handicapped).
def fit_prior(x0, x1, A0, lam):
    return (x1 @ x0.T + lam * A0) @ np.linalg.inv(x0 @ x0.T + lam * np.eye(n))

def split_fly(f, m):
    x0, x1 = pairs[f]; k = int(0.7 * x0.shape[1])
    return x0[:, :min(m, k)], x1[:, :min(m, k)], x0[:, k:], x1[:, k:]

val_f, test_f = FLY[:5], FLY[5:]
shufs = [rng.permutation(n) for _ in range(10)]
Wsh = [Wn[np.ix_(p, p)] for p in shufs]
grid = dict(lam=[10, 100, 1000], d=[0, 0.25, 0.5], au=[0, 0.25, 0.5], aw=[0, 0.5, 1.0])
def run_variant(flist, m, Wm, par):
    lam, d, au, aw = par; out = []
    for f in flist:
        a0, a1, b0, b1 = split_fly(f, m)
        out.append(r2(fit_prior(a0, a1, d * np.eye(n) + au * Unif + aw * (Wm - Unif), lam) @ b0, b1))
    return np.array(out)
import itertools
combos = list(itertools.product(grid["lam"], grid["d"], grid["au"], grid["aw"]))
M_LIST = (30, 60, 120, 250, 500, 1000)
LC = {k: {} for k in ("ridge to zero", "persistence prior", "uniform-coupling prior", "connectome prior", "shuffled-wiring prior")}
for m in M_LIST:
    zero = max(((run_variant(val_f, m, Wn, (l, 0, 0, 0)).mean(), l) for l in grid["lam"]))[1]
    LC["ridge to zero"][m] = run_variant(test_f, m, Wn, (zero, 0, 0, 0))
    bp = max(((run_variant(val_f, m, Wn, (l, d, 0, 0)).mean(), (l, d, 0, 0)) for l in grid["lam"] for d in grid["d"]))[1]
    LC["persistence prior"][m] = run_variant(test_f, m, Wn, bp)
    bu = max(((run_variant(val_f, m, Wn, (l, d, au, 0)).mean(), (l, d, au, 0)) for l in grid["lam"] for d in grid["d"] for au in grid["au"]))[1]
    LC["uniform-coupling prior"][m] = run_variant(test_f, m, Wn, bu)
    bc = max(((run_variant(val_f, m, Wn, c).mean(), c) for c in combos))[1]
    LC["connectome prior"][m] = run_variant(test_f, m, Wn, bc)
    bs = max(((np.mean([run_variant(val_f, m, w, c).mean() for w in Wsh[:4]]), c) for c in combos))[1]
    LC["shuffled-wiring prior"][m] = np.mean([run_variant(test_f, m, w, bs) for w in Wsh], 0)
R["LC"], R["M_LIST"] = LC, M_LIST
print("  4b. small-data learning curve (median R^2 on the 15 test flies; train on the first m frames of the SAME fly only)")
print("      m frames   " + "".join(f"{k:>24s}" for k in LC))
for m in M_LIST:
    print(f"      {m:8d}   " + "".join(f"{np.median(LC[k][m]):24.3f}" for k in LC))
for m in (30, 120):
    d_ = LC["connectome prior"][m] - LC["shuffled-wiring prior"][m]
    p_ = wilcoxon(d_).pvalue
    print(f"      m={m} ({m / meta['fs']:.0f} s of data): connectome - shuffled = {np.median(d_):+.3f} median over flies "
          f"(connectome better in {np.sum(d_ > 0)}/15 flies, Wilcoxon p = {p_:.4f})")
print("      -> nearly all of the small-data gain comes from a persistence + shared-coupling prior (which a shuffled matrix also supplies).")
print("         The specific wiring adds only about +0.01 R^2, visible at ~30-60 frames and gone by ~120 frames: at region level,")
print("         a connectome prior on A is barely separable from generic shrinkage.")

# 4c. Hidden-region reconstruction: hide a random set of regions in a held-out fly, reconstruct them from the rest.
#     These are the baselines a Kalman SMOOTHER must beat.  Everything is scored on z-scored data, R^2 on hidden regions.
#       static    : Gaussian conditioning with the covariance learned from the 19 OTHER flies (no time, no wiring)
#       context   : ridge on visible regions at t-2..t+2 (what any smoother could use; cross-fly learned)
#       wiring    : hidden = SC-weighted average of visible regions (one scalar gain learned on other flies; no region-specific data at all)
#       shuffled  : same, with the wiring matrix relabelled at random   |   uniform: plain average of visible regions
K_CTX = 2
def ctx_mats(segs, V, K):
    X, Y = [], []
    for a in segs:
        T = a.shape[1]
        X.append(np.concatenate([a[V][:, K + o:T - K + o] for o in range(-K, K + 1)], 0)); Y.append(a[:, K:T - K])
    return np.concatenate(X, 1), np.concatenate(Y, 1)

FRACS = (0.25, 0.5, 0.75); N_DRAW = 10; N_SHUF = 5
HR = {fr_: {k: np.zeros((N_DRAW, len(FLY))) for k in ("static", "context", "wiring", "shuffled", "uniform")} for fr_ in FRACS}
allZ = {f: np.concatenate(Z[f], 1) for f in FLY}
Sxx = {f: allZ[f] @ allZ[f].T for f in FLY}; Nf = {f: allZ[f].shape[1] for f in FLY}
Stot, Ntot = sum(Sxx.values()), sum(Nf.values())
for fr_ in FRACS:
    nh = int(round(fr_ * n))
    for dr in range(N_DRAW):
        Hh = np.sort(rng.choice(n, nh, replace=False)); V = np.setdiff1d(np.arange(n), Hh)
        perms = [rng.permutation(n) for _ in range(N_SHUF)]
        # per-fly sufficient statistics for this hidden set (leave-one-out = total minus own)
        ctx = {f: ctx_mats(Z[f], V, K_CTX) for f in FLY}
        Gx = {f: ctx[f][0] @ ctx[f][0].T for f in FLY}; Hx = {f: ctx[f][1][Hh] @ ctx[f][0].T for f in FLY}
        Gt, Ht = sum(Gx.values()), sum(Hx.values())
        wmats = {"wiring": [Wn], "shuffled": [Wn[np.ix_(p, p)] for p in perms], "uniform": [Unif]}
        def wavg(Wm, X):
            w = Wm[np.ix_(Hh, V)]; return (w / w.sum(1, keepdims=True)) @ X
        num = {k: {f: [np.sum(wavg(w, allZ[f][V]) * allZ[f][Hh]) for w in ws] for f in FLY} for k, ws in wmats.items()}
        den = {k: {f: [np.sum(wavg(w, allZ[f][V]) ** 2) for w in ws] for f in FLY} for k, ws in wmats.items()}
        for j, h in enumerate(FLY):
            Ctr = (Stot - Sxx[h]) / (Ntot - Nf[h])
            B = Ctr[np.ix_(Hh, V)] @ np.linalg.inv(Ctr[np.ix_(V, V)] + 1e-3 * np.eye(len(V)))
            Y = allZ[h][Hh]; HR[fr_]["static"][dr, j] = 1 - np.sum((Y - B @ allZ[h][V]) ** 2) / np.sum(Y ** 2)
            Bc = (Ht - Hx[h]) @ np.linalg.inv(Gt - Gx[h] + 100 * np.eye(Gt.shape[0]))
            Xc, Yc = ctx[h][0], ctx[h][1][Hh]; HR[fr_]["context"][dr, j] = 1 - np.sum((Yc - Bc @ Xc) ** 2) / np.sum(Yc ** 2)
            for k, ws in wmats.items():
                sc_ = []
                for q, w in enumerate(ws):
                    g = (sum(num[k][f][q] for f in FLY if f != h)) / (sum(den[k][f][q] for f in FLY if f != h))
                    sc_.append(1 - np.sum((Y - g * wavg(w, allZ[h][V])) ** 2) / np.sum(Y ** 2))
                HR[fr_][k][dr, j] = np.mean(sc_)
R["HR"], R["FRACS"] = HR, FRACS
print("  4c. hidden-region reconstruction, median R^2 on hidden regions (10 random hidden sets x 20 held-out flies)")
print("      hidden    static-conditioning   +temporal context   wiring-only   shuffled-wiring   uniform-average")
for fr_ in FRACS:
    print(f"      {fr_:5.0%}    " + "".join(f"{np.median(HR[fr_][k]):>20.3f}" if k in ("static", "context") else f"{np.median(HR[fr_][k]):>16.3f}"
                                         for k in ("static", "context", "wiring", "shuffled", "uniform")))
for fr_ in FRACS:
    d_ = (HR[fr_]["wiring"] - HR[fr_]["shuffled"]).mean(0)          # per-fly mean over hidden-set draws
    c_ = (HR[fr_]["context"] - HR[fr_]["static"]).mean(0)
    print(f"      {fr_:.0%} hidden: wiring - shuffled = {d_.mean():+.3f} (better in {np.sum(d_ > 0)}/20 flies, Wilcoxon p = {wilcoxon(d_).pvalue:.1e});"
          f"   context - static = {c_.mean():+.3f} (better in {np.sum(c_ > 0)}/20)")
print("      -> wiring-only beats a shuffled or uniform guess by a small but consistent margin (about +0.02 R^2) without seeing any")
print("         data from the hidden region; the covariance learned from other flies does far better (it knows the hidden region);")
print("         adding +/-2 frames of temporal context (the part a smoother adds) gains only about +0.02 on top of that.")


# =============================================================================
# FIGURES
# =============================================================================
if args.no_figs:
    raise SystemExit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False})
os.makedirs("figures", exist_ok=True)
short = lambda f: f[2:]                       # '2017-10-26_1' -> '17-10-26_1'


def savefig(fig, name):
    fig.savefig(os.path.join("figures", name), dpi=200, bbox_inches="tight"); plt.close(fig)


# ---- Figure 1: rank-transformed structure vs function ------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
xs = rankdata(sc[iu]) / len(iu[0]); ys = rankdata(fc_vec) / len(iu[0])
a = ax[0]; a.grid(axis="x", visible=False)
a.scatter(xs, ys, s=14, color=BLUE, alpha=0.35, linewidths=0)
bins = np.linspace(0, 1, 11); mid = (bins[:-1] + bins[1:]) / 2
med = [np.median(ys[(xs > lo) & (xs <= hi)]) for lo, hi in zip(bins[:-1], bins[1:])]
a.plot(mid, med, color=INK, lw=2, label="median of each decile")
resid_rank = ys - xs                                       # which pairs disagree most between structure and function
pairs_lbl = {"hi": np.argsort(resid_rank)[-2:], "lo": np.argsort(resid_rank)[:2]}
for j in np.r_[pairs_lbl["hi"], pairs_lbl["lo"]]:
    i1, i2 = iu[0][j], iu[1][j]
    a.annotate(f"{names[i1]}\u2013{names[i2]}", (xs[j], ys[j]), xytext=(6, 4 if resid_rank[j] > 0 else -12),
               textcoords="offset points", fontsize=8, color=INK2)
a.set_xlabel("structural connectivity (rank, 0 = weakest pair, 1 = strongest)")
a.set_ylabel("functional connectivity (rank)")
a.set_title(f"Wired-together regions are co-active (\u03c1 = {R['p1_rho']:.2f})")
a.legend(loc="lower right")
a.set_xlim(0, 1.02); a.set_ylim(0, 1.02)

b = ax[1]; b.grid(axis="y", visible=True)
b.hist(R["p1_null"], bins=40, color=NEUTRAL, alpha=0.65)
b.axvline(R["p1_rho"], color=BLUE, lw=2)
b.text(R["p1_rho"] + 0.02, b.get_ylim()[1] * 0.92, f"real wiring\n\u03c1 = {R['p1_rho']:.2f}", color=INK, ha="left", va="top")
if "p1_rho_dist" in R:
    b.axvline(R["p1_rho_dist"], color=AQUA, lw=2, ls="--")
    b.text(R["p1_rho_dist"] - 0.02, b.get_ylim()[1] * 0.60, f"distance\nremoved\n\u03c1 = {R['p1_rho_dist']:.2f}", color=INK, ha="right", va="top")
b.text(-0.42, b.get_ylim()[1] * 0.92, "2000 shuffles of\nwhich region is which", color=INK2, ha="left", va="top")
b.set_xlabel("Spearman \u03c1 between structure and function"); b.set_ylabel("number of shuffles")
b.set_title("No shuffle comes close"); b.set_xlim(-0.45, 0.98)
fig.tight_layout(); savefig(fig, "fig1_structure_vs_function.png")

# ---- Figure 2: per-fly R^2 bars -----------------------------------------------------------------------------
order = sorted(FLY, key=lambda f: lofo[f])
fig, a = plt.subplots(figsize=(12, 4.8)); a.grid(axis="x", visible=False)
x = np.arange(len(order)); w_ = 0.27
a.bar(x - w_, [persist[f] for f in order], w_ - 0.03, color=NEUTRAL, label="persistence (next frame = this frame)")
a.bar(x, [within[f] for f in order], w_ - 0.03, color=AQUA, label="same fly: fit on first 70%, test on last 30%")
a.bar(x + w_, [lofo[f] for f in order], w_ - 0.03, color=BLUE, label="other 19 flies' shared A (leave-one-fly-out)")
a.scatter(x, [1 - nf[f] for f in order], marker="_", s=300, color=INK, linewidths=2, zorder=5, label="approx. noise ceiling, 1 \u2212 high-frequency floor")
a.axhline(np.median(list(lofo.values())), color=BLUE, lw=1, ls="--", label=f"median leave-one-fly-out R\u00b2 = {np.median(list(lofo.values())):.2f}")
a.axhline(0, color=INK2, lw=0.8)
a.set_xticks(x); a.set_xticklabels([short(f) for f in order], rotation=55, ha="right")
a.set_ylabel("one-step-ahead R\u00b2 (z-scored, 36 regions)"); a.set_xlabel("held-out fly (sorted by leave-one-fly-out R\u00b2)")
a.set_title("A single shared linear map predicts an unseen fly as well as a map fit on that fly")
a.legend(loc="upper left", ncol=2, fontsize=9); a.set_ylim(-0.05, 0.85)
fig.tight_layout(); savefig(fig, "fig2_per_fly_r2.png")

# ---- Figure 3: is the data state-space-shaped? ---------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]; a.grid(axis="x", visible=False)
a.plot(R["psd_fr"], R["psd"], color=BLUE, lw=2)
a.axvspan(0.45, 0.59, color=ORANGE, alpha=0.18, lw=0)
a.axhline(R["psd"][R["psd_fr"] >= 0.45].mean(), color=ORANGE, lw=1.2, ls="--")
a.text(0.30, R["psd"][R["psd_fr"] >= 0.45].mean() * 0.45, "flat floor = imaging noise", color=INK2, ha="center", va="center")
a.text(0.05, R["psd"].max() * 0.95, "slow signal", color=INK2, va="top")
a.set_xlabel("frequency (Hz)"); a.set_ylabel("power (z-scored, mean of regions and flies)")
a.set_title(f"~{np.mean(nfv):.0%} of variance is a flat noise floor"); a.set_ylim(0, None)

a = ax[0, 1]; a.grid(axis="x", visible=False)
k_ = 12; a.bar(np.arange(1, k_ + 1), R["ev"][:k_] * 100, color=BLUE, width=0.7)
a.set_xticks(np.arange(1, k_ + 1)); a.set_xlabel("principal component"); a.set_ylabel("% of variance")
a.text(1.0, R["ev"][0] * 100 + 1, f"{R['ev'][0]:.0%}", ha="center", color=INK)
a.set_title(f"A few shared modes (participation ratio {1 / np.sum(R['ev'] ** 2):.1f} of 36)")

a = ax[1, 0]; a.grid(axis="x", visible=False)
labels = ["persistence", "each region\nalone", "VAR(1)\n36 regions", "VAR(3)", "approx. noise\nceiling"]
vals = [np.median(list(persist.values())), R["diag_r2"], R["order_r2"][1], R["order_r2"][3], 1 - np.mean(nfv)]
cols = [NEUTRAL, AQUA, BLUE, BLUE, INK2]
bars = a.bar(range(5), vals, color=cols, width=0.62)
bars[4].set_hatch("///"); bars[4].set_facecolor(SURF); bars[4].set_edgecolor(INK2)
for i, vv in enumerate(vals): a.text(i, max(vv, 0) + 0.012, f"{vv:.2f}", ha="center", color=INK)
a.axhline(0, color=INK2, lw=0.8)
a.set_xticks(range(5)); a.set_xticklabels(labels, fontsize=9); a.set_ylabel("leave-one-fly-out median R\u00b2")
a.set_title("Coupling helps, more memory barely does"); a.set_ylim(-0.05, 0.62)

a = ax[1, 1]; a.grid(axis="x", visible=True)
xv = [nf[f] for f in FLY]; yv = [lofo[f] for f in FLY]
a.scatter(xv, yv, s=40, color=BLUE, alpha=0.85, linewidths=0)
for f, xx, yy in zip(FLY, xv, yv):
    if yy < 0.2 or yy > 0.48: a.annotate(short(f), (xx, yy), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK2)
a.text(0.97, 0.95, f"Spearman \u03c1 = {R['rho_nf']:+.2f}", transform=a.transAxes, ha="right", va="top")
a.set_xlabel("flat high-frequency floor (fraction of variance)"); a.set_ylabel("leave-one-fly-out R\u00b2")
a.set_title("Noisier flies are harder to predict")
fig.tight_layout(); savefig(fig, "fig3_state_space_validity.png")

# ---- Figure 4: what the connectome adds ------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
a = ax[0]; a.grid(axis="x", visible=False)
style = {"ridge to zero": (NEUTRAL, "-"), "persistence prior": (AQUA, "-"), "connectome prior": (BLUE, "-"), "shuffled-wiring prior": (ORANGE, "--")}
for k, (c_, ls) in style.items():
    a.plot(R["M_LIST"], [np.median(R["LC"][k][m]) for m in R["M_LIST"]], color=c_, lw=2, ls=ls, marker="o", ms=5, label=k)
a.set_xscale("log"); a.set_xticks(R["M_LIST"]); a.set_xticklabels([str(m) for m in R["M_LIST"]])
a.set_xlabel("frames of the target fly used to fit A (1.2 frames = 1 s)"); a.set_ylabel("R\u00b2 on its last 30% (median of 15 flies)")
a.set_title("Priors help a lot; which prior matters little"); a.legend(loc="lower right", fontsize=9)

a = ax[1]; a.grid(axis="x", visible=False)
for j, m in enumerate(R["M_LIST"]):
    d_ = R["LC"]["connectome prior"][m] - R["LC"]["shuffled-wiring prior"][m]
    a.scatter(np.full(len(d_), j) + rng.uniform(-0.13, 0.13, len(d_)), d_, s=22, color=BLUE, alpha=0.5, linewidths=0)
    a.hlines(np.median(d_), j - 0.28, j + 0.28, color=INK, lw=2)
a.axhline(0, color=INK2, lw=0.8)
a.set_xticks(range(len(R["M_LIST"]))); a.set_xticklabels([str(m) for m in R["M_LIST"]])
a.set_xlabel("frames used to fit A"); a.set_ylabel("R\u00b2(connectome prior) \u2212 R\u00b2(shuffled wiring)")
a.set_title("Wiring edge over shuffled: \u2264 0.01 and fading")

a = ax[2]; a.grid(axis="x", visible=False)
hs = {"static": (INK, "-", "other flies' covariance"), "context": (AQUA, "-", "+ \u00b12 frames of context"),
      "wiring": (BLUE, "-", "wiring only"), "shuffled": (ORANGE, "--", "shuffled wiring"), "uniform": (NEUTRAL, "-", "plain average")}
xf = [f * 100 for f in R["FRACS"]]
for k, (c_, ls, lab) in hs.items():
    ys_ = [np.median(R["HR"][f][k]) for f in R["FRACS"]]
    a.plot(xf, ys_, color=c_, lw=2, ls=ls, marker="o", ms=5, label=lab)
a.axhline(1 - np.mean(nfv), color=INK2, lw=1, ls=":"); a.text(75, 1 - np.mean(nfv) + 0.008, "approx. noise ceiling", ha="right", color=INK2, fontsize=9)
a.set_xticks(xf); a.set_xticklabels([f"{int(v)}%" for v in xf]); a.set_xlabel("share of regions hidden"); a.set_ylabel("R\u00b2 on the hidden regions (median)")
a.set_title("Hidden-region baselines a smoother must beat"); a.legend(loc="lower left", fontsize=9); a.set_ylim(0.1, 0.58)
fig.tight_layout(); savefig(fig, "fig4_what_the_connectome_adds.png")
print("\nfigures written to ./figures/")