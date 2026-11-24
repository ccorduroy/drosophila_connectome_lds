# Protocol and Model Spec: Connectome Prior for a Drosophila LDS

EE 675 project · Nov 22, 2026 · @Caitlin

## 1. Purpose and scope

This document fixes the model, data splits, metrics and statistics before any Kalman/EM code runs on real data, so that no result can steer a later choice. Anything not fixed here is listed in section 16 as an open decision and must be settled before coding.

**What is frozen.** The cleaned dataset (`fly_ito36.npz`, built by `build_dataset.py`), the 36-region ordering, the structural matrices, the random seeds for hidden-region draws, the model family, the hyperparameter grid, the metrics and the statistical tests.

**Status of earlier numbers.** Everything in `analysis_demo.py` (SC–FC correlation, ridge VAR(1), noise floor, hidden-region baselines) is exploratory. All 20 flies were inspected while choosing the cleaning steps and λ = 100, so those numbers are reference points for sanity checks, not test results. They are never cited as evidence for the hypotheses.

**Out of scope unless a stated trigger fires.**

- Nonlinear or non-Gaussian state-space models (trigger: section 14, rule N1).
- Multi-step forecasting as a primary result (stretch goal only).
- Neuron-level models and any use of excitatory/inhibitory signs (the hemibrain tables carry none).

**Labels used throughout.** Models M0–M6 (section 7), baselines B1–B8 (section 12), evaluation regimes R1–R3 and tasks T1–T2 (section 11), hypotheses H1–H6 (section 2).

## 2. Research question and hypotheses

The question is whether a connectome-derived prior on the transition matrix improves a linear state-space model's reconstruction and prediction of whole-brain fly activity, and whether that benefit grows as fewer regions are observed. The feasibility work says the effect, if real, is small (about +0.01 to +0.02 R² over a shuffled prior), so the hypotheses below are written to be decidable at that size.

| ID | Hypothesis | Regime | Metric | Test | Primary? |
| --- | --- | --- | --- | --- | --- |
| H1 | Connectome prior (M3) makes never-observed regions reconstructable: hidden-region r above the persistence-only prior (M1), where hidden regions are decoupled and r is expected near 0 (section 6, gauge freedom) | R3, 25/50/75% hidden | Hidden-region Pearson r (smoothed) | Paired Wilcoxon over 20 flies | Yes |
| H2 | The specific wiring matters: M3 beats both the shuffled prior (M4) and the uniform-coupling prior (M2) | R3, 25/50/75% hidden | Same | Same | Yes |
| H3 | The M3 − M4 gap is larger at 75% hidden than at 25% | R3 | Per-fly gap (75%) minus gap (25%) | Paired Wilcoxon | Yes |
| H4 | In a data-limited fit (few frames of the target fly), M3 beats M0 | R2, m = 30–1000 frames | One-step R² | Paired Wilcoxon per m | No |
| H5 | With 19 fully observed training flies the prior has no measurable effect (a control, expected null) | R1 | One-step R²; hidden r | Paired Wilcoxon | No |
| H6 | A linear-Gaussian model is sufficient: its error is close to the noise ceiling and no worse than the Li et al. nonlinear model on comparable metrics | R1 | One-step R² vs ceiling | Descriptive | No |

**Decision rule.** A primary hypothesis counts as supported only if all three hold: (a) the Holm-adjusted p-value across the primary family (H1–H3, 3 fractions each, 9 tests) is below 0.05; (b) the median paired difference across flies is at least the smallest effect of interest, 0.02 in Pearson r or 0.01 in R²; (c) the 95% bootstrap interval over flies (10,000 resamples) excludes zero. Otherwise it is reported as not supported, with the interval, which is a valid project result.

**Why the regimes differ.** With 19 fully observed flies, about 38,000 transitions already pin down all 1,296 entries of A, so any reasonable prior is overwhelmed; H5 states that expectation. The prior can only matter where the data cannot speak: few frames (R2) or regions never observed in training (R3). The deck's design (train on 19 complete flies, hide regions only at test) is R1, so it needs R3 added to test the research question as asked. See section 11.

## 3. Data and preprocessing

The input is 20 flies, 36 regions, sampled at 1.2 Hz, with 40,700 kept frames after cleaning; every number below comes from the frozen build.

**Pipeline (already implemented in `build_dataset.py`).**

1. Region-averaged raw fluorescence per Ito-atlas region (Turner, Mann & Clandinin 2021); row 0 of each recording is background and is dropped.
2. Merge IB(R) and IB(L) into one region `IB`, weighted by voxel count (3,299 and 4,269 voxels), to match the 36 structural-matrix regions.
3. First-order Butterworth high-pass at 0.01 Hz (sampling rate 1.2 Hz), applied before trimming.
4. Drop the first 100 frames of every recording; apply the authors' trims to three flies. Segments are kept separate and never joined across a gap.
5. Standardise each region within each fly (zero mean, unit variance over that fly's kept frames). All models see only standardised data.

**Per-fly inventory.**

| Fly | Kept frames | Segments (frame ranges) | Known issue |
| --- | --- | --- | --- |
| 2017-10-26\_1 | 1900 | 1 | +7–12% steps between frames 1228 and 1787 (about 20% of filtered variance); not trimmed by the authors |
| 2017-10-30\_1 | 1900 | 1 | — |
| 2017-10-30\_2 | 1900 | 1 | — |
| 2017-11-08\_1 | 3800 | 2: 100–1900, 2000–4000 | Baseline step near frame 1955 (trimmed) |
| 2017-11-08\_2 | 3900 | 1 | — |
| 2017-11-16\_1 | 1900 | 1 | Highest noise floor (about 0.60) |
| 2018-10-19\_1 | 1700 | 2: 100–900, 1100–2000 | Dropouts near frames 1009–1056 (trimmed) |
| 2018-10-19\_2 | 1900 | 1 | — |
| 2018-10-20\_1 | 900 | 1: 100–1000 | Signal lost after about frame 1010 (trimmed) |
| 2018-10-31\_1 | 1900 | 1 | — |
| 2018-11-03\_2 to \_5 | 1900 each | 1 each | Four flies, one recording day |
| 2018-12-12\_2 to \_4 | 1900 each | 1 each | Three flies, one recording day |
| 2018-12-14\_1 to \_3 | 1900 each | 1 each | Three flies, one recording day |

**Frozen artifacts.**

| Artifact | Role |
| --- | --- |
| `fly_ito36.npz` | Cleaned segments, 36 region names, six 36×36 structural matrices, metadata |
| `build_dataset.py` | Rebuilds the npz; `load()` is the only data reader the model code may use |
| `analysis_demo.py` | Feasibility analysis; source of the reference numbers in section 12 |
| `Original_Index_panda_full.csv`, `vfb_68_2018-11-03_5.nii.gz` | Atlas labels and mask; used only for region centroids (distance control M5) |

Store a SHA-256 of each file in the repository and assert it at the start of every run.

**Structural matrices.** Rows and columns follow the same 36-region order as the fluorescence arrays. The primary matrix is the one named `Connectivity` in Turner's computed files, because it is the one every feasibility run used, not because it scored best (`TBars` had the highest SC–FC ρ, 0.714). `CellCount` (Weak + Medium + Strong) is the secondary matrix; the other four are the "if time allows" comparison.

**Definition of `Connectivity`, read from Turner's `computeConnectivityMatrix` (`scfc/anatomical_connectivity.py`, called by `compute_connectivity_matrix.py`, hemibrain v1.2) on Nov 23.** For each pair of hemibrain ROIs (sour, targ) inside atlas regions (i, j), the code takes all traced, uncropped neurons with input synapses in sour and output synapses in targ. Each neuron contributes the square root of (its T-bars in targ × its PSDs in sour), and `Connectivity[i, j]` is the sum of these contributions over neurons and over all (sour, targ) pairs, so it is a sum of square roots, not the square root of summed counts. Atlas regions are Ito regions, mapped to hemibrain ROIs by Turner's Ito-to-neuprint mapping, so the matrix is on the same 36-region Ito parcellation as the fluorescence data. Row i is the input side and column j the output side, so the entry reads as signal flowing from i to j. There is no synapse-confidence threshold and no normalisation, and the diagonal is kept (our recipe zeroes it and takes persistence from d·I). Neurons are deduplicated only in the list of body IDs, never in the matrix sums; see the multiplicity limitation in section 14. The matrix file comes from `compute_connectivity_matrix.py` (confirmed by the author of this project, Nov 23), not from the atlas CSV route (`getAtlasConnectivity`), which this project does not use.

**Recording days are clusters.** Several flies share a recording day (the date prefix of the ID). Same-day flies may share preparation and setup, so leave-one-fly-out may be mildly optimistic. Section 11 adds a leave-one-day-out sensitivity run.

## 4. Notation and dimension bookkeeping

The prior and the data have the same dimension because the hidden state is defined as the 36 regions themselves: the prior is a 36×36 matrix on A, and hiding regions only removes rows from the observation map, never from the state. This is the direct answer to the professor's comment.

| Symbol | Size | Meaning |
| --- | --- | --- |
| n | 36 | Number of regions = state dimension |
| F | 20 | Number of flies; f indexes flies |
| T\_f | 900–3900 | Kept frames of fly f; segments s = 1..S\_f restart the filter |
| x\_t | n×1 | True (noise-free) activity of all 36 regions at frame t, standardised |
| y\_t | m\_t×1 | Observed activity of the visible regions only; m\_t = n − |
| H | subset of 1..n | Hidden regions (empty in R1 training); V = complement = visible |
| C\_t | m\_t×n | Selection matrix: rows of the identity for visible regions |
| A | n×n | Transition matrix; A\_ij = effect of region j at t on region i at t+1 |
| Q | n×n | Process-noise covariance, full, positive definite |
| R | n×n | Observation-noise covariance, diagonal |
| A₀ | n×n | Connectome-derived prior mean for A (section 6) |
| W | n×n | Log-scaled, symmetrised structural matrix |

**What the observation map does.** y\_t = C\_t x\_t + v\_t. In R1 and R2 every region is visible during training, so C\_t = I. In test flies, or in R3 training, C\_t drops the rows of hidden regions. The state x\_t, A, Q and A₀ keep size 36 and 36×36 throughout, so the prior always matches A exactly.

**Units.** The structural matrix counts cells (about 1 to 10³); A holds dimensionless coefficients of magnitude roughly 0–0.5 on standardised data. The mismatch is units, not dimension, and is resolved by a fixed recipe (section 6): log scaling, symmetrisation, zero diagonal, row normalisation, then two fitted scalars (coupling strength and persistence).

**Orientation.** The primary prior is symmetrised, so it does not depend on which axis is source and which is target. A directed prior would need the matrix transposed for x\_{t+1} = A x\_t and its orientation confirmed in Turner's code; that is an optional extension (open decision 5).

**Why not a smaller latent state.** A latent state of size k < n would need the prior projected down (A\_k = P A₀ P⁺), which has no clean connectome interpretation. The feasibility work gives no reason to compress: extra lags add at most 0.01 R², and the 36 regions are all directly measured. Keep n = 36.

## 5. The linear dynamical system

The model is a zero-mean, time-invariant linear-Gaussian state-space model on the 36 regions, with a full process-noise covariance and a diagonal observation-noise covariance. Parameters are shared across flies.

```latex
x_{t+1} = A\,x_t + w_t,\quad w_t \sim \mathcal{N}(0, Q)
```

```latex
y_t = C_t\,x_t + v_t,\quad v_t \sim \mathcal{N}(0, R_{V_t V_t}),\quad R = \mathrm{diag}(r_1,\dots,r_n)
```

```latex
x_1^{(f,s)} \sim \mathcal{N}(0, \Sigma_0)
```

**Design choices and why.**

- **No intercept, no gain.** Data are standardised per fly and region, so the mean is zero and C\_t is a pure selection. Fixing C removes the scale ambiguity between x and C. Without it, A, Q and C trade off freely.
- **Full Q, diagonal R.** The feasibility residuals are correlated across regions (median |r| = 0.16), so Q must be full. Imaging noise is taken as independent per region, so R is diagonal. A full R with C = I would be confounded with Q.
- **Identifiability.** With C = I the lagged covariances are C₀ = Σₓ + R, C₁ = AΣₓ, C₂ = A²Σₓ. Observation noise appears only in C₀, so A = C₂C₁⁻¹, Σₓ = A⁻¹C₁ and R = diag(C₀ − Σₓ) are identified from lags 0–2. The catch: the data are close to low rank (participation ratio 7.8 of 36), so C₁ is nearly singular and the ML estimate is fragile. That fragility is the statistical case for a prior.
- **Initial state.** Σ₀ is fixed, not learned: Σ₀ = P\_data − R₀ floored to positive-definite, where P\_data is the covariance of the training data and R₀ the initial observation noise. With about 1,900 frames per segment, the first-step term is negligible and is excluded from the A and Q updates.
- **Segments restart.** Each kept segment starts a fresh filter from N(0, Σ₀). Gaps in this data are 100–200 frames, long relative to the \~0.5 lag-1 autocorrelation, so restarting loses almost nothing and avoids propagating through dropout artifacts.
- **Stability.** Report the spectral radius ρ(A) after every fit. If any fit returns ρ(A) ≥ 1, shrink A by 0.99 / ρ(A) and record it as a fit flag.
- **Shared parameters.** A, Q and R are shared by all flies in a fit. Per-fly noise scales are a pre-registered ablation (section 10), because noise floors vary from 0.32 to 0.60 across flies.

**Scope of what "hidden" means in R3.** A region hidden in R3 appears in no row of any C\_t. Its state is latent in every fly, so A, Q and Σ₀ restricted to it are determined only by the prior and by how it couples to visible regions. Its absolute scale is not identified from data; section 6 explains how the prior fixes it, and section 11 uses a scale-free metric for that reason.

## 6. Connectome prior construction

The prior mean is one 36×36 matrix built from the structural matrix by a fixed recipe with three fitted scalars; the same recipe feeds every control, so any difference between models comes from which regions are wired together and nothing else.

**Recipe (applied to one structural matrix S).**

1. Symmetrise: M = (S + Sᵀ)/2.
2. Compress: W = log10(1 + M), then set the diagonal to zero.
3. Row-normalise: W̃ᵢⱼ = Wᵢⱼ / Σₖ≠ᵢ Wᵢₖ, so every row sums to 1.
4. Uniform reference: U = (11ᵀ − I)/(n − 1), also with row sums of 1.
5. Assemble the prior mean:

```latex
A_0(d, a_u, a_w) = d\,I + a_u\,U + a_w\,(\tilde W - U)
```

The three scalars have separate jobs. d is persistence (how much a region predicts its own next value), a\_u is generic brain-wide coupling, and a\_w scales how much the wiring pattern moves coupling above or below uniform. Row sums of A₀ are d + a\_u, which bounds its gain.

**Prior distribution.** A given Q is matrix-normal with mean A₀ and column precision λI:

```latex
A \mid Q \sim \mathcal{MN}\!\left(A_0,\; Q,\; (\lambda I)^{-1}\right),\qquad \lambda = \kappa\,N
```

N is the number of transitions in the fit and κ is a dimensionless relative strength: κ = 1 means the prior counts as much as the data. Scaling by N keeps κ comparable between R1 (N about 38,000) and R2 (N as small as 30). This form is chosen because Q cancels from the A update, which stays closed form (section 9).

**Hyperparameter grid (identical for every variant).**

| Parameter | Grid |
| --- | --- |
| κ | 0, 0.001, 0.01, 0.1, 1 |
| d | 0, 0.25, 0.5, 0.75 |
| a\_u | 0, 0.25, 0.5 |
| a\_w | 0, 0.5, 1, 2 (zero for M0–M2) |

Discard any grid point where d + a\_u > 0.9 or the spectral radius of A₀ is 0.98 or higher. Record the fraction of negative off-diagonal entries of A₀; no non-negativity is imposed, so the data can overrule the prior.

**Shuffled control (M4).** Draw a permutation π of the 36 regions and use W̃\[π\]\[:, π\], the same relabelling on rows and columns, applied after row normalisation. This keeps the weight distribution, row sums, symmetry and zero diagonal, and destroys only which region has which wiring. Average every score over 20 permutations (seeds fixed in section 11) and also record where the real matrix ranks among them. The deck's wording ("shuffled order on each neuron intersection") should be changed to this definition.

**Other controls built with the same recipe.**

- **M5, distance prior.** Replace W by exp(−Dᵢⱼ/ℓ), D = distance between region centroids in the template brain, ℓ = median pairwise distance. Tests whether wiring beats mere proximity (feasibility: distance explains little of SC–FC, ρ 0.69 to 0.66 after removal).
- **M6, functional-connectivity ceiling.** Replace W by the training flies' mean Fisher-z FC, clipped at zero. An empirical upper bound on what a structural prior could supply. Only defined where the data exist (R1, R2).

**Gauge freedom in R3.** If a region is never observed, rescaling its state by s and adjusting A, Q accordingly leaves every observed quantity unchanged, and so does flipping its sign. Only the prior breaks this symmetry. Consequences: a flat prior (M0) or a persistence-only prior (M1) leaves hidden regions decoupled from the visible ones, so their reconstruction is expected to be near chance; the prior's scale (d, a\_u, a\_w) sets the hidden-state scale, so only scale-free metrics (Pearson r) are meaningful for R3. This is a derivation, not a measurement; simulation test S0 in section 13 must confirm it before R3 results are trusted.

## 7. Model variants and controls

All variants are the same LDS fitted by the same EM code; they differ only in the prior mean A₀ and which hyperparameters are free. The deck's three models are M0, M3 and M4; M1, M2, M5 and M6 are added because the feasibility results show a shuffled prior alone cannot separate the specific wiring from any generic coupling.

| ID | Name | Prior mean A₀ | Free hyperparameters | Question it answers |
| --- | --- | --- | --- | --- |
| M0 | Flat | 0 | κ | The pitch's "no prior" model (ridge-to-zero); κ = 0 is plain maximum-likelihood EM |
| M1 | Persistence | d·I | κ, d | How much does "each region keeps its own value" alone give? |
| M2 | Uniform coupling | d·I + a\_u·U | κ, d, a\_u | How much does a brain-wide shared mode give, with no wiring? |
| M3 | Connectome | d·I + a\_u·U + a\_w·(W̃ − U) | κ, d, a\_u, a\_w | The model under test |
| M4 | Shuffled wiring | Same as M3 with W̃\[π\]\[:, π\] | Same as M3 | Does the real wiring beat a random wiring of equal statistics? |
| M5 | Distance | Same as M3 with proximity weights | Same as M3 | Tier B. Does wiring beat mere closeness? |
| M6 | FC ceiling | Same as M3 with training-fly FC weights | Same as M3 | Cut for time (undefined in R3). Upper bound on any structural prior (R1, R2 only) |

**Fairness rules.**

- Every variant gets the same grid size per parameter it uses, the same inner-validation budget and the same EM settings (section 10). M4 is tuned on a permutation average, not on one favourable draw: the average over exactly the shuffles it is scored on. Tuning on a subset let M4 pick a\_w from lucky draws in simulation S2 (Nov 23).
- M3 and M4 are scored with the same hidden sets, the same flies and the same random seeds for initialisation.
- Primary comparisons: M3 vs M2 and M3 vs M4 (is the wiring itself informative), M3 vs M1 in R3 (does the prior make hidden regions reconstructable). M0 is reported everywhere but is not the control for wiring.

**Secondary connectome versions** ("if time allows"): rerun M3 and M4 with `CellCount`, `TBars`, `WeightedSynapseNumber`, `StrongConnections` and the other tables. Report them as a sensitivity table; do not pick the best one for the headline.

## 8. Inference: Kalman filter and RTS smoother

With parameters frozen, the Kalman filter gives the causal state estimate at each frame and the Rauch–Tung–Striebel smoother revises it using all frames of the segment; hidden regions are handled by dropping their rows from C\_t, so no imputation is needed.

**Filter, per segment, starting from the prior x̂₁|₀ = 0, P₁|₀ = Σ₀.** Let C = C\_t, R\_t = R restricted to visible regions.

```latex
S_t = C P_{t|t-1} C^\top + R_t,\qquad K_t = P_{t|t-1} C^\top S_t^{-1}
```

```latex
\hat x_{t|t} = \hat x_{t|t-1} + K_t\,(y_t - C\hat x_{t|t-1}),\qquad P_{t|t} = (I - K_t C)\,P_{t|t-1}\,(I - K_t C)^\top + K_t R_t K_t^\top
```

```latex
\hat x_{t+1|t} = A\,\hat x_{t|t},\qquad P_{t+1|t} = A P_{t|t} A^\top + Q
```

Use the Joseph form for P\_{t|t} and symmetrise every covariance after each step. If m\_t = 0 (all observations missing), skip the update and set x̂\_{t|t} = x̂\_{t|t-1}.

**Smoother (backward pass).**

```latex
J_t = P_{t|t}\,A^\top P_{t+1|t}^{-1},\quad \hat x_{t|T} = \hat x_{t|t} + J_t(\hat x_{t+1|T} - \hat x_{t+1|t}),\quad P_{t|T} = P_{t|t} + J_t(P_{t+1|T} - P_{t+1|t})J_t^\top
```

```latex
P_{t+1,t|T} = P_{t+1|T}\,J_t^\top
```

The last line is the lag-one cross-covariance the M-step needs.

**Log-likelihood.** From the filter's prediction errors e\_t = y\_t − C x̂\_{t|t-1}:

```latex
\log p(y_{1:T}) = -\tfrac12 \sum_t \left[\, m_t \log 2\pi + \log|S_t| + e_t^\top S_t^{-1} e_t \,\right]
```

**Outputs by task.**

| Task | Quantity | Uses future frames? |
| --- | --- | --- |
| T1 one-step prediction | C x̂\_{t+1\|t} compared with y\_{t+1}, visible regions | No (causal) |
| T2 hidden-region reconstruction, filtered | x̂\_{t\|t} on hidden rows | No (causal) |
| T2 hidden-region reconstruction, smoothed | x̂\_{t\|T} on hidden rows | Yes; not a forecast |

Filtered and smoothed results are always reported separately. A smoothed number must never be described as prediction.

**Numerics and speed.**

- Double precision throughout. Compute S\_t⁻¹ by Cholesky solves, not explicit inverses.
- When the hidden set is fixed across a fit (R3, and R1 with C = I), C\_t is constant, so the filter converges to a steady state within about 50 frames. Either run the exact time-varying recursion, or use the steady-state gain after a burn-in and verify against the exact version on three flies (maximum relative difference below 1e-4) before relying on it.
- A compiled scan (JAX `lax.scan` or numba) is recommended: a pure-Python loop over 40,000 frames per EM iteration is the compute bottleneck (section 15).

## 9. Learning: EM with the prior

EM alternates the smoother (E-step) with closed-form parameter updates (M-step) that pool expected sufficient statistics over all flies and segments; the prior enters the M-step as a one-line change, and with it the algorithm maximises the log posterior instead of the likelihood.

**E-step.** For every segment of every fly, run the filter and smoother (section 8) with the current (A, Q, R). Accumulate over all transitions t → t+1 inside segments, N of them in total:

```latex
S_{00} = \sum \left(\hat x_{t|T}\hat x_{t|T}^\top + P_{t|T}\right),\quad S_{11} = \sum \left(\hat x_{t+1|T}\hat x_{t+1|T}^\top + P_{t+1|T}\right),\quad S_{10} = \sum \left(\hat x_{t+1|T}\hat x_{t|T}^\top + P_{t+1,t|T}\right)
```

**M-step.**

```latex
A = (S_{10} + \lambda A_0)\,(S_{00} + \lambda I)^{-1}
```

```latex
Q = \frac{1}{N+n}\left[\, S_{11} - A S_{10}^\top - S_{10} A^\top + A S_{00} A^\top + \lambda\,(A - A_0)(A - A_0)^\top \right]
```

```latex
r_i = \frac{1}{N_i} \sum_{t:\, i \in V_t} \left[ (y_{t,i} - \hat x_{t,i|T})^2 + (P_{t|T})_{ii} \right]
```

N\_i counts the frames where region i is observed. Floor the eigenvalues of Q at 1e-6 and every r\_i at 1e-3. The A update follows from setting the derivative of the expected log posterior to zero; Q cancels because the likelihood and the matrix-normal prior both carry Q⁻¹. The Q update includes the prior's own log-determinant term, hence N + n in the denominator; with N near 38,000 the difference from N is negligible but keeps the monotonicity guarantee exact.

**Objective being maximised.**

```latex
\mathcal{L} = \log p(y_{1:T}\mid \theta) \;-\; \tfrac{n}{2}\log|Q| \;-\; \tfrac{\lambda}{2}\,\mathrm{tr}\!\left[Q^{-1}(A-A_0)(A-A_0)^\top\right] + \text{const}
```

**Algorithm.**

1. Initialise (section 10).
2. Repeat E-step, M-step, recompute ℒ.
3. Stop when the relative gain in ℒ per observation is below 1e-6 for 5 consecutive iterations, or at 300 iterations.
4. Fail loudly if ℒ ever decreases by more than 1e-8 relative (a bug, not noise).
5. Return θ = (A, Q, R), the iteration count, ρ(A) and a convergence flag.

**Bridge to the feasibility results.** If R is set to nearly 0 and C = I, the E-step returns the data themselves and the A update becomes exactly the ridge-with-prior formula used in `analysis_demo.py` (`fit_prior`). EM with observation noise is therefore a strict generalisation of the ridge VAR whose numbers (median R² 0.340) serve as the reference; unit test U8 in section 13 checks the equivalence numerically.

**Hidden regions in training (R3).** Rows of C\_t for hidden regions are absent in every fly. The E-step infers their states from visible regions and the dynamics; the M-step updates A, Q for the full 36×36 system as usual. No r\_i exists for a hidden region and none is needed. The initial covariance Σ₀ for the hidden block uses the mean visible variance on its diagonal and zero off-diagonals.

## 10. Hyperparameters, initialisation and ablations

Hyperparameters are chosen only from data that never include the fly or the hidden entries being scored, and every variant gets the same search budget.

**Selection by regime.**

| Regime | Tuning data | Objective | Search |
| --- | --- | --- | --- |
| R1 (19 full training flies, hidden at test) | Inner 3-fold cross-validation over the 19 training flies (folds of 7, 6, 6) | Inner mean one-step R² (T1) or hidden-region r with the same hidden fraction (T2) | Coarse grid (section 15); tier B only |
| R2 (m frames of the target fly) | The other 19 flies, each cut the same way (first m frames to fit, last 30% to score) | Mean one-step R² across those 19 | Full grid |
| R3 (regions never observed) | Nested hiding over a 3-fold split of the visible regions (9, 6, 3 regions per fold at 25, 50, 75% hidden; folds committed in seeds.json): hide each fold in turn together with the test set, fit, score on that fold | Mean hidden-region r over the 3 folds (fallback: folds 0 and 1 only, declared in seeds.json before any run) | Random search, 16 configurations per variant, EM capped at 30 iterations warm-started from M0, then a refit of at most 150 iterations at the chosen setting |

The feasibility run held out 5 flies for tuning and scored the other 15. This protocol tunes on the other 19 flies instead, so all 20 flies are scored.

**Initialisation (all variants identically).**

- A⁰: ridge VAR(1) (λ = 100) on the visible block; entries touching hidden regions set to A₀.
- rᵢ⁰: the region's flat high-frequency power (0.45–0.59 Hz, Welch, 128-frame windows) as a fraction of its variance, floored at 0.05.
- Σₓ = Cov(y) − diag(r⁰), eigenvalues floored at 1e-4; Q⁰ = Σₓ − A⁰ΣₓA⁰ᵀ, projected to positive semi-definite, then floored at 1e-4.
- Restarts: R3 production fits use a single ridge initialisation. The phase 4 sanity fits (M0 in R1) use three, ridge, A₀ and ridge with 10% multiplicative jitter on A, keeping the highest final ℒ, to check that the answer does not depend on initialisation.

**Seeds.** Global seed 675. Hidden-set seeds, shuffle seeds and restart jitter seeds are all derived from it by fixed rules in section 11 and stored in a `seeds.json` that is committed before any run.

**Pre-registered ablations (secondary, uncorrected, labelled exploratory).**

- **A1.** Cut for time: per-fly scalar noise scale s\_f, so R\_f = s\_f R (noise floors differ from 0.32 to 0.60 across flies). For a test fly, s\_f is estimated by EM on its visible regions with A and Q frozen.
- **A2.** Tier B if time: diagonal Q instead of full.
- **A3.** Cut for time: directed prior (transposed structural matrix, orientation confirmed first).
- **A4.** Tier B if time: CellCount only in place of the primary (the other four matrices are cut).
- **A5.** State-dependent noise or a non-Gaussian emission, only if rule N1 (section 14) fires.
- **A6.** Tier B if time: drop the six left-side regions and refit on the 30 right-side regions (section 14).

**Fit diagnostics logged for every fit.** Iterations to converge, final ℒ, ρ(A), condition number of S₀₀ + λI, smallest eigenvalue of Q, largest and smallest rᵢ, fraction of negative entries of A₀, restart that won. A fit with ρ(A) ≥ 0.98 or a convergence failure is kept in the results and flagged, never silently dropped.

## 11. Evaluation protocol

Every comparison is made fly by fly on identical splits, hidden sets and seeds, with the fly as the unit of analysis; three regimes are needed because the prior can only matter where the data cannot already determine A.

**Regimes.**

| Regime | Training data | Test | What it tests |
| --- | --- | --- | --- |
| R1 Population | The other 19 flies, all 36 regions observed | The held-out fly; T1 on all regions, T2 with a fraction hidden at test only | The deck's design. Secondary, run if time allows. Expected: priors do not matter (H5) |
| R2 Data-limited | The first m transitions of the target fly alone (m = 30, 60, 120, 250, 500, 1000 frames; 25 s to 14 min) | The same fly's last 30% of transitions | Whether a prior helps when A must be learned from little data (H4) |
| R3 Unseen regions | Every fly's visible entries; the hidden set H is missing in all flies (k = 0 only; a dose-response over k is dropped) | Hidden entries of all 20 flies | Whether the prior lets a model reconstruct regions it has never seen (H1–H3). Primary regime |

In R3 the fit is transductive on visible entries: a test fly's own visible regions are in the training data, but its hidden regions, and those of every other fly, never enter any fit, tuning step or standardisation of the model. This is imputation of regions nobody measured, not generalisation to a new animal; R1 covers the latter.

**Hidden sets.** Fractions p = 25%, 50%, 75% (9, 18, 27 regions). Ten random sets per fraction, drawn once from `np.random.default_rng([675, int(100*p), j])` and committed to `seeds.json`. Every model and every fly sees the same sets. Validation folds for R3 tuning (adopted Nov 23): for hidden set j of fraction p, take the sorted visible regions, permute them with default\_rng(\[675, 300, int(100\*p), j\]) and split into 3 folds with np.array\_split; all folds are committed to seeds.json. This replaces hiding the same fraction again, which at 75% hidden would have left 2 visible regions during tuning. Shuffles use `default_rng([675, 9, s])`, s = 0..19. Tier A uses sets 0–4 and shuffles 0–4 (reduced from 0–9 on Nov 23, recorded in the seeds.json reductions list); tier B uses all of them (section 15). Any further reduction is written to `seeds.json` before any results are seen.

**Tasks and metrics.**

| Task | Primary metric | Secondary metrics |
| --- | --- | --- |
| T1 one-step-ahead prediction of visible regions (filtered, causal) | R² = 1 − SSE / SS over the visible entries of the fly, where SSE sums (y at t+1 minus C times the filtered one-step prediction)² and SS sums (y at t+1)² | Held-out log-likelihood per observed entry (R1, R2); R² divided by the fly's approximate noise ceiling, 1 − floor |
| T2 hidden-region reconstruction (smoothed) | Pearson r between the smoothed state estimate and the observed hidden series, per hidden region, averaged over regions (Fisher-z mean) | Same with the filtered estimate; R² against standardised truth (R1 only); per-region r |

Pearson r is primary for T2 because hidden-region scale is unidentified in R3 (section 6) and, even in R1, the standardisation of a test fly's hidden regions uses their own mean and variance. The feasibility baselines for hidden regions (section 12) share that advantage, so their R² values are optimistic in absolute terms; comparisons between models stay fair because every model has it.

**Aggregation.** One number per fly, model, fraction and metric: the mean over the ten hidden sets (T2), or the single split (T1). The fly, not the frame or the hidden set, is the unit of analysis, since frames within a fly are strongly autocorrelated.

**Statistics.**

- Paired two-sided Wilcoxon signed-rank test across flies (exact when untied); effect = Hodges–Lehmann median paired difference with a 95% percentile bootstrap interval over flies (10,000 resamples).
- Primary family H1–H3 × 3 fractions = 9 tests, Holm-corrected at 0.05. Everything else is secondary and labelled exploratory, uncorrected.
- For M4, each fly's score is the mean over the shuffled priors of its tier (5 in tier A, 20 in tier B), and M4 is tuned on the mean over exactly those shuffles; also report the rank of the real wiring among them per fly.
- H3 is tested on the per-fly difference (M3 − M4 at 75% hidden) − (M3 − M4 at 25% hidden). H3 is kept as written. Simulation S1 (Nov 23) found the opposite sign when a true connectome effect is planted (−0.032), and that result is reported alongside the test.
- All results in the final report come from one frozen run of the committed code; no re-tuning after seeing test scores.

**Sensitivity analyses (all reported, none used to choose a headline).**

- Leave-one-recording-day-out instead of leave-one-fly-out for R1 (10 recording days), with a cluster bootstrap over days.
- Drop 2017-10-26\_1 (untrimmed steps), then drop the three lowest-R² flies from the feasibility run.
- Secondary structural matrix (`CellCount`) in place of `Connectivity`.
- Filtered versus smoothed estimates side by side for every T2 result.

## 12. Baselines and reference numbers

The feasibility run gives eight non-Kalman baselines with medians over flies; the Kalman models must beat the matching baseline, and the noise ceiling of about 0.53 on one-step R² (flat high-frequency floor 0.47 of variance on average) bounds what any model can score against noisy frames.

| ID | Baseline | Regime / task | Reference median (feasibility) |
| --- | --- | --- | --- |
| B1 | Persistence: next frame = this frame | T1 | R² −0.019 |
| B2 | Each region alone, pooled AR(1) (diagonal A) | T1, leave-one-fly-out | R² 0.274 |
| B3 | Ridge VAR(1), λ = 100, 19 training flies | T1, leave-one-fly-out | R² 0.340 (range 0.152–0.517); VAR(3) 0.350; same fly 70/30: 0.342 |
| B4 | Static Gaussian conditioning, covariance from the other 19 flies | T2, R1-like | R² 0.413 / 0.397 / 0.313 at 25 / 50 / 75% hidden |
| B5 | B4 plus ±2 frames of context, ridge | T2, R1-like | R² 0.428 / 0.421 / 0.329 |
| B6 | Wiring-weighted average of visible regions, one gain from other flies | T2, R3-like | R² 0.285 / 0.295 / 0.243 |
| B7 | B6 with shuffled wiring (5 shuffles) | T2, R3-like | R² 0.269 / 0.267 / 0.222 |
| B8 | Plain average of visible regions | T2, R3-like | R² 0.270 / 0.271 / 0.227 |

**Small-data learning curve (R2 proxy, ridge VAR with prior, 15 test flies, median R²).**

| Frames m | Flat (to zero) | Persistence | Uniform coupling | Connectome | Shuffled wiring |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.162 | 0.255 | 0.280 | 0.288 | 0.280 |
| 60 | 0.196 | 0.262 | 0.281 | 0.290 | 0.281 |
| 120 | 0.239 | 0.291 | 0.298 | 0.300 | 0.298 |
| 250 | 0.265 | 0.306 | 0.304 | 0.310 | 0.304 |
| 500 | 0.311 | 0.339 | 0.339 | 0.340 | 0.339 |
| 1000 | 0.343 | 0.347 | 0.347 | 0.347 | 0.347 |

**Structure–function reference (all 20 flies, 630 region pairs).** Spearman ρ between structure and mean FC: `Connectivity` 0.689, `CellCount` 0.682, `TBars` 0.714, `WeightedSynapseNumber` 0.671 (all label-permutation p = 0.0005); 0.662 for `Connectivity` after removing centroid distance; per-fly median 0.58 (0.43–0.68); FC split-half repeatability 0.934; symmetrised fitted A versus wiring ρ = 0.38.

**Caveats on these numbers.**

- They were computed after looking at all 20 flies, so they are exploratory; the final run recomputes B1–B8 with the final harness and metrics before any Kalman model is scored.
- B4–B8 are R² on hidden regions standardised with their own fly's statistics; the primary metric in section 11 is Pearson r, so recompute them as r.
- B6–B8 learn one scalar gain from the other flies' hidden-region data, so they are only R3-like; a strict R3 baseline fixes the gain from a rule that never touches hidden data (for example, by matching the variance of visible regions).
- B4 and B5 use cross-fly covariance of the hidden regions and have no counterpart in R3.

**Guard rails for the first real fit (not targets).** M0 in R1 for T1 should land near 0.34–0.40. Above the ceiling of about 0.53 on a typical fly means leakage; below about 0.30 means a bug. In R2 the ordering flat < persistence < connectome ≈ shuffled at small m would reproduce the feasibility pattern.

## 13. Verification before real data

The Kalman/EM code must pass every unit test (U1–U9) and every simulation check (S0–S3; S4 dropped Nov 23) below before it touches a real fly; each has a numeric pass criterion so "works" is not a judgement call.

**Unit tests (small synthetic systems, n = 3, T = 10 to 200).**

| ID | Test | Pass criterion |
| --- | --- | --- |
| U1 | Filtered and smoothed means and covariances against brute-force Gaussian conditioning on the stacked joint distribution | Max abs difference below 1e-9 |
| U2 | Log-likelihood from the filter against the log-density of the stacked observations under the joint Gaussian | Difference below 1e-8 |
| U3 | Missing observations: dropping rows of C\_t against marginalising the joint Gaussian | Same as U1, for random masks including all-missing frames |
| U4 | Lag-one cross-covariance P\_{t+1,t\|T} against brute force | Below 1e-9 |
| U5 | EM monotonicity: ℒ non-decreasing at every iteration, for κ = 0 and κ > 0, with and without missing data | No decrease beyond 1e-8 relative |
| U6 | Joseph-form covariance stays symmetric and positive semi-definite over 40,000 steps at n = 36 | Min eigenvalue ≥ −1e-10 |
| U7 | Standardisation invariance: rescaling a region's data by a constant and refitting rescales that state without changing r or R² | Differences below 1e-6 |
| U8 | Reduction: R = 1e-6·I, C = I, one EM iteration equals the closed-form ridge-with-prior fit (`fit_prior` in `analysis_demo.py`) | Max abs difference in A below 1e-5 |
| U9 | Steady-state filter and smoother (frozen gain and covariances after convergence) against the exact time-varying recursion: synthetic n = 36 (C = I and 50% hidden), then three real flies (2017-10-26\_1, 2017-11-08\_1 with both segments, 2018-10-20\_1), C = I and 50% hidden, parameters from the section 10 initialisation | Max relative difference below 1e-4 in every filtered, predicted and smoothed mean and covariance, the lag-one cross-covariance and the log-likelihood |

**Simulation checks (n = 36, structure from the real matrices).**

| ID | Setup | Expected outcome |
| --- | --- | --- |
| S0 | Gauge test: simulate from known (A, Q, R) with regions hidden in all flies; fit M0 and M1 | Rescaling or sign-flipping a hidden region leaves the log-likelihood unchanged (relative change below 1e-8); M0 and M1 leave hidden regions decoupled, so their hidden-region r is at chance (\|r\| below 0.05 for every fly) |
| S1 | True A₀-aligned: A\_true = A₀(d, a\_u, a\_w) plus noise of 20% of its norm; R1, R2, R3 | M3 beats M4 and M2 in R3 and in R2 at small m; the gap shrinks with data |
| S2 | Wiring-independent: A\_true generated with no relation to W̃ | M3 ≈ M4 ≈ M2 within bootstrap error; no false positive for H2 |
| S3 | Wrong-wiring: A\_true built from a permuted W̃ | M4 with the matching permutation wins; M3 ≤ M2 |
| S4 | Recovery: T = 40,000, all regions visible, no prior | Dropped Nov 23. The original criterion (relative error of A below 5%, R within 10%, ρ(A) within 0.02) cannot be met: from the section 10 initialisation, EM is still about 2,900 log-likelihood units below the true parameters after 300 iterations, and with the 47% noise floor even the maximum-likelihood fit is about 20% from the true A at T = 20,000. Model comparisons stay fair because every model has the same iteration budget; check M0 against the guard rails in phase 4 |

S1–S3 are the pipeline's positive and negative controls: S1 shows it can detect a real effect of the size in H2, S2 and S3 show it does not invent one. If S1 cannot detect an effect at the feasibility size (about 0.01–0.02), the real-data test of H2 has no power and the write-up must say so.

**Simulation results, Nov 23 (full run on the real Connectivity matrix).** S0, S1 and S3 passed. S2 failed under the first pass criterion, which used p < 0.05 alone: M3 − M4 had a median of 0.0045, significant but far below the 0.02 smallest effect of interest. The cause was M4 tuned on 2 of its 5 shuffles: it picked a\_w = 0.5 from lucky draws while M3 picked a\_w = 0 (identical to M2), so M4 scored 0.0045 below M2. Two fixes followed: G2 is judged by the section 2 decision rule, and M4 is tuned on exactly the shuffles it is scored on. A G2 rerun with both fixes is pending.

| Fraction hidden | M3 − M4 in S1 (median r) |
| --- | --- |
| 25% | 0.054 |
| 50% | 0.038 |
| 75% | 0.023 |

The gap shrinks as more regions are hidden, so the H3 quantity in S1 is −0.032, the opposite of H3's prediction; at 75% the planted effect is only just above 0.02. In R2 every M3 − M4 difference is at most 0.001 in R², far below the 0.01 threshold, so R2 is expected to be null.

**Real-data sanity checks (run once the tests pass, before any hypothesis test).**

- Log-posterior curves flatten and the fit reports ρ(A) < 1 for all 20 LOFO fits of M0.
- Learned observation-noise variances are consistent with the high-frequency floor: on standardised data the per-fly average of rᵢ should fall in 0.2–0.7 (floors were 0.32–0.60); a value far outside that range is a red flag.
- Fitted A off-diagonal correlates with the wiring at about ρ = 0.3–0.4 (feasibility 0.38).
- Filtered one-step R² of M0 is at least that of B3 (0.340): an LDS with a noise model should not predict worse than the ridge VAR it generalises.
- Smoothed hidden-region r is at least filtered hidden-region r for every model.

## 14. Threats to validity and known limitations

The largest risks are that the effect is small relative to 20-fly noise, that the data have already been looked at, and that most of any prior benefit may come from a generic brain-wide coupling mode rather than from wiring; each is addressed below by design or by explicit reporting.

| Threat | Why it matters | Mitigation or reporting |
| --- | --- | --- |
| Small effect, 20 units | Feasibility gaps are +0.01 to +0.02; 20 flies limits power | S1 power check; effect sizes with intervals; negative results reported as such |
| Data already explored | Cleaning choices, λ = 100 and the primary matrix were chosen having seen all flies | Freeze everything now; recompute baselines under the final harness; say so in the report |
| Generic coupling mode | A shuffled or uniform matrix supplies the same brain-wide mode (feasibility: M2 ≈ M4 ≈ M3 at m ≥ 120) | M2 and M4 are primary controls; do not interpret M3 vs M0 alone |
| Connectome from another animal | A single hemibrain fly from another lab; individual wiring differs | State as a limitation; the shuffle test asks only whether wiring carries shared information |
| Right-hemisphere bias | The hemibrain covers mostly the right central brain; six regions are left-side (ATL(L), BU(L), CRE(L), GOR(L), MBML(L), SMP(L)) | Sensitivity run A6 on the 30 right-side regions |
| No synaptic signs | The prior is positive-leaning; inhibitory coupling is invisible | Allow negative entries in A₀ via a\_w; let data overrule through κ |
| Region averaging and slow indicator | GCaMP6s at 1.2 Hz blurs timing; regions mix cell types | Treated as part of the observation model; interpret A as coupling of slow region activity, not synapses |
| Coupling ≠ causation | Alignment of A with wiring can reflect shared drive | Claim only predictive benefit, not mechanism |
| Hidden-scale leakage | Per-fly standardisation gives each fly's hidden regions unit variance | Pearson r primary; R² only as secondary, flagged optimistic |
| Recording-day clustering | Same-day flies may share preparation | Leave-one-day-out sensitivity, cluster bootstrap |
| Artifacts in single flies | 2017-10-26\_1 has untrimmed steps; three flies have trims | Sensitivity drops; trims frozen from the authors' code |
| Local optima and weak identification | The data are near low rank; EM can stall | Three restarts, fit diagnostics, ρ(A) check; slow EM convergence found in simulation (section 13, S4), so M0 is checked against the guard rails in phase 4 |
| Grid edge | A chosen value on the edge of a grid is not a verified optimum | If more than 20% of flies select an edge value, extend the grid once, record it, and rerun all variants |
| Hyperparameter budget unfairness | M4 has more draws than other variants | Same configuration budget and tuning protocol; M4 tuned on a permutation average |
| Comparison with Li et al. | A single fly, neuron-level model, different metrics and region sets | Compare qualitatively only unless metric definitions can be matched exactly |
| Compute | Tier A needs about 82,000 EM iterations and tier B about 230,000 (section 15) | Steady-state filter, compiled scan, pre-declared reduction of draws and shuffles |

**Structural-matrix multiplicity bias.** Turner's `Connectivity` sums per-neuron contributions without deduplicating neurons: a neuron with inputs and outputs in several regions fills several cells, and a neuron spanning several mapped hemibrain sub-ROIs is counted once per (sour, targ) pair within one cell. Large, multi-ROI regions are therefore weighted up. Row normalisation of the prior removes part of this, and the shuffled-wiring control is not affected by it because it permutes the same matrix. Report it as a limitation; the `CellCount` and `TBars` sensitivity runs show whether the conclusions depend on the choice of matrix.

**Rule N1: when to leave the linear-Gaussian family.** Do not extend the model unless, after M0 passes section 13, one of these holds: (a) M0's R1 one-step R² is more than 0.01 below B3 in more than half of the flies; (b) the standardised innovations have median absolute lag-1 autocorrelation above 0.1 or median excess kurtosis above 3. Feasibility residuals (median lag-1 −0.01, kurtosis 0.61) make this unlikely. If N1 fires, the extension (ablation A5) is reported separately and never replaces the primary analysis.

## 15. Work plan and deliverables

Work runs in six phases separated by five gates; a phase starts only when the previous gate's checks pass, and the single frozen run in phase 5 is the only source of reported results. Dates assume the talk on Tuesday Dec 1 and the report about a week later (decision 7).

&#91;embedded content: work plan · 6 phases, 5 gates\]

Each gate is a checklist, not a date: if a gate slips, what it requires does not change.

**Compute budget (estimates, not measurements; time one E-step on Nov 27).** R3 at k = 0 dominates. Tier A (for the Dec 1 talk) uses 5 hidden sets per fraction, 5 shuffles, 16 random-search configurations per variant, 30 EM iterations per tuning fit, one initialisation and at most 150 iterations per final fit. For M0–M4 that is about 5,500 EM iterations per fraction-and-set, so about 82,000 in total over 3 fractions × 5 sets. Serially that is about 1.1 hours at 0.05 s per iteration (compiled steady-state filter) or about 11 hours at 0.5 s (pure NumPy). On the Ryzen 7 9800X3D (8 cores) with 8 worker processes, one BLAS thread each, expect roughly 10 minutes to 2 hours. Revised Nov 23 for the 3-fold validation split and for M4 tuned on all 5 of its shuffles: an upper bound of about 13,000 EM iterations per fraction-and-set (over half of it M4's tuning), about 200,000 in total, or roughly 4 hours on 8 workers at the measured 0.56 s per E-step; early stopping will cut this. If that does not fit, the declared fallback is folds 0 and 1 only. Tuning M4 on all 20 shuffles in tier B multiplies its cost again; re-estimate before tier B. Tier B (for the report) uses all 10 sets and 20 shuffles and adds M5, R1 and R2: about 230,000 iterations in total (tier A fits can be reused), roughly 25 minutes to 5 hours on 8 workers. R1 is tier B only; its inner cross-validation is the largest part of that cost. No GPU is used.

**Tiers and reduction knobs (fixed now, written to `seeds.json` before any run).** All 10 hidden sets per fraction and all 20 shuffle permutations are drawn and committed up front, so nothing is re-drawn later. Tier A (talk) uses sets 0–4 and shuffles 0–4 (reduced from 0–9 on Nov 23; M4 tuned and scored on the same 5), models M0–M4, R3 at k = 0. Tier B (report) adds sets 5–9, shuffles 5–19, M5, and R1 and R2 as time allows. Tier A results are labelled preliminary; the confirmatory tests of section 11 are run once on the complete tier B set, and any change from tier A is reported. Settings common to both tiers: 16 random-search configurations per variant, 30 EM iterations per tuning fit, one ridge initialisation, at most 150 iterations per final fit, warm-started from M0. R1 and R2 use a coarse grid (κ in 0, 0.01, 1; d in 0, 0.5; a\_u in 0, 0.25; a\_w in 0, 1) and a 3-fold inner split. Cut for time: per-fly noise scale (A1), directed prior (A3), the FC-ceiling model (M6). The report states that the protocol is reduced.

**Deliverables.**

- Code: separate modules for the filter and smoother, EM, prior construction, and the evaluation harness; the test suite from section 13.
- `seeds.json`, file hashes, and one JSON record per fit (configuration, code version, diagnostics from section 10).
- Results tables as CSV, one row per fly × model × regime × fraction × metric.
- Figures: per-fly paired differences (M3 vs M4, M3 vs M2), hidden-region r versus fraction hidden, learning curve over m, per-region reconstruction r against wiring to visible neighbours.
- Final report and slides stating which hypotheses were supported, with intervals, and which sensitivity runs changed the conclusion.

**Cut lines, decided now so they are not argued under deadline pressure.**

| By | If this is not done | Then |
| --- | --- | --- |
| Tue Nov 24 (G1) | Filter and smoother tests not passing | Use the exact time-varying filter (no steady-state speed-up) and run 3 hidden sets, not 5 |
| Thu Nov 26 (G2) | EM or the simulation checks failing | Present the simulation results (S1–S3) plus the existing ridge results (R2-style learning curve, baselines B4–B8) as preliminary; EM results follow in the report |
| Sat Nov 28 | Tier A run not started | Run 3 hidden sets and 5 shuffles at 50% hidden only; H3 is then untestable and the talk says so |
| Mon Nov 30 | Any result missing | Build the slides from what exists; label the rest as in progress |

Nov 26 is US Thanksgiving; the schedule assumes you can work that day. S4 was dropped on Nov 23 (section 13) and U9 passed the same day, so neither cut is needed.

**Presentation, Tue Dec 1.** Length is 6-7 minutes speaking time and at most 10 slides: (1) question and what changed since the pitch; (2) model and prior, with the dimension answer; (3) the controls M0–M4; (4) protocol, R3 and the metric; (5) simulation sanity checks; (6–7) R3 results, correlation against fraction hidden and per-fly paired differences with intervals; (8) limitations; (9) next steps, tier B and R1.

**Report, about Tue Dec 8 (assumed; "the week after"), 10 pages hard maximum including figures and references.**

| Part | Pages |
| --- | --- |
| Abstract and introduction | 1.0 |
| Related work | 0.5 |
| Dataset and connectome | 0.75 |
| Model, prior and EM | 1.75 |
| Protocol | 1.0 |
| Results (4 figures, 1 table) | 3.0 |
| Discussion and limitations | 1.0 |
| References | 0.5 |
| Total, leaving 0.5 page of margin | 9.5 |

Figures: F1 model and prior schematic; F2 simulation checks S1–S3 (detects a planted effect, no false positive); F3 hidden-region correlation against fraction hidden for M0–M4; F4 per-fly paired differences M3 − M4 and M3 − M2 with intervals. The R1 null result goes in a table row. Full grids, the hypothesis table and fit diagnostics move to the code README, not the report. Check the compiled PDF page count before submission.

## 16. Open decisions

All ten decisions now have a setting, chosen for the Dec 1 talk and the 10-page report limit; decisions 2 and 4 change what the experiments mean, so change those first if you disagree.

| # | Decision | Setting | Why |
| --- | --- | --- | --- |
| 1 | Primary regime | Confirmed: R3 at k = 0 primary. R1 secondary (tier B, if time allows) to show the prior has no effect on that kind of prediction. R2 optional, tier B | The deck's design (R1) is expected to show no prior effect because 19 full flies pin down A; R3 is where the prior can act |
| 2 | Primary hidden-region metric | Pearson r; R² secondary | Hidden-region scale is unidentified in R3 and leaks through standardisation in R1; the deck listed R² |
| 3 | Compute plan | CPU only: numba, steady-state filter, 8 worker processes with one BLAS thread each; no GPU; tiers and knobs as in section 15 | Nine days leave no room for GPU engineering; the 8-core CPU is enough if the timing on Nov 27 matches the estimate |
| 4 | Primary structural matrix | `Connectivity`; definition and source confirmed against Turner's code on Nov 23 (section 6) | The headline prior depends on it; `TBars` scored highest on SC–FC, so choosing by that would be a forking path |
| 5 | Directed prior | Cut; symmetrised only (ablation A3 dropped) | A wrong transpose would silently invert the prior, and there is no time to verify orientation |
| 6 | Extra controls and gap handling | M5 (distance) in tier B only; M6 (FC ceiling) cut because it is undefined in R3; the filter restarts at segment boundaries | Keeps tier A to M0–M4; restart versus gap-as-missing is a modelling choice with little expected effect |
| 7 | Deadline and report format | Talk Tue Dec 1; 10-page report (hard maximum, figures and references included) the following week, assumed Tue Dec 8; code submitted separately; talk length assumed about 10 minutes | Sets the tiers, cut lines and page budget in section 15; confirm the report date and talk length |
| 8 | Comparison with Li et al. | One qualitative sentence in the discussion; no numeric comparison | Their model is neuron-level on one fly with different metrics, and there is no page budget for matching them |
| 9 | Per-fly noise scale | Cut (ablation A1 dropped) | Not needed for the primary comparison; first thing to add back if tier B finishes early |
| 10 | Changes to tell the professor | One slide at the talk, "what changed since the pitch", listing the five changes below | The pitch no longer matches the protocol in five places |

**What this spec changes relative to the pitch deck.**

- The shuffled control is a joint relabelling of regions on rows and columns of the row-normalised prior.
- Two controls are added: persistence-only (M1) and uniform coupling (M2), so the wiring is compared with generic coupling and not only with a shuffle.
- A third regime (R3, regions unseen in training) is added, because R1 alone cannot test the prior.
- Hidden-region reconstruction is scored primarily by Pearson r, with R² secondary.
- The hypothesis about benefit growing with hidden fraction (H3) is tested on the M3 − M4 gap, not on M3 against a flat prior.
