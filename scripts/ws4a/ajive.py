"""
AJIVE -- Angle-based Joint and Individual Variation Explained.

Reference implementation in numpy/scipy only.

  Feng, Q., Jiang, M., Hannig, J., & Marron, J.S. (2018).
  "Angle-based joint and individual variation explained."
  Journal of Multivariate Analysis, 166, 241-265.

Algorithm (K blocks, all with the same n rows):

  Step 1  Per block: column-center, take a rank-(r_i+1) SVD.  Keep the first
          r_i left singular vectors U_i (an orthonormal basis for that block's
          estimated signal ROW space).  Record a singular-value threshold
          tau_i = (d_{r_i} + d_{r_i+1}) / 2.

  Step 2  Stack M = [U_1 | ... | U_K]  (n x sum r_i) and take its SVD.
          The squared singular values of M live in [0, K]; s^2 = K means a
          direction shared perfectly by every block.  For K = 2,
          s_j^2 = 1 + cos(theta_j) with theta_j the j-th principal angle
          between the two signal row spaces -- this is the "angle-based" part.

  Step 3  Joint rank = #{ j : s_j^2 > tau_joint }, where tau_joint is the max
          of two data-driven bounds:
            (a) random-direction bound -- the 95th percentile of the largest
                squared singular value obtained by stacking K INDEPENDENT
                random orthonormal bases of the same ranks.  This is the null:
                how much apparent sharing arises from unrelated subspaces at
                this n and these ranks.  It is the binding bound when n is
                small relative to sum r_i.
            (b) Wedin bound -- the 5th percentile of K - sum_i sin^2(angle
                between estimated and true signal subspace of block i),
                each angle bounded by a sample-project resampling.

  Step 4  Identifiability filter: drop any joint direction whose projection
          into some block has norm below that block's tau_i.

  Step 5  Per block: J_i = U_J U_J^T X_i ; X_i^perp = X_i - J_i ;
          individual rank = #{ singular values of X_i^perp > tau_i } ;
          I_i = that truncation ; E_i = X_i - J_i - I_i.
"""
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import svd as dense_svd


# --------------------------------------------------------------------------
def _svd(X, rank=None):
    """Truncated SVD returning (U, d, V) with V as (p, rank), d decreasing."""
    X = np.asarray(X, dtype=float)
    if rank is None or rank >= min(X.shape):
        U, d, Vt = dense_svd(X, full_matrices=False)
        if rank is not None:
            U, d, Vt = U[:, :rank], d[:rank], Vt[:rank]
        return U, d, Vt.T
    U, d, Vt = svds(X, k=rank)
    order = np.argsort(-d)
    return U[:, order], d[order], Vt.T[:, order]


def _randdir_cutoff(n, signal_ranks, n_samples=1000, pct=95, rng=None):
    """Step 3(a). Null distribution of the top squared singular value of
    [Q_1 | ... | Q_K] for INDEPENDENT random orthonormal bases Q_i."""
    rng = np.random.default_rng(rng)
    out = np.empty(n_samples)
    for s in range(n_samples):
        Q = [np.linalg.qr(rng.standard_normal((n, r)))[0] for r in signal_ranks]
        out[s] = _svd(np.hstack(Q), rank=1)[1][0] ** 2
    return np.percentile(out, pct), out


def _project_norms(X, basis, n_samples, rng):
    """Sample-project step of the Wedin bound (Feng et al. 2018, sec. 3.2).

    Draw an isotropic (m, r) block, project it onto the orthogonal complement
    of `basis`, orthonormalise it, and take the operator 2-norm of X @ vecs.
    This estimates how large X looks along directions the signal basis misses,
    i.e. the numerator of the perturbation angle bound.

    Requires X.shape[1] == basis.shape[0] == m.
    """
    rng = np.random.default_rng(rng)
    m, r = basis.shape
    out = np.empty(n_samples)
    for s in range(n_samples):
        V = rng.standard_normal((m, r))
        V -= basis @ (basis.T @ V)            # project off the signal space
        V, _ = np.linalg.qr(V)                # orthonormalise
        out[s] = np.linalg.norm(X @ V, ord=2)  # operator 2-norm
    return out


def _wedin_cutoff(Xs, Us, ds, Vs, ranks, n_samples=1000, pct=5, rng=None):
    """Step 3(b). K - sum_i (resampled sin angle)^2, lower percentile."""
    rng = np.random.default_rng(rng)
    K = len(Xs)
    per_block = []
    for X, U, d, V, r in zip(Xs, Us, ds, Vs, ranks):
        un = _project_norms(X.T, U[:, :r], n_samples, rng)   # row-space side
        vn = _project_norms(X,   V[:, :r], n_samples, rng)   # col-space side
        smin = d[r - 1]
        per_block.append(np.minimum(np.maximum(un, vn) / smin, 1.0))
    samples = K - np.sum(np.square(per_block), axis=0)
    return np.percentile(samples, pct), samples


# --------------------------------------------------------------------------
class AJIVE:
    """Angle-based JIVE. Feng et al. (2018)."""

    def __init__(self, init_signal_ranks, joint_rank=None,
                 n_randdir_samples=1000, n_wedin_samples=1000,
                 randdir_percentile=95, wedin_percentile=5,
                 reconsider_joint_components=True, random_state=None):
        self.init_signal_ranks = list(init_signal_ranks)
        self.joint_rank = joint_rank
        self.n_randdir_samples = n_randdir_samples
        self.n_wedin_samples = n_wedin_samples
        self.randdir_percentile = randdir_percentile
        self.wedin_percentile = wedin_percentile
        self.reconsider_joint_components = reconsider_joint_components
        self.random_state = random_state

    def fit(self, Xs):
        rng = np.random.default_rng(self.random_state)
        Xs = [np.asarray(X, dtype=float) for X in Xs]
        K, n = len(Xs), Xs[0].shape[0]
        assert all(X.shape[0] == n for X in Xs), "blocks must share rows"

        # ---- Step 1: centre + per-block signal SVD -----------------------
        self.means_ = [X.mean(axis=0) for X in Xs]
        Xc = [X - m for X, m in zip(Xs, self.means_)]

        Us, ds, Vs, self.sv_thresholds_ = [], [], [], []
        ranks = []
        for X, r in zip(Xc, self.init_signal_ranks):
            r = min(r, min(X.shape) - 1)
            ranks.append(r)
            U, d, V = _svd(X, r + 1)
            self.sv_thresholds_.append((d[r - 1] + d[r]) / 2.0)
            Us.append(U[:, :r]); ds.append(d[:r]); Vs.append(V[:, :r])
        self.init_signal_ranks_ = ranks

        # ---- Step 2: SVD of the stacked bases -> principal angles --------
        M = np.hstack(Us)
        Uj_all, sj_all, _ = _svd(M)
        self.all_joint_svals_ = sj_all
        self.joint_svalsq_ = sj_all ** 2          # in [0, K]

        # ---- Step 3: threshold -------------------------------------------
        if self.joint_rank is None:
            self.rand_cutoff_, self.random_sv_samples_ = _randdir_cutoff(
                n, ranks, self.n_randdir_samples, self.randdir_percentile, rng)
            self.wedin_cutoff_, self.wedin_sv_samples_ = _wedin_cutoff(
                Xc, Us, ds, Vs, ranks, self.n_wedin_samples,
                self.wedin_percentile, rng)
            self.svalsq_cutoff_ = max(self.rand_cutoff_, self.wedin_cutoff_)
            rank_j = int(np.sum(self.joint_svalsq_ > self.svalsq_cutoff_))
        else:
            self.rand_cutoff_ = self.wedin_cutoff_ = self.svalsq_cutoff_ = None
            rank_j = self.joint_rank
        self.joint_rank_wedin_est_ = rank_j

        # ---- Step 4: identifiability filter ------------------------------
        keep = list(range(rank_j))
        if self.reconsider_joint_components and rank_j > 0:
            keep = [j for j in range(rank_j)
                    if all(np.linalg.norm(X.T @ Uj_all[:, j]) >= tau
                           for X, tau in zip(Xc, self.sv_thresholds_))]
        self.joint_rank_ = len(keep)
        self.joint_scores_ = Uj_all[:, keep] if keep else np.zeros((n, 0))

        # ---- Step 5: per-block joint / individual / noise -----------------
        self.joint_mats_, self.individual_mats_, self.noise_mats_ = [], [], []
        self.individual_scores_, self.individual_svals_ = [], []
        self.individual_loadings_, self.individual_ranks_ = [], []
        self.joint_loadings_ = []

        for X, tau in zip(Xc, self.sv_thresholds_):
            if self.joint_rank_ > 0:
                J = self.joint_scores_ @ (self.joint_scores_.T @ X)
                self.joint_loadings_.append(_svd(J, self.joint_rank_)[2])
            else:
                J = np.zeros_like(X)
                self.joint_loadings_.append(np.zeros((X.shape[1], 0)))
            Xp = X - J

            maxr = min(Xp.shape) - self.joint_rank_
            U, d, V = _svd(Xp, maxr)
            r_ind = int(np.sum(d > tau))
            I = (U[:, :r_ind] * d[:r_ind]) @ V[:, :r_ind].T if r_ind else np.zeros_like(X)

            self.joint_mats_.append(J)
            self.individual_mats_.append(I)
            self.noise_mats_.append(X - J - I)
            self.individual_ranks_.append(r_ind)
            self.individual_scores_.append(U[:, :r_ind])
            self.individual_svals_.append(d[:r_ind])
            self.individual_loadings_.append(V[:, :r_ind])
        return self

    def fit_transform(self, Xs):
        return self.fit(Xs).joint_scores_


# ==========================================================================
#  Practical helpers for the n ~ 100, p ~ 40k regime
# ==========================================================================
def pca_reduce(X, q=None, return_loadings=False):
    """Rotate a block into its own row space. LOSSLESS for AJIVE.

    AJIVE only ever touches each block through its left singular vectors and
    singular values, and those are unchanged by an orthonormal rotation of the
    columns.  With n << p, q = n-1 keeps everything and makes the Wedin
    resampling ~7x faster, because that step draws (p, r) Gaussian blocks.

    Returns scores (n, q); with return_loadings, also W (p, q) so a loading
    computed in reduced space maps back as  W @ loading_reduced.
    """
    Xc = np.asarray(X, dtype=float)
    Xc = Xc - Xc.mean(axis=0)
    q = min(Xc.shape[0] - 1, Xc.shape[1]) if q is None else q
    U, d, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = U[:, :q] * d[:q]
    return (scores, Vt[:q].T) if return_loadings else scores


def permutation_null_joint_rank(Xs, init_signal_ranks, n_perm=20,
                                random_state=0, n_jobs=1, inner_threads=4, **kw):
    """DESTRUCTIVE CONTROL: break the row correspondence and refit.

    Shuffles the rows of every block after the first, which destroys the
    compound-level pairing while preserving each block's own covariance
    structure entirely.  Any joint rank that survives this is an artefact of
    the block geometry, not of cross-modal correspondence.

    Returns (observed_rank, array_of_null_ranks).
    """
    obs = AJIVE(init_signal_ranks, random_state=random_state, **kw).fit(Xs).joint_rank_

    # Each replicate gets an INDEPENDENT seeded stream rather than successive draws
    # from one generator, so the result does not depend on completion order and the
    # parallel and serial paths agree exactly.
    ss = np.random.SeedSequence(random_state)
    seeds = [int(c.generate_state(1)[0]) for c in ss.spawn(n_perm)]

    def _one(sd):
        r = np.random.default_rng(sd)
        Xp = [Xs[0]] + [X[r.permutation(X.shape[0])] for X in Xs[1:]]
        return AJIVE(init_signal_ranks, random_state=random_state, **kw).fit(Xp).joint_rank_

    if n_jobs == 1 or n_perm == 1:
        null = [_one(sd) for sd in seeds]
    else:
        from joblib import Parallel, delayed, parallel_config
        # Each refit is BLAS-heavy, so give workers a few threads rather than one.
        with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
            null = Parallel(n_jobs=min(n_jobs, n_perm))(delayed(_one)(sd) for sd in seeds)
    return obs, np.array(null)
