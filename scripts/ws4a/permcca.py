"""
permcca.py -- permutation inference for CCA, following Winkler, Renaud, Smith &
Nichols (2020), "Permutation inference for canonical correlation analysis",
NeuroImage 220:117065 (PMC7573815).  Pure numpy + scipy.

WHAT THE SCHEME IS
------------------
1.  Canonical correlations from orthonormal bases (Bjorck-Golub):
        r = svdvals(orth(X)' orth(Y)).
2.  Statistic for the k-th hypothesis H_k^0 = "canonical correlations k..K are all zero":
        Wilks   lambda_k = -sum_{i>=k} log(1 - r_i^2)     <- signal spread over variates
        Roy     theta_k  = r_k^2                          <- signal in one variate
3.  STEPWISE CONDITIONING.  To test H_k^0 the first k-1 canonical WEIGHT vectors are
    projected out of both blocks (feature space, not sample space), and CCA is refit in
    the complement.  The statistic compared against the null is then the LEADING
    statistic of that reduced problem.  Omitting this step is the main error in naive
    permutation: variance already explained by earlier variates is still in the data,
    so the k-th permuted correlation is not a null draw for the k-th observed one.
4.  ONE SIDE is permuted.  The identity permutation is included as permutation #1, so
        p = (1 + #{null >= obs}) / (1 + n_perm),
    which makes the test exact and bounds p away from 0.
5.  MULTIPLICITY across canonical variates is handled by closed testing, implemented as
    the running maximum
        p_k^FWER = max(p_1, ..., p_k).
    This controls FWER in the strong sense and forces monotonicity in k.  Raw
    per-variate p-values are NOT monotone in k and admit the inadmissible result
    "r_2 significant while the larger r_1 is not".
6.  NUISANCE covariates use Huh-Jhun: residuals are rotated by a semi-orthogonal basis
    of the residual-forming matrix into the (n - rank(Z))-dimensional space where they
    are exchangeable.  Permuting raw residuals in n dimensions is not valid -- Winkler
    measure 83.9% false positives at a nominal 5% for that shortcut.

SCOPE
-----
Give this function blocks that are ALREADY dimension-reduced (PCA scores, a handful of
components).  It works on the reduced feature space directly, so cost is O(K * n_perm)
tiny SVDs; feeding it 41,780 raw columns is both meaningless (r == 1 by construction
when p + q >= n) and slow.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd, null_space


# --------------------------------------------------------------------------- #
def _center(A):
    return A - A.mean(axis=0, keepdims=True)


def _orth(A, rcond=1e-10):
    """Orthonormal basis of the column space of A, numerically rank-truncated."""
    U, s, _ = svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return U[:, :0]
    return U[:, s > s[0] * rcond]


def canonical_correlations(X, Y, center=True):
    """Sample canonical correlations of X (n,p) and Y (n,q), descending."""
    if center:
        X, Y = _center(X), _center(Y)
    Qx, Qy = _orth(X), _orth(Y)
    if Qx.shape[1] == 0 or Qy.shape[1] == 0:
        return np.zeros(0)
    return np.clip(svd(Qx.T @ Qy, compute_uv=False), 0.0, 1.0)


def cca_full(X, Y, center=True):
    """Returns (r, A, B, U, V): correlations, feature-space weights, sample scores."""
    if center:
        X, Y = _center(X), _center(Y)
    Ux, sx, Vxt = svd(X, full_matrices=False)
    Uy, sy, Vyt = svd(Y, full_matrices=False)
    kx = sx > sx[0] * 1e-10 if sx.size else np.zeros(0, bool)
    ky = sy > sy[0] * 1e-10 if sy.size else np.zeros(0, bool)
    Qx, Qy = Ux[:, kx], Uy[:, ky]
    Wx, s, Wyt = svd(Qx.T @ Qy, full_matrices=False)
    A = Vxt[kx].T @ (Wx / sx[kx][:, None])      # (p, K) weights, X @ A = U
    B = Vyt[ky].T @ (Wyt.T / sy[ky][:, None])   # (q, K)
    return np.clip(s, 0, 1), A, B, Qx @ Wx, Qy @ Wyt.T


def _stats_from_r(r, kind):
    r2 = np.clip(np.asarray(r) ** 2, 0.0, 1.0 - 1e-12)
    if kind == "wilks":
        return np.cumsum((-np.log1p(-r2))[::-1])[::-1]
    if kind == "roy":
        return r2
    raise ValueError("stat must be 'wilks' or 'roy'")


def huh_jhun_basis(Z):
    """Semi-orthogonal Q (n, n-rank(Z)) with Q Q' = I - Z Z^+  and  Q'Q = I."""
    n = Z.shape[0]
    Qz = _orth(Z)
    w, V = np.linalg.eigh(np.eye(n) - Qz @ Qz.T)
    return V[:, w > 0.5]


# --------------------------------------------------------------------------- #
def permcca(X, Y, n_perm=999, stat="wilks", n_components=None,
            Z=None, W=None, seed=0):
    """Permutation test for every canonical correlation, with FWER control.

    Returns dict(r, p_unc, p_fwer, stat_obs, null).  Report p_fwer.
    """
    rng = np.random.default_rng(seed)
    X = _center(np.asarray(X, float))
    Y = _center(np.asarray(Y, float))
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")

    # ---- nuisance covariates: Huh-Jhun ------------------------------------ #
    if Z is not None or W is not None:
        Zc = _center(np.asarray(Z, float)) if Z is not None else None
        Wc = _center(np.asarray(W, float)) if W is not None else None
        if Wc is None:
            Wc = Zc
        if Zc is None:
            Zc = Wc
        if Zc.shape != Wc.shape or not np.allclose(Zc, Wc):
            raise NotImplementedError(
                "bipartial CCA (different nuisance on each side) is not implemented; "
                "pass the same Z for both sides")
        Q = huh_jhun_basis(Zc)
        Qz = _orth(Zc)
        X = Q.T @ (X - Qz @ (Qz.T @ X))
        Y = Q.T @ (Y - Qz @ (Qz.T @ Y))

    n = X.shape[0]
    r_obs, A, B, _, _ = cca_full(X, Y, center=False)
    K = r_obs.size if n_components is None else min(int(n_components), r_obs.size)
    if K == 0:
        raise ValueError("no canonical variates: a block has rank 0")
    stat_obs = _stats_from_r(r_obs, stat)[:K]

    # Pre-build, per k, the orthonormal bases of the two blocks after projecting out
    # the first k-1 canonical weight directions.  A row permutation of a matrix with
    # orthonormal columns still has orthonormal columns, so the permutation loop is
    # just one small matmul + svdvals.
    bases = []
    for k in range(K):
        if k == 0:
            Xk, Yk = X, Y
        else:
            Na = null_space(A[:, :k].T)
            Nb = null_space(B[:, :k].T)
            if Na.shape[1] == 0 or Nb.shape[1] == 0:
                bases.append(None)
                continue
            Xk, Yk = X @ Na, Y @ Nb
        bases.append((_orth(Xk), _orth(Yk)))

    null = np.full((n_perm, K), -np.inf)
    for j in range(n_perm):
        idx = rng.permutation(n)
        for k in range(K):
            if bases[k] is None:
                null[j, k] = np.inf          # degenerate -> never reject
                continue
            Qx, Qy = bases[k]
            rp = np.clip(svd(Qx.T @ Qy[idx], compute_uv=False), 0, 1)
            null[j, k] = _stats_from_r(rp, stat)[0]

    p_unc = (1.0 + (null >= stat_obs[None, :]).sum(0)) / (n_perm + 1.0)
    p_fwer = np.maximum.accumulate(p_unc)
    return dict(r=r_obs[:K], p_unc=p_unc, p_fwer=p_fwer,
                stat_obs=stat_obs, null=null)


# --------------------------------------------------------------------------- #
def pc_budget(n, k, n_sim=2000, seed=0):
    """Null distribution of r1 for two INDEPENDENT Gaussian blocks of k dims each."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_sim)
    for i in range(n_sim):
        out[i] = canonical_correlations(rng.standard_normal((n, k)),
                                        rng.standard_normal((n, k)))[0]
    return dict(k=k, mean=out.mean(), q95=np.quantile(out, .95),
                q99=np.quantile(out, .99), max=out.max())


def cv_canonical_correlation(X, Y, k_pc, n_splits=5, n_repeats=10, seed=0):
    """Out-of-sample r1: PCA and CCA fitted on the training fold only.

    In-sample r1 near 1 with CV r1 near 0 is the signature of a fit that is all
    estimator and no data."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    scores = []
    for rep in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed + rep).split(X):
            px = PCA(n_components=k_pc, svd_solver='full').fit(X[tr])
            py = PCA(n_components=k_pc, svd_solver='full').fit(Y[tr])
            Xtr, Ytr = px.transform(X[tr]), py.transform(Y[tr])
            mx, my = Xtr.mean(0), Ytr.mean(0)
            _, Wx, Wy, _, _ = cca_full(Xtr - mx, Ytr - my, center=False)
            u = (px.transform(X[te]) - mx) @ Wx[:, 0]
            v = (py.transform(Y[te]) - my) @ Wy[:, 0]
            if u.std() > 1e-12 and v.std() > 1e-12:
                scores.append(np.corrcoef(u, v)[0, 1])
    return np.asarray(scores)
