"""
matrix_agreement.py -- Mantel, RV and Procrustes agreement between two blocks
measured on the same objects, with permutation nulls that stay valid at n ~ 100
and p up to 41,780.

Dependencies: numpy, scipy, statsmodels.  Nothing else.  Tested on
Python 3.12.11 / numpy 2.5.2 / pandas 3.0.5 / scipy 1.18.1 / statsmodels 0.15.0.

THE ONE THING TO REMEMBER
-------------------------
Every raw agreement statistic here inflates toward its maximum as p grows with
n fixed.  Measured at n=94 on INDEPENDENT random blocks, where the true value
is exactly 0:

    p=615 vs q=41780     plain RV = 0.930      Procrustes r = 0.980
    p=41780 vs q=41780   plain RV = 0.998      Procrustes r = 0.9995

Never report a raw RV or a raw Procrustes r.  Report the adjusted value
(`rv_adj`, `r_adj`), which measured -0.0002 +- 0.0034 and 0.000 on the same
null data, and always report the permutation p-value beside it.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

__all__ = [
    "clean_block", "scale_block", "distance_matrix", "distance_concentration",
    "mantel_test", "rv_coefficients", "rv_permutation_test",
    "procrustes_stat", "protest", "shared_permutations", "westfall_young",
]


# --------------------------------------------------------------------------- #
# block preparation
# --------------------------------------------------------------------------- #

def clean_block(X, max_abs=1e6, verbose=False, name=""):
    """Drop columns that would otherwise decide the answer on their own:
    non-finite, blown-up, and zero-variance.  Returns (X_clean, keep_mask).

    The blow-up guard is not hypothetical.  On the real a549 morphology block,
    40 of 615 features reach 1.5e19 (the pycytominer mad_robustize zero-MAD /
    epsilon=1e-18 artefact) and hold 100.0000000000% of the block's total sum
    of squares.  Un-scaled, every Gram matrix and every distance on that block
    is a function of 40 broken columns and nothing else.
    """
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    keep = finite.all(axis=0)
    n_nonfinite = int((~keep).sum())

    amax = np.zeros(X.shape[1])
    amax[keep] = np.abs(X[:, keep]).max(axis=0)
    blown = keep & (amax > max_abs)
    keep &= ~blown

    sd = np.zeros(X.shape[1])
    sd[keep] = X[:, keep].std(axis=0)
    zerovar = keep & (sd <= 0)
    keep &= ~zerovar

    if verbose:
        print(f"clean_block[{name}]: {X.shape[1]} -> {int(keep.sum())} columns "
              f"(non-finite {n_nonfinite}, |x|>{max_abs:.0e} {int(blown.sum())}, "
              f"zero-variance {int(zerovar.sum())})")
    return X[:, keep], keep


def scale_block(X, how="zscore"):
    """'zscore' (default) | 'center' | 'none'.

    z-scoring is what makes 615 morphology features and 41,780 expression
    values commensurable, and it is also what stops any single feature from
    dominating.  Use 'center' only when the native variance profile is itself
    the signal, and never on a block that has not been through clean_block.
    """
    X = np.asarray(X, float)
    if how == "none":
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    if how == "center":
        return Xc
    if how == "zscore":
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        return Xc / sd
    raise ValueError(f"unknown scaling {how!r}")


def distance_matrix(X, metric="euclidean"):
    """Square distance matrix.  metric: any scipy pdist metric, plus
    'spearman_corr' (rank each object's feature vector, then correlation).

    For binary fingerprints use 'jaccard' -- it is Tanimoto, and unlike cosine
    or correlation it survives an all-zero fingerprint row (measured: one zero
    row yields 93 NaNs under cosine and correlation, 0 under jaccard).
    """
    X = np.asarray(X, float)
    if metric == "spearman_corr":
        X = rankdata(X, axis=1)
        metric = "correlation"
    D = squareform(pdist(X, metric=metric))
    np.fill_diagonal(D, 0.0)          # scikit-bio rejects a non-hollow diagonal
    return D


def distance_concentration(D):
    """sd/mean of the off-diagonal distances.  Near zero means every object sits
    at the same distance from every other and the matrix carries little
    information -- the high-dimensional concentration failure mode."""
    v = D[np.triu_indices_from(D, k=1)]
    return float(v.std() / v.mean())


# --------------------------------------------------------------------------- #
# 1. Mantel
# --------------------------------------------------------------------------- #

def _mantel_prep(D, method):
    D = np.asarray(D, float)
    if D.ndim == 1:
        D = squareform(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("need a square (or condensed) distance matrix")
    v = squareform(D, checks=False)               # strict upper triangle
    if method == "spearman":
        v = rankdata(v)
    elif method != "pearson":
        raise ValueError("method must be 'pearson' or 'spearman'")
    v = v - v.mean()
    return squareform(v), float(np.linalg.norm(v))


def mantel_test(Dx, Dy, method="spearman", permutations=9999,
                alternative="greater", seed=0, return_null=False, perms=None):
    """Mantel test: correlation between the off-diagonal upper triangles of two
    distance matrices, with a row+column permutation null.

    Cross-checked to <1e-12 against skbio.stats.distance.mantel 0.7.3 and the
    `mantel` PyPI package 2.2.3 for both methods.

    method='spearman' is the default and should stay the default.  Measured at
    n=94 with 5 of 94 objects made outliers in one block only, power at
    alpha=0.05 was 0.303 for Spearman and 0.033 for Pearson -- Pearson did not
    merely lose power, it fell below nominal size because the outlier pairs
    dominate the product-moment sum.

    Exactness note: a simultaneous row+column permutation only reorders the
    off-diagonal entries, so ranks and the centred vector's norm are permutation
    invariant.  Both are computed once; the loop is one gather and one dot.
    """
    Xc, nx = _mantel_prep(Dx, method)
    Yc, ny = _mantel_prep(Dy, method)
    n = Xc.shape[0]
    if Yc.shape[0] != n:
        raise ValueError("distance matrices must be the same size")

    iu = np.triu_indices(n, k=1)
    xv = Xc[iu]
    denom = nx * ny
    r = float(xv @ Yc[iu] / denom)

    if perms is None:
        rng = np.random.default_rng(seed)
        perms = (rng.permutation(n) for _ in range(permutations))
    else:
        perms = np.asarray(perms)
        permutations = perms.shape[0]

    null = np.empty(permutations)
    for i, pm in enumerate(perms):
        null[i] = xv @ Yc[np.ix_(pm, pm)][iu] / denom

    p = _pvalue(r, null, alternative, permutations)
    sd = null.std(ddof=1)
    out = dict(r=r, p=p, n=n, method=method, alternative=alternative,
               permutations=permutations, null_mean=float(null.mean()),
               null_sd=float(sd),
               z=float((r - null.mean()) / sd) if sd > 0 else np.nan)
    if return_null:
        out["null"] = null
    return out


def _pvalue(obs, null, alternative, permutations):
    if alternative == "greater":
        c = np.count_nonzero(null >= obs)
    elif alternative == "less":
        c = np.count_nonzero(null <= obs)
    elif alternative == "two-sided":
        c = np.count_nonzero(np.abs(null) >= abs(obs))
    else:
        raise ValueError(alternative)
    return float((c + 1) / (permutations + 1))


# --------------------------------------------------------------------------- #
# 2. RV coefficient
# --------------------------------------------------------------------------- #

def _trprod(A, B):
    """tr(A @ B) without forming the product."""
    return float(np.sum(A * B.T))


def _gram(X, center=True):
    Xc = X - X.mean(axis=0, keepdims=True) if center else np.asarray(X, float)
    return Xc @ Xc.T


def rv_coefficients(X, Y, center=True):
    """Escoufier's RV and its two bias corrections.

    rv      -- plain RV.  DO NOT REPORT IT.  Measured on independent blocks at
               n=94, p=615, q=41780, where the truth is 0: rv = 0.9303.
    rv2     -- Smilde et al. 2009 modified RV (zero the Gram diagonals).  Better
               but NOT a fix in this regime: 0.2338 on the same null data, and
               0.8282 at p=q=41780.
    rv_adj  -- Mayer et al. 2011 adjusted RV, (rv - E0)/(1 - E0) with
               E0 = sqrt(beta_x beta_y)/(n-1) and beta = (tr S)^2/tr(S^2).
               Measured -0.0002 +- 0.0034 on the same null data.  This is the
               one to report.

    adj_stability = 1 - E0 is the denominator of the correction.  It shrinks
    toward 0 as the blocks approach isotropic full rank; below ~0.05 prefer the
    permutation-based `rv_emp_adj` from rv_permutation_test, which needs no
    such division to be well conditioned.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")
    n = X.shape[0]
    Sx, Sy = _gram(X, center), _gram(Y, center)

    rv = _trprod(Sx, Sy) / np.sqrt(_trprod(Sx, Sx) * _trprod(Sy, Sy))
    Sxu = Sx - np.diag(np.diag(Sx))
    Syu = Sy - np.diag(np.diag(Sy))
    rv2 = _trprod(Sxu, Syu) / np.sqrt(_trprod(Sxu, Sxu) * _trprod(Syu, Syu))

    bx = float(np.trace(Sx) ** 2 / _trprod(Sx, Sx))
    by = float(np.trace(Sy) ** 2 / _trprod(Sy, Sy))
    e0 = float(np.sqrt(bx * by) / (n - 1))
    rv_adj = (rv - e0) / (1.0 - e0) if abs(1.0 - e0) > 1e-12 else np.nan

    return dict(rv=float(rv), rv2=float(rv2), rv_adj=float(rv_adj),
                e0_expected_rv=e0, adj_stability=float(1.0 - e0),
                beta_x=bx, beta_y=by, n=n, p=X.shape[1], q=Y.shape[1])


def rv_permutation_test(X, Y, kind="rv", permutations=9999, seed=0,
                        center=True, return_null=False, perms=None):
    """Permutation test on RV ('rv') or the modified RV ('rv2').

    Permuting the rows of Y is exactly a simultaneous row+column permutation of
    its Gram matrix, so p and q never enter the loop: the cost is O(n^2) per
    permutation whether q is 615 or 41,780.

    The p-value is calibrated even for the plain RV whose point estimate is
    useless -- measured type-I error 0.040 at nominal 0.05, KS vs uniform
    p=0.66 over 200 null replicates at n=94, p=615, q=41780.  It is the EFFECT
    SIZE that is broken, not the test.

    `rv_emp_adj` = (obs - null_mean)/(1 - null_mean) is the assumption-free
    analogue of Mayer's correction.  Measured agreement with the analytic
    rv_adj: max |difference| 0.00033, correlation 0.9995.

    Pass `perms` (from shared_permutations) to reuse one permutation scheme
    across a whole family of tests -- required for westfall_young.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.shape[0]
    Sx, Sy = _gram(X, center), _gram(Y, center)
    if kind == "rv2":
        Sx = Sx - np.diag(np.diag(Sx))
        Sy = Sy - np.diag(np.diag(Sy))
    elif kind != "rv":
        raise ValueError(kind)

    den_x = _trprod(Sx, Sx)
    obs = _trprod(Sx, Sy) / np.sqrt(den_x * _trprod(Sy, Sy))

    if perms is None:
        rng = np.random.default_rng(seed)
        perms = (rng.permutation(n) for _ in range(permutations))
    else:
        perms = np.asarray(perms)
        permutations = perms.shape[0]

    null = np.empty(permutations)
    for i, pm in enumerate(perms):
        Syp = Sy[np.ix_(pm, pm)]
        null[i] = _trprod(Sx, Syp) / np.sqrt(den_x * _trprod(Syp, Syp))

    p = _pvalue(obs, null, "greater", permutations)
    m, sd = null.mean(), null.std(ddof=1)
    out = dict(kind=kind, stat=float(obs), p=p, n=n,
               null_mean=float(m), null_sd=float(sd),
               z=float((obs - m) / sd) if sd > 0 else np.nan,
               rv_emp_adj=float((obs - m) / (1.0 - m)) if abs(1 - m) > 1e-12 else np.nan)
    if return_null:
        out["null"] = null
    return out


# --------------------------------------------------------------------------- #
# 3. Procrustes
# --------------------------------------------------------------------------- #

def _proc_prep(X):
    X = np.asarray(X, float)
    X = X - X.mean(axis=0, keepdims=True)
    nrm = np.linalg.norm(X)
    if nrm == 0:
        raise ValueError("block has no variance")
    return X / nrm


def _proc_basis(X, tol=1e-7):
    """Left singular vectors and singular values.  When p > n this goes through
    the n x n Gram matrix (16x faster than SVD-ing 94 x 41780, measured
    5.03s -> 0.31s).  The Gram squares the condition number, so the rank
    tolerance must be sqrt(eps)-scale (1e-7); at 1e-12 the centring
    null-direction survives as a junk component."""
    n, p = X.shape
    if p > n:
        w, V = np.linalg.eigh(X @ X.T)
        w = np.clip(w[::-1], 0.0, None)
        V, S = V[:, ::-1], np.sqrt(w)
    else:
        V, S, _ = np.linalg.svd(X, full_matrices=False)
    keep = S > tol * S.max()
    return V[:, keep], S[keep]


def procrustes_stat(X, Y):
    """Symmetric Procrustes agreement.  Returns (m2, r).

    Blocks are centred and scaled to unit Frobenius norm; s is the nuclear norm
    of the cross-product, m2 = 1 - s^2 (scipy's `disparity`), r = s.

    Two things this does that scipy.spatial.procrustes cannot:
      * the blocks may have DIFFERENT widths -- no zero-padding needed;
      * it never SVDs a p x q matrix.  Writing the thin SVDs X = Ux Sx Vx' and
        Y = Uy Sy Vy', the singular values of X'Y equal those of
        Sx (Ux' Uy) Sy, which is at most n x n.  scipy would need a
        41780-square SVD (14.0 GB); this takes 0.74 s.
      Verified identical to scipy.spatial.procrustes to 1e-12 on every
      equal-width shape tested.

    Do not report r on its own.  On independent blocks at n=94 it measured
    0.9616 at p=615 and 0.99945 at p=41780.  Use protest().r_adj.
    """
    Xc, Yc = _proc_prep(X), _proc_prep(Y)
    if Xc.shape[0] != Yc.shape[0]:
        raise ValueError("blocks must have the same number of rows")
    Ux, Sx = _proc_basis(Xc)
    Uy, Sy = _proc_basis(Yc)
    M = (Sx[:, None] * (Ux.T @ Uy)) * Sy[None, :]
    s = float(np.linalg.svd(M, compute_uv=False).sum())
    return 1.0 - s * s, s


def protest(X, Y, permutations=9999, seed=0, return_null=False, perms=None):
    """PROTEST (Peres-Neto & Jackson 2001): permutation test on the Procrustes
    statistic.  The two large SVDs are done once; each permutation costs one
    n x n SVD.

    r_adj = (r - null_mean)/(1 - null_mean) is the mean-corrected effect size;
    report that, not r.
    """
    Ux, Sx = _proc_basis(_proc_prep(X))
    Uy, Sy = _proc_basis(_proc_prep(Y))
    n = Ux.shape[0]
    if Uy.shape[0] != n:
        raise ValueError("blocks must have the same number of rows")
    A = Sx[:, None] * Ux.T

    def stat(Uy_rows):
        return float(np.linalg.svd((A @ Uy_rows) * Sy[None, :], compute_uv=False).sum())

    r = stat(Uy)
    if perms is None:
        rng = np.random.default_rng(seed)
        perms = (rng.permutation(n) for _ in range(permutations))
    else:
        perms = np.asarray(perms)
        permutations = perms.shape[0]

    null = np.empty(permutations)
    for i, pm in enumerate(perms):
        null[i] = stat(Uy[pm])

    p = _pvalue(r, null, "greater", permutations)
    m, sd = null.mean(), null.std(ddof=1)
    out = dict(m2=float(1 - r * r), r=float(r), p=p, n=n,
               null_mean=float(m), null_sd=float(sd),
               z=float((r - m) / sd) if sd > 0 else np.nan,
               r_adj=float((r - m) / (1.0 - m)) if abs(1 - m) > 1e-12 else np.nan)
    if return_null:
        out["null"] = null
    return out


# --------------------------------------------------------------------------- #
# 5. multiple testing
# --------------------------------------------------------------------------- #

def shared_permutations(n, permutations, seed=0):
    """One (permutations, n) permutation matrix reused by every test in a
    family.  Reusing it is what makes the joint null valid, and it is free
    here because every test permutes the same n compounds."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(permutations)])


def westfall_young(obs, null, alternative="greater", studentize=True):
    """Step-down maxT multiple testing (Westfall & Young 1993).

    obs  : (k,) observed statistics.
    null : (k, B) null statistics, column b from the SAME permutation b.

    Controls FWER in the strong sense while using the actual dependence between
    the tests instead of assuming the worst as Bonferroni does.  Measured on
    families of 12 correlated Mantel tests at n=94: FWER 0.055 under the global
    null (Bonferroni/Holm/BH 0.062), and power 0.336 against Holm's 0.304 and
    Bonferroni's 0.295 -- about 10% more power at the same control.

    studentize=True (the default, and do not turn it off for a mixed family):
    each test is converted to (stat - null_mean)/null_sd before the max is
    taken.  Raw maxT is only valid when every test in the family has the same
    null scale.  RV nulls in particular do NOT: across three real block pairs
    the plain-RV null means ranged 0.11 to 0.27, so an un-studentised max would
    be decided by whichever pair had the widest blocks rather than by evidence.
    """
    obs = np.asarray(obs, float)
    null = np.asarray(null, float)
    if null.ndim != 2 or null.shape[0] != obs.shape[0]:
        raise ValueError("null must be (k, B) matching obs")
    if alternative == "two-sided":
        obs, null = np.abs(obs), np.abs(null)
    elif alternative != "greater":
        raise ValueError(alternative)
    if studentize:
        m = null.mean(axis=1, keepdims=True)
        s = null.std(axis=1, ddof=1, keepdims=True)
        s[s == 0] = 1.0
        obs = (obs - m[:, 0]) / s[:, 0]
        null = (null - m) / s
    k, B = null.shape

    p_raw = (np.sum(null >= obs[:, None], axis=1) + 1) / (B + 1)

    order = np.argsort(-obs)
    null_o, obs_o = null[order], obs[order]
    succ_max = np.maximum.accumulate(null_o[::-1], axis=0)[::-1]
    p_adj_o = (np.sum(succ_max >= obs_o[:, None], axis=1) + 1) / (B + 1)
    p_adj_o = np.maximum.accumulate(p_adj_o)
    p_fwer = np.empty(k)
    p_fwer[order] = p_adj_o
    return dict(p_raw=p_raw, p_fwer=p_fwer)


# --------------------------------------------------------------------------- #
# 1b. partial Mantel -- read the calibration warning before using it
# --------------------------------------------------------------------------- #

def partial_mantel(Dx, Dy, Dz, method="spearman", permutations=9999,
                   alternative="greater", seed=0, return_null=False, perms=None):
    """Mantel correlation between Dx and Dy holding a third matrix Dz fixed.

    The obvious use here: does morphology ~ expression survive controlling for
    chemical similarity (Dz from ECFP)?

    Statistic is the first-order partial correlation of the three off-diagonal
    vectors; the null permutes the objects of Y, holding X and Z.

    DO NOT USE THIS TO CLAIM A DIRECT LINK.  Measured at n=94, with X and Y
    connected ONLY through a shared latent that also drives Z, type-I error at
    nominal alpha=0.05 was 0.367 -- a 7.3x inflation, KS vs uniform p=5e-57.
    Permuting the residuals of Y on Z instead (Freedman-Lane / ter Braak) does
    NOT fix it: 0.380 on the same data.  Both have full power (1.000) against a
    genuine direct link, so the statistic is not useless -- its NULL is wrong.

    Report it only as a descriptive quantity beside the three marginal Mantel
    results.  When you need to actually remove a third block's contribution,
    residualise the FEATURE BLOCKS on Z and run an ordinary Mantel/RV on the
    residuals (see the validation notes for its measured calibration).
    """
    Xc, nx = _mantel_prep(Dx, method)
    Yc, ny = _mantel_prep(Dy, method)
    Zc, nz = _mantel_prep(Dz, method)
    n = Xc.shape[0]
    if not (Yc.shape[0] == Zc.shape[0] == n):
        raise ValueError("all three matrices must be the same size")
    iu = np.triu_indices(n, k=1)
    xv, zv = Xc[iu], Zc[iu]

    def pstat(yv):
        rxy = xv @ yv / (nx * np.linalg.norm(yv))
        ryz = yv @ zv / (np.linalg.norm(yv) * nz)
        rxz = xv @ zv / (nx * nz)
        d = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        return float((rxy - rxz * ryz) / d) if d > 0 else np.nan

    r = pstat(Yc[iu])
    if perms is None:
        rng = np.random.default_rng(seed)
        perms = (rng.permutation(n) for _ in range(permutations))
    else:
        perms = np.asarray(perms)
        permutations = perms.shape[0]

    null = np.empty(permutations)
    for i, pm in enumerate(perms):
        null[i] = pstat(Yc[np.ix_(pm, pm)][iu])

    p = _pvalue(r, null, alternative, permutations)
    sd = null.std(ddof=1)
    out = dict(r=r, p=p, n=n, method=method, permutations=permutations,
               null_mean=float(null.mean()), null_sd=float(sd),
               z=float((r - null.mean()) / sd) if sd > 0 else np.nan)
    if return_null:
        out["null"] = null
    return out


__all__.append("partial_mantel")


# --------------------------------------------------------------------------- #
# the recommended end-to-end call
# --------------------------------------------------------------------------- #

def agreement_report(blocks, pairs=None, metric="euclidean", scaling="zscore",
                     permutations=9999, seed=0, clean=True, verbose=False):
    """Run Mantel + RV + Procrustes over a family of block pairs, all driven by
    ONE shared permutation scheme, and return a tidy pandas DataFrame with
    Westfall-Young FWER-adjusted p-values computed within each statistic.

    blocks : {name: array (n, p_name)}  -- same n objects, same order, every block
    pairs  : [(name_a, name_b), ...]    -- default: every unordered pair
    metric : distance metric for the Mantel arm only ('jaccard' for fingerprints)

    Columns: pair, stat, effect (the ADJUSTED effect size), raw (the
    uncorrected statistic, for diagnosis only), p_raw, p_fwer, z, plus the
    RV conditioning diagnostic.
    """
    import pandas as pd

    names = list(blocks)
    n = next(iter(blocks.values())).shape[0]
    prepped = {}
    for k, X in blocks.items():
        X = np.asarray(X, float)
        if clean:
            X, _ = clean_block(X, verbose=verbose, name=k)
        prepped[k] = scale_block(X, scaling)
    if pairs is None:
        pairs = [(names[i], names[j])
                 for i in range(len(names)) for j in range(i + 1, len(names))]

    perms = shared_permutations(n, permutations, seed=seed)
    rows, nulls = [], {"mantel": [], "rv": [], "procrustes": []}

    for a, b in pairs:
        Xa, Xb = prepped[a], prepped[b]
        Da = distance_matrix(Xa, metric)
        Db = distance_matrix(Xb, metric)

        Xc, nx = _mantel_prep(Da, "spearman")
        Yc, ny = _mantel_prep(Db, "spearman")
        iu = np.triu_indices(n, k=1)
        xv, den = Xc[iu], nx * ny
        mr = float(xv @ Yc[iu] / den)
        mnull = np.array([xv @ Yc[np.ix_(pm, pm)][iu] / den for pm in perms])

        rvp = rv_permutation_test(Xa, Xb, kind="rv", perms=perms, return_null=True)
        rvc = rv_coefficients(Xa, Xb)
        prt = protest(Xa, Xb, perms=perms, return_null=True)

        msd = mnull.std(ddof=1)
        rows += [
            dict(pair=f"{a} ~ {b}", stat="mantel", effect=mr, raw=mr,
                 z=(mr - mnull.mean()) / msd, diagnostic=np.nan),
            dict(pair=f"{a} ~ {b}", stat="rv", effect=rvc["rv_adj"], raw=rvc["rv"],
                 z=rvp["z"], diagnostic=rvc["adj_stability"]),
            dict(pair=f"{a} ~ {b}", stat="procrustes", effect=prt["r_adj"], raw=prt["r"],
                 z=prt["z"], diagnostic=np.nan),
        ]
        nulls["mantel"].append(mnull)
        nulls["rv"].append(rvp["null"])
        nulls["procrustes"].append(prt["null"])

    df = pd.DataFrame(rows)
    df["p_raw"] = np.nan
    df["p_fwer"] = np.nan
    for stat in ("mantel", "rv", "procrustes"):
        sel = df["stat"] == stat
        obs = df.loc[sel, "raw"].to_numpy()
        wy = westfall_young(obs, np.vstack(nulls[stat]))
        df.loc[sel, "p_raw"] = wy["p_raw"]
        df.loc[sel, "p_fwer"] = wy["p_fwer"]
    return df[["pair", "stat", "effect", "raw", "z", "p_raw", "p_fwer", "diagnostic"]]


__all__.append("agreement_report")


# --------------------------------------------------------------------------- #
# what NOT to do -- three tested-and-rejected ways to "control for" a third block
# --------------------------------------------------------------------------- #
#
# At n=94, with X and Y connected ONLY through a shared latent that also drives
# Z, every attempt to remove Z's contribution and test the remaining X-Y link
# failed.  Measured type-I error at nominal alpha=0.05:
#
#   partial Mantel, naive object permutation .................. 0.367
#   partial Mantel, Freedman-Lane residual permutation ........ 0.380
#   residualise BOTH blocks on Z's top-k object directions,
#     then ordinary Mantel/RV (k=3 and k=10) .................. 1.000
#
# The last one is the worst trap, because it looks principled.  Projecting both
# blocks onto the SAME (n-k)-dimensional subspace manufactures agreement: the
# mean Mantel r ROSE from 0.327 (no residualisation) to 0.536 (k=10) on data
# whose X-Y link was entirely indirect.
#
# What to do instead: report the three MARGINAL agreements (X~Y, X~Z, Y~Z) and
# state the shared-latent explanation as a live alternative.  At n ~ 100 the
# data cannot separate a direct link from an indirect one, and no amount of
# statistic-choice changes that.
