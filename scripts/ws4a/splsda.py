"""
splsda.py -- sparse PLS-DA with mixOmics 'keepX' semantics, in numpy + sklearn.

There is no maintained Python port of mixOmics.  This is the algorithm mixOmics
uses for `splsda(..., keepX=)`: NIPALS-style PLS in regression mode on a dummy-coded
Y, with the X-loading soft-thresholded on every iteration so that exactly `keepX[h]`
variables survive on component h.  Sparsity is controlled by a COUNT, not a penalty,
which is what makes it tunable at n ~ 100 (a lambda has no interpretable scale when
p changes between blocks).

Prediction uses mixOmics' `max.dist` and `centroids.dist` rules on the component
scores.  `centroids.dist` is the one to prefer with unbalanced classes.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


def _soft_threshold_keep(v: np.ndarray, keep: int) -> np.ndarray:
    """Soft-threshold v so that exactly `keep` entries are non-zero.

    mixOmics thresholds at the (keep+1)-th largest |v|, which shrinks the
    survivors as well -- that shrinkage is what makes the selection stable.
    """
    keep = int(min(max(keep, 1), v.size))
    if keep >= v.size:
        return v
    lam = np.sort(np.abs(v))[::-1][keep]          # (keep+1)-th largest magnitude
    return np.sign(v) * np.clip(np.abs(v) - lam, 0.0, None)


def _norm(v: np.ndarray) -> np.ndarray:
    nv = np.linalg.norm(v)
    return v if nv < 1e-12 else v / nv


class SparsePLSDA(BaseEstimator, ClassifierMixin):
    """Sparse PLS-DA.

    Parameters
    ----------
    n_components : number of latent components.
    keep_x : int or list of int, length n_components -- variables kept per component.
    scale : z-score X columns (mixOmics default is TRUE).
    dist : 'centroids' or 'max' -- mixOmics prediction rule.
    max_iter, tol : inner NIPALS loop.

    Attributes
    ----------
    x_weights_   (p, H)  sparse loading vectors, the thing you interpret
    x_scores_    (n, H)
    selected_    list of index arrays, one per component
    """

    def __init__(self, n_components=2, keep_x=20, scale=True,
                 dist="centroids", max_iter=500, tol=1e-9):
        self.n_components = n_components
        self.keep_x = keep_x
        self.scale = scale
        self.dist = dist
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        X = np.asarray(X, float)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        yi = self._le.transform(y)
        Y = np.eye(len(self.classes_))[yi].astype(float)

        self._xm = X.mean(0)
        self._xs = X.std(0, ddof=1) if self.scale else np.ones(X.shape[1])
        self._xs = np.where(self._xs < 1e-12, 1.0, self._xs)
        Xh = (X - self._xm) / self._xs
        self._ym = Y.mean(0)
        self._ys = Y.std(0, ddof=1)
        self._ys = np.where(self._ys < 1e-12, 1.0, self._ys)
        Yh = (Y - self._ym) / self._ys

        H = self.n_components
        keep = self.keep_x
        keep = [keep] * H if np.isscalar(keep) else list(keep)
        if len(keep) != H:
            raise ValueError("keep_x must be a scalar or have length n_components")

        p = Xh.shape[1]
        self.x_weights_ = np.zeros((p, H))
        self.y_weights_ = np.zeros((Yh.shape[1], H))
        self.x_scores_ = np.zeros((Xh.shape[0], H))
        self.x_loadings_ = np.zeros((p, H))

        for h in range(H):
            # init from the leading singular vector of the cross-product
            M = Xh.T @ Yh
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            a, b = U[:, 0], Vt[0]
            for _ in range(self.max_iter):
                a_old = a
                a = _norm(_soft_threshold_keep(Xh.T @ (Yh @ b), keep[h]))
                t = Xh @ a
                denom = t @ t
                if denom < 1e-12:
                    break
                b = _norm(Yh.T @ t)
                if np.linalg.norm(a - a_old) < self.tol:
                    break
            t = Xh @ a
            tt = t @ t
            if tt < 1e-12:
                raise RuntimeError(
                    f"component {h}: X-score collapsed to zero -- keep_x={keep[h]} "
                    "selected only null-variance variables")
            c = (Xh.T @ t) / tt                       # X loading for deflation
            self.x_weights_[:, h] = a
            self.y_weights_[:, h] = b
            self.x_scores_[:, h] = t
            self.x_loadings_[:, h] = c
            # regression-mode deflation (mixOmics default for DA)
            Xh = Xh - np.outer(t, c)
            Yh = Yh - np.outer(t, (Yh.T @ t) / tt)

        self.selected_ = [np.flatnonzero(self.x_weights_[:, h]) for h in range(H)]
        # rotation so transform() is a single matrix multiply
        self.x_rotations_ = self.x_weights_ @ np.linalg.pinv(
            self.x_loadings_.T @ self.x_weights_)
        self._centroids = np.vstack([self.x_scores_[yi == g].mean(0)
                                     for g in range(len(self.classes_))])
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X):
        Xh = (np.asarray(X, float) - self._xm) / self._xs
        return Xh @ self.x_rotations_

    def decision_function(self, X):
        T = self.transform(X)
        if self.dist == "centroids":
            d = ((T[:, None, :] - self._centroids[None, :, :]) ** 2).sum(-1)
            return -d
        elif self.dist == "max":
            B = self.x_rotations_ @ (self.y_weights_ * self._ys).T
            return (np.asarray(X, float) - self._xm) / self._xs @ B + self._ym
        raise ValueError("dist must be 'centroids' or 'max'")

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def selection_frequency(self, X, y, n_boot=200, seed=0, stratify=True):
        """Stability selection: how often each variable is picked over bootstraps.
        A loading that is only ever picked in one resample is noise."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, float); y = np.asarray(y)
        n, p = X.shape
        cnt = np.zeros(p)
        ok = 0
        for _ in range(n_boot):
            if stratify:
                idx = np.concatenate([rng.choice(np.flatnonzero(y == c),
                                                 (y == c).sum(), replace=True)
                                      for c in np.unique(y)])
            else:
                idx = rng.integers(0, n, n)
            try:
                m = SparsePLSDA(**self.get_params()).fit(X[idx], y[idx])
            except Exception:
                continue
            cnt[np.unique(np.concatenate(m.selected_))] += 1
            ok += 1
        return cnt / max(ok, 1)
