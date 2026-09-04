"""
stabsel.py -- Stability selection from the primitive definition.

  MB2010  Meinshausen & Buhlmann (2010) JRSS-B 72(4):417-473.
          B independent subsamples of size floor(n/2), WITHOUT replacement.
          Pi_j = P(j selected somewhere in the regularisation region Lambda).
          Thm 1:  E(V) <= q_Lambda^2 / ((2*pi_thr - 1) * p),  pi_thr in (1/2, 1],
          under EXCHANGEABILITY of the noise variables.

  CPSS    Shah & Samworth (2013) JRSS-B 75(1):55-80.
          B/2 COMPLEMENTARY PAIRS -> 2*(B/2) half-samples, each pair a disjoint
          split of the data. Same number of fits, different sampling scheme.

Design note: each half-sample is summarised by the index of the FIRST lambda at
which each variable enters the path. Lambda is always a prefix of the grid (from
sparse to dense), so the union over Lambda is exactly {j : entry_j < L}. That
makes q_Lambda and Pi computable for EVERY nested Lambda from ONE fit pass, so
you can calibrate Lambda against a target q for free.

Requires: numpy, scikit-learn, joblib.
Verified on Python 3.12.14 / numpy 2.5.2 / scipy 1.18.1 / scikit-learn 1.9.0 /
pandas 3.0.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, lasso_path

# ------------------------------------------------------------- sklearn compat
# sklearn 1.8 deprecated LogisticRegression(penalty=...) for l1_ratio=; removal
# is scheduled for 1.10. penalty='l1' still fits correctly on 1.9 but emits a
# spurious "Inconsistent values: penalty=l1 with l1_ratio=0.0" UserWarning on
# EVERY fit -- thousands of lines in a stability-selection loop.
try:                                              # sklearn >= 1.8
    LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0)._validate_params()
    _L1KW = dict(l1_ratio=1.0)
except Exception:                                 # sklearn <= 1.7
    _L1KW = dict(penalty="l1")


def _l1_logreg(C: float, n_classes: int):
    # sklearn 1.9's liblinear REFUSES n_classes >= 3 ("does not support multiclass
    # classification"). moa-fine has 23 classes, so this branch is not optional.
    if n_classes <= 2:
        return LogisticRegression(solver="liblinear", C=C, **_L1KW)
    return LogisticRegression(solver="saga", C=C, max_iter=3000, tol=1e-3, **_L1KW)


# ------------------------------------------------------------- preprocessing
def _standardise(X: np.ndarray) -> np.ndarray:
    """Column-standardise. Zero-variance columns become zeros -> never selected."""
    sd = X.std(axis=0)
    dead = sd < 1e-12
    Z = (X - X.mean(axis=0)) / np.where(dead, 1.0, sd)
    Z[:, dead] = 0.0
    return Z


def _null_residual(y: np.ndarray, family: str) -> np.ndarray:
    if family == "gaussian":
        return (y - y.mean()).reshape(-1, 1)
    # gradient of the multinomial deviance at the null model, one column per
    # class. Treating 23 categorical labels as a number gives a meaningless
    # alpha_max and the whole path comes back empty.
    cls = np.unique(y)
    Y = (y.reshape(-1, 1) == cls.reshape(1, -1)).astype(float)
    return Y - Y.mean(axis=0, keepdims=True)


def lambda_grid(X, y, family="gaussian", n_lambda=60, eps=1e-3):
    """The regularisation grid, computed ONCE on the full data so every
    half-sample uses the same Lambda (required: Pi is a union over a FIXED
    Lambda). Ordered sparse -> dense. alpha_max is the smallest penalty whose
    solution is all-zero; go WIDE here and choose Lambda later with `select_q`."""
    X = np.asarray(X, float); y = np.asarray(y)
    a_max = float(np.abs(_standardise(X).T @ _null_residual(y, family)).max() / len(y))
    grid = np.logspace(np.log10(max(a_max, 1e-12)),
                       np.log10(max(a_max, 1e-12) * eps), n_lambda)
    return grid if family == "gaussian" else 1.0 / (grid * len(y))


# ------------------------------------------------------------- one half-sample
def _entry_index(X, y, lambdas, family, weights=None, max_iter=5000):
    """uint16 (p,): index of the first lambda at which variable j is non-zero;
    n_lambda if it never enters. `weights` = the RANDOMISED lasso (MB2010 s.2.3):
    penalising column j by lambda/W_j is identical to scaling column j by W_j."""
    nl, p = len(lambdas), X.shape[1]
    Z = _standardise(X)
    if weights is not None:
        Z = Z * weights
    if family == "gaussian":
        _, coefs, _ = lasso_path(Z, y - y.mean(), alphas=lambdas,
                                 max_iter=max_iter, tol=1e-5)
        sup = np.abs(coefs.T) > 1e-10                       # (nl, p)
    else:
        ncls = len(np.unique(y))
        if ncls < 2:                                        # degenerate subsample
            return np.full(p, nl, dtype=np.uint16)
        sup = np.empty((nl, p), dtype=bool)
        for i, C in enumerate(lambdas):
            sup[i] = np.abs(_l1_logreg(C, ncls).fit(Z, y).coef_).max(axis=0) > 1e-10
    ever = sup.any(axis=0)
    return np.where(ever, sup.argmax(axis=0), nl).astype(np.uint16)


# ------------------------------------------------------------- subsampling
def _mb_halfsamples(n, B, rng):
    """MB2010: B independent draws of floor(n/2) indices WITHOUT replacement."""
    m = n // 2
    return [rng.permutation(n)[:m] for _ in range(B)]


def _cp_halfsamples(n, B_pairs, rng):
    """CPSS: B_pairs disjoint (A, A^c) splits -> 2*B_pairs half-samples. For odd
    n one observation is dropped so that both halves have exactly floor(n/2)."""
    m, out = n // 2, []
    for _ in range(B_pairs):
        perm = rng.permutation(n)
        out += [perm[:m], perm[m:2 * m]]
    return out


# ------------------------------------------------------------- result
@dataclass
class StabSelResult:
    entry: np.ndarray           # (n_half, p) uint16 first-entry lambda index
    lambdas: np.ndarray
    mode: str
    L: int                      # Lambda = lambdas[:L]
    names: list | None = None

    # ---- Lambda is a free parameter: everything below is O(n_half * p) ----
    def with_L(self, L: int) -> "StabSelResult":
        return StabSelResult(self.entry, self.lambdas, self.mode,
                             int(np.clip(L, 1, len(self.lambdas))), self.names)

    @property
    def n_halfsamples(self) -> int:
        return self.entry.shape[0]

    @property
    def prob(self) -> np.ndarray:
        """Pi_j = P(j selected somewhere in Lambda), over the half-samples."""
        return (self.entry < self.L).mean(axis=0)

    @property
    def q(self) -> float:
        """q_Lambda = E|S^Lambda|, the average size of the selected set."""
        return float((self.entry < self.L).sum(axis=1).mean())

    def q_curve(self) -> np.ndarray:
        """q_Lambda for every prefix L = 1..n_lambda. Free; use it to pick L."""
        nl = len(self.lambdas)
        return np.array([(self.entry < L).sum(axis=1).mean() for L in range(1, nl + 1)])

    def select_q(self, target_q: float) -> "StabSelResult":
        """Choose Lambda as the WIDEST prefix whose q_Lambda <= target_q.
        This is how you make the MB bound mean something: pick pi_thr and the
        E(V) you are willing to accept, get target_q from `q_for_target_ev`,
        then call this. No refitting."""
        qc = self.q_curve()
        ok = np.flatnonzero(qc <= target_q)
        return self.with_L(int(ok[-1] + 1) if ok.size else 1)

    # ---- selection + error control ----
    def selected(self, pi_thr: float) -> np.ndarray:
        return np.flatnonzero(self.prob >= pi_thr)

    def ev_bound_mb(self, pi_thr: float) -> float:
        """MB2010 Thm 1: E(V) <= q^2 / ((2*pi_thr - 1) * p). Valid only for
        pi_thr in (1/2, 1] and only under exchangeability of the noise variables
        (see the confounder experiment -- on real batch-structured data it fails)."""
        if not 0.5 < pi_thr <= 1.0:
            raise ValueError("pi_thr must be in (0.5, 1]")
        return self.q ** 2 / ((2.0 * pi_thr - 1.0) * self.entry.shape[1])

    # Shah & Samworth's tighter r-concavity bounds (their Thms 2-3, the `minD`
    # function of the R package `stabs`) are NOT implemented and NOT verified
    # here. `ev_bound_mb` is the only bound this module computes.

    def table(self, pi_thr=0.7, top=25):
        order = np.argsort(-self.prob)[:top]
        nm = (lambda j: f"x{j}") if self.names is None else (lambda j: self.names[j])
        return [(nm(j), round(float(self.prob[j]), 3), bool(self.prob[j] >= pi_thr))
                for j in order]


# ------------------------------------------------------------- driver
def stability_selection(
    X, y, *,
    mode: Literal["mb", "cpss"] = "cpss",
    B: int = 100,
    family: Literal["gaussian", "binomial"] = "gaussian",
    lambdas: Sequence[float] | None = None,
    n_lambda: int = 60,
    eps: float = 1e-3,
    target_q: float | None = None,
    weakness: float | None = None,
    random_state: int | None = 0,
    n_jobs: int = -1,
) -> StabSelResult:
    """
    mode="mb"   : B independent floor(n/2) subsamples       -> B half-samples
    mode="cpss" : B/2 complementary pairs (B must be even)  -> B half-samples
    Same number of fits either way, so the two are directly comparable.

    target_q : if given, Lambda is narrowed to the widest prefix with
               q_Lambda <= target_q (no refit). Otherwise Lambda is the whole grid.
    weakness : randomised lasso; W_j iid on {weakness, 1}. None = plain lasso.
    """
    names = list(X.columns) if hasattr(X, "columns") else None
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y = np.asarray(y)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    lam = np.asarray(lambda_grid(X, y, family, n_lambda, eps)
                     if lambdas is None else lambdas, dtype=float)

    if mode == "mb":
        idx = _mb_halfsamples(n, B, rng)
    elif mode == "cpss":
        if B % 2:
            raise ValueError("cpss needs an even B (B/2 complementary pairs)")
        idx = _cp_halfsamples(n, B // 2, rng)
    else:
        raise ValueError(mode)

    W = [None] * len(idx) if weakness is None else \
        [rng.choice([weakness, 1.0], size=p) for _ in idx]
    if weakness is not None and not 0.0 < weakness <= 1.0:
        raise ValueError("weakness must be in (0, 1]")

    ent = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_entry_index)(X[i], y[i], lam, family, w) for i, w in zip(idx, W))

    res = StabSelResult(np.asarray(ent), lam, mode, len(lam), names)
    return res if target_q is None else res.select_q(target_q)


def q_for_target_ev(target_ev: float, pi_thr: float, p: int) -> float:
    """Invert the MB bound: the largest q_Lambda whose bound is <= target_ev."""
    return float(np.sqrt(target_ev * (2.0 * pi_thr - 1.0) * p))


# ------------------------------------------------------------- self-test
if __name__ == "__main__":
    import time, warnings
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    n, p, k = 94, 615, 8                       # the A549 morphology block exactly
    X = np.empty((n, p))
    for s0 in range(0, p, 25):                 # block AR(1), rho = 0.6
        e = min(s0 + 25, p); w = e - s0
        L = 0.6 ** np.abs(np.subtract.outer(np.arange(w), np.arange(w)))
        X[:, s0:e] = rng.normal(size=(n, w)) @ np.linalg.cholesky(L).T
    true = np.arange(0, k * 25, 25)
    b = np.zeros(p); b[true] = 1.0
    y = X @ b + rng.normal(size=n)

    PI = 0.70
    tq = q_for_target_ev(1.0, PI, p)           # accept E(V) <= 1 false positive
    print(f"target E(V)=1.0 at pi={PI} on p={p}  ->  q_Lambda must be <= {tq:.1f}")
    for mode in ("mb", "cpss"):
        t = time.time()
        r = stability_selection(X, y, mode=mode, B=100, random_state=0)
        el = time.time() - t
        for tag, rr in (("wide Lambda", r), ("calibrated", r.select_q(tq))):
            sel = rr.selected(PI)
            print(f"  {mode:4s} {tag:11s} L={rr.L:3d} q={rr.q:5.1f} "
                  f"TP={np.intersect1d(sel, true).size}/{k} "
                  f"V={np.setdiff1d(sel, true).size} "
                  f"bound E(V)={rr.ev_bound_mb(PI):5.2f}   [{el:.2f}s fit]")
