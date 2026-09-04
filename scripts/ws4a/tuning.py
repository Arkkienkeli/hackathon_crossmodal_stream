"""Optuna hyper-parameter search, confined to the inner cross-validation loop.

WHY THE CONFINEMENT IS THE WHOLE POINT
--------------------------------------
Selection bias is not a rounding error at this sample size. On PERMUTED,
uninformative high-dimensional data, choosing the best of 124 classifier variants
by cross-validation yields median error rates of 31-41% against a 50% chance
baseline, and the bias grows as n shrinks (Boulesteix & Strobl). WS4A runs at
n ~ 64 with 14 positives, which is squarely in that regime.

So a search is only admissible if it never sees the data it is scored on:

    for each OUTER fold:
        study = optuna(...)              <- sees the TRAINING part only
        best  = refit on the training part with the study's best parameters
        score = best.score(HELD-OUT part)   <- never influenced the search

That keeps the outer estimate unbiased no matter how large the budget. What a
larger budget DOES do is raise the score on permuted labels -- which is why
`ws4a_ml.py` gives the permuted control the identical budget. Tune the real labels
with 50 trials and the control with a 15-point grid and the reported gap is
inflated by exactly the bias being measured.

SEARCH SPACES ARE DELIBERATELY SMALL
------------------------------------
At n=64 a sprawling space fits noise. Each space below spans the range where the
parameter changes behaviour and stops. The XGBoost space in particular starts
`min_child_weight` at 1: the previous fixed value of 5, against 14 positives,
blocked every split and made the model return exactly chance with zero variance.
"""
from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np

LOG = logging.getLogger("ws4a")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAVE_OPTUNA = True
except ImportError:                                              # pragma: no cover
    HAVE_OPTUNA = False


# --------------------------------------------------------------------------- #
# search spaces. Each returns sklearn-style params for the "model" pipeline step.
# --------------------------------------------------------------------------- #
def _space_elastic_net(trial, kind: str, n: int, p: int) -> dict:
    if kind == "classification":
        # NOTE: no `penalty=`. sklearn 1.9 deprecates it (its default is literally
        # the string "deprecated") and removes it in 1.10; l1_ratio alone selects
        # the penalty, with 0 = ridge and 1 = lasso.
        return {"model__C": trial.suggest_float("C", 1e-4, 1e2, log=True),
                "model__l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}
    return {"model__alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "model__l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0)}


def _space_linear_svm(trial, kind: str, n: int, p: int) -> dict:
    return {"model__C": trial.suggest_float("C", 1e-5, 1e2, log=True)}


def _space_sparse_plsda(trial, kind: str, n: int, p: int) -> dict:
    # keep_x is a COUNT of retained variables (mixOmics semantics), so it must be
    # bounded by the block width -- 200 on a 636-feature block is meaningful, on a
    # 64-bit ECFP block it is not.
    hi = max(5, min(int(p), 300))
    return {"model__n_components": trial.suggest_int("n_components", 1, 3),
            "model__keep_x": trial.suggest_int("keep_x", 5, hi, log=True)}


def _space_xgboost(trial, kind: str, n: int, p: int) -> dict:
    return {
        "model__max_depth": trial.suggest_int("max_depth", 1, 4),
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 400, step=50),
        "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "model__subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        # starts at 1 on purpose -- 5 blocked every split at 14 positives
        "model__min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "model__reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "model__reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


SEARCH_SPACES: dict[str, Callable] = {
    "elastic_net": _space_elastic_net,
    "linear_svm": _space_linear_svm,
    "sparse_plsda": _space_sparse_plsda,
    "xgboost": _space_xgboost,
}


def space_size(model_name: str) -> int:
    """How many parameters a model's space has -- used to sanity-check the budget."""
    return {"elastic_net": 2, "linear_svm": 1, "sparse_plsda": 2, "xgboost": 8}.get(
        model_name, 1)


# --------------------------------------------------------------------------- #
def tune_fit(est, model_name, X, y, kind, inner_cv, scoring, n_trials, seed,
             n_jobs: int = 1):
    """Search inside ONE outer-fold training set, then refit on it.

    Returns (fitted_estimator, info). `X`/`y` must ALREADY be the training part of
    the outer split -- this function must never see held-out rows.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    space = SEARCH_SPACES.get(model_name)
    if not HAVE_OPTUNA or space is None:
        return clone(est).fit(X, y), {"tuned": False,
                                      "reason": "optuna missing" if not HAVE_OPTUNA
                                      else f"no space for {model_name}"}

    n, p = X.shape

    def objective(trial):
        params = space(trial, kind, n, p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                sc = cross_val_score(clone(est).set_params(**params), X, y, cv=inner_cv,
                                     scoring=scoring, n_jobs=1, error_score="raise")
            except Exception:                                     # noqa: BLE001
                # A degenerate parameter set (e.g. keep_x above the block width) is a
                # bad trial, not a crashed run. Prune it and let the search continue.
                raise optuna.TrialPruned()
        return float(np.mean(sc))

    # Seeded per call, so a fold's search is reproducible and independent of the
    # order folds happen to run in.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(seed) % (2**31)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=int(n_trials), n_jobs=n_jobs,
                       show_progress_bar=False, catch=(Exception,))

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        LOG.warning("tuning     : every trial failed for %s -- falling back to defaults",
                    model_name)
        return clone(est).fit(X, y), {"tuned": False, "reason": "all trials failed",
                                      "n_trials": int(n_trials)}

    best_params = {f"model__{k}": v for k, v in study.best_params.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = clone(est).set_params(**best_params).fit(X, y)
    return fitted, {
        "tuned": True,
        "n_trials": int(n_trials),
        "n_complete": len(completed),
        "best_inner_score": float(study.best_value),
        "best_params": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in study.best_params.items()},
    }


# --------------------------------------------------------------------------- #
def is_degenerate(scores, chance: float, tol: float = 1e-9) -> bool:
    """A model that returns exactly chance with zero variance predicted one class.

    That is a broken configuration, not a null result, and it must never be reported
    as though it were a measurement. XGBoost did exactly this on every block with
    min_child_weight=5 against 14 positives: no split satisfied the constraint, so
    every fold returned the majority-class prediction and balanced accuracy was
    exactly 0.5 by construction.
    """
    s = np.asarray(scores, dtype=float)
    if s.size < 2:
        return False
    return bool(np.std(s) < tol and abs(float(np.mean(s)) - chance) < 1e-6)
