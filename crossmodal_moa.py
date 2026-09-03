from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0


def filter_rare_classes(X_blocks, y, min_count=3):
    y = np.asarray(y)
    counts = pd.Series(y).value_counts()
    keep_classes = set(counts[counts >= min_count].index)

    mask = np.array([
        lbl in keep_classes
        for lbl in y
    ])

    print(
        f"filter_rare_classes: kept {mask.sum()}/{len(y)} compounds, "
        f"{len(keep_classes)}/{len(counts)} classes "
        f"(dropped {len(counts)-len(keep_classes)} classes "
        f"with < {min_count} members)"
    )

    return [
        X[mask]
        for X in X_blocks
    ], y[mask], mask


def check_alignment(
    X_morph,
    X_expr,
    y,
    drugs=None
):
    assert (
        X_morph.shape[0]
        ==
        X_expr.shape[0]
        ==
        len(y)
    ), "row counts differ"

    if drugs is not None:
        assert len(drugs) == len(y)
        assert len(set(drugs)) == len(drugs)

    assert np.isfinite(X_morph).all()
    assert np.isfinite(X_expr).all()

    print(
        f"ok: {len(y)} compounds, "
        f"morph {X_morph.shape[1]}d, "
        f"expr {X_expr.shape[1]}d"
    )


class BlockCCA(
    BaseEstimator,
    TransformerMixin
):
    def __init__(
        self,
        n_morph,
        n_components=10,
        morph_pcs=30,
        expr_pcs=30,
        mode="concat"
    ):
        self.n_morph = n_morph
        self.n_components = n_components
        self.morph_pcs = morph_pcs
        self.expr_pcs = expr_pcs
        self.mode = mode

    def _split(self, X):
        return (
            X[:, :self.n_morph],
            X[:, self.n_morph:]
        )

    def fit(self, X, y=None):
        Xm, Xe = self._split(X)
        n = X.shape[0]

        self.sm_ = StandardScaler().fit(Xm)
        self.se_ = StandardScaler().fit(Xe)

        Zm = self.sm_.transform(Xm)
        Ze = self.se_.transform(Xe)

        km = min(
            self.morph_pcs,
            Zm.shape[1],
            n - 1
        )

        ke = min(
            self.expr_pcs,
            Ze.shape[1],
            n - 1
        )

        self.pm_ = PCA(
            n_components=km,
            random_state=RANDOM_STATE
        ).fit(Zm)

        self.pe_ = PCA(
            n_components=ke,
            random_state=RANDOM_STATE
        ).fit(Ze)

        Zm = self.pm_.transform(Zm)
        Ze = self.pe_.transform(Ze)

        k = min(
            self.n_components,
            km,
            ke,
            n - 1
        )

        self.k_ = k

        self.cca_ = CCA(
            n_components=k,
            max_iter=2000
        ).fit(
            Zm,
            Ze
        )

        return self

    def transform(self, X):
        Xm, Xe = self._split(X)

        Zm = self.pm_.transform(
            self.sm_.transform(Xm)
        )

        Ze = self.pe_.transform(
            self.se_.transform(Xe)
        )

        Um, Ue = self.cca_.transform(
            Zm,
            Ze
        )

        if self.mode == "concat":
            return np.hstack([
                Um,
                Ue
            ])

        if self.mode == "mean":
            return (
                Um + Ue
            ) / 2.0

        if self.mode == "morph":
            return Um

        raise ValueError(
            f"unknown mode {self.mode}"
        )


def _clf():
    return LogisticRegression(
        C=0.1,
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def single_modality_pipeline(
    n_pcs=30
):
    return Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n_pcs,
                random_state=RANDOM_STATE
            )
        ),
        (
            "clf",
            _clf()
        )
    ])


def concat_pipeline(
    n_pcs=30
):
    return Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "pca",
            PCA(
                n_components=n_pcs,
                random_state=RANDOM_STATE
            )
        ),
        (
            "clf",
            _clf()
        )
    ])


def cca_pipeline(
    n_morph,
    n_components=10,
    mode="concat"
):
    return Pipeline([
        (
            "cca",
            BlockCCA(
                n_morph=n_morph,
                n_components=n_components,
                mode=mode
            )
        ),
        (
            "scale",
            StandardScaler()
        ),
        (
            "clf",
            _clf()
        )
    ])


SCORING = {
    "balanced_accuracy":
        "balanced_accuracy",
    "macro_f1":
        "f1_macro"
}


def evaluate(
    pipe,
    X,
    y,
    n_splits=5,
    n_repeats=10,
    label=""
):
    y = np.asarray(y)

    min_class = (
        pd.Series(y)
        .value_counts()
        .min()
    )

    splits = min(
        n_splits,
        int(min_class)
    )

    if splits < n_splits:
        print(
            f"  [{label}] reducing to "
            f"{splits} folds "
            f"(rarest class has {min_class})"
        )

    cv = RepeatedStratifiedKFold(
        n_splits=splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE
    )

    res = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        n_jobs=-1,
        error_score="raise"
    )

    out = {}

    for name in SCORING:
        vals = res[
            f"test_{name}"
        ]

        out[name] = (
            float(vals.mean()),
            float(vals.std())
        )

    return out


def permutation_control(
    pipe,
    X,
    y,
    n_perm=20,
    **kw
):
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    y = np.asarray(y)

    scores = []

    for _ in range(n_perm):
        yp = rng.permutation(y)

        result = evaluate(
            pipe,
            X,
            yp,
            n_repeats=1,
            **kw
        )

        scores.append(
            result[
                "balanced_accuracy"
            ][0]
        )

    return (
        float(np.mean(scores)),
        float(np.std(scores))
    )


def crossmodal_retrieval(
    X_morph,
    X_expr,
    n_splits=5,
    n_components=10
):
    n = X_morph.shape[0]

    X = np.hstack([
        X_morph,
        X_expr
    ])

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    top1 = []
    top5 = []
    ranks = []

    for tr, te in kf.split(X):

        al = BlockCCA(
            n_morph=X_morph.shape[1],
            n_components=n_components
        ).fit(
            X[tr]
        )

        Zm = al.pm_.transform(
            al.sm_.transform(
                X_morph[te]
            )
        )

        Ze = al.pe_.transform(
            al.se_.transform(
                X_expr[te]
            )
        )

        Um, Ue = al.cca_.transform(
            Zm,
            Ze
        )

        Um = Um / (
            np.linalg.norm(
                Um,
                axis=1,
                keepdims=True
            )
            +
            1e-9
        )

        Ue = Ue / (
            np.linalg.norm(
                Ue,
                axis=1,
                keepdims=True
            )
            +
            1e-9
        )

        sim = Um @ Ue.T

        order = np.argsort(
            -sim,
            axis=1
        )

        for i in range(
            len(te)
        ):
            rank = int(
                np.where(
                    order[i]
                    ==
                    i
                )[0][0]
            )

            ranks.append(
                rank + 1
            )

            top1.append(
                rank == 0
            )

            top5.append(
                rank < 5
            )

    return {
        "top1":
            float(
                np.mean(top1)
            ),

        "top5":
            float(
                np.mean(top5)
            ),

        "median_rank":
            float(
                np.median(ranks)
            ),

        "n_candidates_per_fold":
            int(
                np.ceil(
                    n / n_splits
                )
            )
    }


def run_all(
    X_morph,
    X_expr,
    y_moa,
    drugs=None,
    min_count=3,
    n_pcs=30,
    cca_components=10,
    run_permutation=True
):
    check_alignment(
        X_morph,
        X_expr,
        y_moa,
        drugs
    )

    (
        X_morph,
        X_expr
    ), y, _ = filter_rare_classes(
        [
            X_morph,
            X_expr
        ],
        y_moa,
        min_count
    )

    n_m = X_morph.shape[1]

    X_cat = np.hstack([
        X_morph,
        X_expr
    ])

    experiments = {
        "A morphology only":
            (
                single_modality_pipeline(
                    min(
                        n_pcs,
                        n_m
                    )
                ),
                X_morph
            ),

        "B expression only":
            (
                single_modality_pipeline(
                    min(
                        n_pcs,
                        X_expr.shape[1]
                    )
                ),
                X_expr
            ),

        "C concat (early fusion)":
            (
                concat_pipeline(
                    n_pcs
                ),
                X_cat
            ),

        "D CCA shared (concat)":
            (
                cca_pipeline(
                    n_m,
                    cca_components,
                    "concat"
                ),
                X_cat
            ),

        "D CCA shared (mean)":
            (
                cca_pipeline(
                    n_m,
                    cca_components,
                    "mean"
                ),
                X_cat
            )
    }

    rows = []

    for label, (
        pipe,
        X
    ) in experiments.items():

        result = evaluate(
            pipe,
            X,
            y,
            label=label
        )

        row = {
            "experiment":
                label,

            "bal_acc":
                result[
                    "balanced_accuracy"
                ][0],

            "bal_acc_sd":
                result[
                    "balanced_accuracy"
                ][1],

            "macro_f1":
                result[
                    "macro_f1"
                ][0],

            "macro_f1_sd":
                result[
                    "macro_f1"
                ][1]
        }

        if run_permutation:

            mean_perm, sd_perm = (
                permutation_control(
                    pipe,
                    X,
                    y,
                    label=label
                )
            )

            row[
                "perm_bal_acc"
            ] = mean_perm

            row[
                "perm_sd"
            ] = sd_perm

        rows.append(
            row
        )

        print(
            f"{label:28s} "
            f"bal_acc "
            f"{row['bal_acc']:.3f} "
            f"+/- "
            f"{row['bal_acc_sd']:.3f} "
            f"macro_f1 "
            f"{row['macro_f1']:.3f}"
        )

    table = pd.DataFrame(
        rows
    )

    retrieval = crossmodal_retrieval(
        X_morph,
        X_expr,
        n_components=cca_components
    )

    print(
        "\ncross-modal retrieval "
        "(held-out):",
        retrieval
    )

    return (
        table,
        retrieval
    )
