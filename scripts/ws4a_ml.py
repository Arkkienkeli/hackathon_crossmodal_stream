#!/usr/bin/env python
"""WS4A Tier 2 — supervised models, with nested CV and a pre-declared model set.

    bash scripts/ws4a.sh python /work/scripts/ws4a_ml.py --cell-line hepg2 --target moa
    python scripts/ws4a_ml.py --cell-line hepg2 --target moa --quick

WHY THIS IS SHAPED THE WAY IT IS
--------------------------------
At n ~ 60 labelled compounds with p up to 41,780, the danger is not underfitting.
Picking the best of 124 classifier variants by cross-validation on PERMUTED,
uninformative data gives median error rates of 31-41% against a 50% chance baseline
(Boulesteix & Strobl), and the bias is WORST at small n. So:

  * the model list is PRE-DECLARED in the config and every model is always reported;
  * cross-validation is genuinely nested -- scaling, feature selection and every
    hyper-parameter are fitted inside the inner loop only;
  * a permuted-label control runs for every (target, block, model) combination, and
    the honest effect is the gap between the real score and that control, not the
    real score itself;
  * the ECFP-only block runs first, as the QSAR control. If chemical structure alone
    predicts the label as well as morphology does, then morphology added nothing,
    and that is the finding.

The OpenScreen data has three independent sites measuring the same 118 compounds, so
where sites are available the outer loop is LEAVE-ONE-SITE-OUT: genuine external
validation, which is far stronger evidence than a CV fold at this sample size.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
from ws4a import splsda as SP                     # noqa: E402
from ws4a import tuning as TU                     # noqa: E402

warnings.filterwarnings("ignore")
LOG = C.LOG


# --------------------------------------------------------------------------- #
# models. Each returns an unfitted sklearn-compatible estimator plus its grid.
# Scaling lives INSIDE the pipeline so it is refitted per training split.
# --------------------------------------------------------------------------- #
def build_model(name: str, kind: str, cfg, device: str, n_classes: int = 2,
                pos_weight: float | None = None, seed: int = 0):
    from sklearn.linear_model import ElasticNet, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC, LinearSVR

    ml = cfg.section("ml")
    sc = ("scale", StandardScaler())

    if name == "elastic_net":
        p = ml.get("elastic_net", {})
        # saga is the only solver supporting an elastic-net penalty for
        # classification and is slow on wide standardised data; 5000 iterations cost
        # minutes per fold and bought nothing measurable. Configurable, so a run that
        # emits ConvergenceWarning can raise it.
        max_iter = int(p.get("max_iter", 1000))
        tol = float(p.get("tol", 1e-3))
        if kind == "classification":
            # No `penalty=`. In sklearn 1.9 its default is literally the string
            # "deprecated" and it is REMOVED in 1.10; l1_ratio alone selects the
            # penalty (0 = ridge, 1 = lasso). The old form emitted a FutureWarning on
            # every single fit and would break outright on upgrade.
            est = LogisticRegression(solver="saga", max_iter=max_iter, tol=tol,
                                     random_state=seed)
            grid = {"model__C": [0.01, 0.1, 1.0],
                    "model__l1_ratio": list(p.get("l1_ratios", [0.5, 0.9]))}
        else:
            est = ElasticNet(max_iter=max_iter, tol=tol, random_state=seed)
            grid = {"model__alpha": np.logspace(-3, 1, 12).tolist(),
                    "model__l1_ratio": list(p.get("l1_ratios", [0.5, 0.9]))}
        return Pipeline([sc, ("model", est)]), grid

    if name == "linear_svm":
        p = ml.get("linear_svm", {})
        est = LinearSVC(max_iter=20000, dual="auto", random_state=seed) \
            if kind == "classification" \
            else LinearSVR(max_iter=20000, dual="auto", random_state=seed)
        return Pipeline([sc, ("model", est)]), {"model__C": list(p.get("C", [0.01, 0.1, 1.0]))}

    if name == "sparse_plsda":
        if kind != "classification":
            return None, None
        p = ml.get("sparse_plsda", {})
        est = SP.SparsePLSDA(n_components=int(p.get("n_components", 2)))
        return Pipeline([sc, ("model", est)]), {"model__keep_x": list(p.get("keep_x", [25, 50]))}

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError:
            LOG.warning("xgboost not installed — skipping")
            return None, None
        p = dict(ml.get("xgboost", {}))
        p.pop("early_stopping_rounds", None)
        p.pop("device", None)
        common = dict(
            max_depth=int(p.get("max_depth", 2)),
            n_estimators=int(p.get("n_estimators", 200)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            subsample=float(p.get("subsample", 0.8)),
            colsample_bytree=float(p.get("colsample_bytree", 0.3)),
            # 1, not 5. MEASURED on the real HepG2 moa target (14 positives / 50
            # negatives): at min_child_weight=5 only 2 of 200 trees made a real split
            # and the model predicted a single class, giving balanced accuracy of
            # exactly 0.500 +- 0.000 on every feature block. At 1, all 200 split.
            min_child_weight=float(p.get("min_child_weight", 1)),
            reg_alpha=float(p.get("reg_alpha", 1.0)),
            reg_lambda=float(p.get("reg_lambda", 5.0)),
            tree_method="hist",
            device=device,
            n_jobs=1,
            verbosity=0,
        )
        if kind == "classification":
            extra = {"objective": "multi:softprob" if n_classes > 2 else "binary:logistic"}
            # 14 vs 50 is imbalanced enough that the majority class dominates the
            # objective; scale_pos_weight rebalances it, which matters because the
            # score is balanced accuracy.
            if n_classes == 2 and pos_weight:
                extra["scale_pos_weight"] = float(pos_weight)
            est = XGBClassifier(**common, **extra)
        else:
            est = XGBRegressor(**common)
        # Grid kept deliberately tiny: every extra candidate inflates the
        # optimistic selection bias, which is worst exactly at this n.
        return Pipeline([sc, ("model", est)]), {"model__max_depth": [1, 2]}

    raise KeyError(f"unknown model {name!r}")


# --------------------------------------------------------------------------- #
def outer_splits(y, groups, kind, cfg, seed):
    """Leave-one-site-out when sites exist, else stratified/grouped K-fold.

    Splits are always by COMPOUND, so replicate rows of one compound can never
    straddle a split.
    """
    from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                         RepeatedStratifiedKFold, StratifiedKFold)
    cv = cfg.section("ml").get("cv", {})
    if groups is not None and len(np.unique(groups)) > 1 and cv.get("leave_one_site_out", True):
        LOG.info("cv         : leave-one-site-out over %d sites", len(np.unique(groups)))
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups)), "leave_one_site_out"

    k = int(cv.get("outer_folds", 5))
    # cv.repeats was in the config and the docs from the start and was NOT applied
    # until 2026-09-03: RepeatedStratifiedKFold was imported and never called, so
    # every earlier run (job 1349676 included) was a single k-fold. The scheme
    # label below says which one a table came from.
    reps = max(1, int(cv.get("repeats", 1)))
    tag = f"_x{reps}" if reps > 1 else ""
    if kind == "classification":
        _, counts = np.unique(y, return_counts=True)
        k = max(2, min(k, int(counts.min())))
        if k < int(cv.get("outer_folds", 5)):
            C.log_cap("cv.outer_folds", k, "reduced to the smallest class size")
        sp = (RepeatedStratifiedKFold(n_splits=k, n_repeats=reps, random_state=seed)
              if reps > 1 else StratifiedKFold(n_splits=k, shuffle=True, random_state=seed))
        return list(sp.split(np.zeros(len(y)), y)), f"stratified_{k}fold{tag}"
    from sklearn.model_selection import KFold, RepeatedKFold
    sp = (RepeatedKFold(n_splits=k, n_repeats=reps, random_state=seed)
          if reps > 1 else KFold(n_splits=k, shuffle=True, random_state=seed))
    return list(sp.split(np.zeros(len(y)))), f"kfold_{k}{tag}"


def score_of(kind):
    return "balanced_accuracy" if kind == "classification" else "neg_root_mean_squared_error"


def evaluate(X, y, kind, model_name, cfg, device, seed, groups=None, permuted=False,
             n_jobs=1, n_trials=0, pre_steps=None):
    """One nested-CV estimate. Everything selectable is fitted in the inner loop.

    `n_trials > 0` replaces the fixed grid with an Optuna search, still confined to
    the inner loop so the outer estimate stays unbiased. The caller MUST pass the
    same n_trials for the permuted control -- tuning the real labels harder than the
    control inflates the reported gap by exactly the selection bias being measured.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold

    if permuted:
        rng = np.random.default_rng(seed)
        y = rng.permutation(y)

    n_classes = len(np.unique(y)) if kind == "classification" else 0
    if kind == "classification" and n_classes < 2:
        return None

    pos_weight = None
    if kind == "classification" and n_classes == 2:
        _, cnt = np.unique(y, return_counts=True)
        pos_weight = float(cnt.max() / max(cnt.min(), 1))
    est, grid = build_model(model_name, kind, cfg, device, n_classes, pos_weight,
                            seed=seed)
    if est is None:
        return None
    if pre_steps:
        # Prepended INSIDE the pipeline, so anything they do is fitted on the
        # training part of each fold only -- which is the whole point when the step
        # is a feature selector.
        from sklearn.pipeline import Pipeline
        est = Pipeline(list(pre_steps) + list(est.steps))

    splits, scheme = outer_splits(y, groups, kind, cfg, seed)
    inner_k = int(cfg.section("ml").get("cv", {}).get("inner_folds", 3))
    scoring = score_of(kind)

    y_enc = y
    if kind == "classification":
        from sklearn.preprocessing import LabelEncoder
        y_enc = LabelEncoder().fit_transform(y)

    def _one_split(item):
        """One outer fold, self-contained so the folds can run in parallel.

        Seeded by fold index, not by call order, so the result is identical whether
        the folds run serially or on 25 workers -- asserted by the selftest.
        """
        fold_i, (tr, te) = item
        ytr = y_enc[tr]
        if kind == "classification":
            _, cnt = np.unique(ytr, return_counts=True)
            k_in = max(2, min(inner_k, int(cnt.min())))
            inner = StratifiedKFold(n_splits=k_in, shuffle=True, random_state=seed)
        else:
            inner = KFold(n_splits=inner_k, shuffle=True, random_state=seed)
        try:
            if n_trials > 0:
                # The search sees ONLY X[tr]. X[te] never influences it, which is
                # what keeps this outer score honest at any budget.
                best, tinfo = TU.tune_fit(est, model_name, X[tr], ytr, kind, inner,
                                          scoring, n_trials, seed + fold_i, n_jobs=1)
                return float(_score_of(best, X[te], y_enc[te], scoring)), tinfo
            # Inner fits are serial ON PURPOSE: the parallelism is across the outer
            # folds, one worker each. A nested pool here would oversubscribe.
            gs = GridSearchCV(est, grid, cv=inner, scoring=scoring, n_jobs=1,
                              refit=True)
            gs.fit(X[tr], ytr)
            return float(gs.score(X[te], y_enc[te])), None
        except Exception as exc:                                  # noqa: BLE001
            LOG.warning("   %s: fold %d failed (%s)", model_name, fold_i,
                        str(exc)[:110])
            return None

    # The outer folds are the parallel unit. 25 of them (5 x 5 repeats) run in ~the
    # time of one, instead of 25x that -- measured 819 s -> ~35 s on the 41,780-
    # feature block. Order is preserved, so the mean is bit-identical to serial.
    t0 = time.time()
    results = C.parallel_map(_one_split, list(enumerate(splits)),
                             n_jobs=max(1, min(n_jobs, len(splits))))
    scores = [r[0] for r in results if r is not None]
    tuned_info = [r[1] for r in results if r is not None and r[1] is not None]
    if not scores:
        return None

    out = {
        "model": model_name, "n": int(len(y)), "n_features": int(X.shape[1]),
        "kind": kind, "cv_scheme": scheme, "scoring": scoring,
        "score_mean": float(np.mean(scores)), "score_std": float(np.std(scores)),
        "n_folds": len(scores), "permuted": bool(permuted),
        "n_trials": int(n_trials),
        "seconds": round(time.time() - t0, 1),
    }
    if tuned_info:
        ok = [t for t in tuned_info if t.get("tuned")]
        out["tuned_folds"] = len(ok)
        if ok:
            out["mean_inner_best"] = float(np.mean([t["best_inner_score"] for t in ok]))
            out["best_params_fold0"] = ok[0].get("best_params")
    return out


def bias_sweep(X, y, kind, model_name, cfg, device, seed, chance, budgets, repeats,
               n_jobs=1):
    """How much apparent performance does a bigger search MANUFACTURE from noise?

    Runs the PERMUTED labels only -- there is no signal in them by construction, so
    every point above `chance` is selection bias and nothing else. Sweeping the
    budget turns that bias into a measured curve instead of a caveat, which is the
    only way to say what a given n_trials costs in honesty.

    Repeats matter: one permutation at one budget is a single draw from a wide
    distribution. `repeats` independent permutations per budget are averaged, and
    the spread is reported alongside.
    """
    rows = []
    for b in budgets:
        vals = []
        for rep in range(repeats):
            r = evaluate(X, y, kind, model_name, cfg, device, seed + 1000 * rep,
                         permuted=True, n_jobs=n_jobs, n_trials=b)
            if r:
                vals.append(r["score_mean"])
        if not vals:
            continue
        rows.append({"model": model_name, "n_trials": int(b), "n_repeats": len(vals),
                     "permuted_score_mean": float(np.mean(vals)),
                     "permuted_score_std": float(np.std(vals)),
                     "chance": float(chance),
                     "bias_over_chance": float(np.mean(vals) - chance)})
        LOG.info("    bias sweep  %-13s n_trials=%-4d permuted=%.3f +-%.3f "
                 "(chance %.3f, bias %+.3f)", model_name, b, np.mean(vals),
                 np.std(vals), chance, np.mean(vals) - chance)
    return rows


def write_tables(df: pd.DataFrame, out_dir: Path, cell_line: str) -> pd.DataFrame:
    """ml_<cl>.csv (every row) and ml_summary_<cl>.csv (real rows + honest gap).

    Shared with ws4a_merge.py so a sharded array run produces byte-for-byte the
    same files as a single-job run.
    """
    C.save_table(df, out_dir / f"ml_{cell_line}.csv")
    real = df[~df.permuted].copy()
    perm = df[df.permuted][["target", "block", "model", "score_mean"]].rename(
        columns={"score_mean": "permuted_score"})
    merged = real.merge(perm, on=["target", "block", "model"], how="left")
    merged["gap_vs_permuted"] = merged.score_mean - merged.permuted_score
    merged = merged.sort_values(["target", "gap_vs_permuted"], ascending=[True, False])
    C.save_table(merged, out_dir / f"ml_summary_{cell_line}.csv")
    return merged


def _resolve_target(ml: dict, name: str):
    """A top-level key under ml.targets, or ONE column of a multi-column target.

    Returns (spec, column_or_None). Letting a single column be addressed is what
    makes the work shardable: `toxicity` alone is 5/6 of the ML stage.
    """
    targets = ml.get("targets", {})
    if name in targets:
        return targets[name], None
    for spec in targets.values():
        cols = spec.get("columns", [spec.get("column")])
        if name in cols:
            return spec, name
    return None, None


def _score_of(est, X, y, scoring):
    """Score a fitted estimator the same way GridSearchCV would."""
    from sklearn.metrics import get_scorer
    return get_scorer(scoring)(est, X, y)


def chance_level(y, kind):
    if kind != "classification":
        return 0.0
    return 1.0 / len(np.unique(y))          # balanced accuracy chance


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--target", default="moa",
                    help="a key under ml.targets, ONE of its columns (e.g. "
                         "tox_renal_toxicity), or 'all'")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated subset of ml.feature_blocks by name; a "
                         "concatenated block is written a+b, e.g. morphology+ecfp")
    ap.add_argument("--list-units", default=None, metavar="PATH",
                    help="write one usable (target column, block) pair per line, "
                         "tab-separated, to PATH and exit -- the work list for "
                         "slurm/ws4a_ml_array.sbatch. (A file, not stdout: the log "
                         "also goes to stdout.)")
    ap.add_argument("--models", default=None, help="comma-separated subset")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="parallel workers (default: compute.n_jobs, capped by SLURM_CPUS_PER_TASK)")
    ap.add_argument("--n-trials", type=int, default=None,
                    help="Optuna trials per outer fold (0 = fixed grid). Applied "
                         "IDENTICALLY to the permuted control.")
    ap.add_argument("--bias-sweep", default=None,
                    help="comma-separated budgets, e.g. 0,5,15,50 -- runs PERMUTED "
                         "labels only at each budget, measuring how much apparent "
                         "performance a larger search manufactures on data with no signal")
    ap.add_argument("--no-permuted-control", action="store_true")
    ap.add_argument("--quick", action="store_true", help="one repeat, tiny grids")
    args = ap.parse_args()

    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    ml = cfg.section("ml")
    seed = args.seed if args.seed is not None else int(ml.get("cv", {}).get("seed", 0))
    device = C.resolve_device(args.device or ml.get("xgboost", {}).get("device", "auto"))
    n_jobs, _inner = C.resolve_jobs(cfg, args.n_jobs)
    tune = cfg.section("ml").get("tuning", {})
    n_trials = args.n_trials if args.n_trials is not None else int(tune.get("n_trials", 0))
    if n_trials and not TU.HAVE_OPTUNA and not args.list_units:
        LOG.error("--n-trials %d requested but optuna is not installed", n_trials)
        return 1
    if n_trials:
        C.log_cap("tuning.n_trials", n_trials,
                  "Optuna, inner loop only; the permuted control gets the SAME budget")
    else:
        C.log_cap("tuning.n_trials", 0, "fixed pre-declared grid (no search)")
    out_dir = Path(args.out) if args.out else C.outputs_dir(cfg, "ml")

    blocks = C.load_mudata_blocks(cfg, args.cell_line)
    models = [m.strip() for m in (args.models.split(",") if args.models
                                  else ml.get("models", []))]
    targets = list(ml.get("targets", {})) if args.target == "all" else [args.target]
    feature_blocks = ml.get("feature_blocks", ["ecfp", "morphology"])
    if args.blocks:
        wanted = [b.strip() for b in args.blocks.split(",") if b.strip()]
        by_name = {(b if isinstance(b, str) else "+".join(b)): b for b in feature_blocks}
        missing = [w for w in wanted if w not in by_name]
        if missing:
            LOG.error("--blocks %s not in ml.feature_blocks %s", missing, list(by_name))
            return 2
        feature_blocks = [by_name[w] for w in wanted]
    if args.quick:
        feature_blocks = feature_blocks[:2]
        models = models[:2]
        cfg.setdefault("ml", {}).setdefault("cv", {})["repeats"] = 1
        # The budget has to shrink too. Trials multiply fits per inner fold, so a
        # "quick" run at the full budget is slower than the FULL untuned run -- which
        # defeats the point of a smoke test.
        if n_trials > 5:
            LOG.warning("QUICK MODE — n_trials %d -> 5. Tuning is the dominant cost.",
                        n_trials)
            n_trials = 5
        LOG.warning("QUICK MODE — %d blocks, %d models, n_trials=%d. Smoke test only.",
                    len(feature_blocks), len(models), n_trials)

    C.log_cap("models", ",".join(models), "pre-declared; every one is reported")
    C.log_cap("feature_blocks", str(feature_blocks))

    budgets = None
    sweep_reps = int(tune.get("bias_sweep_repeats", 10))
    if args.bias_sweep:
        budgets = sorted({int(b) for b in args.bias_sweep.split(",") if b.strip() != ""})
        if not TU.HAVE_OPTUNA and any(b > 0 for b in budgets):
            LOG.error("--bias-sweep needs optuna for any budget > 0"); return 1
        LOG.warning("BIAS SWEEP: permuted labels only, budgets %s x %d repeats. "
                    "No real-label score is produced by this mode.",
                    budgets, sweep_reps)

    rows, skipped, sweep_rows, units = [], [], [], []
    for tname in targets:
        spec, only_col = _resolve_target(ml, tname)
        if spec is None:
            LOG.warning("no target spec or column named %r — skipping", tname)
            continue
        # a multi-column target (toxicity) expands into one run per column
        columns = [only_col] if only_col else spec.get("columns", [spec.get("column", tname)])
        for col in columns:
            sub = dict(spec); sub["column"] = col
            try:
                y_all, mask, info = C.prepare_target(blocks.obs, sub, col)
            except KeyError as exc:
                LOG.warning("%s: %s", col, exc)
                continue
            ok, why = C.target_is_usable(y_all, mask, sub.get("kind", "classification"),
                                         ml.get("target_guards", {}), col)
            if not ok:
                LOG.warning("SKIP %-28s %s", col, why)
                skipped.append({"target": col, "reason": why, "n_labelled": int(mask.sum())})
                continue

            y = y_all[mask]
            kind = sub.get("kind", "classification")
            chance = chance_level(y, kind)

            if args.list_units:
                units += [(col, blk if isinstance(blk, str) else "+".join(blk))
                          for blk in feature_blocks]
                continue

            for blk in feature_blocks:
                Xb, names = blocks.get(blk)
                X = Xb[mask]
                bname = blk if isinstance(blk, str) else "+".join(blk)
                for mname in models:
                    if budgets is not None:
                        sweep_rows += [dict(rw, target=col, block=bname)
                                       for rw in bias_sweep(
                                           X, y, kind, mname, cfg, device, seed, chance,
                                           budgets, sweep_reps, n_jobs=n_jobs)]
                        continue
                    r = evaluate(X, y, kind, mname, cfg, device, seed,
                                 n_jobs=n_jobs, n_trials=n_trials)
                    if r is None:
                        continue
                    r.update(target=col, block=bname, chance=chance)
                    r["degenerate"] = bool(
                        r["score_std"] < 1e-9 and abs(r["score_mean"] - chance) < 1e-6)
                    if r["degenerate"]:
                        LOG.warning("  %-12s %-22s %-13s DEGENERATE: exactly chance with "
                                    "zero variance -- the model predicted one class every "
                                    "fold. This is a broken configuration, not a null result.",
                                    col, bname, mname)
                    LOG.info("  %-12s %-22s %-13s score=%.3f +-%.3f (chance %.3f) %ds",
                             col, bname, mname, r["score_mean"], r["score_std"],
                             chance, r["seconds"])
                    rows.append(r)

                    if not args.no_permuted_control:
                        # SAME n_trials, deliberately. A control tuned less hard
                        # than the real model turns selection bias into a fake gap.
                        rp = evaluate(X, y, kind, mname, cfg, device, seed,
                                      permuted=True, n_jobs=n_jobs, n_trials=n_trials)
                        if rp:
                            rp.update(target=col, block=bname, chance=chance)
                            rows.append(rp)
                            LOG.info("  %-12s %-22s %-13s PERMUTED=%.3f  -> honest gap %+.3f",
                                     col, bname, mname, rp["score_mean"],
                                     r["score_mean"] - rp["score_mean"])

    if args.list_units:
        # One line per unit, tab-separated: the array wrapper reads this file
        # line-by-line by SLURM_ARRAY_TASK_ID.
        Path(args.list_units).write_text("".join(f"{c}\t{b}\n" for c, b in units))
        LOG.info("%d usable (target, block) unit(s) -> %s; %d target(s) guarded out",
                 len(units), args.list_units, len(skipped))
        return 0

    if budgets is not None:
        if not sweep_rows:
            LOG.error("bias sweep produced nothing — every target failed its guard")
            return 1
        sdf = pd.DataFrame(sweep_rows)
        C.save_table(sdf, out_dir / f"bias_sweep_{args.cell_line}.csv")
        print("\n" + "=" * 96)
        print(f"SELECTION-BIAS SWEEP — {args.cell_line}, PERMUTED LABELS ONLY")
        print("  There is no signal in these labels. Every point above `chance` was")
        print("  manufactured by the search, and `bias_over_chance` is its price.")
        print("=" * 96)
        with pd.option_context("display.width", 200):
            print(sdf.sort_values(["target", "block", "model", "n_trials"]).to_string(index=False))
        print(f"\nwrote -> {out_dir}")
        return 0

    if skipped:
        C.save_table(pd.DataFrame(skipped), out_dir / f"skipped_targets_{args.cell_line}.csv")
    if not rows:
        LOG.error("nothing was evaluated — every target failed its guard:")
        for sk in skipped:
            LOG.error("   %-30s %s", sk["target"], sk["reason"])
        return 1

    df = pd.DataFrame(rows)
    merged = write_tables(df, out_dir, args.cell_line)

    print("\n" + "=" * 104)
    print(f"TIER 2 — supervised, {args.cell_line}")
    print("  The honest effect is `gap_vs_permuted`, NOT score_mean. A model that scores well")
    print("  on permuted labels is measuring selection bias, which is worst at this n.")
    print("  `ecfp` is the QSAR control: if morphology does not beat it, morphology added nothing.")
    print("=" * 104)
    cols = ["target", "block", "model", "n", "n_features", "score_mean",
            "permuted_score", "gap_vs_permuted", "chance", "cv_scheme"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(merged[[c for c in cols if c in merged.columns]].to_string(index=False))
    if skipped:
        print("\nskipped targets (guarded, not silently scored):")
        for sk in skipped:
            print(f"  {sk['target']:30s} {sk['reason']}")
    print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
