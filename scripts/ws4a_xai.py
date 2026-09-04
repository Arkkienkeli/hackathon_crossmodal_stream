#!/usr/bin/env python
"""WS4A — explainability. What the model used, and what that does and does not mean.

    bash scripts/ws4a.sh python /work/scripts/ws4a_xai.py --cell-line hepg2 \
        --target tox_cardiotoxicity --block morphology --model xgboost

Produces, for one (target, block, model):

  1. stability_selection_<...>.png   selection probability per feature, with the
                                     Meinshausen-Buhlmann error bound drawn on
  2. shap_beeswarm_<...>.png         interventional SHAP, per-compound attributions
  3. shap_bar_<...>.png              mean |SHAP| ranking
  4. coefficients_<...>.png          model coefficients / PLS loadings where linear
  5. feature_grammar_<...>.png       attribution aggregated by compartment / family /
                                     channel -- the level at which a biological
                                     statement is defensible
  6. xai_<...>.csv / .json           every number behind those figures

EACH METHOD CARRIES ITS OWN CAVEAT, ON THE FIGURE
-------------------------------------------------
* SHAP is computed with the INTERVENTIONAL estimator against a background sample.
  The default path-dependent TreeSHAP assumes feature independence, and under the
  block correlation CellProfiler features always have that produces attributions
  that are qualitatively wrong, not merely noisy (Aas et al. 2021).
* Impurity and permutation importances are NOT produced at all. Under 0.9
  block-correlation, variables with zero true effect are selected as often as
  genuinely influential ones (Strobl et al. 2008). The config forbids them.
* Stability selection controls false positives; it does NOT promise completeness.
  Its bound assumes exchangeability, which is unlikely to hold on real biological
  data, and it carries a beta-min condition, so genuinely small effects are missed.
* A feature name is not a mechanism. The Carpenter lab's own example: DNA-damaging
  drugs show ACTIN features as the most strongly affected, because the cells detach
  and round up. Every figure that names features says so.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
from ws4a import stabsel as SS                    # noqa: E402

LOG = C.LOG

CAVEAT = ("A feature name is not a mechanism: DNA-damaging drugs show actin features as "
          "most affected (cells detach and round up). Carpenter-Singh Lab.")


def _style(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cfg.section("plots")
    plt.rcParams["figure.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["axes.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["font.size"] = 9
    return plt, list(p.get("palette", ["#2a78d6"])), int(p.get("dpi", 150))


def _footer(fig, text, extra=""):
    fig.text(0.005, 0.005, (text + (" " + extra if extra else "")), fontsize=6.5,
             color="#666", ha="left", va="bottom", wrap=True)


# --------------------------------------------------------------------------- #
def run_stability(X, y, names, kind, cfg, out, tag):
    """Selection probability per feature, with the error bound stated."""
    xs = cfg.section("xai").get("stability_selection", {})
    if not xs.get("enabled", True):
        return None
    B = int(xs.get("B", 200))
    if X.shape[1] > int(xs.get("large_block_threshold", 5000)):
        B = int(xs.get("B_large_block", 100))
        C.log_cap("stability.B", B, f"reduced for a {X.shape[1]}-feature block")
    pi_thr = float(xs.get("pi_thr", 0.7))
    family = "binomial" if kind == "classification" else "gaussian"

    y_num = y
    if kind == "classification":
        from sklearn.preprocessing import LabelEncoder
        y_num = LabelEncoder().fit_transform(y)
        if len(np.unique(y_num)) > 2:
            LOG.warning("stability  : >2 classes; using the largest class one-vs-rest")
            top = pd.Series(y_num).value_counts().index[0]
            y_num = (y_num == top).astype(int)

    t0 = time.time()
    res = SS.stability_selection(X, y_num, mode=xs.get("mode", "cpss"), B=B,
                                 family=family, random_state=0)

    # The error bound is E(V) <= q^2 / ((2*pi_thr - 1) * p), so q -- the average number
    # of variables the base selector picks -- is what controls it. Choose the
    # regularisation region to hit the tolerated E(V) instead of reporting whatever
    # bound the default region happens to give.
    target_ev = float(xs.get("target_ev", 5.0))
    q_target = SS.q_for_target_ev(target_ev, pi_thr, X.shape[1])
    res = res.select_q(q_target)
    C.log_cap("stability.target_ev", target_ev, f"-> q<={q_target:.1f}, achieved q={res.q:.1f}")

    prob = res.prob
    sel = res.selected(pi_thr)
    bound = res.ev_bound_mb(pi_thr)
    LOG.info("stability  : %d/%d selected at pi>=%.2f | q=%.1f | E(V)<=%.2f | %.1fs",
             len(sel), X.shape[1], pi_thr, res.q, bound, time.time() - t0)

    df = pd.DataFrame({"feature": names, "selection_probability": prob,
                       "selected": prob >= pi_thr}).sort_values(
        "selection_probability", ascending=False)
    C.save_table(df, out / f"stability_{tag}.csv")

    plt, palette, dpi = _style(cfg)
    top = df.head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.26 * len(top))))
    colors = [palette[0] if s else "#b0b0ae" for s in top.selected]
    ax.barh(range(len(top)), top.selection_probability, color=colors, edgecolor="#444", lw=0.4)
    ax.axvline(pi_thr, color="#c0392b", ls="--", lw=1.2)
    ax.text(pi_thr, len(top), f" pi_thr={pi_thr}", color="#c0392b", fontsize=7, va="top")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n.split(":")[-1][:52] for n in top.feature], fontsize=7)
    ax.set_xlabel("selection probability over half-samples")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Stability selection — {tag}\n"
                 f"{len(sel)} selected, expected false selections E(V) <= {bound:.2f}",
                 fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _footer(fig, "Controls false positives, does NOT promise completeness: the bound assumes "
                 "exchangeability (unlikely on biological data) and a beta-min condition means "
                 "small true effects are missed.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = out / f"stability_selection_{tag}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return {"n_selected": int(len(sel)), "pi_thr": pi_thr, "q": float(res.q),
            "ev_bound": float(bound), "B": B,
            "selected_features": [str(names[i]) for i in sel[:50]]}


# --------------------------------------------------------------------------- #
def run_shap(model, X, names, cfg, out, tag, kind):
    """Interventional SHAP. The estimator choice is the whole point."""
    xs = cfg.section("xai").get("shap", {})
    if not xs.get("enabled", True):
        return None
    try:
        import shap
    except ImportError:
        LOG.warning("shap not installed — skipping SHAP")
        return None

    bg_n = min(int(xs.get("background_size", 50)), X.shape[0])
    rng = np.random.default_rng(0)
    bg = X[rng.choice(X.shape[0], bg_n, replace=False)]
    C.log_cap("shap.background_size", bg_n, "interventional estimator needs a background sample")

    estimator_used = "interventional"
    caveat_extra = ""
    try:
        expl = shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")
        sv = expl.shap_values(X, check_additivity=False)
    except Exception as exc:                                      # noqa: BLE001
        # XGBoost trees with categorical splits are not supported by the
        # interventional estimator in shap 0.52. Falling back is better than no
        # explanation, but the fallback assumes feature independence, so the figure
        # must say so rather than look identical to a correct one.
        LOG.warning("shap       : interventional unavailable (%s)", str(exc)[:110])
        LOG.warning("shap       : FALLING BACK to tree_path_dependent, which assumes feature "
                    "independence -- attributions among correlated features are unreliable")
        estimator_used = "tree_path_dependent (FALLBACK)"
        caveat_extra = ("ESTIMATOR FALLBACK: interventional was unavailable, so this uses the "
                        "path-dependent estimator, which assumes feature independence. Under "
                        "block-correlated CellProfiler features these attributions can be "
                        "qualitatively wrong -- read the stability-selection figure instead.")
        try:
            expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            sv = expl.shap_values(X, check_additivity=False)
        except Exception as exc2:                                 # noqa: BLE001
            LOG.error("shap       : unavailable for this model (%s)", str(exc2)[:120])
            return None

    sv = np.asarray(sv)
    if sv.ndim == 3:                       # (n, p, classes) or (classes, n, p)
        sv = sv[..., -1] if sv.shape[-1] <= sv.shape[0] else sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)

    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False)
    C.save_table(df, out / f"shap_{tag}.csv")

    plt, palette, dpi = _style(cfg)
    k = int(xs.get("max_display", 25))
    top = df.head(k).iloc[::-1]
    idx = [names.index(f) for f in top.feature]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
    ax.barh(range(len(top)), top.mean_abs_shap, color=palette[0], edgecolor="#444", lw=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n.split(":")[-1][:52] for n in top.feature], fontsize=7)
    ax.set_xlabel("mean |SHAP| (interventional)")
    ax.set_title(f"Feature attribution — {tag}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _footer(fig, (caveat_extra or
                  "Interventional estimator against a background sample; the default "
                  "path-dependent TreeSHAP assumes feature independence and is qualitatively "
                  "wrong under block-correlated features.") + "  " + CAVEAT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / f"shap_bar_{tag}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)

    # beeswarm: direction and spread per compound, not just magnitude
    fig, ax = plt.subplots(figsize=(9, max(4, 0.30 * len(top))))
    for row, j in enumerate(idx):
        vals = sv[:, j]
        colour_by = X[:, j]
        rngj = np.ptp(colour_by)
        norm = (colour_by - colour_by.min()) / (rngj if rngj else 1.0)
        jitter = (np.random.default_rng(row).random(len(vals)) - 0.5) * 0.34
        ax.scatter(vals, np.full(len(vals), row) + jitter, c=norm, cmap="coolwarm",
                   s=11, alpha=0.75, linewidths=0)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n.split(":")[-1][:52] for n in top.feature], fontsize=7)
    ax.set_xlabel("SHAP value  (colour = feature value, blue low / red high)")
    ax.set_title(f"Per-compound attribution — {tag}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _footer(fig, "Each point is one compound. " + CAVEAT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / f"shap_beeswarm_{tag}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return {"top_features": df.head(25).to_dict("records"),
            "estimator": estimator_used, "background_size": bg_n,
            "fallback_used": estimator_used != "interventional"}


# --------------------------------------------------------------------------- #
def run_grammar(attr: pd.DataFrame, cfg, out, tag, value_col):
    """Aggregate attribution to compartment / family / channel.

    This is the level at which a morphology statement is defensible. An individual
    feature name is not a mechanism; a family-level pattern at least describes what
    kind of measurement moved.
    """
    names = attr["feature"].tolist()
    g = C.feature_table(names, cfg)
    g[value_col] = attr[value_col].to_numpy()
    if g["compartment"].isna().all() and g["family"].isna().all():
        LOG.info("grammar    : features are not CellProfiler-shaped — skipping")
        return None

    plt, palette, dpi = _style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, key in zip(axes, ["compartment", "family", "channel"]):
        s = g.groupby(key, dropna=True)[value_col].sum().sort_values(ascending=True)
        if s.empty:
            ax.set_visible(False)
            continue
        ax.barh(range(len(s)), s.to_numpy(),
                color=[palette[i % len(palette)] for i in range(len(s))],
                edgecolor="#444", lw=0.4)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([str(i) for i in s.index], fontsize=8)
        ax.set_xlabel(f"summed {value_col}")
        ax.set_title(f"by {key}", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Attribution by feature grammar — {tag}", fontsize=11)
    _footer(fig, CAVEAT + "  Family-level patterns are more defensible than single features.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    p = out / f"feature_grammar_{tag}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    C.save_table(g.sort_values(value_col, ascending=False), out / f"grammar_{tag}.csv")
    return {k: g.groupby(k, dropna=True)[value_col].sum().sort_values(ascending=False)
            .head(6).to_dict() for k in ("compartment", "family", "channel")}


def run_coefficients(model, names, cfg, out, tag):
    """Linear coefficients / PLS loadings, where the model has them."""
    est = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    # Order matters. For SparsePLSDA the sparsity lives in x_weights_; x_loadings_
    # stays DENSE by construction, so reading loadings first would draw a dense
    # figure for a model that actually selected 20 of 636 features.
    coef, source = None, None
    for attr in ("coef_", "x_weights_", "x_loadings_"):
        if hasattr(est, attr):
            coef = np.asarray(getattr(est, attr))
            source = attr
            break
    if coef is None:
        return None
    if coef.ndim > 1:
        coef = coef[0] if coef.shape[0] < coef.shape[-1] else coef[:, 0]
    coef = np.ravel(coef)
    if coef.size != len(names):
        LOG.warning("coefficients: length %d != %d features — skipping", coef.size, len(names))
        return None

    df = pd.DataFrame({"feature": names, "coefficient": coef})
    df["abs"] = df.coefficient.abs()
    df = df.sort_values("abs", ascending=False)
    C.save_table(df.drop(columns="abs"), out / f"coefficients_{tag}.csv")

    plt, palette, dpi = _style(cfg)
    top = df.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(top))))
    ax.barh(range(len(top)), top.coefficient,
            color=[palette[0] if v > 0 else palette[1] for v in top.coefficient],
            edgecolor="#444", lw=0.4)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([n.split(":")[-1][:52] for n in top.feature], fontsize=7)
    ax.set_xlabel("coefficient  (blue positive, orange negative)")
    ax.set_title(f"Model coefficients ({source}) — {tag}", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _footer(fig, "Coefficients of correlated features split arbitrarily between them; read "
                 "the stability-selection figure alongside this one. " + CAVEAT)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / f"coefficients_{tag}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    LOG.info("coefficients: read from %s (%d non-zero of %d)",
             source, int((np.abs(coef) > 0).sum()), coef.size)
    return {"source": source, "n_nonzero": int((np.abs(coef) > 0).sum()),
            "top": df.head(25).drop(columns="abs").to_dict("records")}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--target", required=True, help="an obs column, e.g. tox_cardiotoxicity")
    ap.add_argument("--block", default="morphology",
                    help="modality, or several joined with '+' (e.g. morphology+ecfp)")
    ap.add_argument("--model", default="xgboost",
                    choices=["xgboost", "elastic_net", "linear_svm", "sparse_plsda"])
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()

    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    if cfg.section("xai").get("forbid_impurity_importance", True):
        LOG.info("xai        : impurity/permutation importance is DISABLED by config "
                 "(selects zero-effect correlated variables as often as real ones)")
    device = C.resolve_device(args.device or cfg.section("ml").get("xgboost", {}).get("device", "auto"))
    out = Path(args.out) if args.out else C.outputs_dir(cfg, "xai")

    blocks = C.load_mudata_blocks(cfg, args.cell_line)
    spec = None
    for tspec in cfg.section("ml").get("targets", {}).values():
        cols = tspec.get("columns", [tspec.get("column")])
        if args.target in cols:
            spec = dict(tspec); spec["column"] = args.target
            break
    if spec is None:
        spec = {"column": args.target, "kind": "classification"}

    y_all, mask, info = C.prepare_target(blocks.obs, spec, args.target)
    ok, why = C.target_is_usable(y_all, mask, spec.get("kind", "classification"),
                                 cfg.section("ml").get("target_guards", {}), args.target)
    if not ok:
        LOG.error("target %s is not usable: %s", args.target, why)
        return 1

    blk = args.block.split("+") if "+" in args.block else args.block
    Xb, names = blocks.get(blk)
    X, y = Xb[mask], y_all[mask]
    kind = spec.get("kind", "classification")
    tag = f"{args.cell_line}_{args.target}_{args.block.replace('+','-')}_{args.model}"
    LOG.info("xai        : %s  X=%s", tag, X.shape)

    # Fit ONE model on all usable rows. This is an explanation of a fitted model,
    # not a performance estimate -- ws4a_ml.py owns performance, with nested CV.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ws4a_ml import build_model
    y_enc = y
    if kind == "classification":
        from sklearn.preprocessing import LabelEncoder
        y_enc = LabelEncoder().fit_transform(y)
    est, _ = build_model(args.model, kind, cfg, device, len(np.unique(y_enc)))
    if est is None:
        LOG.error("model %s unavailable", args.model)
        return 1
    est.fit(X, y_enc)
    LOG.info("fitted     : %s on %d rows x %d features", args.model, *X.shape)

    results = {"tag": tag, "target": args.target, "block": args.block,
               "model": args.model, "n": int(X.shape[0]),
               "n_features": int(X.shape[1]), "target_info": info,
               "note": "explanation of a model fitted on all usable rows; "
                       "performance estimates come from ws4a_ml.py with nested CV"}

    if not args.no_stability:
        results["stability"] = run_stability(X, y, names, kind, cfg, out, tag)

    Xs = est.named_steps["scale"].transform(X) if hasattr(est, "named_steps") else X
    inner = est.named_steps["model"] if hasattr(est, "named_steps") else est
    if not args.no_shap:
        sh = run_shap(inner, Xs, names, cfg, out, tag, kind)
        results["shap"] = sh
        if sh:
            attr = pd.DataFrame(sh["top_features"])
            full = pd.read_csv(out / f"shap_{tag}.csv")
            results["grammar"] = run_grammar(full, cfg, out, tag, "mean_abs_shap")

    results["coefficients"] = run_coefficients(est, names, cfg, out, tag)
    C.save_json(results, out / f"xai_{tag}.json")

    print("\n" + "=" * 92)
    print(f"xAI — {tag}")
    print("=" * 92)
    if results.get("stability"):
        s = results["stability"]
        print(f"  stability selection : {s['n_selected']} features at pi>={s['pi_thr']}, "
              f"E(V) <= {s['ev_bound']:.2f} false selections")
        for f in s["selected_features"][:8]:
            print(f"      {f}")
    if results.get("grammar"):
        for k, v in results["grammar"].items():
            if v:
                print(f"  attribution by {k:12s}: "
                      + ", ".join(f"{kk}={vv:.3g}" for kk, vv in list(v.items())[:4]))
    print(f"\n  {CAVEAT}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
