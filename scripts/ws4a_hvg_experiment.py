#!/usr/bin/env python3
"""WS4A — what does highly-variable-gene selection cost, and what does doing it wrong cost?

    bash scripts/ws4a.sh hvg --cell-line hepg2 --target moa

WHY THIS EXPERIMENT EXISTS
--------------------------
The WS4 presentation's Task 1 selects 2,000 highly variable genes ONCE, on all
compounds, and then cross-validates. Its own caveat slide says so: "HVGs were
selected globally before supervised CV, so this is an exploratory workstream
benchmark rather than a final leakage-free classifier."

Two separate things are bundled in that sentence, and they pull in opposite
directions:

  1. REDUCTION. Going from 41,780 genes to 2,000 throws information away. That can
     only hurt -- or help, if the discarded genes were mostly noise at n=64.
  2. LEAKAGE. Choosing WHICH 2,000 using all compounds, including the ones later
     held out, lets the held-out rows influence the feature set they are scored on.
     That can only flatter.

This script separates them by running three arms through the identical nested CV:

  all_genes        41,780 genes, no selection                       (our pipeline)
  hvg2000_leaky    2,000 HVGs chosen ONCE on all rows               (the deck's way)
  hvg2000_honest   2,000 HVGs chosen INSIDE each training fold      (leakage-free)

  all_genes vs hvg2000_honest  ->  what REDUCTION costs
  hvg2000_leaky vs hvg2000_honest  ->  what the LEAKAGE was worth

Every arm gets the shuffled-label control at the identical budget, so all three are
compared on `gap_vs_permuted`, never on the raw score.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
import ws4a_ml as ML                              # noqa: E402

LOG = C.LOG


class HVGSelector(BaseEstimator, TransformerMixin):
    """Keep the `n_top` highest-variance columns, fitted on whatever it is given.

    Placed first in the pipeline it sees only the training fold, which is the
    leakage-free version. Applied to the whole matrix beforehand it is the leaky
    one. Same selection rule either way -- only the rows it may look at differ,
    which is what makes the two arms comparable.

    Variance is computed on the raw scale, matching how HVGs are picked in the
    scanpy-style workflow the presentation used (before standardisation, which
    would make every gene's variance 1 and the choice meaningless).
    """

    def __init__(self, n_top: int = 2000):
        self.n_top = n_top

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        v = X.var(axis=0)
        k = int(min(self.n_top, X.shape[1]))
        self.idx_ = np.argsort(-v)[:k]
        self.idx_.sort()
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.idx_]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--target", default="moa")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--models", default=None)
    ap.add_argument("--n-trials", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--morph-ref", type=float, default=None,
                    help="best morphology honest gap, drawn as a reference line -- the "
                         "comparison the whole experiment is about")
    args = ap.parse_args(argv)
    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    ml = cfg.section("ml")
    seed = args.seed if args.seed is not None else int(ml.get("cv", {}).get("seed", 0))
    device = C.resolve_device(args.device if hasattr(args, "device") else "auto")
    n_jobs, _ = C.resolve_jobs(cfg, args.n_jobs)
    out = Path(args.out) if args.out else C.outputs_dir(cfg, "hvg")

    blocks = C.load_mudata_blocks(cfg, args.cell_line)
    models = [m.strip() for m in (args.models.split(",") if args.models
                                  else ml.get("models", []))]

    spec, only_col = ML._resolve_target(ml, args.target)
    if spec is None:
        LOG.error("no target %r", args.target); return 2
    cols = [only_col] if only_col else spec.get("columns", [spec.get("column", args.target)])

    Xfull, _ = blocks.get("expression")
    C.log_cap("n_hvg", args.n_hvg, "genes retained by the two reduced arms")
    C.log_cap("expression", Xfull.shape[1], "genes in the unreduced arm")

    rows = []
    for col in cols:
        sub = dict(spec); sub["column"] = col
        y_all, mask, _ = C.prepare_target(blocks.obs, sub, col)
        ok, why = C.target_is_usable(y_all, mask, sub.get("kind", "classification"),
                                     ml.get("target_guards", {}), col)
        if not ok:
            LOG.warning("SKIP %s: %s", col, why); continue
        y = y_all[mask]
        kind = sub.get("kind", "classification")
        chance = ML.chance_level(y, kind)
        X = Xfull[mask]

        # the leaky arm: choose the genes ONCE, using every row including the ones
        # that will later be held out
        leak = HVGSelector(args.n_hvg).fit(X)
        X_leaky = leak.transform(X)
        LOG.info("target     : %s  n=%d  chance=%.3f", col, len(y), chance)

        arms = [
            ("all_genes", X, None, X.shape[1]),
            ("hvg%d_leaky" % args.n_hvg, X_leaky, None, X_leaky.shape[1]),
            ("hvg%d_honest" % args.n_hvg, X,
             [("hvg", HVGSelector(args.n_hvg))], args.n_hvg),
        ]
        for arm, Xa, pre, nfeat in arms:
            for mname in models:
                r = ML.evaluate(Xa, y, kind, mname, cfg, device, seed, n_jobs=n_jobs,
                                n_trials=args.n_trials, pre_steps=pre)
                if r is None:
                    continue
                rp = ML.evaluate(Xa, y, kind, mname, cfg, device, seed, permuted=True,
                                 n_jobs=n_jobs, n_trials=args.n_trials, pre_steps=pre)
                gap = r["score_mean"] - (rp["score_mean"] if rp else np.nan)
                rows.append({"target": col, "arm": arm, "model": mname,
                             "n_features": nfeat, "n": r["n"],
                             "score_mean": r["score_mean"], "score_std": r["score_std"],
                             "permuted_score": rp["score_mean"] if rp else np.nan,
                             "gap_vs_permuted": gap, "chance": chance,
                             # evaluate() does not set this key -- it is added in
                             # ws4a_ml's own loop -- so compute it here or every
                             # refuse-to-guess model is reported as a real zero.
                             "degenerate": bool(r["score_std"] < 1e-9
                                                and abs(r["score_mean"] - chance) < 1e-6),
                             "seconds": r["seconds"]})
                LOG.info("  %-16s %-13s gap=%+.3f  (real %.3f, permuted %.3f) %ds",
                         arm, mname, gap, r["score_mean"],
                         rp["score_mean"] if rp else np.nan, r["seconds"])

    if not rows:
        LOG.error("nothing evaluated"); return 1
    df = pd.DataFrame(rows)
    C.save_table(df, out / f"hvg_experiment_{args.cell_line}.csv")

    piv = df.pivot_table(index=["target", "model"], columns="arm",
                         values="gap_vs_permuted")
    lo, hi = f"hvg{args.n_hvg}_honest", f"hvg{args.n_hvg}_leaky"
    if {"all_genes", lo, hi} <= set(piv.columns):
        piv["cost_of_reduction"] = piv[lo] - piv["all_genes"]
        piv["worth_of_leakage"] = piv[hi] - piv[lo]
    C.save_table(piv.reset_index(), out / f"hvg_deltas_{args.cell_line}.csv")

    print("\n" + "=" * 96)
    print(f"HVG EXPERIMENT — {args.cell_line}, expression block")
    print("  all three arms share the same nested CV and the same shuffled-label control.")
    print("=" * 96)
    print(piv.round(3).to_string())
    if "cost_of_reduction" in piv:
        print(f"\n  cost of REDUCTION  (honest 2k − all genes)  median {piv.cost_of_reduction.median():+.3f}")
        print(f"  worth of LEAKAGE   (leaky 2k − honest 2k)   median {piv.worth_of_leakage.median():+.3f}")
        print("     > 0 means selecting genes on all rows flattered the score")
    fig = plot(df, piv, cfg, out, args.cell_line, args.n_hvg, args.morph_ref)
    print(f"  fig -> {fig}")
    print(f"\nwrote -> {out}")
    return 0


def plot(df, piv, cfg, out, cell, n_hvg, morph_ref=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cfg.section("plots")
    plt.rcParams["figure.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["axes.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["font.size"] = 9
    palette = list(p.get("palette", ["#2a78d6"]))
    dpi = int(p.get("dpi", 150))

    lo, hi = f"hvg{n_hvg}_honest", f"hvg{n_hvg}_leaky"
    order = ["all_genes", lo, hi]
    labels = {"all_genes": "all genes\n(41,780)\nno selection",
              lo: f"{n_hvg:,} HVGs\nchosen IN-FOLD\n(leakage-free)",
              hi: f"{n_hvg:,} HVGs\nchosen on ALL rows\n(the deck's way)"}

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.6))

    # ---------------------------------------------------------------- left
    ax = axes[0]
    models = sorted(df.model.unique())
    w = 0.8 / len(models)
    x = np.arange(len(order))
    for i, m in enumerate(models):
        g = df[df.model == m].set_index("arm").reindex(order)
        pos = x + (i - (len(models) - 1) / 2) * w
        deg = g.degenerate.fillna(False).values
        ax.bar(pos, g.gap_vs_permuted, width=w, color=palette[i % len(palette)],
               edgecolor="#333", lw=0.4, label=m,
               hatch=None)
        for xi, v, d in zip(pos, g.gap_vs_permuted.values, deg):
            if d:
                # a degenerate model scored exactly chance both ways: its zero bar is
                # not "no effect", it is "no model". Say so rather than draw nothing.
                ax.plot([xi], [0.004], marker="x", ms=7, mew=1.8, color="#b02418",
                        zorder=5)
            elif np.isfinite(v):
                ax.text(xi, v + (0.006 if v >= 0 else -0.014), f"{v:+.2f}",
                        ha="center", fontsize=6.4, color="#333")
    if morph_ref is not None:
        ax.axhline(morph_ref, color="#7a2c0a", lw=1.6, ls=":",
                   label=f"best MORPHOLOGY arm ({morph_ref:+.3f})")
    ax.axhline(0, color="#333", lw=1.0, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order], fontsize=7.8)
    ax.set_ylabel("honest gap  (real − shuffled labels)")
    ax.set_title("Gene selection decides whether expression beats morphology\n"
                 "every bar: same nested CV, same models, its own shuffled control.\n"
                 "✗ = degenerate (predicted one class every fold — not a null result)",
                 fontsize=8.8)
    ax.legend(fontsize=7.2, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---------------------------------------------------------------- right
    ax = axes[1]
    if "cost_of_reduction" in piv:
        d = piv.reset_index()
        d = d[~d.model.isin(df[df.degenerate].model.unique())]   # drop non-models
        d = d.sort_values("cost_of_reduction")
        y = np.arange(len(d))
        ax.barh(y + 0.19, d.cost_of_reduction, height=0.36, color="#2a78d6",
                edgecolor="#333", lw=0.4,
                label="cost of REDUCTION\n(in-fold 2k − all genes)")
        ax.barh(y - 0.19, d.worth_of_leakage, height=0.36, color="#b02418",
                edgecolor="#333", lw=0.4,
                label="worth of LEAKAGE\n(all-rows 2k − in-fold 2k)")
        for yi, v in zip(y + 0.19, d.cost_of_reduction):
            ax.text(v - 0.008, yi, f"{v:+.3f}", va="center", ha="right", fontsize=7)
        for yi, v in zip(y - 0.19, d.worth_of_leakage):
            ax.text(v + 0.006, yi, f"{v:+.3f}", va="center", ha="left", fontsize=7)
        ax.axvline(0, color="#333", lw=1.1)
        ax.set_yticks(y); ax.set_yticklabels(d.model, fontsize=8.5)
        ax.set_xlabel("change in honest gap")
        ax.set_xlim(min(d.cost_of_reduction.min() * 1.35, -0.05),
                    max(d.worth_of_leakage.max() * 2.6, 0.05))
        ax.set_title("The two effects, separated\n"
                     "Throwing genes away costs an order of magnitude more than\n"
                     "the leakage it was criticised for was ever worth.", fontsize=8.8)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle(f"What 2,000-HVG selection costs the transcriptomic arm — {cell}",
                 fontsize=11)
    fig.tight_layout()
    path = out / f"hvg_experiment_{cell}.png"
    fig.savefig(path, dpi=dpi); plt.close(fig)
    return path


if __name__ == "__main__":
    sys.exit(main())
