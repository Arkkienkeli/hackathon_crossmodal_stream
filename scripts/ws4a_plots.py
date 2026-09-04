#!/usr/bin/env python
"""WS4A — integration figures.

    bash scripts/ws4a.sh python /work/scripts/ws4a_plots.py --cell-line hepg2

Reads what ws4a_stats.py and ws4a_ml.py wrote and draws:

  1. agreement_<cl>.png       effect + permutation p per modality pair, with the
                              destructive controls plotted alongside as the reference
  2. embedding_<cl>.png       each modality's compounds in 2-D, coloured by a shared
                              label, so "do the modalities agree" is visible not just
                              tabulated
  3. joint_structure_<cl>.png AJIVE joint vs individual variation
  4. ml_performance_<cl>.png  score against the PERMUTED-LABEL control, per block
  5. modality_overlap_<cl>.png which compounds are active in each modality

Design rule followed throughout: a figure never shows a statistic whose null is
not also on the plot. Plain RV is 0.93 on independent blocks at this shape and raw
Procrustes 0.9995, so a bare bar of either would be actively misleading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
from ws4a import matrix_agreement as MA           # noqa: E402

LOG = C.LOG


def _style(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cfg.section("plots")
    plt.rcParams["figure.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["axes.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["font.size"] = 9
    return plt, list(p.get("palette", ["#2a78d6"])), int(p.get("dpi", 150))


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
def plot_agreement(df, cfg, out, cell_line):
    """Observed effect against its destructive controls, on one axis."""
    plt, palette, dpi = _style(cfg)
    obs = df[df.variant == "observed"]
    ctl = df[df.variant != "observed"]
    if obs.empty:
        return None

    pairs = list(obs["pair"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    y = np.arange(len(pairs))
    ax.barh(y, obs.rv_adj, color=palette[0], edgecolor="#444", lw=0.5, label="observed")
    for i, pr in enumerate(pairs):
        for j, (v, colour) in enumerate((("ctl_scrambled", "#b0b0ae"),
                                         ("ctl_random", "#d8d8d4"))):
            sub = ctl[(ctl.pair == pr) & (ctl.variant == v)]
            if len(sub):
                ax.scatter(sub.rv_adj, np.full(len(sub), i) + (j - 0.5) * 0.22,
                           s=14, color=colour, edgecolors="#666", linewidths=0.3,
                           zorder=3, label=v if i == 0 else None)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(pairs)
    ax.invert_yaxis()
    ax.set_xlabel("adjusted RV  (Mayer et al. — the only unbiased effect size at p >> n)")
    ax.set_title("Cross-modal agreement, with destructive controls", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _despine(ax)

    ax = axes[1]
    for i, pr in enumerate(pairs):
        r = obs[obs.pair == pr].iloc[0]
        for j, (col, lab) in enumerate((("rv_adj_p", "adjusted RV"),
                                        ("mantel_p", "Mantel"),
                                        ("protest_p", "PROTEST"))):
            ax.scatter(r[col], i + (j - 1) * 0.2, s=52, color=palette[j % len(palette)],
                       edgecolors="#444", linewidths=0.4, label=lab if i == 0 else None,
                       zorder=3)
    ax.axvline(0.05, color="#c0392b", ls="--", lw=1.2)
    ax.text(0.05, -0.6, " p=0.05", color="#c0392b", fontsize=7)
    ax.set_xscale("log")
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(pairs)
    ax.invert_yaxis()
    ax.set_xlabel("permutation p-value (log scale)")
    ax.set_title("The p-values are trustworthy even where the raw statistic is not", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="best")
    _despine(ax)

    fig.suptitle(f"Tier 1 — matrix agreement, {cell_line}", fontsize=12)
    fig.text(0.005, 0.005,
             "Plain RV is 0.93 and raw Procrustes 0.9995 on INDEPENDENT blocks at this shape, "
             "so neither raw statistic is plotted as an effect. Controls must sit at zero.",
             fontsize=6.5, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    p = out / f"agreement_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return p


def plot_embeddings(blocks, cfg, out, cell_line, colour_by="moa-fine"):
    """Each modality's compounds in 2-D, same points, same colours."""
    from sklearn.decomposition import PCA
    plt, palette, dpi = _style(cfg)

    mods = [m for m in ("morphology", "expression", "ecfp") if m in blocks.X]
    lab = blocks.obs[colour_by].astype(str) if colour_by in blocks.obs.columns \
        else pd.Series(["all"] * blocks.n)
    top = lab.value_counts()
    top = [c for c in top.index if c != "unclear"][:6]
    lab = lab.where(lab.isin(top), "other")

    fig, axes = plt.subplots(1, len(mods), figsize=(5.0 * len(mods), 4.6), squeeze=False)
    for ax, m in zip(axes[0], mods):
        Z = PCA(n_components=2, random_state=0).fit_transform(
            MA.scale_block(blocks.X[m], how="zscore"))
        for i, c in enumerate(list(top) + ["other"]):
            sel = (lab == c).to_numpy()
            if not sel.any():
                continue
            ax.scatter(Z[sel, 0], Z[sel, 1], s=26 if c != "other" else 12,
                       color=palette[i % len(palette)] if c != "other" else "#c9c9c7",
                       alpha=0.85 if c != "other" else 0.4, linewidths=0,
                       label=c[:26], zorder=3 if c != "other" else 1)
        ax.set_title(f"{m}  ({blocks.X[m].shape[1]} features)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        _despine(ax)
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="center left",
                       bbox_to_anchor=(1.01, 0.5), title=colour_by)
    fig.suptitle(f"Same {blocks.n} compounds in each modality — {cell_line}", fontsize=12)
    fig.text(0.005, 0.005, "PCA per modality. Agreement means the same compounds sit together "
                           "in more than one panel; it is not implied by any single panel.",
             fontsize=6.5, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    p = out / f"embedding_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return p


def plot_joint(stats_json, cfg, out, cell_line):
    aj = (stats_json or {}).get("ajive")
    if not aj:
        return None
    plt, palette, dpi = _style(cfg)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels, values, colours = ["joint"], [aj.get("joint_rank", 0)], [palette[0]]
    for i, (blk, r) in enumerate((aj.get("individual_ranks") or {}).items()):
        labels.append(f"{blk}\nindividual"); values.append(r)
        colours.append(palette[(i + 1) % len(palette)])
    ax.bar(labels, values, color=colours, edgecolor="#444", lw=0.5)
    if "null_joint_rank_mean" in aj:
        ax.axhline(aj["null_joint_rank_mean"], color="#c0392b", ls="--", lw=1.2)
        ax.text(0.02, aj["null_joint_rank_mean"], " permutation null (mean)",
                color="#c0392b", fontsize=7, va="bottom", transform=ax.get_yaxis_transform())
    for i, v in enumerate(values):
        ax.text(i, v, f" {v}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("rank (number of dimensions)")
    ax.set_title(f"AJIVE — shared vs modality-specific structure, {cell_line}", fontsize=11)
    _despine(ax)
    fig.text(0.005, 0.005, "The joint rank IS the answer to 'how much is shared'. A joint rank "
                           "at or below the permutation null means nothing is shared.",
             fontsize=6.5, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = out / f"joint_structure_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return p


def plot_ml(ml_csv, cfg, out, cell_line):
    """Score against the permuted-label control. The gap is the result."""
    if not ml_csv.exists():
        return None
    df = pd.read_csv(ml_csv)
    if "permuted_score" not in df.columns or df.empty:
        return None
    plt, palette, dpi = _style(cfg)

    targets = list(dict.fromkeys(df.target))
    fig, axes = plt.subplots(1, len(targets), figsize=(max(6, 4.2 * len(targets)), 4.8),
                             squeeze=False)
    for ax, t in zip(axes[0], targets):
        sub = df[df.target == t].sort_values("gap_vs_permuted", ascending=True)
        lbl = [f"{b}\n{m}" for b, m in zip(sub.block, sub.model)]
        y = np.arange(len(sub))
        ax.barh(y, sub.score_mean, color=palette[0], edgecolor="#444", lw=0.4, label="real labels")
        ax.scatter(sub.permuted_score, y, s=38, color="#c0392b", zorder=3,
                   edgecolors="#444", linewidths=0.4, label="permuted labels")
        if "chance" in sub.columns and sub.chance.notna().any():
            ax.axvline(float(sub.chance.iloc[0]), color="#666", ls=":", lw=1.2, label="chance")
        ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=7)
        ax.set_xlabel("balanced accuracy")
        ax.set_title(t, fontsize=10)
        _despine(ax)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(f"Tier 2 — score vs the permuted-label control, {cell_line}", fontsize=12)
    fig.text(0.005, 0.005,
             "The result is the GAP between the bar and the red point. A model scoring well on "
             "permuted labels is measuring selection bias, which is worst at this n.",
             fontsize=6.5, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    p = out / f"ml_performance_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return p


def plot_modality_overlap(blocks, cfg, out, cell_line):
    """Which compounds move in each modality, and how much they overlap.

    Published on A549: only 11-34% of compounds give signal in BOTH Cell Painting
    and L1000. If that holds here, it caps what any integration can recover.
    """
    plt, palette, dpi = _style(cfg)
    mods = [m for m in ("morphology", "expression") if m in blocks.X]
    if len(mods) < 2:
        return None

    mag = {}
    for m in mods:
        Z = MA.scale_block(blocks.X[m], how="zscore")
        mag[m] = np.linalg.norm(Z, axis=1) / np.sqrt(Z.shape[1])

    a, b = mag[mods[0]], mag[mods[1]]
    thr_a, thr_b = np.percentile(a, 66), np.percentile(b, 66)
    both = (a > thr_a) & (b > thr_b)
    only_a = (a > thr_a) & ~(b > thr_b)
    only_b = ~(a > thr_a) & (b > thr_b)
    neither = ~(a > thr_a) & ~(b > thr_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for sel, colour, lab in ((neither, "#c9c9c7", "neither"),
                             (only_a, palette[1], f"{mods[0]} only"),
                             (only_b, palette[2], f"{mods[1]} only"),
                             (both, palette[0], "both")):
        ax.scatter(a[sel], b[sel], s=26, color=colour, alpha=0.85, linewidths=0,
                   label=f"{lab} ({int(sel.sum())})")
    ax.axvline(thr_a, color="#666", ls=":", lw=1); ax.axhline(thr_b, color="#666", ls=":", lw=1)
    ax.set_xlabel(f"{mods[0]} response magnitude")
    ax.set_ylabel(f"{mods[1]} response magnitude")
    ax.set_title("Per-compound response in each modality", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _despine(ax)

    ax = axes[1]
    counts = [int(both.sum()), int(only_a.sum()), int(only_b.sum()), int(neither.sum())]
    labs = ["both", f"{mods[0]}\nonly", f"{mods[1]}\nonly", "neither"]
    ax.bar(labs, counts, color=[palette[0], palette[1], palette[2], "#c9c9c7"],
           edgecolor="#444", lw=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c, f" {c}\n {c / len(a):.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("compounds")
    ax.set_title("Overlap of responders", fontsize=10)
    _despine(ax)

    fig.suptitle(f"Do the modalities respond to the same compounds? — {cell_line}", fontsize=12)
    fig.text(0.005, 0.005,
             "Thresholds are the 66th percentile of each modality, so 'both' would be ~33% under "
             "independence. Published on A549: only 11-34% of compounds give signal in both "
             "Cell Painting and L1000, which caps what any integration can recover.",
             fontsize=6.5, color="#666")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    p = out / f"modality_overlap_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    LOG.info("wrote      : %s", p)
    return p


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--colour-by", default="moa-fine")
    ap.add_argument("--stats-dir", default=None)
    ap.add_argument("--ml-dir", default=None)
    args = ap.parse_args()

    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    out = Path(args.out) if args.out else C.outputs_dir(cfg, "figures")
    stats_dir = Path(args.stats_dir) if args.stats_dir else C.outputs_dir(cfg, "stats")
    ml_dir = Path(args.ml_dir) if args.ml_dir else C.outputs_dir(cfg, "ml")
    cl = args.cell_line

    made = []
    agr = stats_dir / f"agreement_{cl}.csv"
    if agr.exists():
        made.append(plot_agreement(pd.read_csv(agr), cfg, out, cl))
    else:
        LOG.warning("no %s — run ws4a_stats.py first", agr.name)

    sj = stats_dir / f"stats_{cl}.json"
    if sj.exists():
        made.append(plot_joint(json.load(open(sj)), cfg, out, cl))

    try:
        blocks = C.load_mudata_blocks(cfg, cl)
        made.append(plot_embeddings(blocks, cfg, out, cl, args.colour_by))
        made.append(plot_modality_overlap(blocks, cfg, out, cl))
    except C.ContaminatedBlockError as exc:
        LOG.error("cannot draw modality figures: %s", str(exc).splitlines()[0])

    made.append(plot_ml(ml_dir / f"ml_summary_{cl}.csv", cfg, out, cl))

    made = [m for m in made if m]
    print(f"\n{len(made)} figure(s) -> {out}")
    for m in made:
        print(f"  {m.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
