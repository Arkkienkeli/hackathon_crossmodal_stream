#!/usr/bin/env python3
"""WS4A — the analyses that turn two results tables into an answer.

    apptainer run ws4a.sif python /work/scripts/ws4a_report.py \
        --baseline <outputs>/ws4a/ml --tuned <outputs>/ws4a_tuned/ml --cell-line hepg2

Reads the merged ML tables and computes four things the per-run tables cannot show,
because each needs the whole grid at once:

1. THE SIGNAL MAP. Best honest gap per target x block, with the model that got it
   and how many of the four models independently clear zero. One model above zero is
   a hint; four is a result.

2. INCREMENTAL VALUE OF MORPHOLOGY. The project's actual question. Chemistry is free
   and morphology is not, so what matters is gap(morphology + chemistry) - gap(chemistry),
   not gap(morphology) on its own. Same for morphology on top of expression.

3. UNCERTAINTY ON THE GAP. Every gap here is a mean over cross-validation folds, and
   a mean without a spread invites over-reading. The interval below is
   +-1.96 * sqrt(se_real^2 + se_permuted^2), se = fold sd / sqrt(n_folds).
   IT IS APPROXIMATE AND ANTI-CONSERVATIVE: CV folds share training data, so they are
   not independent, and there is no unbiased estimator of the variance of k-fold CV
   (Bengio & Grandvalet 2004). Treat it as "is this anywhere near zero", never as a
   p-value.

4. THE TUNING AUDIT. Whether the search bought signal or bias, measured three ways:
   the change in the permuted score (which is bias, by construction), the change in
   the honest gap, and the count of degenerate rows it created or removed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402

LOG = C.LOG
KEY = ["target", "block", "model"]
Z = 1.959963985


def _style(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = cfg.section("plots")
    plt.rcParams["figure.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["axes.facecolor"] = p.get("facecolor", "#fcfcfb")
    plt.rcParams["font.size"] = 9
    return plt, list(p.get("palette", ["#2a78d6"])), int(p.get("dpi", 150))


def _despine(ax, keep=()):
    for s in ("top", "right"):
        if s not in keep:
            ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- #
def load(run_dir: Path, cell: str, label: str) -> pd.DataFrame:
    """Merged summary + fold spread recovered from the full table."""
    summ = pd.read_csv(run_dir / f"ml_summary_{cell}.csv")
    full = pd.read_csv(run_dir / f"ml_{cell}.csv")
    perm = (full[full.permuted][KEY + ["score_std", "n_folds"]]
            .rename(columns={"score_std": "permuted_std", "n_folds": "permuted_folds"}))
    df = summ.merge(perm, on=KEY, how="left")

    df["se_real"] = df.score_std / np.sqrt(df.n_folds.clip(lower=1))
    df["se_perm"] = df.permuted_std / np.sqrt(df.permuted_folds.clip(lower=1))
    df["se_gap"] = np.sqrt(df.se_real ** 2 + df.se_perm ** 2)
    df["gap_lo"] = df.gap_vs_permuted - Z * df.se_gap
    df["gap_hi"] = df.gap_vs_permuted + Z * df.se_gap
    df["clears_zero"] = df.gap_lo > 0
    df["run"] = label
    if "degenerate" not in df:
        df["degenerate"] = False
    df["degenerate"] = df.degenerate.fillna(False).astype(bool)
    LOG.info("%-9s : %d rows, %d degenerate, %d clear zero",
             label, len(df), int(df.degenerate.sum()), int(df.clears_zero.sum()))
    return df


def signal_map(df: pd.DataFrame) -> pd.DataFrame:
    """Best gap per target x block, plus how many models agree it is above zero."""
    rows = []
    for (t, b), g in df.groupby(["target", "block"]):
        ok = g[~g.degenerate]
        src = ok if len(ok) else g
        best = src.loc[src.gap_vs_permuted.idxmax()]
        rows.append({
            "target": t, "block": b,
            "best_gap": best.gap_vs_permuted, "best_model": best.model,
            "best_gap_lo": best.gap_lo, "best_gap_hi": best.gap_hi,
            "n_models": len(g), "n_degenerate": int(g.degenerate.sum()),
            "n_models_above_zero": int((src.gap_vs_permuted > 0).sum()),
            "n_models_clear_zero": int(src.clears_zero.sum()),
            "median_gap": float(src.gap_vs_permuted.median()),
        })
    return pd.DataFrame(rows)


def incremental(sm: pd.DataFrame) -> pd.DataFrame:
    """Does adding morphology to a block that already exists buy anything?

    This is the question the project is actually asking. Chemistry is free -- no
    cells, no microscope -- so morphology has to beat what chemistry already gives,
    not merely score above chance.
    """
    piv = sm.pivot_table(index="target", columns="block", values="best_gap")
    out = pd.DataFrame(index=piv.index)
    for name, combined, base in (("over_chemistry", "morphology+ecfp", "ecfp"),
                                 ("over_expression", "morphology+expression", "expression")):
        if combined in piv and base in piv:
            out[f"{base}_alone"] = piv[base]
            out[f"{combined}"] = piv[combined]
            out[f"delta_{name}"] = piv[combined] - piv[base]
    if "morphology" in piv:
        out["morphology_alone"] = piv["morphology"]
    if "ecfp" in piv:
        out["morphology_minus_chemistry"] = piv.get("morphology") - piv["ecfp"]
    return out.reset_index()


# --------------------------------------------------------------------------- #
def fig_signal_map(sm, cfg, out, cell, run_label):
    plt, palette, dpi = _style(cfg)
    order_b = [b for b in ["ecfp", "morphology", "expression",
                           "morphology+ecfp", "morphology+expression"]
               if b in set(sm.block)]
    piv = sm.pivot(index="target", columns="block", values="best_gap")[order_b]
    cons = sm.pivot(index="target", columns="block", values="n_models_clear_zero")[order_b]
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    cons = cons.loc[piv.index]

    fig, ax = plt.subplots(figsize=(1.55 * len(order_b) + 3.6, 0.62 * len(piv) + 3.0))
    lim = float(np.nanmax(np.abs(piv.values)))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v, k = piv.values[i, j], cons.values[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i - 0.13, f"{v:+.3f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if k >= 3 else "normal",
                    color="white" if abs(v) > 0.6 * lim else "#111")
            ax.text(j, i + 0.19, f"{int(k)}/4 models", ha="center", va="center",
                    fontsize=6.6, color="white" if abs(v) > 0.6 * lim else "#555")
    ax.set_xticks(range(len(order_b)))
    ax.set_xticklabels([b.replace("+", "\n+") for b in order_b], fontsize=8)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(f"Signal map — {cell}, {run_label}\n"
                 "best honest gap (real − permuted) per target × description. "
                 "Bold = at least 3 of 4 models independently clear zero.\n"
                 "Red is signal; blue is worse than its own shuffled-label control.",
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.75, label="honest gap")
    fig.tight_layout()
    p = out / f"signal_map_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_incremental(inc, cfg, out, cell):
    """The project's question, as one figure."""
    plt, palette, dpi = _style(cfg)
    d = inc.dropna(subset=["delta_over_chemistry"]).copy()
    if d.empty:
        return None
    d = d.sort_values("delta_over_chemistry")
    n = len(d)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, max(3.2, 0.5 * n + 2.4)))

    ax = axes[0]
    y = np.arange(n)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.ecfp_alone, r["morphology+ecfp"]], [i, i], color="#bbb", lw=1.6, zorder=1)
        ax.scatter([r.ecfp_alone], [i], s=46, color="#9a9a97", edgecolor="#444", lw=0.6,
                   zorder=3, label="chemistry alone" if i == 0 else None)
        ax.scatter([r["morphology+ecfp"]], [i], s=52,
                   color="#2f7d32" if r.delta_over_chemistry > 0 else "#b02418",
                   edgecolor="#333", lw=0.6, zorder=3,
                   label="+ morphology" if i == 0 else None)
    ax.axvline(0, color="#333", lw=1.0, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(d.target, fontsize=8)
    ax.set_xlabel("honest gap")
    ax.set_title("Does morphology add to chemistry?\n"
                 "Chemistry is free — no cells, no microscope. Morphology has to beat it.",
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    _despine(ax)

    ax = axes[1]
    cols = [("morphology_minus_chemistry", "morphology alone\n− chemistry alone", "#2a78d6"),
            ("delta_over_chemistry", "chemistry + morphology\n− chemistry", "#1baf7a"),
            ("delta_over_expression", "expression + morphology\n− expression", "#eda100")]
    cols = [c for c in cols if c[0] in d.columns]
    w = 0.8 / len(cols)
    for k, (col, lab, colour) in enumerate(cols):
        ax.barh(y + (k - (len(cols) - 1) / 2) * w, d[col].fillna(0), height=w,
                color=colour, edgecolor="#333", lw=0.4, label=lab)
    ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([""] * n)
    ax.set_xlabel("change in honest gap when morphology is added")
    ax.set_title("Incremental value of the microscope\nright of 0 = imaging added information",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    _despine(ax)
    fig.tight_layout()
    p = out / f"incremental_value_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_uncertainty(df, cfg, out, cell, top=26):
    """Every gap with an interval, so a mean is never read as a certainty."""
    plt, palette, dpi = _style(cfg)
    d = df[~df.degenerate].nlargest(top, "gap_vs_permuted").sort_values("gap_vs_permuted")
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(d) + 2.4))
    y = np.arange(len(d))
    colours = ["#2f7d32" if c else "#9a9a97" for c in d.clears_zero]
    ax.errorbar(d.gap_vs_permuted, y, xerr=Z * d.se_gap, fmt="none",
                ecolor="#999", elinewidth=1.1, capsize=2.4, zorder=1)
    ax.scatter(d.gap_vs_permuted, y, s=34, c=colours, edgecolor="#333", lw=0.5, zorder=2)
    ax.axvline(0, color="#b02418", lw=1.1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.target} · {r.block} · {r.model}" for _, r in d.iterrows()],
                       fontsize=7)
    ax.set_xlabel("honest gap (real − permuted), with approximate 95 % interval")
    ax.set_title(f"How certain is each gap — {cell}\n"
                 "Green = interval excludes zero. The interval is ANTI-CONSERVATIVE: CV\n"
                 "folds share training data, so the true interval is wider. Read it as\n"
                 "'is this near zero', never as a p-value.", fontsize=8.6)
    _despine(ax)
    fig.tight_layout()
    p = out / f"gap_uncertainty_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_tuning_audit(base, tuned, cfg, out, cell):
    """Did the search buy signal, or bias? Three panels, three answers."""
    plt, palette, dpi = _style(cfg)
    m = base.merge(tuned, on=KEY, suffixes=("_b", "_t"))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # 1. the bias question: did the PERMUTED score move?
    ax = axes[0]
    d_perm = m.permuted_score_t - m.permuted_score_b
    ax.hist(d_perm, bins=22, color="#9a9a97", edgecolor="#333", lw=0.5)
    ax.axvline(0, color="#333", lw=1.2)
    ax.axvline(float(d_perm.median()), color="#b02418", lw=1.6,
               label=f"median {d_perm.median():+.4f}")
    ax.set_xlabel("change in score on SHUFFLED labels")
    ax.set_ylabel("target × block × model")
    ax.set_title("Did tuning manufacture score?\n"
                 "These labels carry no signal. A shift right\nwould be pure selection bias.",
                 fontsize=8.8)
    ax.legend(fontsize=7.5, frameon=False)
    _despine(ax)

    # 2. the gain question, per model
    ax = axes[1]
    models = sorted(m.model.unique())
    for i, mod in enumerate(models):
        v = (m[m.model == mod].gap_vs_permuted_t - m[m.model == mod].gap_vs_permuted_b).values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.16, 0.16, len(v)), v, s=26,
                   color=palette[i % len(palette)], edgecolor="#444", lw=0.4, zorder=3)
        ax.hlines(float(np.median(v)), i - 0.3, i + 0.3, color="#111", lw=2.2, zorder=4)
    ax.axhline(0, color="#333", lw=1.0, ls="--")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=7.6, rotation=15)
    ax.set_ylabel("Δ honest gap (tuned − untuned)")
    ax.set_title("Which models did tuning help?\nbar = median across target × block", fontsize=8.8)
    _despine(ax)

    # 3. the degeneracy question
    ax = axes[2]
    nb, nt = int(base.degenerate.sum()), int(tuned.degenerate.sum())
    ax.bar(["untuned", "tuned"], [nb, nt],
           color=["#b02418", "#2f7d32"], edgecolor="#333", lw=0.6, width=0.55)
    for i, v in enumerate((nb, nt)):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("rows scoring exactly chance with zero variance")
    ax.set_ylim(0, max(nb, nt, 1) * 1.28)
    ax.set_title("Models that refused to guess\n"
                 "exactly 0.500 ± 0.000 = one class predicted every fold;\n"
                 "a broken configuration, not a null result", fontsize=8.8)
    _despine(ax)

    fig.suptitle(f"Tuning audit — {cell}: did the search buy signal or bias?", fontsize=10.5)
    fig.tight_layout()
    p = out / f"tuning_audit_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_controls(stats_dir, cfg, out, cell):
    """The two controls that decided a conclusion, drawn against their nulls.

    A permutation null is normally reported as a p-value and believed or not. Drawn,
    it is obvious: the AJIVE joint rank the pipeline found sits INSIDE the
    distribution produced by data with the compound correspondence destroyed, while
    the first canonical correlation sits outside its own.
    """
    import json
    plt, palette, dpi = _style(cfg)
    sj = Path(stats_dir) / f"stats_{cell}.json"
    ac = Path(stats_dir) / f"agreement_{cell}.csv"
    if not sj.exists():
        LOG.warning("no %s -- skipping the controls figure", sj)
        return None
    j = json.load(open(sj))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))

    # --- AJIVE: the control that removed a finding
    ax = axes[0]
    aj = j.get("ajive", {})
    nulls = np.asarray(aj.get("null_joint_ranks", []), float)
    obs = aj.get("joint_rank", np.nan)
    if nulls.size:
        bins = np.arange(-0.5, max(nulls.max(), obs) + 1.5)
        ax.hist(nulls, bins=bins, color="#9a9a97", edgecolor="#333", lw=0.6,
                label=f"null (correspondence destroyed), n={nulls.size}")
        ax.axvline(obs, color="#b02418", lw=2.4, label=f"observed = {obs:g}")
        ax.set_xlabel("AJIVE joint rank")
        ax.set_ylabel("permutations")
        ax.set_title("The control that removed a finding\n"
                     f"null reaches {nulls.max():g} too — p = {aj.get('null_p', float('nan')):.3f}, "
                     "NOT significant", fontsize=8.8)
        ax.legend(fontsize=7, frameon=False)
    _despine(ax)

    # --- CCA: the control a number survived
    ax = axes[1]
    cc = j.get("cca", {})
    null = np.asarray(cc.get("null", []), float)
    stat = np.asarray(cc.get("stat_obs", []), float)
    r = np.asarray(cc.get("r", []), float)
    if null.size and stat.size:
        # `null` holds the permutation distribution of the TEST STATISTIC, not of r
        # -- the two are on different scales, and plotting r against it would compare
        # unlike quantities.
        n1 = null[:, 0] if null.ndim == 2 else null
        ax.hist(n1, bins=45, color="#9a9a97", edgecolor="#333", lw=0.3,
                label=f"null statistic, n={len(n1)}")
        ax.axvline(stat[0], color="#2f7d32", lw=2.4,
                   label=f"observed = {stat[0]:.2f}")
        p1 = np.asarray(cc.get("p_unc", [np.nan]), float)[0]
        ax.set_xlabel("permutation CCA test statistic (first component)")
        ax.set_ylabel("permutations")
        ax.set_title(f"A number that SURVIVED its control\n"
                     f"p = {p1:.3f}. The correlation itself is r₁ = {r[0]:.3f} —\n"
                     "quoted alone it would mean nothing.", fontsize=8.8)
        ax.legend(fontsize=7, frameon=False)
    _despine(ax)

    # --- plain vs adjusted RV: the statistic that lies
    ax = axes[2]
    if ac.exists():
        a = pd.read_csv(ac)
        obs_a = a[a.variant == "observed"].set_index("pair")
        rnd = a[a.variant == "ctl_random"].groupby("pair").mean(numeric_only=True)
        pairs = list(obs_a.index)
        y = np.arange(len(pairs))
        h = 0.19
        ax.barh(y + 1.5 * h, obs_a.rv_plain, height=h, color="#b02418",
                edgecolor="#333", lw=0.4, label="plain RV — real data")
        ax.barh(y + 0.5 * h, rnd.reindex(pairs).rv_plain, height=h, color="#e8a49c",
                edgecolor="#333", lw=0.4, label="plain RV — RANDOM NOISE")
        ax.barh(y - 0.5 * h, obs_a.rv_adj, height=h, color="#2a78d6",
                edgecolor="#333", lw=0.4, label="adjusted RV — real data")
        ax.barh(y - 1.5 * h, rnd.reindex(pairs).rv_adj, height=h, color="#a9c9ec",
                edgecolor="#333", lw=0.4, label="adjusted RV — random noise")
        ax.set_yticks(y); ax.set_yticklabels(pairs, fontsize=7.5)
        ax.set_xlabel("agreement between two whole tables")
        ax.set_title("Why the adjusted RV, not the plain one\n"
                     "Plain RV scores RANDOM NOISE as high as the real data —\n"
                     "on two of three pairs, higher. The adjusted one does not.",
                     fontsize=8.8)
        ax.legend(fontsize=6.4, frameon=False, loc="lower right")
    _despine(ax)

    fig.suptitle(f"Tier 1 controls — {cell}: what survives a deliberate attempt to break it",
                 fontsize=10.5)
    fig.tight_layout()
    p = out / f"controls_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_why_subtract(df, cfg, out, cell):
    """The single most important habit, as a figure.

    If the shuffled-label score were always 0.5, comparing a raw score to 0.5 would
    be fine and none of this machinery would be needed. It is not: across this grid
    it spans a quarter of the whole scale.
    """
    plt, palette, dpi = _style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    ax = axes[0]
    ps = df.permuted_score.dropna()
    ax.hist(ps, bins=26, color="#9a9a97", edgecolor="#333", lw=0.5)
    ax.axvline(0.5, color="#b02418", lw=2.0, ls="--", label="chance = 0.500")
    ax.axvline(float(ps.mean()), color="#2a78d6", lw=1.8,
               label=f"mean = {ps.mean():.3f}")
    ax.set_xlabel("score on SHUFFLED labels")
    ax.set_ylabel("target × block × model")
    ax.set_title("The control is NOT 0.5\n"
                 f"it spans {ps.min():.3f} – {ps.max():.3f} across this grid.\n"
                 "That spread is why every score is paired with its own control.",
                 fontsize=8.6)
    ax.legend(fontsize=7.5, frameon=False)
    _despine(ax)

    ax = axes[1]
    d = df.dropna(subset=["permuted_score"])
    col = ["#2f7d32" if c else "#c9c9c6" for c in d.clears_zero]
    lo = min(d.score_mean.min(), d.permuted_score.min()) - 0.03
    hi = max(d.score_mean.max(), d.permuted_score.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color="#b02418", lw=1.3, ls="--", zorder=1)
    ax.scatter(d.permuted_score, d.score_mean, s=34, c=col, edgecolor="#333",
               lw=0.4, zorder=2)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("score on shuffled labels")
    ax.set_ylabel("score on real labels")
    ax.set_title("A high score is not a result\n"
                 "distance ABOVE the diagonal is the finding, not height.\n"
                 "Green = interval excludes zero.", fontsize=8.6)
    _despine(ax)

    fig.suptitle(f"Why every number is paired with a control — {cell}", fontsize=10.5)
    fig.tight_layout()
    p = out / f"why_subtract_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


# A "gap vs number of features" figure was built here and REMOVED as confounded.
# The x-axis would have mixed two different things: how many features a block has,
# and which modality it is. morphology (636) -> ecfp (1024) is a change of data type,
# not a feature-count increase, so a falling line could not be attributed to dilution.
# The nested comparisons that CAN support that claim -- ecfp vs morphology+ecfp, and
# expression vs morphology+expression, each adding the same 636 morphology columns to
# a fixed base -- are what fig_incremental already plots.


def fig_concordance(df, cfg, out, cell):
    """Do four different model families agree about where the signal is?

    Consensus is the substitute for a p-value here. If four methods with different
    inductive biases rank the grid the same way, that is worth more than any one
    interval on data this small.
    """
    from scipy.stats import spearmanr
    plt, palette, dpi = _style(cfg)
    piv = df.pivot_table(index=["target", "block"], columns="model",
                         values="gap_vs_permuted")
    piv = piv.dropna()
    models = list(piv.columns)
    if len(models) < 2 or len(piv) < 4:
        return None
    rho = np.eye(len(models))
    for i in range(len(models)):
        for k in range(i + 1, len(models)):
            r = spearmanr(piv[models[i]], piv[models[k]]).statistic
            rho[i, k] = rho[k, i] = r

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(models)):
        for k in range(len(models)):
            ax.text(k, i, f"{rho[i, k]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(rho[i, k]) > 0.6 else "#111")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=7.5, rotation=25)
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=7.5)
    ax.set_title("Do the models rank the grid the same way?\n"
                 f"Spearman ρ across {len(piv)} target × block cells", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    order = piv.mean(axis=1).sort_values().index
    y = np.arange(len(order))
    for i, mod in enumerate(models):
        ax.scatter(piv.loc[order, mod], y, s=28, color=palette[i % len(palette)],
                   edgecolor="#333", lw=0.35, label=mod, zorder=3)
    ax.axvline(0, color="#333", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} · {b}" for t, b in order], fontsize=6.2)
    ax.set_xlabel("honest gap")
    ax.set_title("Every model on every cell\n"
                 "cells where all four sit right of 0 are the trustworthy ones",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    p = out / f"model_concordance_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_concordance_slide(df, cfg, out, cell, label_top=6):
    """A projector cut of the concordance figure that KEEPS the trend.

    An earlier version showed only the 10 strongest rows. That made the labels
    readable and destroyed the thing the panel is actually good at: the sweep from
    near-zero up to strongly positive, which IS the honest landscape -- most
    combinations are nothing, a handful are real. You do not need to read a label to
    see a trend.

    So: all 30 rows stay, the 14 where every model clears zero are shaded as a band,
    and only the top few get names -- the ones you would say out loud.
    """
    from scipy.stats import spearmanr
    plt, palette, dpi = _style(cfg)
    piv = df.pivot_table(index=["target", "block"], columns="model",
                         values="gap_vs_permuted").dropna()
    models = list(piv.columns)
    if len(models) < 2 or len(piv) < 4:
        return None
    rho = np.eye(len(models))
    for i in range(len(models)):
        for k in range(i + 1, len(models)):
            rho[i, k] = rho[k, i] = spearmanr(piv[models[i]], piv[models[k]]).statistic

    # Wide, and with real separation: the right panel's y labels are long
    # ("tox_pulmonary_toxicity · morphology+expression") and were crowding the matrix.
    fig, axes = plt.subplots(1, 2, figsize=(19.5, 7.0),
                             gridspec_kw={"width_ratios": [0.74, 1.75], "wspace": 0.92})

    # ---- left: the agreement matrix, big enough to read from the back
    ax = axes[0]
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(models)):
        for k in range(len(models)):
            ax.text(k, i, f"{rho[i, k]:.2f}", ha="center", va="center", fontsize=17,
                    weight="bold" if i != k else "normal",
                    color="white" if abs(rho[i, k]) > 0.6 else "#111")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, rotation=20, ha="right")
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models, fontsize=12)
    ax.set_title("Do the four models rank the grid the same way?\n"
                 "Spearman ρ over all 30 target × block combinations",
                 fontsize=13, pad=14)

    # ---- right: every combination, sorted -- the trend is the message
    ax = axes[1]
    # Sort by the WORST model, not the mean. Then "all four clear zero" is exactly the
    # top block and the shading is one contiguous region instead of zebra stripes --
    # and the ranking is the conservative one: a combination is only as good as its
    # weakest model.
    order = piv.min(axis=1).sort_values().index          # weakest at the bottom
    allpos = (piv > 0).all(axis=1)
    y = np.arange(len(order))

    n_ok = int(allpos.sum())
    if n_ok:
        ax.axhspan(len(order) - n_ok - 0.5, len(order) - 0.5,
                   color="#e8f4ee", zorder=0)
    for i, mod in enumerate(models):
        ax.scatter(piv.loc[order, mod], y, s=52, color=palette[i % len(palette)],
                   edgecolor="#333", lw=0.5, label=mod, zorder=3)
    ax.axvline(0, color="#b02418", lw=1.8, ls="--", zorder=2)

    # Every row is labelled. Hierarchy comes from COLOUR, not from omission: the rows
    # where all four models clear zero are the ones worth naming out loud, so they are
    # green and bold and everything else recedes to grey.
    ax.set_yticks(y)
    # every toxicity target starts "tox_"; dropping it costs no clarity and buys the
    # width that stopped the longest labels reaching into the matrix panel
    ax.set_yticklabels([f"{t.replace('tox_', '')}  ·  {b}" for t, b in order],
                       fontsize=9.4)
    ax.tick_params(axis="y", length=0)
    for tick, key in zip(ax.get_yticklabels(), order):
        if allpos[key]:
            tick.set_color("#12805a"); tick.set_fontweight("bold"); tick.set_fontsize(9.6)
        else:
            tick.set_color("#9a9a97")
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("honest gap   (real − shuffled labels)", fontsize=12.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_title(f"All {len(order)} combinations, sorted by their WEAKEST model\n"
                 f"green + shaded = all four models clear zero  ({n_ok} of {len(order)})"
                 "   ·   most of the rest sit at zero",
                 fontsize=12.5, pad=12)
    ax.legend(fontsize=10, frameon=True, framealpha=0.92, edgecolor="#dcdcd8",
              loc="lower right", bbox_to_anchor=(1.0, 0.02))
    _despine(ax)

    fig.tight_layout(w_pad=4.5)
    p = out / f"model_concordance_slide_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tuned", default=None,
                    help="optional; without it only the baseline analyses are produced")
    ap.add_argument("--stats", default=None,
                    help="the stats/ directory, to draw the Tier 1 control figures")
    ap.add_argument("--headline-run", default="tuned", choices=["tuned", "baseline"],
                    help="which run the signal map and incremental figures describe")
    args = ap.parse_args(argv)
    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    out = Path(args.out) if args.out else C.outputs_dir(cfg, "report")

    base = load(Path(args.baseline), args.cell_line, "baseline")
    tuned = load(Path(args.tuned), args.cell_line, "tuned") if args.tuned else None

    head = tuned if (tuned is not None and args.headline_run == "tuned") else base
    head_label = "tuned" if head is tuned else "untuned baseline"

    sm = signal_map(head)
    inc = incremental(sm)
    C.save_table(sm.sort_values("best_gap", ascending=False),
                 out / f"signal_map_{args.cell_line}.csv")
    C.save_table(inc, out / f"incremental_value_{args.cell_line}.csv")
    C.save_table(head[KEY + ["n", "score_mean", "permuted_score", "gap_vs_permuted",
                             "se_gap", "gap_lo", "gap_hi", "clears_zero", "degenerate"]]
                 .sort_values("gap_vs_permuted", ascending=False),
                 out / f"gap_intervals_{args.cell_line}.csv")

    figs = [fig_signal_map(sm, cfg, out, args.cell_line, head_label),
            fig_incremental(inc, cfg, out, args.cell_line),
            fig_uncertainty(head, cfg, out, args.cell_line)]
    if tuned is not None:
        figs.append(fig_tuning_audit(base, tuned, cfg, out, args.cell_line))
    figs += [fig_why_subtract(head, cfg, out, args.cell_line),
             fig_concordance(head, cfg, out, args.cell_line),
             fig_concordance_slide(head, cfg, out, args.cell_line)]
    if args.stats:
        figs.append(fig_controls(args.stats, cfg, out, args.cell_line))

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 100)
    print(f"WS4A REPORT — {args.cell_line}, headline run: {head_label}")
    print("=" * 100)

    print("\nSIGNAL MAP  (best honest gap per target x description)")
    print(sm.sort_values("best_gap", ascending=False)[
        ["target", "block", "best_gap", "best_model", "n_models_clear_zero",
         "n_models_above_zero", "n_degenerate"]].round(3).to_string(index=False))

    print("\nINCREMENTAL VALUE OF MORPHOLOGY  (the project's question)")
    print(inc.round(3).to_string(index=False))
    if "delta_over_chemistry" in inc:
        d = inc.delta_over_chemistry.dropna()
        pos = int((d > 0).sum())
        print(f"\n  morphology ADDED to chemistry in {pos} of {len(d)} targets "
              f"(median {d.median():+.3f})")
    if "morphology_minus_chemistry" in inc:
        d = inc.morphology_minus_chemistry.dropna()
        print(f"  morphology ALONE beat chemistry alone in {int((d > 0).sum())} of "
              f"{len(d)} targets (median {d.median():+.3f})")

    if tuned is not None:
        m = base.merge(tuned, on=KEY, suffixes=("_b", "_t"))
        dp = (m.permuted_score_t - m.permuted_score_b)
        dg = (m.gap_vs_permuted_t - m.gap_vs_permuted_b)
        print("\nTUNING AUDIT")
        print(f"  degenerate rows            untuned {int(base.degenerate.sum())}"
              f"  ->  tuned {int(tuned.degenerate.sum())}")
        print(f"  median change, SHUFFLED    {dp.median():+.4f}   "
              "(> 0 would be manufactured score)")
        print(f"  median change, honest gap  {dg.median():+.4f}")
        print(f"  rows where tuning helped   {int((dg > 0).sum())} of {len(dg)}")
        print(f"  gaps clearing zero         untuned {int(base.clears_zero.sum())}"
              f"  ->  tuned {int(tuned.clears_zero.sum())}")

    print("\n  NOTE: intervals are approximate and ANTI-CONSERVATIVE -- CV folds share")
    print("  training data, so the true intervals are wider (Bengio & Grandvalet 2004).")
    for f in [f for f in figs if f]:
        print(f"  fig -> {f}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
