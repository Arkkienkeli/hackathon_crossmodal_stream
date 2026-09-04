#!/usr/bin/env python3
"""WS4A — compare an untuned baseline run against a tuned one.

    apptainer run ws4a.sif compare \
        --baseline /path/ws4a --tuned /path/ws4a_tuned --cell-line hepg2

WHAT IS BEING COMPARED, AND WHAT IS NOT
---------------------------------------
NOT `score_mean`. Tuning raises the score on data that contains no signal at all --
that is the definition of selection bias, and at n ~ 64 it is worth tens of points.
A tuned run whose real score went up by 0.06 while its PERMUTED score went up by
0.06 learned nothing; it just searched harder against noise.

The comparison is therefore on `gap_vs_permuted` = real - permuted, with both sides
of that subtraction given the identical search budget (enforced in ws4a_ml.py). Every
figure here plots the gap, and every figure also shows the permuted score that was
subtracted, so a gap that moved for the wrong reason is visible rather than hidden.

VERDICTS
--------
  improved   gap grew, and the permuted score did not grow more than the real one
  bias       real score grew but the permuted score grew as much or more -- the
             search bought apparent performance, not signal
  degraded   gap shrank
  unchanged  |delta gap| below --tol (default 0.02, roughly one fold's noise here)

Nothing in this script re-fits a model. It reads the two runs' CSVs, so it can be
run on a laptop against results copied off the cluster.
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
def load_run(d: Path, cell_line: str, label: str) -> pd.DataFrame:
    """Read one run's ml_summary_<cell>.csv, or rebuild it from ml_<cell>.csv."""
    summ = d / f"ml_summary_{cell_line}.csv"
    raw = d / f"ml_{cell_line}.csv"
    if summ.exists():
        df = pd.read_csv(summ)
    elif raw.exists():
        # Older runs, or a run stopped before the summary was written.
        full = pd.read_csv(raw)
        real = full[~full.permuted].copy()
        perm = full[full.permuted][KEY + ["score_mean"]].rename(
            columns={"score_mean": "permuted_score"})
        df = real.merge(perm, on=KEY, how="left")
        df["gap_vs_permuted"] = df.score_mean - df.permuted_score
        LOG.warning("%s: no ml_summary — rebuilt from %s", label, raw.name)
    else:
        raise FileNotFoundError(
            f"{label}: neither {summ.name} nor {raw.name} in {d}\n"
            f"  the ml stage writes both; point --{label} at the run's ml/ subdirectory")
    missing = [c for c in KEY if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: {summ.name} lacks {missing} — not a WS4A ml table")
    if "permuted_score" not in df.columns:
        raise ValueError(f"{label}: no permuted_score column. That run was made with "
                         "--no-permuted-control, so there is no honest gap to compare.")
    df["run"] = label
    LOG.info("%-9s : %d rows from %s", label, len(df), d)
    return df


def merge_runs(base: pd.DataFrame, tuned: pd.DataFrame, tol: float) -> pd.DataFrame:
    m = base.merge(tuned, on=KEY, how="outer", suffixes=("_base", "_tuned"),
                   indicator=True)
    only = m[m._merge != "both"]
    if len(only):
        LOG.warning("%d combination(s) present in only one run — reported as NaN, "
                    "never silently dropped:", len(only))
        for _, r in only.iterrows():
            LOG.warning("   %-26s %-22s %-13s only in %s", r.target, r.block, r.model,
                        "baseline" if r._merge == "left_only" else "tuned")
    m = m.drop(columns="_merge")

    m["d_gap"] = m.gap_vs_permuted_tuned - m.gap_vs_permuted_base
    m["d_score"] = m.score_mean_tuned - m.score_mean_base
    m["d_permuted"] = m.permuted_score_tuned - m.permuted_score_base

    def verdict(r):
        if not np.isfinite(r.d_gap):
            return "incomparable"
        if abs(r.d_gap) < tol:
            return "unchanged"
        if r.d_gap < 0:
            # A real score that rose while the gap fell means the control rose faster.
            return "bias" if r.d_score > tol else "degraded"
        return "improved"

    m["verdict"] = m.apply(verdict, axis=1)
    return m.sort_values("d_gap", ascending=False)


# --------------------------------------------------------------------------- #
def plot_slopes(m, cfg, out, cell_line, top=None, suffix=""):
    """One line per target x block x model: baseline gap -> tuned gap.

    `top` keeps only the N largest absolute movers. The full 120-row version is an
    audit trail -- correct, and illegible on a projector. The trimmed one is the
    slide.
    """
    plt, palette, dpi = _style(cfg)
    d = m[np.isfinite(m.d_gap)]
    if d.empty:
        return None
    if top:
        d = d.reindex(d.d_gap.abs().sort_values(ascending=False).index).head(int(top))
    d = d.sort_values("gap_vs_permuted_base")
    n = len(d)
    # Width must accommodate the y labels, which are long ("target · block · model").
    # At 7.2 in the axes get pushed right and the title and x label run off the canvas.
    lab_w = max((len(f"{r.target} · {r.block} · {r.model}") for _, r in d.iterrows()),
                default=30)
    width = 6.0 + 0.085 * lab_w
    fig, ax = plt.subplots(figsize=(width, max(3.4, (0.30 if top else 0.22) * n + 2.0)))

    colours = {"improved": "#2f7d32", "bias": "#c2410c",
               "degraded": "#b02418", "unchanged": "#9a9a97"}
    y = np.arange(n)
    for i, (_, r) in enumerate(d.iterrows()):
        c = colours.get(r.verdict, "#9a9a97")
        ax.plot([r.gap_vs_permuted_base, r.gap_vs_permuted_tuned], [i, i],
                color=c, lw=1.4, zorder=1, alpha=0.85)
        ax.scatter([r.gap_vs_permuted_base], [i], s=22, facecolor="white",
                   edgecolor="#555", lw=0.9, zorder=2)
        ax.scatter([r.gap_vs_permuted_tuned], [i], s=30, color=c, zorder=3)

    ax.axvline(0, color="#333", lw=1.0, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.target} · {r.block} · {r.model}" for _, r in d.iterrows()],
                       fontsize=7 if not top else 8.5)
    ax.set_xlabel("gap vs permuted  (real − permuted, same search budget)")
    ax.set_title(f"Tuning effect on the honest gap — {cell_line}\n"
                 "hollow = untuned, filled = tuned\n"
                 "left of the dashed line: worse than its own shuffled control",
                 fontsize=9)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="-", label=k)
               for k, c in colours.items()]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    p = out / f"compare_gap_slopes{suffix}_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def plot_real_vs_permuted(m, cfg, out, cell_line):
    """The figure that catches a fake win: did the CONTROL move too?"""
    plt, palette, dpi = _style(cfg)
    d = m[np.isfinite(m.d_gap)]
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))

    for ax, (col, title) in zip(axes, (
            ("score_mean", "real labels"), ("permuted_score", "PERMUTED labels"))):
        xb, yt = d[f"{col}_base"], d[f"{col}_tuned"]
        lo = float(np.nanmin([xb.min(), yt.min()])) - 0.03
        hi = float(np.nanmax([xb.max(), yt.max()])) + 0.03
        ax.plot([lo, hi], [lo, hi], color="#999", lw=1.0, ls="--", zorder=1)
        ax.scatter(xb, yt, s=34, c=[{"improved": "#2f7d32", "bias": "#c2410c",
                                     "degraded": "#b02418"}.get(v, "#9a9a97")
                                    for v in d.verdict],
                   edgecolor="#444", lw=0.5, zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("untuned baseline"); ax.set_ylabel("tuned")
        med = float(np.nanmedian(yt - xb))
        ax.set_title(f"{title}\nmedian shift {med:+.3f}", fontsize=9)
        _despine(ax)

    axes[1].text(0.03, 0.95,
                 "Points above the diagonal HERE are the warning:\n"
                 "these labels carry no signal, so any rise is\n"
                 "manufactured by the search, not learned.",
                 transform=axes[1].transAxes, va="top", fontsize=7.4, color="#7a2c0a")
    fig.suptitle(f"Tuned vs untuned, real and permuted — {cell_line}", fontsize=10)
    fig.tight_layout()
    p = out / f"compare_real_vs_permuted_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def plot_per_model(m, cfg, out, cell_line):
    """Which models did tuning actually help? Grouped by model, gap deltas."""
    plt, palette, dpi = _style(cfg)
    d = m[np.isfinite(m.d_gap)]
    if d.empty:
        return None
    models = sorted(d.model.unique())
    fig, ax = plt.subplots(figsize=(1.55 * len(models) + 3.2, 4.4))
    for i, mod in enumerate(models):
        v = d[d.model == mod].d_gap.values
        ax.scatter(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v,
                   s=30, color=palette[i % len(palette)], edgecolor="#444", lw=0.5,
                   zorder=3)
        ax.hlines(float(np.median(v)), i - 0.28, i + 0.28, color="#222", lw=2, zorder=4)
    ax.axhline(0, color="#333", lw=1.0, ls="--")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Δ gap  (tuned − untuned)")
    ax.set_title(f"Did tuning help, per model — {cell_line}\n"
                 "bar = median across target × block. Above 0 the tuned run has a "
                 "larger honest gap.", fontsize=9)
    _despine(ax)
    fig.tight_layout()
    p = out / f"compare_per_model_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def plot_bias_sweep(path: Path, cfg, out, cell_line):
    """Apparent performance on PERMUTED labels as a function of search budget."""
    plt, palette, dpi = _style(cfg)
    df = pd.read_csv(path)
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, (key, g) in enumerate(df.groupby(["model", "target", "block"])):
        g = g.sort_values("n_trials")
        ax.errorbar(g.n_trials, g.permuted_score_mean, yerr=g.permuted_score_std,
                    marker="o", ms=4, lw=1.3, capsize=2.5,
                    color=palette[i % len(palette)],
                    label=" · ".join(str(k) for k in key))
    ch = float(df.chance.median())
    ax.axhline(ch, color="#b02418", lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1], ch, "  chance", color="#b02418", va="center", fontsize=8)
    ax.set_xlabel("Optuna trials per outer fold")
    ax.set_ylabel("score on PERMUTED labels")
    ax.set_title(f"Price of a bigger search — {cell_line}\n"
                 "These labels contain no signal. Everything above the dashed line "
                 "was manufactured by the search itself.", fontsize=9)
    ax.legend(fontsize=6.6, frameon=False, ncol=1)
    _despine(ax)
    fig.tight_layout()
    p = out / f"bias_sweep_{cell_line}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--baseline", required=True,
                    help="directory holding the UNTUNED run's ml_summary_<cell>.csv")
    ap.add_argument("--tuned", required=True,
                    help="directory holding the TUNED run's ml_summary_<cell>.csv")
    ap.add_argument("--bias-sweep-csv", default=None,
                    help="optional bias_sweep_<cell>.csv from `ml --bias-sweep`")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="|Δ gap| below this is 'unchanged' (default 0.02, ~one fold "
                         "of noise at n~64)")
    args = ap.parse_args(argv)
    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)

    base_d, tuned_d = Path(args.baseline), Path(args.tuned)
    if base_d.resolve() == tuned_d.resolve():
        LOG.error("--baseline and --tuned are the same directory (%s). The tuned run "
                  "overwrote the baseline; there is nothing to compare.", base_d)
        return 1

    try:
        base = load_run(base_d, args.cell_line, "baseline")
        tuned = load_run(tuned_d, args.cell_line, "tuned")
    except (FileNotFoundError, ValueError) as exc:
        # A clean message, not a traceback: under `set -euo pipefail` in the sbatch
        # the traceback is the last thing in the log and buries the actual cause.
        LOG.error("%s", exc)
        return 2

    # A tuned run whose n_trials is 0 was made with the wrong config -- say so rather
    # than plot an identical pair of runs and call it a null result.
    if "n_trials" in tuned.columns and float(pd.to_numeric(
            tuned.n_trials, errors="coerce").fillna(0).max()) == 0:
        LOG.error("the --tuned run records n_trials=0 for every row: it used the fixed "
                  "grid, not Optuna. Re-run with --config configs/ws4a_tuned.yaml "
                  "(or --n-trials N).")
        return 1

    out = Path(args.out) if args.out else C.outputs_dir(cfg, "compare")
    m = merge_runs(base, tuned, args.tol)

    keep = KEY + ["n", "chance",
                  "score_mean_base", "score_mean_tuned", "d_score",
                  "permuted_score_base", "permuted_score_tuned", "d_permuted",
                  "gap_vs_permuted_base", "gap_vs_permuted_tuned", "d_gap", "verdict"]
    tbl = m[[c for c in keep if c in m.columns]]
    C.save_table(tbl, out / f"compare_tuned_vs_untuned_{args.cell_line}.csv")

    figs = [plot_slopes(m, cfg, out, args.cell_line),
            plot_slopes(m, cfg, out, args.cell_line, top=25, suffix="_top25"),
            plot_real_vs_permuted(m, cfg, out, args.cell_line),
            plot_per_model(m, cfg, out, args.cell_line)]
    sweep = Path(args.bias_sweep_csv) if args.bias_sweep_csv else \
        tuned_d / f"bias_sweep_{args.cell_line}.csv"
    if sweep.exists():
        figs.append(plot_bias_sweep(sweep, cfg, out, args.cell_line))
    else:
        LOG.info("no bias_sweep CSV at %s — run `ml --bias-sweep 0,10,40,150` to "
                 "measure what the budget costs", sweep)

    counts = m.verdict.value_counts().to_dict()
    print("\n" + "=" * 108)
    print(f"TUNED vs UNTUNED — {args.cell_line}")
    print("  Read `d_gap`, not `d_score`. d_score rises on pure noise; d_gap is what")
    print("  survives subtracting a control given the IDENTICAL search budget.")
    print("  verdict `bias` = the real score rose but the permuted control rose as much.")
    print("=" * 108)
    with pd.option_context("display.width", 230, "display.max_rows", 300,
                           "display.float_format", lambda v: f"{v: .3f}"):
        print(tbl.to_string(index=False))
    print("\nverdicts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    fin = m[np.isfinite(m.d_gap)]
    if len(fin):
        print(f"median Δ gap      {float(np.median(fin.d_gap)):+.3f}")
        print(f"median Δ real     {float(np.median(fin.d_score)):+.3f}")
        print(f"median Δ permuted {float(np.median(fin.d_permuted)):+.3f}"
              "   <- if this matches Δ real, tuning bought bias, not signal")
    for f in [f for f in figs if f]:
        print(f"  fig -> {f}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
