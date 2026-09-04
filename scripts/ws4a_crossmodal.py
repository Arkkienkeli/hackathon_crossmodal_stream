#!/usr/bin/env python3
"""WS4A — the shared morphology / expression / chemistry space, and what defines it.

    bash scripts/ws4a.sh crossmodal --cell-line hepg2

THE ONE AXIS THAT EARNED THE RIGHT TO BE DRAWN
----------------------------------------------
Of the three modality pairs, only morphology~expression has a shared axis that beat
its own permutation control: the first canonical correlation is r = 0.903 with
p = 0.017 (Winkler-style permutation CCA on a 10-PC budget). AJIVE's joint rank of 4
did NOT -- the destroyed-correspondence null also reached 4, p = 0.095 -- so the AJIVE
joint space is not drawn here as though it were structure.

So this script builds ONE thing: the leading morphology~expression canonical axis, the
compounds' positions on it, and the individual features at each end of it. Chemistry
is then related to that axis rather than folded into it, which is what lets the figure
say something about all three.

WHY LOADINGS AND NOT WEIGHTS
----------------------------
CCA runs on PCA-reduced blocks, so its weights live in PC space and mean nothing as
biology. What is plotted instead is the canonical LOADING: the correlation between an
original feature (one gene, one CellProfiler measurement, one ECFP bit) and the
canonical variate. That is interpretable, and it is far more stable than
back-transformed weights.

THE THRESHOLD IS NOT A CONVENTION, IT IS MEASURED
-------------------------------------------------
At n=119 with 41,780 genes, the largest loading you can obtain from pure noise is
large. Every panel therefore draws a threshold from a permutation null on the MAXIMUM
absolute loading -- destroy the compound correspondence, recompute the whole pipeline,
record the biggest loading anywhere, repeat. The 95th percentile of that distribution
is the line. A feature below it is not evidence, however large it looks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import common as C                      # noqa: E402
from ws4a import ajive as AJ                      # noqa: E402
from ws4a import matrix_agreement as MA           # noqa: E402
from ws4a import permcca as PC                    # noqa: E402

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


def _corr_with(Xs: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Correlation of every column of Xs with the vector v. Xs may be very wide."""
    xc = Xs - Xs.mean(0, keepdims=True)
    vc = v - v.mean()
    sx = np.sqrt((xc ** 2).sum(0))
    sv = np.sqrt((vc ** 2).sum())
    denom = sx * sv
    out = np.zeros(Xs.shape[1])
    ok = denom > 1e-12
    out[ok] = (xc[:, ok].T @ vc) / denom[ok]
    return out


def _pretty_feature(name: str, comps, chans) -> str:
    """compartment · family · channel, keeping whatever tokens the parser did not use.

    parse_feature returns compartment / family / channel(_2) only. Three DNA texture
    features differing solely in scale and angle would therefore render as three
    identical labels, so the leftover tokens (measurement name, scale, angle, grey
    levels) are appended rather than dropped.
    """
    pf = C.parse_feature(name, comps, chans)
    used = {pf.get("compartment"), pf.get("family"),
            pf.get("channel"), pf.get("channel_2")} - {None}
    rest = [t for t in str(name).split("_") if t not in used]
    head = " · ".join(str(b) for b in (pf.get("compartment"), pf.get("family"))
                      if b)
    chan = "/".join(str(c) for c in (pf.get("channel"), pf.get("channel_2")) if c)
    parts = [p for p in (head, " ".join(rest), chan) if p]
    return " · ".join(parts) if parts else str(name)


# --------------------------------------------------------------------------- #
def shared_axis(blocks, cfg, mods, seed, n_perm_null=200):
    """The leading canonical axis of two blocks, its loadings, and a null threshold."""
    st = cfg.section("stats").get("cca", {})
    k = int(st.get("pc_budget", 10))
    scaling = cfg.section("stats").get("scaling", "zscore")

    Xs = {m: MA.scale_block(blocks.X[m], how=scaling) for m in mods}
    P = {m: AJ.pca_reduce(Xs[m], q=k) for m in mods}
    LOG.info("cca        : %s x %s on a %d-PC budget", mods[0], mods[1], k)

    r, A, B, U, V = PC.cca_full(P[mods[0]], P[mods[1]])
    u1, v1 = U[:, 0], V[:, 0]
    LOG.info("cca        : r1=%.3f r2=%.3f r3=%.3f", *np.round(r[:3], 3))

    load = {mods[0]: _corr_with(Xs[mods[0]], u1),
            mods[1]: _corr_with(Xs[mods[1]], v1)}

    # --- the null: destroy the compound correspondence, redo everything --------
    rng = np.random.default_rng(seed)
    n = blocks.n
    max_abs = {m: [] for m in mods}
    r1_null = []
    for _ in range(n_perm_null):
        idx = rng.permutation(n)
        rp, _, _, Up, Vp = PC.cca_full(P[mods[0]], P[mods[1]][idx])
        r1_null.append(rp[0])
        max_abs[mods[0]].append(np.abs(_corr_with(Xs[mods[0]], Up[:, 0])).max())
        max_abs[mods[1]].append(np.abs(_corr_with(Xs[mods[1]][idx], Vp[:, 0])).max())
    thr = {m: float(np.quantile(max_abs[m], 0.95)) for m in mods}
    LOG.info("null       : r1 %.3f (95th pct of %d permutations)",
             float(np.quantile(r1_null, 0.95)), n_perm_null)
    for m in mods:
        LOG.info("null       : %-11s max |loading| threshold %.3f  -- %d of %d real "
                 "features exceed it", m, thr[m], int((np.abs(load[m]) > thr[m]).sum()),
                 load[m].size)
    return dict(r=r, u1=u1, v1=v1, load=load, thr=thr, r1_null=np.asarray(r1_null),
                scaled=Xs, mods=mods)


def relate_third(blocks, res, third, seed, n_perm_null=200):
    """How does the third modality sit on an axis it did not help define?

    Chemistry is deliberately NOT part of the canonical fit. If ECFP bits correlate
    with an axis built only from morphology and expression, that is a three-way link;
    if none exceed the null, that is an answer too.
    """
    if third not in blocks.X:
        return None
    Xs = MA.scale_block(blocks.X[third], how="zscore")
    axis = res["u1"]
    load = _corr_with(Xs, axis)
    rng = np.random.default_rng(seed + 1)
    mx = [np.abs(_corr_with(Xs, axis[rng.permutation(len(axis))])).max()
          for _ in range(n_perm_null)]
    thr = float(np.quantile(mx, 0.95))
    LOG.info("null       : %-11s max |loading| threshold %.3f  -- %d of %d exceed it",
             third, thr, int((np.abs(load) > thr).sum()), load.size)
    return dict(load=load, thr=thr, names=blocks.var_names.get(third))


# --------------------------------------------------------------------------- #
def fig_embedding(res, blocks, obs, cfg, out, cell):
    plt, palette, dpi = _style(cfg)
    m0, m1 = res["mods"]
    u1, v1 = res["u1"], res["v1"]
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.7))

    # --- compounds on the shared axis, coloured by mechanism
    ax = axes[0]
    moa = obs.get("moa-fine")
    if moa is not None:
        moa = pd.Series(moa).astype(str).values
        top = [c for c, _ in pd.Series(moa).value_counts().items()
               if c not in ("unclear", "nan", "NaN", "")][:4]
        ax.scatter(u1, v1, s=26, color="#d5d5d2", edgecolor="#999", lw=0.3,
                   label="other / unannotated", zorder=2)
        for i, cls in enumerate(top):
            sel = moa == cls
            ax.scatter(u1[sel], v1[sel], s=52, color=palette[i % len(palette)],
                       edgecolor="#222", lw=0.5, zorder=3,
                       label=f"{cls[:28]} (n={int(sel.sum())})")
        ax.legend(fontsize=6.2, frameon=False, loc="best")
    else:
        ax.scatter(u1, v1, s=30, color=palette[0], edgecolor="#333", lw=0.4)
    ax.set_xlabel(f"{m0} canonical variate 1")
    ax.set_ylabel(f"{m1} canonical variate 1")
    ax.set_title(f"The shared axis, {res['r'][0]:.3f} correlation\n"
                 "each point is one compound, placed by BOTH modalities", fontsize=8.8)
    _despine(ax)

    # --- the same, correspondence destroyed
    ax = axes[1]
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(v1))
    ax.scatter(u1, v1[idx], s=26, color="#c9c9c6", edgecolor="#777", lw=0.3)
    ax.set_xlabel(f"{m0} canonical variate 1")
    ax.set_ylabel(f"{m1} canonical variate 1 (SCRAMBLED)")
    ax.set_title("The same plot with the compound pairing destroyed\n"
                 "this is what no relationship looks like", fontsize=8.8)
    _despine(ax)

    # --- r1 against its null
    ax = axes[2]
    ax.hist(res["r1_null"], bins=32, color="#9a9a97", edgecolor="#333", lw=0.4,
            label=f"null, n={len(res['r1_null'])}")
    ax.axvline(res["r"][0], color="#2f7d32", lw=2.4, label=f"observed {res['r'][0]:.3f}")
    ax.set_xlabel("first canonical correlation")
    ax.set_ylabel("permutations")
    ax.set_title("Why r alone is not the result\n"
                 "the null reaches high correlations on its own", fontsize=8.8)
    ax.legend(fontsize=7, frameon=False)
    _despine(ax)

    fig.suptitle(f"Shared morphology–expression space — {cell}", fontsize=10.5)
    fig.tight_layout()
    p = out / f"crossmodal_embedding_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


def fig_features(res, blocks, third_res, cfg, out, cell, top=18):
    """Which individual features define the shared axis — with the null drawn."""
    plt, palette, dpi = _style(cfg)
    m0, m1 = res["mods"]
    xai = cfg.section("xai").get("feature_grammar", {})
    comps = xai.get("compartments", []); chans = xai.get("channels", [])

    panels = [(m0, res["load"][m0], res["thr"][m0], blocks.var_names.get(m0)),
              (m1, res["load"][m1], res["thr"][m1], blocks.var_names.get(m1))]
    if third_res:
        panels.append(("ecfp", third_res["load"], third_res["thr"], third_res["names"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 6.2))
    axes = np.atleast_1d(axes)
    for ax, (name, load, thr, names) in zip(axes, panels):
        names = (list(names) if names is not None
                 else [f"{name}_{i}" for i in range(len(load))])
        order = np.argsort(-np.abs(load))[:top][::-1]
        vals = load[order]
        labs = []
        for i in order:
            lab = str(names[i])
            if name == m0:
                lab = _pretty_feature(lab, comps, chans)
            labs.append(lab[:52])
        y = np.arange(len(vals))
        cols = ["#2f7d32" if abs(v) > thr else "#c9c9c6" for v in vals]
        ax.barh(y, vals, color=cols, edgecolor="#333", lw=0.4)
        ax.axvline(thr, color="#b02418", lw=1.2, ls="--")
        ax.axvline(-thr, color="#b02418", lw=1.2, ls="--",
                   label=f"null threshold ±{thr:.2f}")
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.4)
        ax.set_xlabel("correlation with the shared axis")
        n_sig = int((np.abs(load) > thr).sum())
        ax.set_title(f"{name}\n{n_sig} of {load.size} features exceed the null",
                     fontsize=9)
        ax.legend(fontsize=6.6, frameon=False, loc="lower right")
        _despine(ax)

    fig.suptitle(f"What defines the shared axis — {cell}. Grey bars are BELOW the "
                 "permutation threshold: large, and not evidence.", fontsize=10)
    fig.tight_layout()
    p = out / f"crossmodal_features_{cell}.png"
    fig.savefig(p, dpi=dpi); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--modalities", default="morphology,expression",
                    help="the two blocks whose shared axis is built")
    ap.add_argument("--third", default="ecfp",
                    help="related to that axis afterwards, never folded into it")
    ap.add_argument("--null-reps", type=int, default=200)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args(argv)
    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    seed = args.seed if args.seed is not None else int(cfg.section("stats").get("seed", 0))
    out = Path(args.out) if args.out else C.outputs_dir(cfg, "crossmodal")

    blocks = C.load_mudata_blocks(cfg, args.cell_line)
    mods = [m.strip() for m in args.modalities.split(",")][:2]
    C.log_cap("null_reps", args.null_reps, "permutations for the loading threshold")

    res = shared_axis(blocks, cfg, mods, seed, n_perm_null=args.null_reps)
    third = relate_third(blocks, res, args.third, seed, n_perm_null=args.null_reps)

    rows = []
    for m in mods:
        names = blocks.var_names.get(m)
        for i, v in enumerate(res["load"][m]):
            if abs(v) > res["thr"][m]:
                rows.append({"block": m, "feature": str(names[i]) if names is not None else i,
                             "loading": float(v), "threshold": res["thr"][m]})
    if third:
        for i, v in enumerate(third["load"]):
            if abs(v) > third["thr"]:
                nm = third["names"]
                rows.append({"block": args.third,
                             "feature": str(nm[i]) if nm is not None else i,
                             "loading": float(v), "threshold": third["thr"]})
    tbl = pd.DataFrame(rows).sort_values("loading", key=abs, ascending=False) \
        if rows else pd.DataFrame(columns=["block", "feature", "loading", "threshold"])
    C.save_table(tbl, out / f"crossmodal_loadings_{args.cell_line}.csv")

    figs = [fig_embedding(res, blocks, blocks.obs, cfg, out, args.cell_line),
            fig_features(res, blocks, third, cfg, out, args.cell_line, top=args.top)]

    print("\n" + "=" * 92)
    print(f"SHARED AXIS — {args.cell_line}: {mods[0]} ~ {mods[1]}")
    print("=" * 92)
    print(f"  first canonical correlation   r = {res['r'][0]:.3f}")
    print(f"  95th percentile of its null   r = {np.quantile(res['r1_null'], 0.95):.3f}"
          "   <- r means nothing without this")
    for m in mods:
        n_sig = int((np.abs(res["load"][m]) > res["thr"][m]).sum())
        print(f"  {m:<11} {n_sig:>6} of {res['load'][m].size:>6} features exceed the "
              f"null threshold ±{res['thr'][m]:.3f}")
    if third:
        n_sig = int((np.abs(third["load"]) > third["thr"]).sum())
        print(f"  {args.third:<11} {n_sig:>6} of {third['load'].size:>6} exceed "
              f"±{third['thr']:.3f}  (related to the axis, not part of it)")
    if len(tbl):
        print("\n  strongest surviving features:")
        print(tbl.head(12).to_string(index=False))
    else:
        print("\n  NOTHING survived the permutation threshold in any block.")
        print("  That is a result: at this n the shared axis cannot be attributed to")
        print("  individual features, only to the blocks as wholes.")
    for f in figs:
        print(f"  fig -> {f}")
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
