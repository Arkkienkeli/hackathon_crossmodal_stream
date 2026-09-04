#!/usr/bin/env python
"""WS4A Tier 1 — matrix-level agreement between modalities. No learning.

    bash scripts/ws4a.sh python /work/scripts/ws4a_stats.py --cell-line hepg2
    python scripts/ws4a_stats.py --config configs/ws4a.yaml --cell-line hepg2 --quick

This is the workstream's actual question — how much of morphology is shared with
expression — and it may be the whole deliverable. Everything here is estimation and
testing with a permutation null; nothing is fitted to a label.

WHAT IS REPORTED, AND WHY
-------------------------
The headline effect size is the **adjusted RV coefficient** (Mayer et al. 2011).
This is not a stylistic preference. Measured on independent random blocks at our
exact shape (n=94, p=615, q=41,780), where the true value is 0:

    plain RV      0.930      <- would be reported as a near-perfect result
    modified RV2  0.234      <- Smilde et al. 2009; still badly biased here
    adjusted RV  -0.0002     <- the only one that survives p >> n

Raw Procrustes behaves even worse: 0.9995 on pure noise, and FLAT across signal
strength. So for both, the permutation p-value is the trustworthy part and the raw
statistic is not. Plain RV and raw Procrustes are still written to the CSV, because
hiding them would make this page's claim unverifiable — they are simply never the
headline.

CONTROLS
--------
Two destructive controls run by default and both must collapse:
  scrambled  -- compound correspondence between the blocks is permuted
  random     -- one block replaced by Gaussian noise of the same shape
If a cross-modal statistic survives either, the pipeline is wrong, not the biology.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ws4a import ajive as AJ                      # noqa: E402
from ws4a import common as C                      # noqa: E402
from ws4a import matrix_agreement as MA           # noqa: E402
from ws4a import permcca as PC                    # noqa: E402

LOG = C.LOG


# --------------------------------------------------------------------------- #
def pairwise_agreement(Xa, Xb, name_a, name_b, scaling, metric, permutations, seed,
                       label="observed"):
    """Mantel + RV (plain, RV2, adjusted) + PROTEST for one block pair."""
    A = MA.scale_block(Xa, how=scaling)
    B = MA.scale_block(Xb, how=scaling)

    Da, Db = MA.distance_matrix(A, metric=metric), MA.distance_matrix(B, metric=metric)
    man = MA.mantel_test(Da, Db, method="spearman", permutations=permutations, seed=seed)
    rv = MA.rv_permutation_test(A, B, kind="rv", permutations=permutations, seed=seed)
    rvc = MA.rv_coefficients(A, B)
    pro = MA.protest(A, B, permutations=permutations, seed=seed)

    return {
        "pair": f"{name_a}~{name_b}",
        "variant": label,
        "n": int(A.shape[0]),
        "p_a": int(A.shape[1]), "p_b": int(B.shape[1]),
        # --- headline
        "rv_adj": rvc["rv_adj"],
        "rv_adj_p": rv["p"],
        # --- test statistics whose p-values are trustworthy
        "mantel_r": man["r"], "mantel_p": man["p"],
        "protest_r": pro["r"], "protest_p": pro["p"],
        # --- raw statistics, kept for transparency, never the headline
        "rv_plain": rvc["rv"], "rv2": rvc["rv2"],
        "rv_emp_adj": rv.get("rv_emp_adj", np.nan),
        "permutations": permutations,
    }


def run_controls(Xa, Xb, name_a, name_b, cfg, scaling, metric, permutations, seed, n_rep,
                 n_jobs=1, inner_threads=1):
    """Destructive controls. Both must collapse to the null.

    This is the dominant cost of Tier 1 -- 40 replicates per pair, each a full
    agreement computation -- and every replicate is independent, so it parallelises
    perfectly. Each gets its own seeded stream rather than successive draws from one
    generator, so the numbers do not depend on completion order.
    """
    ctl = cfg.section("controls")
    tasks = []
    if ctl.get("scrambled_pairing", True):
        tasks += [("ctl_scrambled", sd) for sd in C.child_seeds(seed, n_rep)]
    if ctl.get("random_block", True):
        tasks += [("ctl_random", sd) for sd in C.child_seeds(seed + 1, n_rep)]

    def _one(task):
        label, sd = task
        r = np.random.default_rng(sd)
        Xother = Xb[r.permutation(Xb.shape[0])] if label == "ctl_scrambled" \
            else r.standard_normal(Xb.shape)
        return pairwise_agreement(Xa, Xother, name_a, name_b, scaling, metric,
                                  permutations, int(sd) % (2**31), label=label)

    return C.parallel_map(_one, tasks, n_jobs, inner_threads,
                          desc=f"controls {name_a}~{name_b}")


def run_ajive(blocks: C.Blocks, cfg, mods, seed, n_jobs=1, inner_threads=4):
    """Joint rank: how many dimensions morphology and expression genuinely share."""
    st = cfg.section("stats").get("ajive", {})
    if not st.get("enabled", True):
        return None
    ranks = st.get("init_signal_ranks", {})
    scaling = cfg.section("stats").get("scaling", "zscore")
    # AJIVE takes ORDERED LISTS of blocks and of their initial signal ranks -- passing
    # dicts silently iterates the KEYS and fails with "could not convert string to
    # float". Keep `used` alongside so the results stay labelled.
    used = [m for m in mods if m in blocks.X]
    if len(used) < 2:
        return None
    Xs = [MA.scale_block(blocks.X[m], how=scaling) for m in used]
    init = [int(ranks.get(m, 20)) for m in used]

    C.log_cap("ajive.init_signal_ranks", dict(zip(used, init)))
    t0 = time.time()
    aj = AJ.AJIVE(
        init_signal_ranks=init,
        n_randdir_samples=int(st.get("n_randdir_samples", 1000)),
        n_wedin_samples=int(st.get("n_wedin_samples", 1000)),
        random_state=seed,
    ).fit(Xs)
    dt = time.time() - t0
    ind = np.atleast_1d(getattr(aj, "individual_ranks_", [])).tolist()
    LOG.info("ajive      : joint_rank=%d individual=%s  (%.1fs)", aj.joint_rank_, ind, dt)

    res = {"joint_rank": int(aj.joint_rank_), "blocks": used,
           "init_signal_ranks": dict(zip(used, init)), "seconds": round(dt, 1),
           "individual_ranks": {m: int(r) for m, r in zip(used, ind)}}

    n_perm = int(st.get("permutation_null_reps", 0))
    if n_perm:
        C.log_cap("ajive.permutation_null_reps", n_perm,
                  "joint rank under destroyed row correspondence")
        t0 = time.time()
        # Returns a TUPLE (observed_rank, array_of_null_ranks) -- it refits the
        # observed model as well. Passing the tuple straight to np.asarray raises
        # "inhomogeneous shape", because it is a scalar next to an n_perm array.
        obs_rank, null = AJ.permutation_null_joint_rank(
            Xs, init, n_perm=n_perm, random_state=seed,
            n_jobs=n_jobs, inner_threads=inner_threads)
        null = np.atleast_1d(np.asarray(null, dtype=float))

        # The refit should reproduce the joint rank we already have. If it does not,
        # the fit is not deterministic and neither number can be trusted.
        if int(obs_rank) != int(aj.joint_rank_):
            LOG.warning("ajive      : refit gave joint_rank=%d but the first fit gave %d "
                        "-- the fit is not deterministic", int(obs_rank), int(aj.joint_rank_))
        res["null_joint_rank_refit"] = int(obs_rank)
        res["null_joint_rank_mean"] = float(np.mean(null))
        res["null_joint_rank_max"] = float(np.max(null))
        res["null_joint_ranks"] = [int(v) for v in null]
        # An empirical p-value for the joint rank: how often does destroying the
        # compound correspondence produce a joint rank at least this large?
        res["null_p"] = float((np.sum(null >= aj.joint_rank_) + 1) / (len(null) + 1))
        res["null_seconds"] = round(time.time() - t0, 1)
        LOG.info("ajive      : permutation null joint rank mean=%.2f max=%.0f p=%.3f (%.0fs)",
                 res["null_joint_rank_mean"], res["null_joint_rank_max"],
                 res["null_p"], res["null_seconds"])
    return res


def run_cca(blocks: C.Blocks, cfg, mods, seed):
    """Regularised CCA via a PC budget, with Winkler-style permutation inference.

    PC-CCA reaches r ~ 0.9 on PERMUTED NULL data when too many PCs are retained, so
    the budget is deliberately small relative to n and the p-value is the result.
    """
    st = cfg.section("stats").get("cca", {})
    if not st.get("enabled", True):
        return None
    if not all(m in blocks.X for m in mods):
        return None

    k = int(st.get("pc_budget", 10))
    n = blocks.n
    C.log_cap("cca.pc_budget", k, f"PCs per block at n={n}; large budgets inflate r on null data")
    scaling = cfg.section("stats").get("scaling", "zscore")
    A = AJ.pca_reduce(MA.scale_block(blocks.X[mods[0]], how=scaling), q=k)
    B = AJ.pca_reduce(MA.scale_block(blocks.X[mods[1]], how=scaling), q=k)

    n_perm = int(st.get("permutations", 999))
    t0 = time.time()
    res = PC.permcca(A, B, n_perm=n_perm, n_components=int(st.get("n_components", 3)), seed=seed)
    # permcca returns r, stat_obs, p_unc, p_fwer, null -- NOT a key called "p".
    # Filtering for "p" printed the canonical correlations with no significance at
    # all, which is the one presentation this pipeline must never produce: PC-CCA
    # reaches r ~ 0.9 on PERMUTED NULL data, so an r without its p means nothing.
    def _fmt(v):
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        return [round(float(x), 4) for x in arr]

    r_ = _fmt(res.get("r", []))
    p_unc = _fmt(res.get("p_unc", []))
    p_fwer = _fmt(res.get("p_fwer", []))
    LOG.info("cca        : r=%s  p_unc=%s  p_fwer=%s  (%.1fs)",
             r_, p_unc, p_fwer, time.time() - t0)
    if r_ and p_fwer and p_fwer[0] > 0.05:
        LOG.warning("cca        : r1=%.3f is NOT significant (p_fwer=%.3f). At this PC "
                    "budget a null pair can reach r ~ 0.9 -- read the p, not the r.",
                    r_[0], p_fwer[0])
    out = {"blocks": list(mods), "pc_budget": k, "permutations": n_perm}
    for kk, vv in res.items():
        out[kk] = vv.tolist() if isinstance(vv, np.ndarray) else vv
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--permutations", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="parallel workers (default: compute.n_jobs, capped by SLURM_CPUS_PER_TASK)")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--no-ajive", action="store_true")
    ap.add_argument("--no-cca", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny permutation budgets — for checking the code runs, not for results")
    args = ap.parse_args()

    C.setup_logging(args.log_level)
    cfg = C.load_config(args.config, args.root)
    st = cfg.section("stats")
    seed = args.seed if args.seed is not None else int(st.get("seed", 0))
    permutations = args.permutations or int(st.get("permutations", 9999))
    n_rep = int(cfg.section("controls").get("n_repeats", 20))

    if args.quick:
        permutations, n_rep = 199, 3
        # Shrink the AJIVE null too. Each replicate is a full AJIVE refit -- ~45 s on
        # the real HepG2 blocks -- so 20 of them is 15 minutes, which is not a smoke
        # test. This keeps --quick to something you will actually run before submitting.
        aj = dict(cfg["stats"].get("ajive", {}))
        aj.update(permutation_null_reps=2, n_randdir_samples=100, n_wedin_samples=100)
        cfg["stats"] = dict(cfg["stats"], ajive=aj)
        LOG.warning("QUICK MODE — permutations=%d, control repeats=%d, ajive null reps=2. "
                    "Smoke test only; do not report these numbers.", permutations, n_rep)

    out_dir = Path(args.out) if args.out else C.outputs_dir(cfg, "stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs, inner_threads = C.resolve_jobs(cfg, args.n_jobs)

    blocks = C.load_mudata_blocks(cfg, args.cell_line)
    C.log_cap("permutations", permutations)
    C.log_cap("control_repeats", n_rep)

    scaling = st.get("scaling", "zscore")
    metric = st.get("distance", "euclidean")
    rows = []

    for pair in st.get("modality_pairs", []):
        a, b = pair
        if a not in blocks.X or b not in blocks.X:
            LOG.warning("skip pair  : %s~%s (missing modality)", a, b)
            continue
        LOG.info("--- %s ~ %s", a, b)
        t0 = time.time()
        rows.append(pairwise_agreement(blocks.X[a], blocks.X[b], a, b,
                                       scaling, metric, permutations, seed))
        r = rows[-1]
        LOG.info("    rv_adj=%+.4f p=%.4f | mantel r=%+.4f p=%.4f | protest r=%.4f p=%.4f (%.0fs)",
                 r["rv_adj"], r["rv_adj_p"], r["mantel_r"], r["mantel_p"],
                 r["protest_r"], r["protest_p"], time.time() - t0)

        if cfg.section("controls").get("enabled", True) and not args.no_controls:
            rows += run_controls(blocks.X[a], blocks.X[b], a, b, cfg,
                                 scaling, metric, permutations, seed, n_rep,
                                 n_jobs=n_jobs, inner_threads=inner_threads)

    df = pd.DataFrame(rows)
    C.save_table(df, out_dir / f"agreement_{args.cell_line}.csv")

    extra = {}
    mods = ["morphology", "expression"]
    if not args.no_ajive:
        extra["ajive"] = run_ajive(blocks, cfg, mods, seed,
                                   n_jobs=n_jobs, inner_threads=max(inner_threads, 4))
    if not args.no_cca:
        extra["cca"] = run_cca(blocks, cfg, mods, seed)

    C.save_json(
        {"cell_line": args.cell_line, "n": blocks.n,
         "modalities": {m: list(blocks.X[m].shape) for m in blocks.X},
         "hygiene": blocks.reports,
         "caps": {"permutations": permutations, "control_repeats": n_rep,
                  "quick": bool(args.quick)},
         **extra},
        out_dir / f"stats_{args.cell_line}.json")

    # ---- summary the user actually reads --------------------------------- #
    obs = df[df.variant == "observed"]
    print("\n" + "=" * 96)
    print(f"TIER 1 — matrix agreement, {args.cell_line}, n={blocks.n}")
    print("  headline effect = adjusted RV. Plain RV and raw Procrustes are in the CSV")
    print("  but are NOT interpretable at p >> n (0.93 and 0.9995 respectively on pure noise).")
    print("=" * 96)
    cols = ["pair", "rv_adj", "rv_adj_p", "mantel_r", "mantel_p", "protest_p", "rv_plain"]
    with pd.option_context("display.width", 200):
        print(obs[cols].to_string(index=False))

    ctl = df[df.variant != "observed"]
    if len(ctl):
        print("\ncontrols (must collapse — if these do not, the pipeline is wrong):")
        g = ctl.groupby(["pair", "variant"]).agg(
            rv_adj_mean=("rv_adj", "mean"),
            frac_p_below_05=("rv_adj_p", lambda s: float((s < 0.05).mean())),
            mantel_r_mean=("mantel_r", "mean")).reset_index()
        print(g.to_string(index=False))
    if extra.get("ajive"):
        aj = extra["ajive"]
        print(f"\nAJIVE joint rank = {aj['joint_rank']}"
              + (f"   (null mean {aj['null_joint_rank_mean']:.2f}, "
                 f"max {aj['null_joint_rank_max']:.0f}, p={aj['null_p']:.3f})"
                 if "null_joint_rank_mean" in aj else ""))
        print(f"      individual ranks: {aj.get('individual_ranks')}")
    print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
