#!/usr/bin/env python
"""WS4A pre-flight — call every vendored API exactly as the pipeline calls it.

    python scripts/ws4a_selftest.py            # ~20 s on tiny synthetic data
    bash scripts/ws4a.sh python /work/scripts/ws4a_selftest.py

WHY THIS EXISTS
---------------
The five algorithms in scripts/ws4a/ are vendored, and their return contracts are
not obvious from the call site. Three separate mismatches reached a real run before
this file existed:

  * mantel_test takes `permutations=`, not `n_perm=`
  * AJIVE.fit takes ORDERED LISTS; a dict silently iterates the KEYS and dies with
    "could not convert string to float: 'a'"
  * permutation_null_joint_rank returns a TUPLE (observed, null_array); passing the
    tuple to np.asarray raises "inhomogeneous shape after 1 dimensions"

The last of those surfaced 40 minutes into an HPC job, after the expensive part had
already completed. Every check here runs in seconds on synthetic data shaped like
the real thing, and asserts the SHAPE AND TYPE of what comes back -- not merely
that the call did not raise.

Run it before submitting. The sbatch runs it automatically as stage 0.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = "ok  ", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    """Decorator: run a contract check, record pass/fail, never abort the suite."""
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            results.append((PASS, name, f"{detail}  ({time.time() - t0:.1f}s)"))
        except Exception as exc:                                  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()[-1]
            results.append((FAIL, name, tb))
        return fn
    return deco


# --------------------------------------------------------------------------- #
# synthetic data shaped like the real blocks, but small enough to be instant
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
N = 60
Zc = rng.standard_normal((N, 3))
MORPH = Zc @ rng.standard_normal((3, 80)) + 0.8 * rng.standard_normal((N, 80))
EXPR = Zc @ rng.standard_normal((3, 150)) + 0.8 * rng.standard_normal((N, 150))
ECFP = rng.integers(0, 2, (N, 64)).astype(float)
NOISE_A = rng.standard_normal((N, 80))
NOISE_B = rng.standard_normal((N, 150))
YBIN = (Zc[:, 0] > 0).astype(int)
YCLS = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 20)


@check("matrix_agreement.scale_block / distance_matrix")
def _t1():
    from ws4a import matrix_agreement as MA
    A = MA.scale_block(MORPH, how="zscore")
    assert A.shape == MORPH.shape, A.shape
    D = MA.distance_matrix(A, metric="euclidean")
    assert D.shape == (N, N), D.shape
    assert np.allclose(np.diag(D), 0), "distance diagonal must be zero"
    return f"D {D.shape}"


@check("matrix_agreement.mantel_test  -> dict with r, p")
def _t2():
    from ws4a import matrix_agreement as MA
    Da = MA.distance_matrix(MA.scale_block(MORPH))
    Db = MA.distance_matrix(MA.scale_block(EXPR))
    # NOTE the kwarg name: `permutations`, not `n_perm`.
    out = MA.mantel_test(Da, Db, method="spearman", permutations=99, seed=0)
    for k in ("r", "p"):
        assert k in out, f"missing key {k}: {list(out)}"
        assert np.isscalar(out[k]) or np.ndim(out[k]) == 0, f"{k} is not scalar"
    assert 0 <= out["p"] <= 1, out["p"]
    return f"r={out['r']:+.3f} p={out['p']:.3f}"


@check("matrix_agreement.rv_coefficients -> rv, rv2, rv_adj")
def _t3():
    from ws4a import matrix_agreement as MA
    out = MA.rv_coefficients(MA.scale_block(MORPH), MA.scale_block(EXPR))
    for k in ("rv", "rv2", "rv_adj"):
        assert k in out, f"missing key {k}: {list(out)}"
    # The whole reason Tier 1 is trustworthy: adjusted RV must be ~0 on
    # independent blocks where the plain RV is large.
    null = MA.rv_coefficients(MA.scale_block(NOISE_A), MA.scale_block(NOISE_B))
    assert abs(null["rv_adj"]) < 0.2, f"adjusted RV not ~0 on noise: {null['rv_adj']}"
    assert null["rv"] > null["rv_adj"], "plain RV should exceed adjusted on noise"
    return f"planted rv_adj={out['rv_adj']:+.3f}, noise rv_adj={null['rv_adj']:+.3f} (plain {null['rv']:.3f})"


@check("matrix_agreement.rv_permutation_test -> stat, p, rv_emp_adj")
def _t4():
    from ws4a import matrix_agreement as MA
    out = MA.rv_permutation_test(MA.scale_block(MORPH), MA.scale_block(EXPR),
                                 kind="rv", permutations=99, seed=0)
    for k in ("stat", "p", "rv_emp_adj"):
        assert k in out, f"missing key {k}: {list(out)}"
    # 'rv_adj' is NOT a valid `kind` -- the adjusted value comes back in the dict.
    try:
        MA.rv_permutation_test(MORPH, EXPR, kind="rv_adj", permutations=9)
    except ValueError:
        pass
    else:
        raise AssertionError("kind='rv_adj' should raise; the contract changed")
    return f"stat={out['stat']:.3f} p={out['p']:.3f}"


@check("matrix_agreement.protest -> r, m2, p  (NOT 'stat')")
def _t5():
    from ws4a import matrix_agreement as MA
    out = MA.protest(MA.scale_block(MORPH), MA.scale_block(EXPR),
                     permutations=99, seed=0)
    for k in ("r", "m2", "p"):
        assert k in out, f"missing key {k}: {list(out)}"
    assert "stat" not in out, "protest gained a 'stat' key; check the call sites"
    return f"r={out['r']:.3f} p={out['p']:.3f}"


@check("ajive.AJIVE.fit  -> takes ORDERED LISTS, not dicts")
def _t6():
    from ws4a import ajive as AJ
    aj = AJ.AJIVE(init_signal_ranks=[8, 8], random_state=0).fit([MORPH, EXPR])
    assert hasattr(aj, "joint_rank_"), "no joint_rank_ attribute"
    assert int(aj.joint_rank_) >= 1, f"planted rank-3 gave joint_rank={aj.joint_rank_}"
    ind = np.atleast_1d(aj.individual_ranks_).tolist()
    assert len(ind) == 2, ind
    # a dict must fail loudly, not silently produce nonsense
    try:
        AJ.AJIVE(init_signal_ranks=[8, 8], random_state=0).fit({"a": MORPH, "b": EXPR})
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("AJIVE accepted a dict; the call sites assume it cannot")
    return f"joint_rank={aj.joint_rank_} individual={ind}"


@check("ajive.AJIVE on noise -> joint_rank 0 (destructive control)")
def _t7():
    from ws4a import ajive as AJ
    aj = AJ.AJIVE(init_signal_ranks=[8, 8], random_state=0).fit([NOISE_A, NOISE_B])
    assert int(aj.joint_rank_) == 0, f"noise gave joint_rank={aj.joint_rank_}, must be 0"
    return "joint_rank=0"


@check("ajive.permutation_null_joint_rank -> TUPLE (observed, null_array)")
def _t8():
    from ws4a import ajive as AJ
    out = AJ.permutation_null_joint_rank([MORPH, EXPR], [8, 8], n_perm=4, random_state=0)
    assert isinstance(out, tuple) and len(out) == 2, f"expected a 2-tuple, got {type(out)}"
    obs, null = out
    null = np.atleast_1d(np.asarray(null, dtype=float))
    assert null.ndim == 1 and null.size == 4, null.shape
    assert np.isscalar(obs) or np.ndim(obs) == 0, "observed rank should be scalar"
    return f"observed={obs} null={null.astype(int).tolist()}"


@check("ajive.pca_reduce -> (n, q)")
def _t9():
    from ws4a import ajive as AJ
    Z = AJ.pca_reduce(EXPR, q=10)
    assert Z.shape == (N, 10), Z.shape
    return f"{Z.shape}"


@check("stabsel.stability_selection -> .prob / .selected(pi) / .ev_bound_mb(pi)")
def _t10():
    from ws4a import stabsel as SS
    beta = np.zeros(MORPH.shape[1]); beta[:4] = 3.0
    y = MORPH @ beta + rng.standard_normal(N)
    res = SS.stability_selection(MORPH, y, mode="cpss", B=30, family="gaussian",
                                 random_state=0)
    prob = res.prob
    assert prob.shape == (MORPH.shape[1],), prob.shape
    # `selected` is a METHOD taking pi_thr, not an attribute
    sel = res.selected(0.7)
    assert np.ndim(sel) == 1, "selected(pi) should return a 1-D index array"
    bound = res.ev_bound_mb(0.7)
    assert np.isfinite(bound), bound
    q = SS.q_for_target_ev(5.0, 0.7, MORPH.shape[1])
    assert np.isfinite(q) and q > 0, q
    res2 = res.select_q(q)
    assert hasattr(res2, "prob"), "select_q must return a StabSelResult"
    planted = set(range(4)) & set(res.selected(0.6).tolist())
    return f"{len(sel)} selected, E(V)<={bound:.2f}, planted found={len(planted)}/4"


@check("stabsel binomial family (classification path)")
def _t11():
    from ws4a import stabsel as SS
    res = SS.stability_selection(MORPH, YBIN, mode="cpss", B=20, family="binomial",
                                 random_state=0)
    assert res.prob.shape == (MORPH.shape[1],), res.prob.shape
    return f"q={res.q:.1f}"


@check("splsda.SparsePLSDA fit/predict -> classes_, x_loadings_, selected_")
def _t12():
    from ws4a import splsda as SP
    sp = SP.SparsePLSDA(n_components=2, keep_x=20).fit(MORPH, YCLS)
    for a in ("classes_", "x_loadings_", "selected_"):
        assert hasattr(sp, a), f"missing {a}"
    pred = sp.predict(MORPH)
    assert len(pred) == N, len(pred)
    # Sparsity lives in x_weights_ / selected_, NOT in x_loadings_, which stays dense
    # by construction. Any xAI figure must read the weights, or it will show a dense
    # model where a sparse one was fitted.
    nz = int((np.abs(sp.x_weights_[:, 0]) > 0).sum())
    assert nz <= 20, f"keep_x=20 but component 1 kept {nz} weights"
    assert len(sp.selected_[0]) == nz, (len(sp.selected_[0]), nz)
    dense = int((np.abs(sp.x_loadings_[:, 0]) > 0).sum())
    return f"weights keep {nz}/{MORPH.shape[1]} (loadings dense at {dense}, as expected)"


@check("permcca.permcca / pc_budget -> dict")
def _t13():
    from ws4a import ajive as AJ
    from ws4a import permcca as PC
    A = AJ.pca_reduce(MORPH, q=5)
    B = AJ.pca_reduce(EXPR, q=5)
    out = PC.permcca(A, B, n_perm=49, n_components=2, seed=0)
    assert isinstance(out, dict), type(out)
    assert out, "permcca returned an empty dict"
    return f"keys={sorted(out)[:5]}"


@check("common.parse_feature handles BOTH naming conventions")
def _t14():
    from ws4a import common as C
    comps = ["Cells", "Cytoplasm", "Nuclei", "Cyto", "Nuc"]
    chans = ["AGP", "DNA", "ER", "Mito", "RNA"]
    a = C.parse_feature("Nuclei_Correlation_RWC_AGP_DNA", comps, chans)   # LINCS
    b = C.parse_feature("Cyto_Texture_SumAverage_DNA_3_00_256", comps, chans)  # OpenScreen
    assert a["compartment"] == "Nuclei" and a["channel"] == "AGP" and a["channel_2"] == "DNA", a
    assert b["compartment"] == "Cyto" and b["family"] == "Texture" and b["channel"] == "DNA", b
    return "LINCS + OpenScreen conventions both parse"


@check("common.target_is_usable rejects a degenerate binary target")
def _t15():
    from ws4a import common as C
    guards = {"min_usable_n": 30, "min_minority_count": 10, "min_minority_fraction": 0.15}
    y = np.array(["1"] * 68 + ["0"] * 2)
    mask = np.ones(70, dtype=bool)
    ok, why = C.target_is_usable(y, mask, "classification", guards, "tox_derm")
    assert not ok, "a 68/2 target must be rejected"
    y2 = np.array(["1"] * 37 + ["0"] * 33)
    ok2, _ = C.target_is_usable(y2, np.ones(70, dtype=bool), "classification",
                                guards, "tox_cardio")
    assert ok2, "a 37/33 target must be accepted"
    return f"68/2 rejected ({why[:40]}...), 37/33 accepted"


@check("PARALLEL == SERIAL: ajive permutation null")
def _t17():
    from ws4a import ajive as AJ
    a = AJ.permutation_null_joint_rank([MORPH, EXPR], [8, 8], n_perm=4,
                                       random_state=0, n_jobs=1)
    b = AJ.permutation_null_joint_rank([MORPH, EXPR], [8, 8], n_perm=4,
                                       random_state=0, n_jobs=4, inner_threads=1)
    assert int(a[0]) == int(b[0]), (a[0], b[0])
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1])), (a[1], b[1])
    return f"identical: observed={a[0]}, null={np.asarray(a[1]).astype(int).tolist()}"


@check("common.parallel_map preserves ORDER and values")
def _t18():
    from ws4a import common as C
    items = list(range(17))
    ser = C.parallel_map(lambda x: x * x, items, n_jobs=1)
    par = C.parallel_map(lambda x: x * x, items, n_jobs=4)
    assert ser == par == [x * x for x in items], (ser[:5], par[:5])
    assert C.child_seeds(7, 5) == C.child_seeds(7, 5), "child_seeds not reproducible"
    return "order and values preserved"


@check("common.resolve_device never raises on 'auto'")
def _t16():
    from ws4a import common as C
    dev = C.resolve_device("auto")
    assert dev in ("cpu", "cuda"), dev
    return dev


@check("PARALLEL == SERIAL: ml.evaluate outer folds")
def _t23():
    """The outer CV folds now run on a worker pool. Same numbers, or it is a bug.

    Each fold is seeded by its index rather than by call order, so the mean and
    the std must be bit-identical between n_jobs=1 and n_jobs=8 -- for the grid
    path AND the Optuna path, whose TPE sampler is seeded per fold too.
    """
    import numpy as np
    import ws4a_ml as ML
    from ws4a import common as C, tuning as TU
    rng = np.random.default_rng(3)
    n, p = 60, 30
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + 0.8 * rng.standard_normal(n) > 0).astype(int)
    cfg = C.Config({"ml": {"cv": {"outer_folds": 4, "inner_folds": 2, "repeats": 2,
                                  "stratify": True, "seed": 0},
                           "elastic_net": {"max_iter": 200, "tol": 0.01,
                                           "l1_ratios": [0.5, 1.0]}}}, "<selftest>")
    out = []
    budgets = [0, 3] if TU.HAVE_OPTUNA else [0]
    for nt in budgets:
        ser = ML.evaluate(X, y, "classification", "elastic_net", cfg, "cpu", 0,
                          n_jobs=1, n_trials=nt)
        par = ML.evaluate(X, y, "classification", "elastic_net", cfg, "cpu", 0,
                          n_jobs=8, n_trials=nt)
        assert ser and par, (ser, par)
        assert ser["score_mean"] == par["score_mean"], (nt, ser, par)
        assert ser["score_std"] == par["score_std"], (nt, ser, par)
        assert ser["n_folds"] == par["n_folds"] and ser["n_folds"] >= 4, \
            (ser["n_folds"], par["n_folds"])
        out.append(f"n_trials={nt}: {ser['score_mean']:.4f} both ways")
    return "; ".join(out)


@check("tuning: Optuna NEVER sees an outer fold's held-out rows")
def _t19():
    """The property that makes tuning admissible at n~64, tested at the call site.

    Every row is tagged with a unique id, tune_fit is replaced by a spy, and
    evaluate() is run for real. The spy records which ids each search was handed.
    Two things must hold, and together they ARE the confinement:

      * no single search ever saw all n rows      -> a fold was always held back
      * every row was withheld from >= 1 search   -> the held-out part is scored blind

    A refactor that passed X instead of X[tr] would satisfy neither.
    """
    import numpy as np
    import ws4a_ml as ML
    from ws4a import tuning as TU
    from ws4a import common as C
    if not TU.HAVE_OPTUNA:
        return "optuna absent — skipped (the ml stage refuses --n-trials without it)"

    rng = np.random.default_rng(0)
    n, p = 48, 8
    X = rng.standard_normal((n, p))
    X[:, 0] = np.arange(n)                       # column 0 is the row id
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)

    seen, real_tune = [], TU.tune_fit

    def spy(est, name, Xtr, ytr, *a, **kw):
        seen.append(set(np.round(Xtr[:, 0]).astype(int).tolist()))
        return real_tune(est, name, Xtr, ytr, *a, **kw)

    cfg = C.Config({"ml": {"cv": {"outer_folds": 4, "inner_folds": 2, "repeats": 1,
                                  "stratify": True, "seed": 0},
                           "elastic_net": {"max_iter": 120, "tol": 0.01}}},
                   "<selftest>")
    TU.tune_fit = spy
    try:
        r = ML.evaluate(X, y, "classification", "elastic_net", cfg, "cpu", 0,
                        n_trials=3)
    finally:
        TU.tune_fit = real_tune

    assert r is not None and seen, "evaluate() never reached the tuning path"
    allrows = set(range(n))
    assert all(s < allrows for s in seen), \
        "a search was handed EVERY row — the held-out fold leaked into it"
    withheld = allrows - set.intersection(*seen) if len(seen) > 1 else allrows - seen[0]
    assert set.union(*seen) == allrows, "some row was never trained on"
    assert all(any(i not in s for s in seen) for i in allrows), \
        "some row was in every training set — it was never scored blind"
    sizes = sorted({len(s) for s in seen})
    return (f"{len(seen)} searches, {sizes} rows each of {n}; "
            f"{len(withheld)} rows withheld from at least one")


@check("tuning: every search space is ACCEPTED by its own estimator")
def _t20():
    """A space that emits a parameter the estimator rejects fails only mid-run.

    Optuna catches trial exceptions, so a typo'd parameter name does not crash --
    every trial is silently pruned and the fold falls back to defaults, reporting a
    "tuned" run that tuned nothing. This builds each model exactly as evaluate()
    does, asks its space for one real suggestion, and requires set_params + fit +
    predict to go through.
    """
    import warnings
    import numpy as np
    import ws4a_ml as ML
    from ws4a import common as C, tuning as TU
    if not TU.HAVE_OPTUNA:
        return "optuna absent — skipped"
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    cfg = C.Config({"ml": {"elastic_net": {"max_iter": 150, "tol": 0.01},
                           "sparse_plsda": {"n_components": 2, "keep_x": [10, 25]},
                           "linear_svm": {"C": [0.01, 1.0]},
                           "xgboost": {"max_depth": 2, "n_estimators": 30,
                                       "device": "cpu"}}}, "<selftest>")
    rng = np.random.default_rng(0)
    n, p = 50, 60
    X = rng.standard_normal((n, p))
    ys = {"classification": (rng.random(n) < 0.5).astype(int),
          "regression": rng.standard_normal(n)}
    done = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kind, y in ys.items():
            for name, space in TU.SEARCH_SPACES.items():
                est, _ = ML.build_model(name, kind, cfg, "cpu", 2, 1.0)
                if est is None:
                    continue                      # e.g. sparse_plsda has no regressor
                st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
                params = space(st.ask(), kind, n, p)
                assert all(k.startswith("model__") for k in params), (name, params)
                est.set_params(**params).fit(X, y).predict(X[:3])
                done.append(f"{name}/{kind[:4]}")
    assert len(done) >= 7, done
    return f"{len(done)} model x kind combinations fit and predict: {' '.join(done)}"


@check("tuning: is_degenerate flags exactly-chance-zero-variance")
def _t21():
    from ws4a import tuning as TU
    assert TU.is_degenerate([0.5, 0.5, 0.5, 0.5], 0.5) is True
    assert TU.is_degenerate([0.5, 0.52, 0.48], 0.5) is False      # varies -> real
    assert TU.is_degenerate([0.62, 0.62, 0.62], 0.5) is False     # not at chance
    return "flags the XGBoost min_child_weight=5 failure, not a genuine null"


@check("config `extends:` merges without losing base keys")
def _t22():
    from ws4a import common as C
    from pathlib import Path
    here = Path(__file__).resolve().parent.parent
    base_p, tuned_p = here / "configs/ws4a.yaml", here / "configs/ws4a_tuned.yaml"
    if not tuned_p.exists():
        return "configs/ws4a_tuned.yaml absent — skipped"
    base = C.load_config(base_p, root_override=str(here))
    tuned = C.load_config(tuned_p, root_override=str(here))
    assert tuned.section("ml")["targets"] == base.section("ml")["targets"], \
        "overlay lost the inherited target list"
    assert tuned.section("ml")["models"] == base.section("ml")["models"]
    bo, to = base.get("paths")["outputs"], tuned.get("paths")["outputs"]
    assert bo != to, f"tuned run would OVERWRITE the baseline at {bo}"
    assert int(tuned.section("ml")["tuning"]["n_trials"]) > 0
    return f"targets inherited; outputs {bo} != {to}"


# --------------------------------------------------------------------------- #
def main() -> int:
    width = max(len(n) for _, n, _ in results)
    print("=" * (width + 60))
    print("WS4A pre-flight — vendored API contracts")
    print("=" * (width + 60))
    for status, name, detail in results:
        print(f"  [{status}] {name:<{width}}  {detail}")
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("=" * (width + 60))
    print(f"  {len(results) - n_fail}/{len(results)} passed")
    if n_fail:
        print("\n  A failure here means a call site disagrees with a vendored module.")
        print("  Fix it before submitting: the same mismatch will surface mid-job,")
        print("  after the expensive part has already run.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
