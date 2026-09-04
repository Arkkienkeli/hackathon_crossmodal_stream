"""Shared plumbing for the WS4A toolchain: config, paths, data loading, hygiene.

Nothing here hardcodes a path. Every location comes from configs/ws4a.yaml, and the
config's `root` is resolved against the repository so the same file works on this
workstation and on a cluster after an rsync.

The one opinionated thing in this module is `assert_clean`. The A549 morphology in
the delivered MuData carries 44 features up to 1.5e19 (the drop_outliers defect),
and every distance, correlation and gradient computed on it would be dominated by
those. The default is to REFUSE rather than to quietly clean, because a silent fix
produces numbers nobody can trace back to a decision.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

LOG = logging.getLogger("ws4a")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def log_cap(name: str, value, reason: str = "") -> None:
    """Every cap, subsample and threshold is announced. Never silent."""
    LOG.info("CAP %-24s = %-12s %s", name, value, reason)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class Config(dict):
    """dict with dotted access and path resolution against the config's root."""

    def __init__(self, data: dict, config_path: Path):
        super().__init__(data)
        self.config_path = Path(config_path).resolve()
        self.repo_root = self.config_path.parent.parent
        root = Path(str(self.get("root", ".")))
        self.root = root if root.is_absolute() else (self.repo_root / root)

    def get_path(self, *keys: str, must_exist: bool = True) -> Path:
        """Resolve paths.<...> against the data root."""
        node: Any = self.get("paths", {})
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise KeyError(f"configs: paths.{'.'.join(keys)} is not set")
            node = node[k]
        p = Path(str(node))
        p = p if p.is_absolute() else (self.root / p)
        if must_exist and not p.exists():
            raise FileNotFoundError(
                f"paths.{'.'.join(keys)} -> {p}\n"
                f"  data root is {self.root}\n"
                f"  set `root:` in {self.config_path.name}, or pass --root"
            )
        return p

    def section(self, name: str) -> dict:
        return dict(self.get(name, {}) or {})


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge. A LIST in the overlay replaces the base list whole.

    Replacing rather than concatenating is deliberate: `models: [xgboost]` in an
    overlay must mean *only* xgboost, not the four base models plus xgboost twice.
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml_chain(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load one YAML, following an `extends:` chain oldest-first.

    An overlay config states only what it changes. Copying the whole base file
    instead is how a "tuned" run silently ends up with a stale target list six
    edits later.
    """
    _seen = _seen or set()
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular `extends:` chain at {path}")
    _seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if not parent:
        return data
    ppath = Path(parent)
    if not ppath.is_absolute():
        ppath = path.parent / ppath
    LOG.info("extends    : %s", ppath)
    return _deep_merge(_read_yaml_chain(ppath, _seen), data)


def load_config(path: str | Path, root_override: str | None = None) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = _read_yaml_chain(path)
    if root_override:
        data["root"] = root_override
    cfg = Config(data, path)
    LOG.info("config     : %s", cfg.config_path)
    LOG.info("data root  : %s", cfg.root)
    if not cfg.root.exists():
        raise FileNotFoundError(
            f"data root does not exist: {cfg.root}\n"
            f"  fix `root:` in {path}, or pass --root /path/to/hackathon_crossmodal_stream"
        )
    return cfg


def outputs_dir(cfg: Config, subdir: str | None = None) -> Path:
    p = cfg.get_path("outputs", must_exist=False)
    if subdir:
        p = p / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #
class ContaminatedBlockError(RuntimeError):
    pass


def contamination_report(X: np.ndarray, cutoff: float, names: Sequence[str] | None = None) -> dict:
    X = np.asarray(X, dtype=float)
    amax = np.nanmax(np.abs(X), axis=0) if X.size else np.zeros(0)
    bad = np.flatnonzero(amax > cutoff)
    order = bad[np.argsort(-amax[bad])] if bad.size else bad
    return {
        "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
        "n_over_cutoff": int(bad.size),
        "cutoff": float(cutoff),
        "max_abs": float(np.nanmax(np.abs(X))) if X.size else 0.0,
        "min": float(np.nanmin(X)) if X.size else 0.0,
        "max": float(np.nanmax(X)) if X.size else 0.0,
        "n_nonfinite": int((~np.isfinite(X)).sum()),
        "worst": [
            (str(names[j]) if names is not None else int(j), float(amax[j]))
            for j in order[:6]
        ],
    }


def assert_clean(
    X: np.ndarray,
    name: str,
    cutoff: float,
    policy: str = "abort",
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Enforce the contamination policy. Returns (X, keep_mask, report).

    policy:
      abort  -- raise, naming the offending features (the default, on purpose)
      clean  -- drop the offending features and say so
      ignore -- proceed, but log loudly
    """
    rep = contamination_report(X, cutoff, names)
    keep = np.ones(X.shape[1], dtype=bool)
    if rep["n_over_cutoff"] == 0 and rep["n_nonfinite"] == 0:
        LOG.info("hygiene    : %-12s clean (%d features, range %.4g .. %.4g)",
                 name, rep["n_features"], rep["min"], rep["max"])
        return X, keep, rep

    worst = ", ".join(f"{n}={v:.3g}" for n, v in rep["worst"][:3])
    msg = (f"{name}: {rep['n_over_cutoff']} of {rep['n_features']} features exceed "
           f"|{cutoff:g}| (max {rep['max_abs']:.4g}); worst: {worst}")

    if policy == "abort":
        raise ContaminatedBlockError(
            msg + "\n\n"
            "  This is the drop_outliers defect: pycytominer's mad_robustize divides by\n"
            "  (MAD + 1e-18), so a zero-MAD feature explodes. Every distance, correlation\n"
            "  and gradient computed on this block would be dominated by these features.\n\n"
            "  Rebuild the block with drop_outliers in the feature-selection operations,\n"
            "  or set hygiene.on_contamination: clean to drop them here (and say so in\n"
            "  the write-up). See docs/ws4/finding.md."
        )
    if policy == "clean":
        amax = np.nanmax(np.abs(np.asarray(X, float)), axis=0)
        keep = np.isfinite(amax) & (amax <= cutoff)
        LOG.warning("hygiene    : %s -- CLEANED, dropped %d of %d features (%s)",
                    name, int((~keep).sum()), rep["n_features"], worst)
        return np.asarray(X, float)[:, keep], keep, rep
    LOG.error("hygiene    : %s -- IGNORED: %s", name, msg)
    return X, keep, rep


def translated_blocklist(feature_names: Sequence[str], prefix_map: dict) -> list[str]:
    """pycytominer's blocklist under a compartment-name translation.

    The packaged list is written `Nuclei_...`/`Cytoplasm_...`. OpenScreen features are
    `Nuc_...`/`Cyto_...`, so the raw list matches NOTHING (measured: 0 of 55). After
    translating the prefixes, 11 features match. Without this the `blocklist`
    operation looks like it is filtering and is silently inert.
    """
    try:
        import pycytominer
        bl_path = Path(pycytominer.__file__).parent / "data" / "blocklist_features.txt"
        blocklist = {
            line.strip() for line in bl_path.read_text().splitlines()
            if line.strip() and line.strip() != "blocklist"
        }
    except Exception as exc:                                     # noqa: BLE001
        LOG.warning("hygiene    : could not read pycytominer blocklist (%s)", exc)
        return []

    hits = []
    for nm in feature_names:
        translated = nm
        for src, dst in (prefix_map or {}).items():
            if translated.startswith(src):
                translated = dst + translated[len(src):]
                break
        if translated in blocklist:
            hits.append(nm)
    LOG.info("hygiene    : blocklist matched %d feature(s) after prefix translation", len(hits))
    return hits


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Blocks:
    """Matched modality blocks for one cell line, plus their shared obs."""

    cell_line: str
    obs: pd.DataFrame
    X: dict[str, np.ndarray] = field(default_factory=dict)
    var_names: dict[str, list[str]] = field(default_factory=dict)
    reports: dict[str, dict] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.obs)

    def modalities(self) -> list[str]:
        return list(self.X)

    def get(self, spec) -> tuple[np.ndarray, list[str]]:
        """One modality, or several concatenated (early fusion)."""
        mods = [spec] if isinstance(spec, str) else list(spec)
        missing = [m for m in mods if m not in self.X]
        if missing:
            raise KeyError(f"unknown modality {missing}; have {self.modalities()}")
        X = np.hstack([self.X[m] for m in mods])
        names = [f"{m}:{v}" for m in mods for v in self.var_names[m]]
        return X, names

    def summary(self) -> str:
        return " | ".join(f"{m} {self.X[m].shape}" for m in self.X)


def load_mudata_blocks(cfg: Config, cell_line: str) -> Blocks:
    """Load one cell line's MuData and enforce the hygiene policy on every block."""
    import mudata

    mdir = cfg.get_path("mudata_dir")
    fname = cfg.section("paths").get("mudata", {}).get(cell_line)
    if fname is None:
        raise KeyError(f"paths.mudata.{cell_line} is not set in the config")
    path = mdir / fname
    if not path.exists():
        raise FileNotFoundError(f"MuData not found: {path}")

    LOG.info("loading    : %s", path.name)
    md = mudata.read_h5mu(str(path))
    hyg = cfg.section("hygiene")
    cutoff = float(hyg.get("outlier_cutoff", 500.0))
    policy = str(hyg.get("on_contamination", "abort"))

    blocks = Blocks(cell_line=cell_line, obs=md.obs.copy())
    for mod in md.mod:
        a = md.mod[mod]
        X = a.X
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
        names = [str(v) for v in a.var_names]

        # ECFP is a binary fingerprint; the outlier cutoff is meaningless there.
        if mod == "ecfp":
            blocks.X[mod], blocks.var_names[mod] = X, names
            blocks.reports[mod] = contamination_report(X, cutoff, names)
            LOG.info("hygiene    : %-12s binary fingerprint, cutoff not applied", mod)
            continue

        Xc, keep, rep = assert_clean(X, f"{cell_line}/{mod}", cutoff, policy, names)
        blocks.X[mod] = Xc
        blocks.var_names[mod] = [n for n, k in zip(names, keep) if k]
        blocks.reports[mod] = rep

    LOG.info("loaded     : %s  n=%d  %s", cell_line, blocks.n, blocks.summary())
    return blocks


def load_openscreen(cfg: Config, level: str = "consensus") -> dict[str, "Any"]:
    """OpenScreen HepG2, per site. `level` is 'consensus' or 'well'.

    This is the per-site view of the same morphology the HepG2 MuData collapses —
    identical 636 feature names, 118/119 shared compounds — which is what makes
    leave-one-site-out validation possible.
    """
    import anndata as ad

    key = "openscreen_consensus" if level == "consensus" else "openscreen_sites"
    d = cfg.get_path("openscreen_dir")
    sites = cfg.section("paths").get(key, {})
    out = {}
    for site, fname in sites.items():
        p = d / fname
        if not p.exists():
            LOG.warning("openscreen : %s missing (%s)", site, p.name)
            continue
        a = ad.read_h5ad(p)
        out[site] = a
        LOG.info("openscreen : %-7s %s  drugs=%d", site, a.shape,
                 a.obs["Metadata_Drug"].nunique() if "Metadata_Drug" in a.obs else -1)
    if not out:
        raise FileNotFoundError(f"no OpenScreen {level} files found under {d}")
    return out


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def prepare_target(obs: pd.DataFrame, spec: dict, name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build (y, mask, info) for one target from the config's target spec.

    `mask` marks the usable rows. Classification targets are collapsed: rare classes
    are dropped, because moa-fine has 23 classes of which 42/94 are "unclear" and the
    largest real class has 10 members.
    """
    kind = spec.get("kind", "classification")
    col = spec.get("column", name)
    if col not in obs.columns:
        raise KeyError(f"target column {col!r} not in obs ({list(obs.columns)[:8]}...)")

    s = obs[col]
    if kind == "regression":
        y = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        info = {"kind": kind, "column": col, "n_usable": int(mask.sum()),
                "n_total": len(y)}
        LOG.info("target     : %-12s regression   n=%d/%d", name, mask.sum(), len(y))
        return y, mask, info

    y_raw = s.astype(str).str.strip()
    drop = {str(v).strip().lower() for v in spec.get("drop_values", [])}
    mask = ~y_raw.str.lower().isin(drop) & y_raw.notna() & (y_raw != "")
    counts = y_raw[mask].value_counts()

    min_size = int(spec.get("min_class_size", 5))
    mode = str(spec.get("mode", "multiclass"))

    if mode == "one_vs_rest" and len(counts) >= 2:
        # Multiclass MoA is not viable here: after dropping "unclear" the largest
        # class has 14 members and the rest have <=5. Ask the one question the data
        # can answer -- largest annotated class vs the other ANNOTATED compounds.
        top = str(counts.index[0])
        y = np.where(y_raw.to_numpy() == top, top, f"not_{top}")
        keep_classes = pd.Series({top: int(counts.iloc[0]),
                                  f"not_{top}": int(counts.iloc[1:].sum())})
        LOG.info("target     : %-12s one-vs-rest on %r", name, top)
    else:
        keep_classes = counts[counts >= min_size]
        max_classes = spec.get("max_classes")
        if max_classes:
            keep_classes = keep_classes.head(int(max_classes))
        mask &= y_raw.isin(keep_classes.index)
        y = y_raw.to_numpy()

    info = {
        "kind": kind, "column": col, "mode": mode,
        "n_usable": int(mask.sum()), "n_total": len(y),
        "n_classes": int(len(keep_classes)),
        "class_counts": {str(k): int(v) for k, v in keep_classes.items()},
        "dropped_values": sorted(drop),
        "min_class_size": min_size,
    }
    LOG.info("target     : %-12s %d classes, n=%d/%d  %s", name, len(keep_classes),
             mask.sum(), len(y), dict(list(info["class_counts"].items())[:5]))
    return y, mask.to_numpy(), info


def target_is_usable(y: np.ndarray, mask: np.ndarray, kind: str, guards: dict,
                     name: str) -> tuple[bool, str]:
    """Refuse degenerate targets rather than reporting a meaningless score.

    tox_dermatological_toxicity is 68/2 on HepG2. A classifier predicting the majority
    class scores 0.97 accuracy on it, and even balanced accuracy is unstable when the
    minority class has 2 members -- one misclassification moves it by 0.25. Such a
    target cannot support a conclusion, so it is skipped with a stated reason.
    """
    n = int(mask.sum())
    if n < int(guards.get("min_usable_n", 30)):
        return False, f"only {n} labelled rows (need {guards.get('min_usable_n', 30)})"
    if kind != "classification":
        return True, ""
    vals, counts = np.unique(np.asarray(y)[mask], return_counts=True)
    if len(vals) < 2:
        return False, "fewer than 2 classes present"
    minority = int(counts.min())
    frac = minority / counts.sum()
    if minority < int(guards.get("min_minority_count", 10)):
        return False, (f"smallest class has {minority} members "
                       f"(need {guards.get('min_minority_count', 10)}) -- "
                       f"balance {dict(zip(map(str, vals), map(int, counts)))}")
    if frac < float(guards.get("min_minority_fraction", 0.15)):
        return False, (f"smallest class is {frac:.1%} of labelled rows "
                       f"(need {guards.get('min_minority_fraction', 0.15):.0%})")
    return True, ""


# --------------------------------------------------------------------------- #
# feature grammar (for the xAI layer)
# --------------------------------------------------------------------------- #
def parse_feature(name: str, compartments: Iterable[str], channels: Iterable[str]) -> dict:
    """Split a CellProfiler feature name into compartment / family / channel.

    Two naming conventions are in play and both must work:
      LINCS      Cells_/Cytoplasm_/Nuclei_ , 8 channel tokens, token counts 3/4/5/7
      OpenScreen Cells_/Cyto_/Nuc_        , 4 channel tokens, token counts 3..7

    Deliberately tolerant: an unknown family or channel yields None rather than an
    exception, because feature sets differ across sources and CellProfiler versions.
    """
    raw = name.split(":", 1)[-1]          # strip a "modality:" prefix if present
    toks = raw.split("_")
    comp = toks[0] if toks and toks[0] in set(compartments) else None
    family = toks[1] if len(toks) > 1 else None
    chans = [t for t in toks if t in set(channels)]
    return {
        "feature": raw,
        "compartment": comp,
        "family": family,
        "channel": chans[0] if chans else None,
        "channel_2": chans[1] if len(chans) > 1 else None,
        "n_tokens": len(toks),
    }


def feature_table(names: Sequence[str], cfg: Config) -> pd.DataFrame:
    g = cfg.section("xai").get("feature_grammar", {})
    comps = g.get("compartments", [])
    chans = g.get("channels", [])
    return pd.DataFrame([parse_feature(n, comps, chans) for n in names])


# --------------------------------------------------------------------------- #
# parallelism
# --------------------------------------------------------------------------- #
def resolve_jobs(cfg, override: int | None = None) -> tuple[int, int]:
    """Return (n_workers, blas_threads_per_worker).

    Nesting matters. joblib workers that each spawn a full BLAS thread pool
    oversubscribe catastrophically -- 384 workers x 384 threads on a zen5_dense node
    is how you get a load average in the thousands and a SLOWER run than serial. The
    outer parallelism owns the cores; each worker is pinned to `inner_threads`.

    Honours SLURM_CPUS_PER_TASK so a job never grabs more than it was allocated.
    """
    comp = cfg.section("compute") if hasattr(cfg, "section") else dict(cfg or {})

    # os.cpu_count() reports the MACHINE's cores (384 on a zen5_dense node), not what
    # this job was allocated, and inside `apptainer --cleanenv` the Slurm variables
    # are stripped -- so a 128-core job silently spawned 384 workers. Prefer, in
    # order: the Slurm allocation, the CPU affinity mask (which cgroups do restrict),
    # then the machine.
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
    source = "SLURM_CPUS_PER_TASK"
    if not allocated:
        try:
            allocated = len(os.sched_getaffinity(0))
            source = "cpu affinity mask"
        except (AttributeError, OSError):
            allocated = 0
    if not allocated:
        allocated = os.cpu_count() or 4
        source = "os.cpu_count (machine total -- may exceed your allocation)"

    n_jobs = override if override is not None else int(comp.get("n_jobs", -1))
    if n_jobs is None or n_jobs < 0:
        n_jobs = allocated
    n_jobs = max(1, min(n_jobs, allocated))
    inner = int(comp.get("inner_threads", 1))
    inner = max(1, inner)
    LOG.info("parallel   : %d worker(s) x %d BLAS thread(s)  (%d cpu from %s)",
             n_jobs, inner, allocated, source)
    return n_jobs, inner


def parallel_map(fn, items, n_jobs: int, inner_threads: int = 1, desc: str = ""):
    """joblib map with BLAS pinned inside the workers.

    Falls back to a plain serial map when n_jobs == 1, which keeps the serial path
    available for the equivalence check in ws4a_selftest.py.
    """
    items = list(items)
    if not items:
        return []
    if n_jobs == 1 or len(items) == 1:
        return [fn(x) for x in items]
    from joblib import Parallel, delayed, parallel_config

    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=inner_threads):
        out = Parallel(n_jobs=min(n_jobs, len(items)))(delayed(fn)(x) for x in items)
    if desc:
        LOG.info("parallel   : %-28s %d task(s) in %.1fs on %d worker(s)",
                 desc, len(items), time.time() - t0, min(n_jobs, len(items)))
    return out


def child_seeds(seed: int, n: int) -> list[int]:
    """n independent, reproducible seeds.

    A serial loop drawing from one Generator is order-dependent, so it cannot be
    parallelised while giving identical numbers. Spawning independent streams makes
    each task self-contained AND reproducible regardless of completion order.
    """
    ss = np.random.SeedSequence(seed)
    return [int(c.generate_state(1)[0]) for c in ss.spawn(n)]


# --------------------------------------------------------------------------- #
# device
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> str:
    """'cuda' only if XGBoost can actually train on it.

    Deliberately probes XGBoost rather than torch: torch is a 3 GB dependency this
    toolchain does not otherwise need, and it is XGBoost's own CUDA build that has
    to work. The probe trains a real (tiny) model, because the failure mode here is
    a wheel built against a newer CUDA than the driver supports -- availability
    checks pass and the first real call raises.
    """
    if requested == "cpu":
        return "cpu"

    # XGBoost does NOT raise when asked for device="cuda" on a machine with no GPU --
    # it warns and falls back to CPU. So the training probe alone reported "cuda" on
    # a GPU-less zen5_dense node. Establish that a device is actually visible first.
    import glob
    import shutil
    import subprocess
    visible = bool(glob.glob("/dev/nvidia[0-9]*"))
    if not visible and shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                 text=True, timeout=20)
            visible = out.returncode == 0 and "GPU 0" in out.stdout
        except Exception:                                        # noqa: BLE001
            visible = False
    if not visible:
        if requested == "cuda":
            raise SystemExit("--device cuda requested but no NVIDIA device is visible "
                             "(no /dev/nvidia* and nvidia-smi lists none). Inside a "
                             "container this usually means --nv was not passed.")
        LOG.info("device     : cpu (no NVIDIA device visible)")
        return "cpu"

    try:
        import numpy as _np
        import xgboost as xgb
        rng = _np.random.default_rng(0)
        Xp, yp = rng.standard_normal((32, 4)), rng.integers(0, 2, 32)
        xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist",
                          device="cuda", verbosity=0).fit(Xp, yp)
        info = xgb.build_info()
        LOG.info("device     : cuda (xgboost %s, built against CUDA %s)",
                 xgb.__version__, info.get("CUDA_VERSION"))
        return "cuda"
    except Exception as exc:                                     # noqa: BLE001
        msg = str(exc).split("\n")[0][:140]
        if requested == "cuda":
            raise SystemExit(f"--device cuda requested but XGBoost cannot use it: {msg}")
        LOG.info("device     : cpu (cuda unusable: %s)", msg)
        return "cpu"


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    LOG.info("wrote      : %s  (%d rows)", path, len(df))
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
    LOG.info("wrote      : %s", path)
    return path


def add_common_args(ap) -> None:
    """CLI flags every WS4A entry point shares."""
    here = Path(__file__).resolve().parent.parent.parent
    ap.add_argument("--config", default=str(here / "configs" / "ws4a.yaml"),
                    help="path to ws4a.yaml (default: configs/ws4a.yaml in the repo)")
    ap.add_argument("--root", default=None,
                    help="override the data root from the config")
    ap.add_argument("--out", default=None, help="override paths.outputs")
    ap.add_argument("--cell-line", default="hepg2", choices=["a549", "hepg2"])
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--seed", type=int, default=None)
