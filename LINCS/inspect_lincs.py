import anndata as ad
import numpy as np
from scipy import sparse

files = [
    "LINCS/data/lincs_expression_a549.h5ad",
    "LINCS/data/lincs_morphology_a549_batch1.h5ad",
    "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad",
]

for p in files:
    print("\n" + "=" * 100)
    print("FILE:", p)
    print("=" * 100)

    a = ad.read_h5ad(p, backed="r")

    print("SHAPE:", a.shape)

    print("\nOBS COLUMNS:")
    print(a.obs.columns.tolist())

    print("\nVAR COLUMNS:")
    print(a.var.columns.tolist())

    print("\nLAYERS:")
    print(list(a.layers.keys()))

    print("\nOBSM:")
    print(list(a.obsm.keys()))

    print("\nVARM:")
    print(list(a.varm.keys()))

    print("\nUNS:")
    print(list(a.uns.keys()))

    print("\nRAW PRESENT:", a.raw is not None)

    print("\nINDEX EXAMPLES:")
    print("obs:", list(a.obs_names[:5]))
    print("var:", list(a.var_names[:10]))

    # Important metadata
    candidates = [
        "drug",
        "compound",
        "pert_iname",
        "pert_id",
        "Metadata_pert_iname",
        "Metadata_pert_id",
        "cell_line",
        "cell_line_id",
        "cell_iname",
        "dose",
        "pert_dose",
        "time",
        "pert_time",
        "plate",
        "Metadata_Plate",
        "batch",
        "moa",
    ]

    print("\nIMPORTANT OBS METADATA:")
    for c in candidates:
        if c in a.obs.columns:
            print("\n---", c, "---")
            print("N unique:", a.obs[c].nunique(dropna=False))
            print(a.obs[c].value_counts(dropna=False).head(20))

    # Inspect a small X slice only — do not load whole matrix
    nr = min(200, a.n_obs)
    nc = min(500, a.n_vars)

    try:
        x = a[:nr, :nc].X
        if sparse.issparse(x):
            x = x.toarray()

        x = np.asarray(x, dtype=float)

        print("\nSMALL X SAMPLE:")
        print("shape:", x.shape)
        print("min:", np.nanmin(x))
        print("max:", np.nanmax(x))
        print("mean:", np.nanmean(x))
        print("std:", np.nanstd(x))
        print("fraction zero:", np.mean(x == 0))
        print("fraction negative:", np.mean(x < 0))

        qs = np.nanquantile(
            x,
            [0, .01, .25, .5, .75, .99, 1]
        )
        print("quantiles:", qs)

    except Exception as e:
        print("\nCould not inspect X sample:", repr(e))

    a.file.close()

print("\nDONE")
