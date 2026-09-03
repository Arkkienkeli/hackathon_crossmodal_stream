import anndata as ad

files = {
    "TAHOE HepG2": "OpenScreen/data/hepg2_cells.h5ad",
    "TAHOE A549": "LINCS/data/lincs_expression_a549.h5ad",
    "LINCS morphology": "LINCS/data/lincs_morphology_a549_batch1.h5ad",
    "LINCS morphology consensus": "LINCS/data/lincs_morphology_a549_batch1_consensus.h5ad",
}

for name, path in files.items():
    print("\n" + "=" * 80)
    print(name)
    print(path)

    a = ad.read_h5ad(path, backed="r")

    print("shape:", a.shape)
    print("X dtype:", a.X.dtype)
    print("obs columns:", list(a.obs.columns))
    print("var columns:", list(a.var.columns))
    print("layers:", list(a.layers.keys()))
    print("obsm:", list(a.obsm.keys()))

    a.file.close()
