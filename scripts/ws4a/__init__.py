"""WS4A — cross-modal integration toolchain.

Vendored, tested implementations (no fragile third-party dependency):
  ajive            AJIVE, Feng et al. 2018            -- verified identical to mvlearn
  matrix_agreement Mantel / RV / adjusted RV / PROTEST -- adjusted RV is the only
                                                          unbiased effect size at p >> n
  stabsel          stability selection, MB2010 + CPSS  -- nothing installable works
  splsda           sparse PLS-DA, mixOmics keepX       -- no Python port exists
  permcca          permutation CCA, Winkler et al. 2020
"""
