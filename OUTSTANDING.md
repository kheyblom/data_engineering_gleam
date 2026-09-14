1. ~~verify and clean temporal stores~~ **DONE 2026-09-13**
    > run same verification pipeline to spatial stores. you can update the pipeline as necessary make sure data is copied properly. make sure that metadata is sonsistnet with spatial store.

    152 checks, 0 failures against raw; metadata compared against the spatial
    store (identical coordinates, variable attrs and all shared global attrs);
    `verification` attribute written, gc run (collected nothing), no tag by
    choice since task 4 removes this store; bench_temporal.zarr deleted (123 GB).
    Verifier gained a `metadata` phase, a land-stratified draw and a
    cross-variable absent-tile trace. See TESTING.md, findings 14 and 15.
2. ~~split zarr stores into individual stores per variable~~ **DONE 2026-09-13** (28 stores, ~17 core-hours per layout, tmp_split_gleam_zarr.py)
    > I have updated the end requirement of the data stores and now need them to be divided by variable. I need you to split the the current full stores into one store per variable.
3. ~~verify and clean variable stores~~ **DONE 2026-09-14** (all 28: 0 failures, attrs + tag v4.3a-verified-20260914 + gc; verifier gained --variable and cross-store checks)
    > update verification pipeline to be run for each of the new variable stores and then run and clean as necessary
4. ~~remove original stores (stores containing all variables)~~ **DONE 2026-09-14** (~2.2 T freed; all 28 replacements read through their tags first)
    > remove all of the original all-variable stores after the individual-variable stores have been verified
5. ~~refactor codebase to instead create zarr stores for each variable~~ **DONE 2026-09-14** (gleam_zarr.py builds one variable per store, --variable or the whole family; merge_variables replaced by check_time_axis; title/summary derived; variable_attrs; job script fans out per variable)
    > I want future uses of this codebase to go straight to creating the indivdual-variable stores. refoactor the whole codebase so that this is handled.
6. test codebase refactor
    > test the above refactor to ensure that the codebase works as intended.