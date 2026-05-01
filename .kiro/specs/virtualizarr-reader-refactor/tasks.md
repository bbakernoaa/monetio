# Implementation Plan: VirtualiZarr Reader Refactor

## Overview

This plan implements the VirtualiZarr reader refactor in incremental steps: first the shared infrastructure (deprecation helper, driver enhancements, base class updates), then the legacy module conversions (models, obs, profile, sat), dependency configuration, reader uniformity audit, and finally property-based and integration tests. Each task builds on previous work so there is no orphaned code.

## Tasks

- [x] 1. Create deprecation helper and update pyproject.toml
  - [x] 1.1 Create `monetio/readers/_deprecation.py` with the `deprecated_wrapper` decorator
    - Implement the decorator that emits `DeprecationWarning` with legacy function name, `monetio.load()` equivalent, and target removal version
    - Use `warnings.warn()` with `stacklevel=2` and `DeprecationWarning` category
    - _Requirements: 9.1, 9.2, 9.3_
  - [x] 1.2 Add `[virtualizarr]` and `[icechunk]` optional dependency groups to `pyproject.toml`
    - Add `virtualizarr = ["virtualizarr>=1.0", "obstore", "obspec_utils", "ujson", "zarr>=2.18"]`
    - Add `icechunk = ["icechunk>=0.1", "monetio[virtualizarr]"]`
    - _Requirements: 1.4, 2.2_

- [x] 2. Enhance XarrayDriver with Icechunk backend support
  - [x] 2.1 Add `virtualizarr_backend` and `icechunk_repo` parameters to `XarrayDriver.open()`
    - Add `virtualizarr_backend: str = "kerchunk"` and `icechunk_repo: str | None = None` parameters
    - Add validation that `virtualizarr_backend` is one of `"kerchunk"` or `"icechunk"`
    - _Requirements: 2.1, 2.3, 2.4_
  - [x] 2.2 Implement `_select_store()` helper method in `XarrayDriver`
    - Extract the existing store selection logic (S3Store, HTTPStore, LocalStore) into a reusable `_select_store()` function
    - Ensure local files are prefixed with `file://` as required by VirtualiZarr
    - _Requirements: 1.5, 1.6, 1.7_
  - [x] 2.3 Implement `_open_via_icechunk()` helper method in `XarrayDriver`
    - Create the Icechunk code path: open or create repo, write virtual references, commit, re-open for reading
    - Raise `ImportError` with installation instructions when `icechunk` is not installed
    - _Requirements: 2.1, 2.2, 2.4_
  - [x] 2.4 Wire Icechunk backend into the existing VirtualiZarr code path in `XarrayDriver.open()`
    - After computing `vds`, branch on `virtualizarr_backend`: if `"icechunk"`, call `_open_via_icechunk()`; if `"kerchunk"`, use existing kerchunk JSON path
    - _Requirements: 2.1, 2.3_
  - [x] 2.5 Write property test for store selection by protocol (Property 1)
    - **Property 1: Store Selection by Protocol**
    - Generate random file paths with s3://, http://, https://, or local prefix; mock store constructors; verify correct store type is instantiated and local files are prefixed with `file://`
    - **Validates: Requirements 1.5, 1.6, 1.7**

- [x] 3. Enhance GriddedReader and PointReader base classes
  - [x] 3.1 Update `GriddedReader.open_dataset()` signature in `monetio/readers/base.py`
    - Add `use_virtualizarr`, `virtualizarr_file`, `virtualizarr_backend`, `icechunk_repo`, and `use_dask` as explicit keyword arguments
    - Forward all VirtualiZarr kwargs to `self.driver.open()`
    - _Requirements: 8.2, 8.4_
  - [x] 3.2 Update `PointReader.open_dataset()` signature in `monetio/readers/base.py`
    - Add `use_virtualizarr`, `virtualizarr_file`, `virtualizarr_backend`, `icechunk_repo` as accepted-but-ignored keyword arguments
    - Ensure these kwargs are silently discarded and not forwarded to `PandasDriver`
    - _Requirements: 3.2, 8.3_
  - [x] 3.3 Write property test for VirtualiZarr kwargs forwarding through `monetio.load()` (Property 2)
    - **Property 2: VirtualiZarr Kwargs Forwarding Through Load**
    - Generate random subsets of VZ kwargs; mock reader `open_dataset`; call `monetio.load()`; verify kwargs arrive for GriddedReaders and are ignored for PointReaders
    - **Validates: Requirements 3.1, 3.2, 3.3**

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Convert legacy model modules to thin deprecation wrappers
  - [x] 5.1 Convert `monetio/models/cmaq.py` — add deprecation decorator to existing wrapper functions
    - Import and apply `deprecated_wrapper` to `open_dataset()` and `open_mfdataset()`
    - _Requirements: 4.1, 4.11_
  - [x] 5.2 Convert `monetio/models/camx.py` to thin deprecation wrapper
    - Replace implementation with import from `monetio.readers.camx` + `deprecated_wrapper` decorated functions
    - _Requirements: 4.2, 4.11_
  - [x] 5.3 Convert `monetio/models/chimere.py` to thin deprecation wrapper
    - _Requirements: 4.3, 4.11_
  - [x] 5.4 Convert `monetio/models/hysplit.py` to thin deprecation wrapper
    - _Requirements: 4.4, 4.11_
  - [x] 5.5 Convert `monetio/models/hytraj.py` to thin deprecation wrapper
    - _Requirements: 4.5, 4.11_
  - [x] 5.6 Convert `monetio/models/ncep_grib.py` to thin deprecation wrapper
    - _Requirements: 4.6, 4.11_
  - [x] 5.7 Convert `monetio/models/pardump.py` to thin deprecation wrapper
    - _Requirements: 4.7, 4.11_
  - [x] 5.8 Convert `monetio/models/raqms.py` to thin deprecation wrapper
    - _Requirements: 4.8, 4.11_
  - [x] 5.9 Convert `monetio/models/ufs.py` to thin deprecation wrapper
    - _Requirements: 4.9, 4.11_
  - [x] 5.10 Convert `monetio/models/icap_mme.py` to thin deprecation wrapper
    - _Requirements: 4.10, 4.11_
  - [x] 5.11 Verify `monetio/models/cdump2netcdf.py` is unchanged (no deprecation infrastructure)
    - Confirm the module has no `deprecated_wrapper` import or deprecation warnings
    - _Requirements: 4.12_

- [x] 6. Convert legacy observation modules to thin deprecation wrappers
  - [x] 6.1 Convert `monetio/obs/airnow.py` — add deprecation decorator to existing wrapper functions
    - Apply `deprecated_wrapper` to `add_data()` and `aggregate_files()`
    - _Requirements: 5.1, 5.2_
  - [x] 6.2 Convert `monetio/obs/aeronet.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.3_
  - [x] 6.3 Convert `monetio/obs/aqs.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.4_
  - [x] 6.4 Convert `monetio/obs/cems.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.5_
  - [x] 6.5 Convert `monetio/obs/crn.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.6_
  - [x] 6.6 Convert `monetio/obs/improve.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.7_
  - [x] 6.7 Convert `monetio/obs/ish.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.8_
  - [x] 6.8 Convert `monetio/obs/ish_lite.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.9_
  - [x] 6.9 Convert `monetio/obs/nadp.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.10_
  - [x] 6.10 Convert `monetio/obs/openaq.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.11_
  - [x] 6.11 Convert `monetio/obs/openaq_v2.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.12_
  - [x] 6.12 Convert `monetio/obs/pams.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.13_
  - [x] 6.13 Convert `monetio/obs/ndacc.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.14_
  - [x] 6.14 Convert `monetio/obs/pandora.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.15_
  - [x] 6.15 Convert `monetio/obs/skynet.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.16_
  - [x] 6.16 Convert `monetio/obs/eprofile.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.17_
  - [x] 6.17 Convert `monetio/obs/actris.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.18_
  - [x] 6.18 Convert `monetio/obs/iagos.py` to thin deprecation wrapper
    - _Requirements: 5.1, 5.19_
  - [x] 6.19 Verify `monetio/obs/epa_util.py` is unchanged (no deprecation infrastructure)
    - Confirm the module has no `deprecated_wrapper` import or deprecation warnings
    - _Requirements: 5.20_

- [x] 7. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Migrate TOLNet and GOES implementation code to readers
  - [x] 8.1 Migrate TOLNet implementation from `monetio/profile/tolnet.py` to `monetio/readers/tolnet.py`
    - Move the `TOLNet` class HDF5 reading logic into the existing `TOLNetReader` in `monetio/readers/tolnet.py` (or create it if it only has a stub)
    - Ensure `TOLNetReader` inherits from `BaseReader` or `GriddedReader` and implements `open_dataset()` and `harmonize()`
    - _Requirements: 12.1, 12.3, 12.4_
  - [x] 8.2 Convert `monetio/profile/tolnet.py` to thin deprecation wrapper
    - Replace implementation with `deprecated_wrapper` decorated functions delegating to `TOLNetReader`
    - Retain `tolnet_colormap()` and `tolnet_plot()` as non-deprecated utility functions
    - _Requirements: 6.1, 6.2_
  - [x] 8.3 Migrate GOES implementation from `monetio/sat/goes.py` to `monetio/readers/goes.py`
    - Move the `GOES` class S3 access and grid computation logic into the existing `GOESReader` in `monetio/readers/goes.py` (or create it if it only has a stub)
    - Ensure `GOESReader` inherits from `GriddedReader` and implements `open_dataset()` and `harmonize()`
    - _Requirements: 12.2, 12.3, 12.4_
  - [x] 8.4 Convert `monetio/sat/goes.py` to thin deprecation wrapper
    - Replace implementation with `deprecated_wrapper` decorated functions delegating to `GOESReader`
    - Retain `add_goes_bands()` as a non-deprecated utility function if it exists
    - _Requirements: 7.1, 7.2_

- [x] 9. Convert remaining legacy satellite and profile modules to thin deprecation wrappers
  - [x] 9.1 Convert `monetio/profile/geoms.py` to thin deprecation wrapper
    - _Requirements: 6.1, 6.3_
  - [x] 9.2 Convert `monetio/profile/gml_ozonesonde.py` to thin deprecation wrapper
    - _Requirements: 6.1, 6.4_
  - [x] 9.3 Convert `monetio/profile/icartt.py` to thin deprecation wrapper
    - _Requirements: 6.1, 6.5_
  - [x] 9.4 Convert `monetio/profile/umbc_aerosol.py` to thin deprecation wrapper
    - _Requirements: 6.1, 6.6_
  - [x] 9.5 Convert `monetio/sat/modis_l2.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.3_
  - [x] 9.6 Convert `monetio/sat/modis_ornl.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.4_
  - [x] 9.7 Convert `monetio/sat/nesdis_edr_viirs.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.5_
  - [x] 9.8 Convert `monetio/sat/nesdis_eps_viirs.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.6_
  - [x] 9.9 Convert `monetio/sat/nesdis_frp.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.7_
  - [x] 9.10 Convert `monetio/sat/omps_l3.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.8_
  - [x] 9.11 Convert `monetio/sat/omps_nadir.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.9_
  - [x] 9.12 Convert `monetio/sat/tempo_l2.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.10_
  - [x] 9.13 Convert `monetio/sat/tropomi_l2_no2.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.11_
  - [x] 9.14 Convert `monetio/sat/mopitt_l3.py` to thin deprecation wrapper
    - _Requirements: 7.1, 7.12_
  - [x] 9.15 Convert `monetio/sat/gridded_eos.py` to thin deprecation wrapper or verify it remains as utility
    - _Requirements: 7.1, 7.13_

- [x] 10. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Reader structure uniformity audit
  - [x] 11.1 Audit all 69 reader modules in `monetio/readers/` for uniform structure
    - Verify each reader module (excluding `__init__.py`, `base.py`, `drivers.py`, `*_specs.py`, `*_utils.py`, `time_utils.py`) contains exactly one `@register_reader` decorated class
    - Verify each reader class inherits from `GriddedReader` or `PointReader`
    - Verify each reader class implements `open_dataset()` and `harmonize()`
    - Verify each module has a non-empty module-level docstring
    - Fix any non-conforming readers
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.7_
  - [x] 11.2 Verify `READER_REGISTRY` contains all 69 registered reader names
    - Import all reader modules and check `READER_REGISTRY` has entries matching `_READER_MODULES` in `monetio/__init__.py`
    - _Requirements: 11.3_
  - [x] 11.3 Write property test for reader structure uniformity (Property 5)
    - **Property 5: Reader Structure Uniformity**
    - For each reader module in `monetio/readers/`, inspect structure: verify single `@register_reader` class, correct inheritance, required methods, non-empty docstring
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.7**

- [x] 12. Property-based and integration tests
  - [x] 12.1 Write property test for legacy wrapper delegation (Property 3)
    - **Property 3: Legacy Wrapper Delegation**
    - For each legacy wrapper function, mock the underlying reader, call the wrapper with arguments, verify the reader's `open_dataset()` is called with equivalent arguments
    - **Validates: Requirements 4.1–4.10, 5.2–5.19, 6.2–6.5, 7.2–7.12**
  - [x] 12.2 Write property test for deprecation warning emission (Property 4)
    - **Property 4: Deprecation Warning Emission**
    - For each deprecated legacy wrapper function, call it, capture warnings, verify format contains function name, `monetio.load()` equivalent, and target removal version
    - **Validates: Requirements 4.11, 5.1, 6.1, 7.1, 9.1, 9.2, 9.3**
  - [x] 12.3 Write property test for VirtualiZarr activation producing valid dataset (Property 6)
    - **Property 6: VirtualiZarr Activation Produces Valid Dataset**
    - Generate random file lists (mocked NetCDF); compare VZ path output structure (variable names, dimensions) to standard `open_mfdataset` path output
    - **Validates: Requirements 1.1, 11.1, 11.2**
  - [x] 12.4 Write integration smoke tests
    - Verify all 69 reader modules import without error
    - Verify `monetio.load()` with each source name doesn't crash on import
    - Verify `cdump2netcdf.py` and `epa_util.py` have no deprecation infrastructure
    - _Requirements: 11.1, 11.2, 11.3_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using Hypothesis with `@settings(max_examples=100)`
- Unit tests validate specific examples and edge cases
- The `monetio/models/cdump2netcdf.py` and `monetio/obs/epa_util.py` modules are explicitly excluded from deprecation conversion as standalone utilities
- Legacy modules that already delegate to readers (e.g., `monetio/models/cmaq.py`, `monetio/obs/airnow.py`) only need the deprecation decorator added
- Legacy modules with implementation code (e.g., `monetio/profile/tolnet.py`, `monetio/sat/goes.py`) need their logic migrated to readers first
