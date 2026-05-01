# Requirements Document

## Introduction

This feature extends VirtualiZarr support across all 69 readers in monetio/readers/, converts the remaining legacy modules (monetio/obs/, monetio/models/, monetio/profile/, monetio/sat/) into thin wrappers that delegate to the unified reader system, and standardizes the reader structure for consistency and maintainability. The existing partial VirtualiZarr support in XarrayDriver will be expanded to cover all GriddedReader subclasses uniformly, and an Icechunk backend will be added as an alternative to kerchunk JSON references. Legacy modules that still contain original implementation code (e.g., monetio/profile/tolnet.py, monetio/sat/goes.py) will be refactored so their logic lives in monetio/readers/ and the legacy module becomes a deprecation wrapper.

## Glossary

- **XarrayDriver**: The unified driver class in monetio/readers/drivers.py that opens gridded data (NetCDF, GRIB, HDF) via xarray, with support for local, S3, and HTTP file access.
- **PandasDriver**: The unified driver class in monetio/readers/drivers.py that opens tabular/point data via pandas or dask.
- **GriddedReader**: The base class for gridded data readers (models, satellites) that uses XarrayDriver.
- **PointReader**: The base class for point/tabular data readers (observations, profiles) that uses PandasDriver.
- **BaseReader**: The abstract base class defining the open_dataset() and harmonize() interface for all readers.
- **READER_REGISTRY**: The global dictionary mapping reader names to reader classes, populated by the @register_reader decorator.
- **VirtualiZarr**: A library that creates virtual Zarr datasets from existing file collections without copying data, enabling efficient lazy access to multi-file datasets.
- **Kerchunk**: A reference file format (JSON) that stores byte-range metadata for virtual Zarr access to existing files.
- **Icechunk**: A transactional storage engine for Zarr that provides versioned, ACID-compliant access to array data, usable as an alternative backend to kerchunk JSON references.
- **Legacy_Module**: Any module in monetio/obs/, monetio/models/, monetio/profile/, or monetio/sat/ that contains original implementation code rather than delegating to a reader in monetio/readers/.
- **Thin_Wrapper**: A legacy module that imports a reader class from monetio/readers/ and exposes backward-compatible functions (open_dataset, open_mfdataset, add_data) that delegate to the reader.
- **Deprecation_Warning**: A Python warnings.warn() call with DeprecationWarning category, informing users to migrate to monetio.load().
- **Load_Function**: The monetio.load(source, files, **kwargs) entry point that instantiates a registered reader and calls open_dataset().
- **Harmonize**: The standardization step that normalizes coordinate names, variable names, and metadata to MONETIO conventions.

## Requirements

### Requirement 1: VirtualiZarr Support for All GriddedReaders

**User Story:** As a researcher working with large multi-file gridded datasets, I want all GriddedReader subclasses to support VirtualiZarr so that I can efficiently access data without the overhead of xarray.open_mfdataset.

#### Acceptance Criteria

1. WHEN use_virtualizarr=True is passed to any GriddedReader subclass open_dataset() call, THE XarrayDriver SHALL construct a virtual Zarr dataset from the provided files using the VirtualiZarr library.
2. WHEN virtualizarr_file is provided and the file exists on disk, THE XarrayDriver SHALL load cached kerchunk references from that file instead of recomputing them.
3. WHEN virtualizarr_file is provided and the file does not exist on disk, THE XarrayDriver SHALL compute kerchunk references and save them to the specified path.
4. WHEN use_virtualizarr=True is passed and the required dependencies (virtualizarr, obstore, obspec_utils, ujson, zarr) are not installed, THE XarrayDriver SHALL raise an ImportError with a message listing the missing packages and installation instructions.
5. WHEN use_virtualizarr=True is passed with files on S3, THE XarrayDriver SHALL configure the S3Store with the appropriate credentials and region from storage_options.
6. WHEN use_virtualizarr=True is passed with local files, THE XarrayDriver SHALL configure a LocalStore and prefix file paths with file:// as required by VirtualiZarr.
7. WHEN use_virtualizarr=True is passed with HTTP/HTTPS URLs, THE XarrayDriver SHALL configure an HTTPStore for remote access.

### Requirement 2: Icechunk Backend Support

**User Story:** As a data engineer, I want to store virtual Zarr references in Icechunk instead of kerchunk JSON so that I get versioned, transactional access to my reference data.

#### Acceptance Criteria

1. WHEN use_virtualizarr=True and virtualizarr_backend="icechunk" are passed, THE XarrayDriver SHALL store virtual dataset references in an Icechunk repository instead of a kerchunk JSON file.
2. WHEN virtualizarr_backend="icechunk" is passed and the icechunk package is not installed, THE XarrayDriver SHALL raise an ImportError with installation instructions.
3. WHEN virtualizarr_backend="kerchunk" is passed or no backend is specified, THE XarrayDriver SHALL use the existing kerchunk JSON reference format as the default.
4. WHEN an Icechunk repository path is provided via icechunk_repo parameter, THE XarrayDriver SHALL read from or write to that repository location.

### Requirement 3: VirtualiZarr Passthrough via monetio.load()

**User Story:** As a user of the monetio.load() API, I want to pass use_virtualizarr and related options through the universal load function so that I do not need to instantiate reader classes directly.

#### Acceptance Criteria

1. WHEN use_virtualizarr=True is passed to monetio.load() for a gridded data source, THE Load_Function SHALL forward the parameter to the reader's open_dataset() method.
2. WHEN use_virtualizarr=True is passed to monetio.load() for a point data source (PointReader), THE Load_Function SHALL ignore the parameter and proceed with normal loading.
3. WHEN virtualizarr_file, virtualizarr_backend, or icechunk_repo are passed to monetio.load(), THE Load_Function SHALL forward these parameters to the reader's open_dataset() method.

### Requirement 4: Legacy Model Module Conversion to Thin Wrappers

**User Story:** As a maintainer, I want all modules in monetio/models/ to be thin wrappers that delegate to monetio/readers/ so that implementation logic is centralized.

#### Acceptance Criteria

1. THE monetio/models/cmaq.py Thin_Wrapper SHALL import CMAQReader from monetio.readers.cmaq and delegate open_dataset() and open_mfdataset() calls to CMAQReader.open_dataset().
2. THE monetio/models/camx.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.camx and delegate all calls.
3. THE monetio/models/chimere.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.chimere and delegate all calls.
4. THE monetio/models/hysplit.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.hysplit and delegate all calls.
5. THE monetio/models/hytraj.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.hytraj and delegate all calls.
6. THE monetio/models/ncep_grib.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.ncep_grib and delegate all calls.
7. THE monetio/models/pardump.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.pardump and delegate all calls.
8. THE monetio/models/raqms.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.raqms and delegate all calls.
9. THE monetio/models/ufs.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.ufs and delegate all calls.
10. THE monetio/models/icap_mme.py Thin_Wrapper SHALL import the corresponding reader from monetio.readers.icap_mme and delegate all calls.
11. WHEN any function in a monetio/models/ Thin_Wrapper is called, THE Thin_Wrapper SHALL emit a Deprecation_Warning advising the user to migrate to monetio.load().
12. THE monetio/models/cdump2netcdf.py module SHALL remain unchanged as a standalone conversion utility that does not follow the reader pattern.

### Requirement 5: Legacy Observation Module Conversion to Thin Wrappers

**User Story:** As a maintainer, I want all modules in monetio/obs/ to be thin wrappers that delegate to monetio/readers/ so that implementation logic is centralized.

#### Acceptance Criteria

1. WHEN any function in a monetio/obs/ Thin_Wrapper is called, THE Thin_Wrapper SHALL emit a Deprecation_Warning advising the user to migrate to monetio.load().
2. THE monetio/obs/airnow.py Thin_Wrapper SHALL delegate add_data() and aggregate_files() to AirNowReader.open_dataset().
3. THE monetio/obs/aeronet.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/aeronet.py.
4. THE monetio/obs/aqs.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/aqs.py.
5. THE monetio/obs/cems.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/cems.py.
6. THE monetio/obs/crn.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/crn.py.
7. THE monetio/obs/improve.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/improve.py.
8. THE monetio/obs/ish.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/ish.py.
9. THE monetio/obs/ish_lite.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/ish_lite.py.
10. THE monetio/obs/nadp.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/nadp.py.
11. THE monetio/obs/openaq.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/openaq.py.
12. THE monetio/obs/openaq_v2.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/openaq_v2.py.
13. THE monetio/obs/pams.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/pams.py.
14. THE monetio/obs/ndacc.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/ndacc.py.
15. THE monetio/obs/pandora.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/pandora.py.
16. THE monetio/obs/skynet.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/skynet.py.
17. THE monetio/obs/eprofile.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/eprofile.py.
18. THE monetio/obs/actris.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/actris.py.
19. THE monetio/obs/iagos.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/iagos.py.
20. THE monetio/obs/epa_util.py utility module SHALL remain unchanged as a shared utility that does not follow the reader pattern.

### Requirement 6: Legacy Profile Module Conversion to Thin Wrappers

**User Story:** As a maintainer, I want all modules in monetio/profile/ to be thin wrappers that delegate to monetio/readers/ so that implementation logic is centralized.

#### Acceptance Criteria

1. WHEN any function in a monetio/profile/ Thin_Wrapper is called, THE Thin_Wrapper SHALL emit a Deprecation_Warning advising the user to migrate to monetio.load().
2. THE monetio/profile/tolnet.py Thin_Wrapper SHALL delegate open_dataset() and open_mfdataset() to the TOLNetReader in monetio/readers/tolnet.py, while retaining tolnet_colormap() and tolnet_plot() as non-deprecated utility functions.
3. THE monetio/profile/geoms.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/geoms.py.
4. THE monetio/profile/gml_ozonesonde.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/gml_ozonesonde.py.
5. THE monetio/profile/icartt.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/icartt.py.
6. THE monetio/profile/umbc_aerosol.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/umbc_aerosol.py.

### Requirement 7: Legacy Satellite Module Conversion to Thin Wrappers

**User Story:** As a maintainer, I want all modules in monetio/sat/ to be thin wrappers that delegate to monetio/readers/ so that implementation logic is centralized.

#### Acceptance Criteria

1. WHEN any function in a monetio/sat/ Thin_Wrapper is called, THE Thin_Wrapper SHALL emit a Deprecation_Warning advising the user to migrate to monetio.load().
2. THE monetio/sat/goes.py Thin_Wrapper SHALL delegate open_dataset() to the GOESReader in monetio/readers/goes.py, while retaining add_goes_bands() as a non-deprecated utility function.
3. THE monetio/sat/modis_l2.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/modis_l2.py.
4. THE monetio/sat/modis_ornl.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/modis_ornl.py.
5. THE monetio/sat/nesdis_edr_viirs.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/nesdis_edr_viirs.py.
6. THE monetio/sat/nesdis_eps_viirs.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/nesdis_eps_viirs.py.
7. THE monetio/sat/nesdis_frp.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/nesdis_frp.py.
8. THE monetio/sat/omps.py (omps_l3) Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/omps.py.
9. THE monetio/sat/omps_nadir.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/omps_nadir.py.
10. THE monetio/sat/tempo_l2.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/tempo.py.
11. THE monetio/sat/tropomi_l2_no2.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/tropomi.py.
12. THE monetio/sat/mopitt_l3.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/mopitt.py.
13. THE monetio/sat/gridded_eos.py Thin_Wrapper SHALL delegate to the corresponding reader in monetio/readers/ or remain as a utility if no direct reader mapping exists.

### Requirement 8: Uniform Reader open_dataset() Signature

**User Story:** As a developer adding new readers, I want all readers to follow a consistent open_dataset() signature pattern so that the codebase is predictable and easy to maintain.

#### Acceptance Criteria

1. THE BaseReader.open_dataset() abstract method SHALL accept files as the first positional argument and **kwargs for reader-specific options.
2. THE GriddedReader.open_dataset() method SHALL accept use_virtualizarr, virtualizarr_file, virtualizarr_backend, and use_dask as standard keyword arguments forwarded to XarrayDriver.
3. THE PointReader.open_dataset() method SHALL accept as_xarray, lazy, and wide_fmt as standard keyword arguments.
4. WHEN a GriddedReader subclass overrides open_dataset(), THE subclass SHALL call super().open_dataset() or self.driver.open() to ensure VirtualiZarr and driver options are consistently available.
5. WHEN a PointReader subclass overrides open_dataset(), THE subclass SHALL call super().open_dataset() to ensure driver options and to_xarray conversion are consistently available.

### Requirement 9: Deprecation Warning Infrastructure

**User Story:** As a maintainer, I want a consistent deprecation warning mechanism so that users are informed about the migration path from legacy modules to monetio.load().

#### Acceptance Criteria

1. THE Deprecation_Warning message SHALL include the legacy function name, the recommended monetio.load() equivalent, and a target removal version.
2. WHEN a deprecated function is called, THE Thin_Wrapper SHALL emit the Deprecation_Warning exactly once per function per Python session using warnings.warn() with stacklevel=2.
3. THE Deprecation_Warning SHALL use the Python DeprecationWarning category so that standard warning filters apply.

### Requirement 10: Reader Structure Uniformity

**User Story:** As a maintainer, I want all readers to follow a uniform internal structure so that the codebase is consistent and easy to navigate.

#### Acceptance Criteria

1. THE reader module file SHALL contain exactly one reader class decorated with @register_reader.
2. THE reader class SHALL inherit from GriddedReader or PointReader.
3. THE reader class SHALL implement open_dataset() as the primary entry point.
4. THE reader class SHALL implement harmonize() to apply reader-specific naming conventions.
5. WHEN a reader requires per-file preprocessing, THE reader SHALL define a module-level preprocess function and pass it to the driver via the preprocess keyword argument.
6. WHEN a reader requires helper functions, THE reader module SHALL define them as module-level functions below the reader class.
7. THE reader module SHALL include a module-level docstring identifying the data source.

### Requirement 11: Backward Compatibility for Existing monetio.load() Calls

**User Story:** As an existing user, I want my current monetio.load() calls to continue working without changes after the refactor.

#### Acceptance Criteria

1. WHEN monetio.load(source, files, **kwargs) is called with any currently registered source name, THE Load_Function SHALL return the same data structure (xr.Dataset or pd.DataFrame) as before the refactor.
2. WHEN monetio.load() is called without VirtualiZarr options, THE Load_Function SHALL use the standard xarray/pandas loading path.
3. THE READER_REGISTRY SHALL contain all 69 currently registered reader names after the refactor.

### Requirement 12: Migrate Remaining Legacy Implementation Code to Readers

**User Story:** As a maintainer, I want all data-reading logic to live in monetio/readers/ so that there is a single source of truth for each data source.

#### Acceptance Criteria

1. WHEN monetio/profile/tolnet.py contains the TOLNet class with HDF5 reading logic, THE refactor SHALL move that logic into a TOLNetReader class in monetio/readers/tolnet.py that inherits from BaseReader or GriddedReader.
2. WHEN monetio/sat/goes.py contains the GOES class with S3 access and grid computation logic, THE refactor SHALL move that logic into a GOESReader class in monetio/readers/goes.py that inherits from GriddedReader.
3. WHEN any legacy module in monetio/obs/, monetio/models/, monetio/profile/, or monetio/sat/ contains implementation code beyond simple delegation, THE refactor SHALL move that code into the corresponding reader in monetio/readers/.
4. WHEN implementation code is moved from a legacy module to monetio/readers/, THE legacy module SHALL be replaced with a Thin_Wrapper that delegates to the new reader.
