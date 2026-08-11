# Documentation Architecture

MONETIO is transitioning to a more robust and maintainable documentation and code architecture.

## Unified Reader System

Previously, MONETIO was organized into submodules based on data types: `obs`, `models`, `sat`, and `profile`. These modules contained various readers with inconsistent interfaces.

We are moving to a unified reader system located in `monetio/readers/`.

### Key Benefits

- **Consistency**: All readers follow a common interface (`BaseReader`).
- **Discovery**: The `monetio.load()` function provides a single entry point for all data sources.
- **Maintainability**: Common logic (like S3 access, coordinate renaming, and lazy loading) is centralized in drivers and base classes.
- **Lazy Loading**: Readers are registered and loaded only when needed, reducing initial import time.
- **Virtualization**: MONETIO supports virtualization of large datasets via Kerchunk and Icechunk.
    - **API**: Use `monetio.virtualize(source, files, output, backend)` to pre-compute and store metadata references. This provides a Pythonic way to generate virtualization maps for any supported gridded source.
    - **Kerchunk**: For large gridded datasets (MERRA2, GFS, ICAP), readers leverage `use_virtualizarr=True` via the Xarray driver to bypass `open_mfdataset` overhead. By pre-computing references to `virtualizarr_file`, datasets map into memory virtually via the Zarr engine.
    - **Icechunk**: A transactional Zarr-like storage format is supported via `use_icechunk=True` and `icechunk_url`.
    - **CLI**: The `monetio virtualizarr` command provides a command-line interface for generating these references.
- **Parallelism (Cubed)**: Experimental support for [Cubed](https://github.com/cubed-dev/cubed) as a lazy-loading backend is available via `use_cubed=True` in `monetio.load()` for gridded data sources. This requires `cubed` and `cubed-xarray` to be installed.

## Interaction Modes

MONETIO is designed to support different user interaction patterns:

- **Static (Production Pipeline)**: Optimized for batch processing, utilizing virtualization and standardized diagnostic paths.
- **Interactive (Exploration)**: Flexible for Jupyter notebooks, providing easy-to-use plotting and interactive data discovery.

## Deprecation of Legacy Modules

The following legacy submodules are currently being deprecated in favor of the new `monetio/readers/` architecture:

- `monetio.obs`
- `monetio.models`
- `monetio.sat`
- `monetio.profile`

While these modules still exist as wrappers to maintain backward compatibility, new features and readers should be implemented directly in `monetio/readers/`. Users are encouraged to transition to the `monetio.load()` function.

### Example: Migration

**Legacy way:**

```python
from monetio.obs import airnow

df = airnow.add_data(dates)
```

**New way:**

```python
import monetio as mio

df = mio.load("airnow", files=dates)
```

## MONETIO standards

MONETIO documentation and code follow the **monetio standards** for scientific pipeline code:

1.  **Flexibility**: Lazy by default (using Dask/Xarray), Eager optional.
2.  **Maintainability**: Comprehensive Type hints and NumPy-style docstrings.
3.  **Provenance**: Automatically updating `ds.attrs['history']` to track data transformations.
4.  **Visualization**: Integrated support for both static (Matplotlib/Cartopy) and interactive (HvPlot) visualization.
