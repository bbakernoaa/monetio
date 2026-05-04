# Adding a New Reader

MONETIO uses a unified reader architecture based on a common base class. This ensures consistency across different data sources and simplifies the addition of new models and observation networks.

## Core Concepts

All readers are located in the `monetio/readers/` directory and inherit from `BaseReader`.

- **`BaseReader`**: The abstract base class that defines the `open_dataset` interface.
- **`GriddedReader`**: A base class for gridded data (Models, Satellites) that utilizes the `XarrayDriver`.
- **`PointReader`**: A base class for point/tabular data (Observations) that utilizes the `PandasDriver`.
- **`READER_REGISTRY`**: A global dictionary where all readers must register themselves using the `@register_reader("name")` decorator.

## Design Principles

Architecting scientific pipelines in MONETIO requires balancing flexibility, maintainability, and provenance. All new readers should adhere to these principles:

### 1. Backend Agnostic (Optional Dask)
Code must run **Eagerly (NumPy) by default** and **Lazily (Dask) optionally**.
- **Generic Inputs**: Write functions that accept generic `xr.DataArray` or `xr.Dataset` inputs. Do not assume a specific backend.
- **No Hidden Computes**: Never call `.compute()`, `.load()`, or `.values` inside a processing function. This breaks laziness for Dask users.
- **No Forced Chunking**: Do not hardcode `.chunk()` inside functions. Chunking is the user's responsibility at the I/O stage.
- **Vectorization**: Use `xarray.apply_ufunc` with `dask='parallelized'` to support both backends simultaneously.

### 2. Maintainability and Documentation
- **NumPy Docstrings**: Every function must have a docstring following the [NumPy format](https://numpydoc.readthedocs.io/en/latest/format.html) (Parameters, Returns, Examples).
- **Type Hinting**: Use `xarray.DataArray` or `xarray.Dataset` types. Avoid backend-specific types like `dask.array.Array`.

### 3. Provenance (Scientific Hygiene)
- **History Tracking**: Automatically track data lineage by updating `ds.attrs['history']` whenever data is transformed.
- **Coordinate Integrity**: Never drop coordinates (like vertical levels) unless explicitly requested.

### 4. Quality and Validation
- **Dual-Backend Testing**: Every reader or processing logic should be verified with a unit test that runs twice: once with Eager (NumPy) data and once with Lazy (Dask) data.
- **Pre-Commit**: Use `pre-commit run --all-files` to ensure code style and linting standards are met.

## Steps to Add a New Reader

### 1. Create the Reader Module

Create a new Python file in `monetio/readers/`, for example `monetio/readers/mynewmodel.py`.

### 2. Implement the Reader Class

Inherit from `GriddedReader` (for gridded data) or `PointReader` (for point data).

```python
from typing import List, Union
import xarray as xr
from .base import GriddedReader, register_reader

@register_reader("mynewmodel")
class MyNewModelReader(GriddedReader):
    def open_dataset(
        self,
        files: Union[str, List[str]],
        **kwargs,
    ) -> xr.Dataset:
        # 1. Open the dataset using the driver
        # The driver handles local files, S3, and common Xarray arguments
        ds = self.driver.open(files, **kwargs)

        # 2. Perform reader-specific processing
        # e.g., Rename dimensions, calculate coordinates, handle units
        ds = ds.rename({"old_dim": "x"})

        # 3. Harmonize (Apply standard naming conventions)
        ds = self.harmonize(ds)

        return ds

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        # Optional: Standardize variable names to MONETIO conventions
        return ds
```

### 3. Register the Reader for Lazy Loading

Update `monetio/__init__.py` to add your reader to the `_READER_MODULES` map. This allows users to use `monetio.load("mynewmodel", ...)` without manually importing your module.

```python
_READER_MODULES = {
    # ...
    "mynewmodel": ".readers.mynewmodel",
}
```

### 4. Add to the Universal Load Function

Ensure your reader is listed in the docstring of the `load` function in `monetio/__init__.py`.

## Best Practices

### Lazy Loading

Readers should aim to be lazy by default. Use Xarray and Dask to avoid loading large datasets into memory until computation is requested.

- Avoid calling `.values`, `.compute()`, or `.load()` unless absolutely necessary (e.g., for coordinate construction).
- Use `preprocess` functions with `xarray.open_mfdataset` for per-file processing.

### Provenance and Metadata

Always update the `history` attribute of the returned dataset or dataframe to record the transformation.

```python
import datetime
history = f"{datetime.datetime.now()}: Read MyNewModel data."
ds.attrs["history"] = ds.attrs.get("history", "") + "\n" + history
```

### Harmonization

Use standard coordinate names:

- Gridded: `time`, `x`, `y`, `z`, `latitude`, `longitude`.
- Point: `time`, `latitude`, `longitude`, `siteid`.

### Error Handling

When `error_missing=False` is passed (supported by the drivers), your reader should gracefully handle cases where files are not found, typically by returning an empty Dataset or DataFrame.
