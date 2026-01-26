# Adding a New Reader

MONETIO uses a unified reader architecture based on a common base class. This ensures consistency across different data sources and simplifies the addition of new models and observation networks.

## Core Concepts

All readers are located in the `monetio/readers/` directory and inherit from `BaseReader`.

- **`BaseReader`**: The abstract base class that defines the `open_dataset` interface.
- **`GriddedReader`**: A base class for gridded data (Models, Satellites) that utilizes the `XarrayDriver`.
- **`PointReader`**: A base class for point/tabular data (Observations) that utilizes the `PandasDriver`.
- **`READER_REGISTRY`**: A global dictionary where all readers must register themselves using the `@register_reader("name")` decorator.

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

### Lazy Loading (The Aero Protocol)

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
