import abc
from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import ds_to_2d, force_object_strings

if TYPE_CHECKING:
    import dask.dataframe as dd

from .drivers import PandasDriver, XarrayDriver

# 1. The Registry
READER_REGISTRY = {}


def register_reader(name):
    """Decorator to register a reader class."""

    def _register(cls):
        READER_REGISTRY[name] = cls
        return cls

    return _register


# 2. The Abstract Base Class
class BaseReader(abc.ABC):
    """
    The interface that ALL readers must implement.
    """

    @abc.abstractmethod
    def open_dataset(
        self, files: Union[str, List[str]], **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Main entry point to read data.

        Args:
            files: File path, list of paths, or glob pattern.
            **kwargs: Reader-specific arguments.

        Returns:
            xarray.Dataset (for models/sat) or pandas.DataFrame (for point obs).
        """
        pass

    def harmonize(self, ds):
        """
        Optional: Apply standard naming conventions (middleware).
        Can be overridden by specific readers.
        """
        return ds


class GriddedReader(BaseReader):
    """
    Base class for gridded data (Models, Satellites) that utilizes XarrayDriver.
    """

    def __init__(self):
        self.driver = XarrayDriver()

    def open_dataset(self, files: Union[str, List[str]], **kwargs) -> xr.Dataset:
        """
        Uses XarrayDriver to open files.
        Readers can override this to add pre/post processing.
        """
        ds = self.driver.open(files, **kwargs)
        return self.harmonize(ds)


class PointReader(BaseReader):
    """
    Base class for point/tabular data (Observations) that utilizes PandasDriver.
    """

    def __init__(self):
        self.driver = PandasDriver()

    def open_dataset(
        self,
        files: Union[str, List[str]],
        read_method: str = "read_csv",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load point data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        read_method : str, optional
            The pandas/dask reading method to use, by default "read_csv".
        as_xarray : bool, optional
            If True, return an xarray.Dataset, by default True.
        lazy : bool, optional
            If True, return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded dataset.
        """
        df = self.driver.open(files, read_method=read_method, lazy=lazy, **kwargs)

        df = self.harmonize(df)

        # Consistently force object strings to avoid nullable string issues in Pandas/Dask
        df = force_object_strings(df)

        if as_xarray:
            return self.to_xarray(df)

        return df

    def harmonize(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Harmonize the dataset (standard naming, dropping NaNs).

        Parameters
        ----------
        df : Union[pd.DataFrame, "dd.DataFrame"]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, "dd.DataFrame"]
            Harmonized dataframe.
        """
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])
        return super().harmonize(df)

    def to_xarray(self, df: Union[pd.DataFrame, "dd.DataFrame"]) -> xr.Dataset:
        """
        Convert the DataFrame to an xarray Dataset in UGRID convention.
        Returns a 1D dataset by default to ensure backend consistency and laziness.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.

        Returns
        -------
        xr.Dataset
            The dataset in 1D UGRID convention.
        """
        # 1. Identify backend
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        # 2. Prepare DataFrame (ensure time and siteid are columns)
        if is_dask:
            temp_df = df
        else:
            temp_df = df.copy()

        for name in ["time", "siteid"]:
            try:
                names = temp_df.index.names
            except AttributeError:
                names = [temp_df.index.name]

            if name in names:
                temp_df = temp_df.reset_index()

        # 3. Handle Backends
        # Consistently force object strings for both backends to avoid nullable string issues.
        temp_df = force_object_strings(temp_df)

        if is_dask:
            # 3a. Lazy Path
            ds = xr.Dataset()
            # Exception to "No Hidden Computes": lengths=True is required by Xarray
            # to determine dimension sizes for the Dataset structure.
            for col in temp_df.columns:
                ds[col] = (("node",), temp_df[col].to_dask_array(lengths=True))
        else:
            # 3b. Eager Path
            # Consistently use 1D for both Eager and Lazy by default.
            ds = temp_df.reset_index(drop=True).to_xarray()
            if "index" in ds.dims:
                ds = ds.rename({"index": "node"})

        # Set standard coordinates
        coords = [c for c in ["time", "siteid", "latitude", "longitude"] if c in ds.data_vars]
        ds = ds.set_coords(coords)

        # Ensure node coordinate is a simple integer range for both
        if "node" in ds.dims:
            ds.coords["node"] = (("node",), np.arange(ds.sizes["node"]))

        # 4. Standard Path (Consistently try 2D expansion by default)
        # The user requested 2D UGRID as default.
        ds = ds_to_2d(ds)

        # Add UGRID metadata
        if "node" in ds.dims:
            ds["mesh"] = xr.DataArray(
                data=np.int32(0),
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 0,
                    "node_coordinates": "longitude latitude",
                },
            )

            if "latitude" in ds.coords:
                ds.coords["latitude"].attrs.update(
                    {"units": "degrees_north", "standard_name": "latitude"}
                )
            if "longitude" in ds.coords:
                ds.coords["longitude"].attrs.update(
                    {"units": "degrees_east", "standard_name": "longitude"}
                )

            for var in ds.data_vars:
                if "node" in ds[var].dims:
                    ds[var].attrs.update({"mesh": "mesh", "location": "node"})

        # Update history
        history = (
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            "Converted to xarray Dataset with UGRID convention."
        )
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds
