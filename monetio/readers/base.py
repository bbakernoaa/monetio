import abc
from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import xarray as xr

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

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.

        Returns
        -------
        xr.Dataset
            The dataset in UGRID convention.
        """
        temp_df = df.copy()

        # Handle cases where 'time' or 'siteid' might be in the index already
        for name in ["time", "siteid"]:
            try:
                names = temp_df.index.names
            except AttributeError:
                names = [temp_df.index.name]

            if name in names:
                temp_df = temp_df.reset_index()

        index_cols = [c for c in ["time", "siteid"] if c in temp_df.columns]

        # Standard MONET site metadata columns
        site_meta_cols = [
            "latitude",
            "longitude",
            "site",
            "site_name",
            "state_name",
            "epa_region",
            "msa_name",
            "msa_code",
            "cmsa_name",
            "utcoffset",
        ]

        # Check for dask
        is_dask = False
        try:
            import dask.dataframe as dd

            if isinstance(df, dd.DataFrame):
                is_dask = True
        except ImportError:
            pass

        if is_dask:
            # For dask dataframes, we return a 1D dataset to keep it lazy and avoid shuffles.
            ds = xr.Dataset()
            for col in temp_df.columns:
                ds[col] = (("node",), temp_df[col].to_dask_array(lengths=True))

            # If we have time and siteid, we still name the dimension node for UGRID
            # but it represents individual observations.
        elif "time" in index_cols and "siteid" in index_cols:
            present_meta = [c for c in site_meta_cols if c in temp_df.columns]

            if present_meta:
                # Extract one record per siteid to create 1D coordinates
                meta_df = temp_df[["siteid"] + present_meta].drop_duplicates(subset=["siteid"])
                meta_df = meta_df.set_index("siteid")
                # Remove from main DF so they don't become 2D variables
                temp_df = temp_df.drop(columns=present_meta)
            else:
                meta_df = pd.DataFrame()

            # Create the dense 2D Dataset for observation data
            # Only if the index is unique, otherwise stay 1D
            idx_cols = ["time", "siteid"]
            if temp_df.set_index(idx_cols).index.is_unique:
                ds = temp_df.set_index(idx_cols).to_xarray()
            else:
                # Fallback to 1D
                ds = temp_df.to_xarray()

            # Rename siteid to node for UGRID compliance
            ds = ds.rename({"siteid": "node"})

            # Re-attach site metadata as 1D coords indexed by node
            for col in meta_df.columns:
                # Use .values safely if possible, but for 2D transformation we already computed
                ds.coords[col] = (("node",), meta_df.loc[ds.node.values, col].values)

        elif index_cols:
            ds = temp_df.set_index(index_cols).to_xarray()
            if "siteid" in ds.dims:
                ds = ds.rename({"siteid": "node"})
        else:
            ds = temp_df.to_xarray()

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
