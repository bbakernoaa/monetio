import abc
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

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
        read_method="read_csv",
        as_xarray=True,
        lazy=False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
        Uses PandasDriver to open files.
        Readers can override this to add pre/post processing.
        """
        df = self.driver.open(files, read_method=read_method, lazy=lazy, **kwargs)

        if not lazy:
            df = self.harmonize(df)
            if as_xarray:
                return self.to_xarray(df)
        return df

    def harmonize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize point data: drop NaNs in coordinates.
        """
        if "latitude" in df.columns and "longitude" in df.columns:
            df = df.dropna(subset=["latitude", "longitude"])
        return super().harmonize(df)

    def to_xarray(self, df: pd.DataFrame) -> xr.Dataset:
        """
        Convert the DataFrame to an xarray Dataset in UGRID convention.
        """
        temp_df = df.copy()

        # Handle cases where 'time' or 'siteid' might be in the index already
        for name in ["time", "siteid"]:
            if name in temp_df.index.names:
                temp_df = temp_df.reset_index(name)

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

        if "time" in index_cols and "siteid" in index_cols:
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
            ds = temp_df.set_index(["time", "siteid"]).to_xarray()

            # Rename siteid to node for UGRID compliance
            ds = ds.rename({"siteid": "node"})

            # Re-attach site metadata as 1D coords indexed by node
            for col in meta_df.columns:
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
