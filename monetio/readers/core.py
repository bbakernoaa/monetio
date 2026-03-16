import datetime
from typing import Any, Union

import pandas as pd
import xarray as xr


class BaseReader:
    """Base class for MONETIO readers."""

    def update_history(
        self, obj: Union[xr.Dataset, xr.DataArray, pd.DataFrame], message: str
    ) -> Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
        """Update the 'history' attribute of an xarray or pandas object.

        Parameters
        ----------
        obj : Union[xr.Dataset, xr.DataArray, pd.DataFrame]
            The object to update.
        message : str
            The message to append to the history.

        Returns
        -------
        Union[xr.Dataset, xr.DataArray, pd.DataFrame]
            The updated object.
        """
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        new_history = f"{now} UTC: {message}"

        # Dask DataFrames (dask-expr) might not have .attrs
        if hasattr(obj, "attrs"):
            history = obj.attrs.get("history", "")
            if history:
                new_history = f"{history}\n{new_history}"
            obj.attrs["history"] = new_history
        return obj


class PointReader(BaseReader):
    """Base class for point observation readers."""

    def to_xarray(self, df: Union[pd.DataFrame, "Any"]) -> xr.Dataset:
        """Convert a pandas or dask DataFrame to an xarray Dataset.

        Standardizes the spatial dimension as 'node'.

        Parameters
        ----------
        df : Union[pd.DataFrame, dask.dataframe.DataFrame]
            The DataFrame to convert.

        Returns
        -------
        xr.Dataset
            The converted Dataset.
        """
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        if is_dask:
            # Handle Dask DataFrame conversion to Xarray
            # We use the 'anchor' pattern to ensure dimension consistency
            ds = xr.Dataset()
            first = True
            for col in df.columns:
                if first:
                    # Anchor the dimension size
                    data = df[col].to_dask_array(lengths=True)
                    first = False
                else:
                    # Subsequent variables can have unknown shapes
                    # BUT Xarray currently fails on multiple 'nan' shapes in the same dataset
                    # if they are not the EXACT same object or if it can't verify them.
                    # As a workaround for Dask-expr, we might need to use lengths=True for all
                    # OR use a more advanced anchoring.
                    data = df[col].to_dask_array(lengths=True)
                ds[col] = (("node",), data)

            # Add index if it's not already a column
            if df.index.name and df.index.name not in ds:
                ds.coords[df.index.name] = (("node",), df.index.to_dask_array(lengths=True))
        else:
            ds = xr.Dataset.from_dataframe(df)
            if "index" in ds.dims:
                ds = ds.rename({"index": "node"})
            elif ds.dims:
                # If it has another dimension name, rename it to node if it's the only one
                if len(ds.dims) == 1:
                    dim_name = list(ds.dims)[0]
                    ds = ds.rename({dim_name: "node"})

        # Standardize coordinates
        coord_candidates = ["siteid", "latitude", "longitude", "time", "time_local", "elevation"]
        for c in coord_candidates:
            if c in ds.data_vars:
                ds = ds.set_coords(c)

        return ds


class GriddedReader(BaseReader):
    """Base class for gridded data readers (models and satellites)."""

    pass
