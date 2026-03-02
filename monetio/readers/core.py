import logging
from datetime import datetime

import xarray as xr

logger = logging.getLogger(__name__)


class GriddedReader:
    """Base class for gridded model/satellite readers."""

    def __init__(self):
        pass

    def update_history(self, ds, message):
        """Update the history attribute of the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to update.
        message : str
            The message to append to the history.

        Returns
        -------
        xarray.Dataset
            The updated dataset.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = ds.attrs.get("history", "")
        ds.attrs["history"] = f"{now}: {message}\n{history}"
        return ds


class PointReader:
    """Base class for point observation readers."""

    def __init__(self):
        pass

    def update_history(self, ds, message):
        """Update the history attribute of the dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            The dataset to update.
        message : str
            The message to append to the history.

        Returns
        -------
        xarray.Dataset
            The updated dataset.
        """
        from datetime import datetime

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = ds.attrs.get("history", "")
        ds.attrs["history"] = f"{now}: {message}\n{history}"
        return ds

    def to_xarray(self, df, expand2d=True):
        """Convert a pandas/dask DataFrame to a UGRID-compliant xarray Dataset.

        Parameters
        ----------
        df : pandas.DataFrame or dask.dataframe.DataFrame
            The input dataframe.
        expand2d : bool
            Whether to expand the dataset to (time, node) dimensions.

        Returns
        -------
        xarray.Dataset
        """
        from ..util import ds_to_2d

        # 1. Convert to xarray
        if hasattr(df, "to_xarray"):
            ds = df.to_xarray()
        else:
            # Handle Dask DataFrame which might not have to_xarray

            # Convert to Dataset lazily if it's Dask
            # Note: This is a simplified version, as dask.dataframe doesn't have a direct to_xarray.
            # We convert each column to a dask-backed DataArray.
            data_vars = {}
            for col in df.columns:
                data_vars[col] = (("node",), df[col].to_dask_array(lengths=True))

            ds = xr.Dataset(data_vars)
            ds = ds.assign_coords(node=range(len(ds.node)))

        # 2. Add UGRID mesh topology if missing
        if "mesh" not in ds:
            ds["mesh"] = xr.DataArray(
                0,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 0,
                    "node_coordinates": "longitude latitude",
                },
            )

        # 3. Standardize dimensions to 'node'
        if "index" in ds.dims:
            ds = ds.rename({"index": "node"})
        elif "node" not in ds.dims:
            # If no node dimension, assume it is 'index' or similar
            ds = ds.rename_dims({ds.dims[0]: "node"})

        # 4. Standardize spatial coordinates
        if "latitude" in ds:
            ds.latitude.attrs.update(
                {"standard_name": "latitude", "units": "degrees_north"}
            )
        if "longitude" in ds:
            ds.longitude.attrs.update(
                {"standard_name": "longitude", "units": "degrees_east"}
            )

        # 5. Add UGRID to Conventions
        conv = ds.attrs.get("Conventions", "")
        if "UGRID-1.0" not in conv:
            ds.attrs["Conventions"] = f"{conv} UGRID-1.0".strip()

        # 6. Expand to 2D (time, node) if requested
        if expand2d:
            ds = ds_to_2d(ds)

        return ds
