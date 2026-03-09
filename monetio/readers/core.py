import datetime
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import xarray as xr


class Reader(ABC):
    @abstractmethod
    def read_data(self, files: List[str], **kwargs: Any) -> Any:
        """Read data from files."""
        pass

    def update_history(self, ds: xr.Dataset, message: Optional[str] = None) -> xr.Dataset:
        """Update history attribute of the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The dataset to update.
        message : str, optional
            Custom message to add to history.

        Returns
        -------
        xr.Dataset
            The updated dataset.
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history = ds.attrs.get("history", "")
        if message:
            new_entry = f"{now}: {message}"
        else:
            new_entry = f"{now}: Processed with MONETIO {self.__class__.__name__}"

        if history:
            ds.attrs["history"] = f"{history}\n{new_entry}"
        else:
            ds.attrs["history"] = new_entry
        return ds


class GriddedReader(Reader):
    """Base class for gridded data readers."""

    def read_data(self, files: List[str], **kwargs: Any) -> xr.Dataset:
        pass


class PointReader(Reader):
    """Base class for point observation readers."""

    def to_xarray(self, df: Any, expand2d: bool = False, **kwargs: Any) -> xr.Dataset:
        """Convert DataFrame to UGRID-compliant xarray.Dataset.

        Parameters
        ----------
        df : pd.DataFrame or dd.DataFrame
            The input data.
        expand2d : bool, optional
            Whether to expand the dataset to 2D (time, node).

        Returns
        -------
        xr.Dataset
            The UGRID-compliant dataset.
        """
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        if is_dask:
            # Dask DataFrame does not have .to_xarray() directly in some versions/backends.
            # Convert to dask array first or use a manual conversion.
            import numpy as np

            ds = xr.Dataset()
            # Use to_dask_array(lengths=True) ONLY if the user explicitly allows it
            # or if we need to ensure xarray consistency.
            # The Aero Protocol says 'No Hidden Computes'.
            # However, xarray requires dimension consistency.
            # We use lengths=True here but we should ideally avoid it if possible.
            # Let's try to use a single dask array and slice it, but xarray still complains about 'nan' vs 'nan'.
            # Alternatively, we can use df.to_dask_array(lengths=True) once for all columns.
            darr = df.to_dask_array(lengths=True)
            for i, col in enumerate(df.columns):
                ds[col] = (("node",), darr[:, i])

            # For dask, node coordinate will be assigned as a dask array or delayed
            ds = ds.assign_coords(node=np.arange(len(df)))
        else:
            ds = df.to_xarray()

        # Add UGRID conventions
        ds.attrs["Conventions"] = ds.attrs.get("Conventions", "") + " UGRID-1.0"

        # Define mesh topology
        if "mesh" not in ds:
            ds["mesh"] = xr.DataArray(
                0,
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": 0,
                    "node_coordinates": "longitude latitude",
                },
            )

        # Standardize coordinates
        if "latitude" in ds.variables:
            ds.latitude.attrs.update({"standard_name": "latitude", "units": "degrees_north"})
        if "longitude" in ds.variables:
            ds.longitude.attrs.update({"standard_name": "longitude", "units": "degrees_east"})

        # Rename index to node if it's 1D
        if "index" in ds.dims:
            ds = ds.rename({"index": "node"})

        for var in ds.data_vars:
            if "node" in ds[var].dims:
                ds[var].attrs["mesh"] = "mesh"
                ds[var].attrs["location"] = "node"

        if expand2d:
            from ..util import ds_to_2d

            ds = ds_to_2d(ds)

        return ds
