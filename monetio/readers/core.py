from __future__ import annotations

import xarray as xr
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    import dask.dataframe as dd
    import pandas as pd

class BaseReader:
    """Base class for all MONETIO readers.

    Follows the Aero Protocol:
    - Backend Agnostic
    - Provenance Tracking
    - No Hidden Computes
    """

    def update_history(self, ds: xr.Dataset | xr.DataArray, message: Optional[str] = None) -> xr.Dataset | xr.DataArray:
        """Update the history attribute of the dataset.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            The dataset to update.
        message : str, optional
            The message to add to the history. If None, a default message is used.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            The updated dataset with a timestamped history entry.
        """
        if message is None:
            message = f"Data processed by {self.__class__.__name__}"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = f"{timestamp}: {message}"

        history = ds.attrs.get("history", "")
        if history:
            ds.attrs["history"] = f"{new_entry}\n{history}"
        else:
            ds.attrs["history"] = new_entry

        return ds

class PointReader(BaseReader):
    """Base class for point observation readers.

    Provides utilities to convert tabular data to UGRID-compliant xarray Datasets.
    """

    def to_xarray(self, df: pd.DataFrame | dd.DataFrame, expand2d: bool = True) -> xr.Dataset:
        """Convert a pandas or dask DataFrame to a UGRID-compliant xarray Dataset.

        Parameters
        ----------
        df : pandas.DataFrame or dask.dataframe.DataFrame
            The input tabular data.
        expand2d : bool, optional
            If True, expand the 1D dataset (indexed by 'node') to a 2D dataset
            with dimensions (time, node). Default is True.

        Returns
        -------
        xarray.Dataset
            A UGRID-compliant dataset.
        """
        from ..util import ds_to_2d, force_object_strings

        # 1. Ensure string columns are objects to avoid Pandas 3.0 issues
        df = force_object_strings(df)

        # 2. Convert to Dataset.
        # We use from_dataframe which is the standard Xarray way to convert DataFrames.
        # For Dask DataFrames, this should preserve laziness for the data variables,
        # although it may compute the index to define coordinates.
        ds = xr.Dataset.from_dataframe(df)

        if "index" in ds.dims:
            ds = ds.rename({"index": "node"})
        elif df.index.name and df.index.name in ds.dims:
            ds = ds.rename({df.index.name: "node"})

        # 3. Add UGRID-1.0 Conventions
        conv = ds.attrs.get("Conventions", "")
        if "UGRID-1.0" not in conv:
            ds.attrs["Conventions"] = f"UGRID-1.0 {conv}".strip()

        # 4. Define Mesh Topology
        coords = ["latitude", "longitude", "elevation"]
        available_coords = [c for c in coords if c in ds.variables or c in ds.coords]

        ds["mesh"] = xr.DataArray(
            0,
            attrs={
                "cf_role": "mesh_topology",
                "topology_dimension": 0,
                "node_coordinates": " ".join(available_coords),
            }
        )

        # Link data variables to the mesh
        for v in ds.data_vars:
            if v != "mesh":
                ds[v].attrs["mesh"] = "mesh"
                ds[v].attrs["location"] = "node"

        # 5. Optional 2D Expansion (time, node)
        if expand2d:
            ds = ds_to_2d(ds)

        # 6. Provenance
        ds = self.update_history(ds, "Converted to UGRID-compliant xarray Dataset via PointReader")

        return ds
