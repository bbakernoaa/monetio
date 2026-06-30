"""MADIS (Meteorological Assimilation Data Ingest System) Reader"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from ..util import force_object_strings
from .base import PointReader, register_reader
from .sat_utils import update_history


@register_reader("madis")
class MADISReader(PointReader):
    """
    Reader for NOAA MADIS (Meteorological Assimilation Data Ingest System) data.
    """

    def open_dataset(
        self,
        files: str | list[str],
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        virtualizarr_parser: str | None = None,
        virtualizarr_backend: str = "kerchunk",
        icechunk_repo: str | None = None,
        use_icechunk: bool = False,
        icechunk_url: str | None = None,
        lazy: bool = False,
        use_dask: bool = False,
        as_xarray: bool = True,
        expand2d: bool = True,
        **kwargs,
    ) -> xr.Dataset | pd.DataFrame | dd.DataFrame:
        """
        Reads MADIS NetCDF files.

        Parameters
        ----------
        files : str or list[str]
            File path(s) or glob pattern.
        use_virtualizarr : bool, optional
            Whether to use VirtualiZarr to create a virtual Zarr dataset, by default False.
        virtualizarr_file : str or None, optional
            Path to save/load the VirtualiZarr reference JSON file, by default None.
        virtualizarr_parser : str or None, optional
            The VirtualiZarr parser to use (e.g. 'hdf5', 'netcdf3', 'zarr', 'grib2').
        virtualizarr_backend : str, optional
            Backend for VirtualiZarr references ("kerchunk" or "icechunk"), by default "kerchunk".
        icechunk_repo : str or None, optional
            Path to the Icechunk repository, by default None.
        use_icechunk : bool, optional
            Whether to use Icechunk, by default False.
        icechunk_url : str or None, optional
            Path to the Icechunk repository, by default None.
        lazy : bool, optional
            Whether to use Dask for lazy loading, by default False.
        use_dask : bool, optional
            Alias for ``lazy``.
        as_xarray : bool, optional
            Whether to return an xarray Dataset, by default True.
        expand2d : bool, optional
            Whether to expand to 2D UGRID format, by default True.
        **kwargs : dict
            Additional arguments passed to ``xr.open_dataset`` or ``xr.open_mfdataset``.

        Returns
        -------
        xr.Dataset or pd.DataFrame or dask.dataframe.DataFrame
            UGRID xarray dataset by default, or DataFrame if ``as_xarray=False``.
        """
        if use_dask:
            lazy = True

        # MADIS files are NetCDF but contain point data.
        # We can use xarray to open them and then convert to the MONETIO point format.
        if isinstance(files, str):
            import glob

            files = sorted(glob.glob(files)) if "*" in files else [files]

        if lazy:
            # We must set chunks to an empty dict if not provided to ensure dask-backed
            if "chunks" not in kwargs:
                kwargs["chunks"] = {}
            ds = xr.open_mfdataset(files, combine="nested", concat_dim="recNum", **kwargs)
        else:
            datasets = []
            for f in files:
                ds_single = xr.open_dataset(f, **kwargs)
                datasets.append(ds_single)

            if len(datasets) > 1:
                ds = xr.concat(datasets, dim="recNum")
            else:
                ds = datasets[0]

        # MADIS files often have 'recNum' as the dimension
        if "recNum" in ds.dims:
            ds = ds.rename({"recNum": "node"})

        # Ensure node coordinate is set for both eager and lazy
        if "node" in ds.dims and "node" not in ds.coords:
            if lazy:
                import dask.array as da

                # Match chunking if possible
                chunks = ds.chunks.get("node", (ds.sizes["node"],))
                ds.coords["node"] = (("node",), da.arange(ds.sizes["node"], chunks=chunks[0]))
            else:
                import numpy as np

                ds.coords["node"] = (("node",), np.arange(ds.sizes["node"]))

        ds = self.harmonize(ds)

        if as_xarray:
            # If it's already an xarray, we might still want to apply to_xarray
            # for UGRID expansion if expand2d=True.
            # But we can also do it via ds_to_2d directly to avoid round-trip.
            if expand2d:
                from ..util import ds_to_2d

                ds = ds_to_2d(ds, fixed_location=self.fixed_location)

            ds = update_history(ds, "Read MADIS data into xarray format.")
            return ds

        # Fallback to DataFrame if explicitly requested
        if lazy:
            # Avoid triggering a compute with .to_dataframe()
            import dask.dataframe as dd

            # Identify variables and coordinates with 'node' dimension
            node_vars = [v for v in ds.variables if ds[v].dims == ("node",)]

            if not node_vars:
                return pd.DataFrame()

            # Create dask dataframe
            # We use dd.concat to join 1D dask arrays into a DataFrame
            # All variables here are guaranteed to have dimension ('node',)
            import dask.array as da

            chunks = ds.chunks.get("node", (ds.sizes["node"],))[0]
            df_parts = []
            for v in node_vars:
                data = ds[v].data
                if not hasattr(data, "dask"):
                    data = da.from_array(data, chunks=chunks)
                df_parts.append(dd.from_dask_array(data.flatten(), columns=[v]))

            df = dd.concat(df_parts, axis=1)
        else:
            df = ds.to_dataframe().reset_index()

        df = force_object_strings(df)
        df.attrs = dict(ds.attrs)

        return df

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize MADIS dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # Mapping MADIS variable names to MONET names
        mapping = {
            "latitude": "latitude",
            "longitude": "longitude",
            "observationTime": "time",
            "stationId": "siteid",
            "stationName": "name",
            "elevation": "elevation",
            "temperature": "temperature",
            "dewpoint": "dewpoint",
            "relHumidity": "rel_humidity",
            "windDir": "wind_dir",
            "windSpeed": "wind_speed",
            "altimeter": "altimeter",
            "stationPress": "station_pressure",
            "seaLevelPress": "slp",
            "precip": "precipitation",
        }

        rename_dict = {
            old: new
            for old, new in mapping.items()
            if old in ds.variables and new not in ds.variables
        }
        if rename_dict:
            ds = ds.rename(rename_dict)

        # Handle time if it's in seconds since epoch
        if "time" in ds.variables:
            if ds["time"].attrs.get("units") == "seconds since 1970-01-01 00:00:00.0 +0000":
                # Convert to datetime64[ns] lazily using vectorized operations
                # Epoch is 1970-01-01
                ds["time"] = (ds["time"] * 1e9).astype("datetime64[ns]")

        # Set coordinates
        coords = ["time", "siteid", "latitude", "longitude", "elevation"]
        ds = ds.set_coords([c for c in coords if c in ds.variables])

        # Update history
        ds = update_history(ds, "Harmonized MADIS data.")

        return ds
