"""MADIS (Meteorological Assimilation Data Ingest System) Reader"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd


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
        use_dask: bool = False,
        as_xarray: bool = True,
        expand2d: bool = True,
        lazy: bool = False,
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
        use_dask : bool, optional
            Whether to use Dask for lazy loading, by default False.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        expand2d : bool, optional
            Whether to expand to 2D (time, node) structure, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open or to_xarray.

        Returns
        -------
        xr.Dataset or pd.DataFrame or dd.DataFrame
            UGRID xarray dataset by default, or DataFrame.
        """
        from .drivers import XarrayDriver

        # Handle 'use_dask' as an alias for 'lazy'
        if use_dask:
            lazy = True

        if lazy and "chunks" not in kwargs:
            kwargs["chunks"] = "auto"

        def _madis_preprocess(ds):
            if "recNum" in ds.dims:
                ds = ds.rename({"recNum": "node"})
            # Also handle if it's already named 'node' to prevent double rename/conflicts
            return ds

        # Chain preprocess if provided
        user_preprocess = kwargs.get("preprocess")
        if user_preprocess:

            def chained_preprocess(ds):
                ds = _madis_preprocess(ds)
                return user_preprocess(ds)

            kwargs["preprocess"] = chained_preprocess
        else:
            kwargs["preprocess"] = _madis_preprocess

        # MADIS files are NetCDF. We use XarrayDriver for robustness.
        dr = XarrayDriver()
        ds = dr.open(
            files,
            use_virtualizarr=use_virtualizarr,
            virtualizarr_file=virtualizarr_file,
            virtualizarr_parser=virtualizarr_parser,
            virtualizarr_backend=virtualizarr_backend,
            icechunk_repo=icechunk_repo,
            use_icechunk=use_icechunk,
            icechunk_url=icechunk_url,
            use_dask=use_dask,
            concat_dim="node",
            combine="nested",
            **kwargs,
        )

        ds = self.harmonize(ds)

        if as_xarray:
            # For Xarray output, we can bypass the DataFrame round-trip
            # and use ds_to_2d directly if expand2d is True.
            from ..util import ds_to_2d

            ds_out = ds
            if expand2d:
                # pivot defaults to True in ds_to_2d
                pivot = kwargs.get("wide_fmt", kwargs.get("pivot", True))
                ds_out = ds_to_2d(ds, pivot=pivot, fixed_location=self.fixed_location)

            # Add UGRID metadata and ensure time dimension (consistent with to_xarray)
            ds_out.attrs.update(ds.attrs)
            if "Conventions" not in ds_out.attrs:
                ds_out.attrs["Conventions"] = "CF-1.8 UGRID-1.0"
            elif "UGRID-1.0" not in ds_out.attrs["Conventions"]:
                ds_out.attrs["Conventions"] += " UGRID-1.0"

            ds_out = update_history(ds_out, "Converted MADIS data to UGRID xarray format.")
            from .base import _ensure_time_dimension

            return _ensure_time_dimension(ds_out)

        # For DataFrame output (as_xarray=False)
        if hasattr(ds, "chunks") and ds.chunks:
            # We must ensure all variables are chunked consistently for to_dask_dataframe
            # and that it's 1D on 'node'.
            df = ds.to_dask_dataframe().reset_index()
        else:
            df = ds.to_dataframe().reset_index()

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
            # Use vectorized .astype('datetime64[s]') to avoid .values compute
            if ds["time"].attrs.get("units") == "seconds since 1970-01-01 00:00:00.0 +0000":
                with xr.set_options(keep_attrs=True):
                    ds["time"] = ds["time"].astype("datetime64[s]").astype("datetime64[ns]")

        # Set coordinates
        coords = ["time", "siteid", "latitude", "longitude", "elevation"]
        ds = ds.set_coords([c for c in coords if c in ds.variables])

        # Update history
        ds = update_history(ds, "Harmonized MADIS data.")

        return ds
