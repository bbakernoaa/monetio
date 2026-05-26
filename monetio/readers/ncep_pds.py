"""Base Reader for NCEP products on AWS Public Dataset (PDS) or NOMADS."""

import datetime

import pandas as pd
import xarray as xr

from .base import GriddedReader
from .sat_utils import update_history


class NCEPPDSReader(GriddedReader):
    """
    Base reader for NCEP products on AWS Public Dataset (PDS).
    """

    def open_dataset(
        self,
        files: str | list[str] = None,
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        virtualizarr_parser: str | None = None,
        virtualizarr_backend: str = "kerchunk",
        icechunk_repo: str | None = None,
        use_icechunk: bool = False,
        icechunk_url: str | None = None,
        use_dask: bool = False,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str = None,
        hour: int = 0,
        lead_time: int | list[int] = 0,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NCEP GRIB2 data from AWS S3 or NOMADS.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s), S3 URL(s), or HTTPS URL(s).
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
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        hour : int, optional
            Forecast cycle hour (0, 6, 12, 18). Default is 0.
        lead_time : Union[int, List[int]], optional
            Forecast lead time(s) in hours. Default is 0.
        **kwargs : dict
            Additional arguments. Arguments like 'product', 'source', 'domain' are passed
            to build_urls. Other arguments are passed to the Xarray driver.

        Returns
        -------
        xr.Dataset
            The dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            # Pass all kwargs to build_urls so it can handle 'product', 'source', etc.
            files = self.build_urls(dates, hour=hour, lead_time=lead_time, **kwargs)

        if "engine" not in kwargs:
            kwargs["engine"] = "grib2io"
        # Note: Some kwargs passed to build_urls might not be valid for the driver,
        # but XarrayDriver.open generally handles this by consuming what it knows
        # and ignoring/forwarding the rest.
        ds = super().open_dataset(
            files,
            use_virtualizarr=use_virtualizarr,
            virtualizarr_file=virtualizarr_file,
            virtualizarr_parser="grib2",
            virtualizarr_backend=virtualizarr_backend,
            icechunk_repo=icechunk_repo,
            use_icechunk=use_icechunk,
            icechunk_url=icechunk_url,
            use_dask=use_dask,
            **kwargs,
        )

        # Apply standard harmonization
        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, f"Read {self.__class__.__name__} data.")

        return ds

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Harmonize NCEP metadata to monetio standards.
        """
        # Coordinate Renaming
        rename_dict = {
            "latitude": "latitude",
            "longitude": "longitude",
            "lat": "latitude",
            "lon": "longitude",
            "lat_0": "latitude",
            "lon_0": "longitude",
            "time": "time",
            "valid_time": "time",
            "step": "step",
        }

        actual_rename = {}
        for k, v in rename_dict.items():
            if (k in ds.variables or k in ds.dims) and v not in ds.variables and k != v:
                actual_rename[k] = v

        if actual_rename:
            ds = ds.rename(actual_rename)

        # Variable Mapping
        var_mapping = {
            "O3MR": "ozone",
            "TMP": "temperature",
            "UGRD": "u_wind",
            "VGRD": "v_wind",
            "PRES": "pressure",
            "HGT": "height",
            "RH": "relative_humidity",
            "PRMSL": "mslp",
        }
        actual_var_rename = {}
        for var in ds.variables:
            for k, v in var_mapping.items():
                # Check for exact match or suffix (e.g., 'TMP:isobaricInhPa')
                if (var == k or var.startswith(f"{k}:")) and v not in ds.variables:
                    actual_var_rename[var] = v
                    break
        if actual_var_rename:
            ds = ds.rename(actual_var_rename)

        # Ensure latitude/longitude are coordinates
        coord_vars = [v for v in ["latitude", "longitude", "time"] if v in ds.variables]
        if coord_vars:
            ds = ds.set_coords(coord_vars)

        # Scientific Hygiene: Strip whitespace from string attributes
        for var in ds.variables:
            for attr, val in ds[var].attrs.items():
                if isinstance(val, str):
                    ds[var].attrs[attr] = val.strip()

        return ds
