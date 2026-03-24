"""GFS, GEFS, and GDAS Readers for AWS Open Data"""

import datetime
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


class NCEPPDSReader(GriddedReader):
    """
    Base reader for NCEP products on AWS Public Dataset (PDS).
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "pgrb2.0p25",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NCEP GRIB2 data from AWS S3.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or S3 URL(s).
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        hour : int, optional
            Forecast cycle hour (0, 6, 12, 18). Default is 0.
        lead_time : Union[int, List[int]], optional
            Forecast lead time(s) in hours. Default is 0.
        product : str, optional
            Product string (e.g., 'pgrb2.0p25').
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates, hour=hour, lead_time=lead_time, product=product)

        if "engine" not in kwargs:
            kwargs["engine"] = "grib2io"

        # grib2io engine might need help with S3 URLs.
        # We ensure they are expanded and handled.
        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, f"Read {self.__class__.__name__} data from AWS PDS.")

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

        # Variable Mapping (Aero Protocol)
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
        actual_var_rename = {
            k: v for k, v in var_mapping.items() if k in ds.variables and v not in ds.variables
        }
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


@register_reader("gfs")
class GFSReader(NCEPPDSReader):
    """
    Reader for GFS (Global Forecast System) on AWS.
    """

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "pgrb2.0p25",
    ) -> List[str]:
        """
        Build S3 URLs for GFS data.
        """
        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        bucket = "noaa-gfs-bdp-pds"
        urls = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            h_str = f"{hour:02d}"
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                # s3://noaa-gfs-bdp-pds/gfs.20250324/00/atmos/gfs.t00z.pgrb2.0p25.f000
                url = f"s3://{bucket}/gfs.{d_str}/{h_str}/atmos/gfs.t{h_str}z.{product}.f{lt_str}"
                urls.append(url)
        return urls


@register_reader("gefs")
class GEFSReader(NCEPPDSReader):
    """
    Reader for GEFS (Global Ensemble Forecast System) on AWS.
    """

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "geavg.tHHz.pgrb2a.0p50",
    ) -> List[str]:
        """
        Build S3 URLs for GEFS data.
        Note: product here usually specifies the member and resolution.
        Example: 'geavg.tHHz.pgrb2a.0p50' for ensemble mean 0.5 deg.
        """
        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        bucket = "noaa-gefs-pds"
        urls = []
        h_str = f"{hour:02d}"
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                # The product string might have 'tHHz' as a placeholder
                prod = product.replace("tHHz", f"t{h_str}z")
                # s3://noaa-gefs-pds/gefs.20250324/00/atmos/pgrb2ap5/geavg.t00z.pgrb2a.0p50.f000
                # Note: pgrb2ap5 is a subdirectory for 0.5 deg products
                res_dir = "pgrb2ap5" if "0p50" in prod else "pgrb2bp5"
                url = f"s3://{bucket}/gefs.{d_str}/{h_str}/atmos/{res_dir}/{prod}.f{lt_str}"
                urls.append(url)
        return urls


@register_reader("gdas")
class GDASReader(NCEPPDSReader):
    """
    Reader for GDAS (Global Data Assimilation System) on AWS.
    """

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        hour: int = 0,
        lead_time: Union[int, List[int]] = 0,
        product: str = "pgrb2.0p25",
    ) -> List[str]:
        """
        Build S3 URLs for GDAS data.
        """
        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        if isinstance(lead_time, int):
            lead_times = [lead_time]
        else:
            lead_times = lead_time

        bucket = "noaa-gfs-bdp-pds"
        urls = []
        for d in dates:
            d_str = d.strftime("%Y%m%d")
            h_str = f"{hour:02d}"
            for lt in lead_times:
                lt_str = f"{lt:03d}"
                # s3://noaa-gfs-bdp-pds/gdas.20250324/00/atmos/gdas.t00z.pgrb2.0p25.f000
                url = f"s3://{bucket}/gdas.{d_str}/{h_str}/atmos/gdas.t{h_str}z.{product}.f{lt_str}"
                urls.append(url)
        return urls
