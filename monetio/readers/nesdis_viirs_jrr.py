"""NESDIS VIIRS JRR AOD Reader"""

import datetime
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import add_time_coord, standardize_satellite_coords, update_history


@register_reader("nesdis_viirs_jrr")
class VIIRSJRRAODReader(GriddedReader):
    """
    Reader for NESDIS VIIRS JRR (Joint Polar Satellite System Risk Reduction) AOD data.
    Available on AWS Open Data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        satellite: str = "snpp",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS VIIRS JRR AOD data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        satellite : str, optional
            Satellite identifier: 'snpp', 'j01' (NOAA-20), or 'j02' (NOAA-21).
            Default is 'snpp'.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The VIIRS JRR AOD dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates, satellite=satellite)

        if "preprocess" not in kwargs:
            kwargs["preprocess"] = viirs_jrr_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read NESDIS VIIRS JRR AOD data.")

        return ds

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        satellite: str = "snpp",
    ) -> List[str]:
        """
        Build S3 URLs for NESDIS VIIRS JRR AOD data based on dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        satellite : str, optional
            Satellite identifier ('snpp', 'j01', 'j02').

        Returns
        -------
        List[str]
            List of S3 URLs.
        """
        import s3fs

        if isinstance(dates, (str, datetime.datetime, pd.Timestamp)):
            dates = pd.DatetimeIndex([pd.to_datetime(dates)])
        else:
            dates = pd.to_datetime(dates)

        sat_map = {
            "snpp": "noaa-nesdis-snpp-pds",
            "j01": "noaa-nesdis-j01-pds",
            "j02": "noaa-nesdis-j02-pds",
        }
        bucket = sat_map.get(satellite.lower())
        if not bucket:
            raise ValueError(f"Unknown satellite: {satellite}. Choose from {list(sat_map.keys())}")

        fs = s3fs.S3FileSystem(anon=True)
        urls = []
        for d in dates.floor("D").unique():
            prefix = f"{bucket}/VIIRS-JRR-AOD/{d.strftime('%Y/%m/%d')}/"
            # We use glob to find all granules for the day
            found = fs.glob(f"{prefix}*.nc")
            urls.extend([f"s3://{f}" for f in found])

        return sorted(urls)


def viirs_jrr_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess VIIRS JRR AOD dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    ds = standardize_satellite_coords(ds)
    ds = add_time_coord(ds, time_attr="time_coverage_start")

    # Rename variables to more standard names
    if "AOD550" in ds.data_vars:
        ds = ds.rename({"AOD550": "aod_550"})
        ds["aod_550"].attrs.update(
            {
                "long_name": "Aerosol Optical Thickness at 550nm",
                "units": "1",
                "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol",
            }
        )

    return ds
