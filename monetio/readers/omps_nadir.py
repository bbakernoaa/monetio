"""OMPS Nadir Reader"""

import datetime
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import (
    add_time_coord,
    jpss_time_to_datetime,
    standardize_satellite_coords,
    update_history,
)


@register_reader("omps_nadir")
class OMPSNadirReader(GriddedReader):
    """
    Reader for OMPS (Ozone Mapping and Profiler Suite) Nadir products.
    Supports NOAA JPSS (V8TOZ, NP/TC SDR/GEO) and NASA (NMTO3) products.
    Available on AWS Open Data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        satellite: str = "snpp",
        product: str = "v8toz",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads OMPS Nadir data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        satellite : str, optional
            Satellite identifier: 'snpp', 'n20' (NOAA-20/J01), or 'n21' (NOAA-21/J02).
            Default is 'snpp'.
        product : str, optional
            OMPS product type:
            - 'v8toz': NOAA Total Ozone EDR (default)
            - 'nmto3_l2': NASA Nadir Mapper Total Ozone L2
            - 'nmto3_l3': NASA Nadir Mapper Total Ozone L3
            - 'tc_sdr': NOAA Total Column SDR
            - 'np_sdr': NOAA Nadir Profiler SDR
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The OMPS dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates, satellite=satellite, product=product)

        if "preprocess" not in kwargs:
            from functools import partial

            kwargs["preprocess"] = partial(omps_nadir_preprocess, product=product)

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, f"Read OMPS Nadir {product} data from {satellite}.")

        return ds

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        satellite: str = "snpp",
        product: str = "v8toz",
    ) -> List[str]:
        """
        Build S3 URLs for OMPS Nadir data based on dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        satellite : str, optional
            Satellite identifier ('snpp', 'n20', 'n21', 'j01', 'j02').
        product : str, optional
            OMPS product.

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
            "n20": "noaa-nesdis-n20-pds",
            "n21": "noaa-nesdis-n21-pds",
            "j01": "noaa-nesdis-n20-pds",
            "j02": "noaa-nesdis-n21-pds",
        }
        bucket = sat_map.get(satellite.lower())
        if not bucket:
            raise ValueError(f"Unknown satellite: {satellite}. Choose from {list(sat_map.keys())}")

        prod_map = {
            "v8toz": "OMPS_V8TOZ",
            "tc_sdr": "OMPS-TC-SDR",
            "tc_geo": "OMPS-TC-GEO",
            "np_sdr": "OMPS-NP-SDR",
            "np_geo": "OMPS-NP-GEO",
        }
        dir_name = prod_map.get(product.lower())
        if not dir_name:
            # Fallback for NASA or others if they ever follow this pattern,
            # but usually they don't.
            dir_name = product

        fs = s3fs.S3FileSystem(anon=True)
        urls = []
        for d in dates.floor("D").unique():
            prefix = f"{bucket}/{dir_name}/{d.strftime('%Y/%m/%d')}/"
            try:
                found = fs.glob(f"{prefix}*.nc")
                # Also try .h5 for SDRs
                if not found and "SDR" in dir_name:
                    found = fs.glob(f"{prefix}*.h5")
                urls.extend([f"s3://{f}" for f in found])
            except Exception:
                continue

        return sorted(urls)


def omps_nadir_preprocess(ds: xr.Dataset, product: str = "v8toz") -> xr.Dataset:
    """
    Preprocess OMPS Nadir dataset lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    product : str, optional
        Product type, by default "v8toz".

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    if product == "v8toz":
        ds = _preprocess_v8toz(ds)
    elif product in ["nmto3_l2", "nmto3_l3"]:
        # Use existing logic from omps.py
        from .omps import omps_preprocess

        ds = omps_preprocess(ds, product=product)
        return ds
    elif product in ["tc_sdr", "np_sdr"]:
        ds = _preprocess_sdr(ds)

    # Standardize coordinates and dimensions
    ds = standardize_satellite_coords(ds)

    # Ensure a time dimension exists for stitching if not already present
    if "time" not in ds.dims:
        if "time" in ds.coords:
            if ds.coords["time"].ndim == 1:
                # If time is 1D and matches y or x, swap it
                # But jpss_time_to_datetime usually produces something indexed by y
                pass
            else:
                # Expand a new time dimension
                ds = ds.expand_dims("time")
        else:
            # Try to add from attributes
            ds = add_time_coord(ds, time_attr="time_coverage_start")

    return ds


def _preprocess_v8toz(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess NOAA V8TOZ Total Ozone EDR."""
    mapping = {
        "Latitude": "latitude",
        "Longitude": "longitude",
        "ScanTime": "time_raw",
        "ColumnAmountO3": "ozone_column",
        "AerosolIndex": "aerosol_index",
        "So2Index": "so2_index",
        "CloudFraction": "cloud_fraction",
        "QualityFlag": "quality_flag",
    }

    # Rename variables if they exist
    rename_dict = {old: new for old, new in mapping.items() if old in ds.variables}
    if rename_dict:
        ds = ds.rename(rename_dict)

    # Handle Time (Lazy)
    if "time_raw" in ds.variables:
        ds["time"] = jpss_time_to_datetime(ds["time_raw"])
        ds = ds.set_coords("time")

    return ds


def _preprocess_sdr(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess NOAA SDR (Sensor Data Record)."""
    # SDRs usually have data in groups
    mapping = {
        "All_Data/OMPS-TC-SDR_All/Radiance": "radiance",
        "All_Data/OMPS-NP-SDR_All/Radiance": "radiance",
        "All_Data/OMPS-TC-GEO_All/Latitude": "latitude",
        "All_Data/OMPS-TC-GEO_All/Longitude": "longitude",
        "All_Data/OMPS-NP-GEO_All/Latitude": "latitude",
        "All_Data/OMPS-NP-GEO_All/Longitude": "longitude",
    }
    rename_dict = {old: new for old, new in mapping.items() if old in ds.variables}
    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds
