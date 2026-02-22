"""NESDIS VIIRS JRR Reader"""

import datetime
from typing import List, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import add_time_coord, standardize_satellite_coords, update_history


@register_reader("nesdis_viirs_jrr")
@register_reader("viirs_jrr")
class VIIRSJRRReader(GriddedReader):
    """
    Reader for NESDIS VIIRS JRR (Joint Polar Satellite System Risk Reduction) products.
    Supports AOD, ADP, CloudMask, and others.
    Available on AWS Open Data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        satellite: str = "snpp",
        product: str = "AOD",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS VIIRS JRR data.

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
            JRR product: 'AOD' (default), 'ADP', 'CloudMask', 'CloudHeight', etc.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The VIIRS JRR dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates, satellite=satellite, product=product)

        if "preprocess" not in kwargs:
            from functools import partial

            kwargs["preprocess"] = partial(viirs_jrr_preprocess, product=product)

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, f"Read NESDIS VIIRS JRR {product} data from {satellite}.")

        return ds

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        satellite: str = "snpp",
        product: str = "AOD",
    ) -> List[str]:
        """
        Build S3 URLs for NESDIS VIIRS JRR data based on dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        satellite : str, optional
            Satellite identifier ('snpp', 'n20', 'n21', 'j01', 'j02').
        product : str, optional
            JRR product.

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
            "j01": "noaa-nesdis-n20-pds",
            "n20": "noaa-nesdis-n20-pds",
            "j02": "noaa-nesdis-n21-pds",
            "n21": "noaa-nesdis-n21-pds",
        }
        bucket = sat_map.get(satellite.lower())
        if not bucket:
            raise ValueError(f"Unknown satellite: {satellite}. Choose from {list(sat_map.keys())}")

        fs = s3fs.S3FileSystem(anon=True)
        urls = []
        for d in dates.floor("D").unique():
            prefix = f"{bucket}/VIIRS-JRR-{product}/{d.strftime('%Y/%m/%d')}/"
            # We use glob to find all granules for the day
            try:
                found = fs.glob(f"{prefix}*.nc")
                urls.extend([f"s3://{f}" for f in found])
            except Exception:
                continue

        return sorted(urls)


def viirs_jrr_preprocess(ds: xr.Dataset, product: str = "AOD") -> xr.Dataset:
    """
    Preprocess VIIRS JRR dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    product : str, optional
        Product type, by default "AOD".

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    ds = standardize_satellite_coords(ds)
    ds = add_time_coord(ds, time_attr="time_coverage_start")

    # Product-specific renaming
    if product.upper() == "AOD":
        if "AOD550" in ds.data_vars:
            ds = ds.rename({"AOD550": "aod_550"})
            ds["aod_550"].attrs.update(
                {
                    "long_name": "Aerosol Optical Thickness at 550nm",
                    "units": "1",
                    "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol",
                }
            )
    elif product.upper() == "ADP":
        # ADP already has descriptive names like 'Smoke', 'Dust', 'Ash'
        pass

    return ds
