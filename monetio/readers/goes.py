"""GOES Reader"""

import datetime
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import add_time_coord, standardize_satellite_coords, update_history


@register_reader("goes")
class GOESReader(GriddedReader):
    """
    Reader for GOES-R Series (GOES-16, 17, 18) ABI data.
    Supports local files and S3 (via s3fs).
    """

    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str] = None,
        satellite: str = "16",
        product: str = "ABI-L2-AODF",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads GOES data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve. If files is None, this is used to build URLs.
        satellite : str, optional
            Satellite identifier (e.g., '16', '17', '18'). Default is '16'.
        product : str, optional
            GOES product (e.g., 'ABI-L2-AODF'). Default is 'ABI-L2-AODF'.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The GOES dataset.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files = self.build_urls(dates, satellite=satellite, product=product)

        if "preprocess" not in kwargs:
            kwargs["preprocess"] = goes_preprocess

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, f"Read GOES-{satellite} {product} data.")

        return ds

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        satellite: str = "16",
        product: str = "ABI-L2-AODF",
    ) -> List[str]:
        """
        Build S3 URLs for GOES data based on dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        satellite : str, optional
            Satellite identifier ('16', '17', '18').
        product : str, optional
            GOES product.

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

        fs = s3fs.S3FileSystem(anon=True)
        bucket = f"noaa-goes{satellite}"

        urls = []
        for d in dates:
            # GOES S3 structure: <product>/<year>/<day_of_year>/<hour>/
            prefix = f"{bucket}/{product}/{d.strftime('%Y/%j/%H')}/"
            try:
                found = fs.ls(prefix)
                if not found:
                    continue

                # Find the file closest to the requested time
                file_dates = [
                    pd.to_datetime(f.split("_")[-1].split(".")[0][1:], format="%Y%j%H%M%S%f")
                    for f in found
                ]
                idx = np.argmin([abs(fd - d) for fd in file_dates])
                urls.append(f"s3://{found[idx]}")
            except (FileNotFoundError, ValueError):
                continue

        return sorted(list(set(urls)))


def goes_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess GOES dataset: calculate grid and standardize coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Processed dataset.

    Examples
    --------
    >>> ds = reader.open_dataset(files)
    >>> ds = goes_preprocess(ds)
    """
    # 1. Standardize dimensions and coordinates
    ds = standardize_satellite_coords(ds)

    # 2. Add time coordinate if not present
    if "time" not in ds.coords:
        ds = add_time_coord(ds, time_attr="time_coverage_start")

    # 3. Calculate Latitude/Longitude (Lazy)
    if "latitude" not in ds.coords and "goes_imager_projection" in ds.variables:
        ds = _add_goes_latlon(ds)

    # Update history
    ds = update_history(ds, "Preprocessed GOES data.")

    return ds


def _add_goes_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate latitude and longitude for GOES data lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'x', 'y' and 'goes_imager_projection'.

    Returns
    -------
    xr.Dataset
        Dataset with 'latitude' and 'longitude' coordinates added.
    """
    from pyproj import CRS, Proj

    proj_var = ds.goes_imager_projection
    proj_dict = proj_var.attrs.copy()

    # Ensure all attributes are scalars
    for k, v in proj_dict.items():
        if isinstance(v, (list, np.ndarray)):
            proj_dict[k] = v[0]

    crs = CRS.from_cf(proj_dict)
    ds.attrs["projection"] = crs.to_wkt()
    proj = Proj(crs)

    satellite_height = proj_var.perspective_point_height

    def _calc_latlon(x, y):
        # x and y are in radians in GOES files
        xx, yy = np.meshgrid(x * satellite_height, y * satellite_height)
        lon, lat = proj(xx, yy, inverse=True)
        # Handle out of disk values
        lon = np.where(lon < 400, lon, np.nan)
        lat = np.where(lat < 100, lat, np.nan)
        return lat.astype(np.float32), lon.astype(np.float32)

    # Use standardized approach: unified apply_ufunc for both backends.
    # We use allow_rechunk=True to handle cases where x/y are chunked, as they are
    # core dimensions for the projection calculation.
    lat, lon = xr.apply_ufunc(
        _calc_latlon,
        ds.x,
        ds.y,
        input_core_dims=[["x"], ["y"]],
        output_core_dims=[["y", "x"], ["y", "x"]],
        dask="parallelized",
        output_dtypes=[np.float32, np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    ds = ds.assign_coords(
        latitude=lat.assign_attrs({"units": "degrees_north", "standard_name": "latitude"}),
        longitude=lon.assign_attrs({"units": "degrees_east", "standard_name": "longitude"}),
    )

    return ds
