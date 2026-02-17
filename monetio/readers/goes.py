"""GOES Reader"""

from datetime import datetime
from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("goes")
class GOESReader(GriddedReader):
    """
    Reader for GOES-16/17/18 NetCDF data from Amazon S3 or local files.
    """

    def open_dataset(
        self,
        date: Optional[Union[pd.Timestamp, str, datetime]] = None,
        files: Optional[Union[str, List[str]]] = None,
        satellite: str = "16",
        product: str = "ABI-L2-AODF",
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads GOES data (S3 or local).

        Parameters
        ----------
        date : Union[pd.Timestamp, str, datetime], optional
            Target date for S3 retrieval. Closest file will be selected.
        files : Union[str, List[str]], optional
            Local file path(s). If provided, `date` and `satellite` are ignored
            for discovery but used for metadata.
        satellite : str, optional
            GOES satellite number ('16', '17', or '18'), by default "16".
        product : str, optional
            GOES product name, by default "ABI-L2-AODF".
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open (e.g., chunks).

        Returns
        -------
        xr.Dataset
            The loaded GOES dataset with lazy coordinates.
        """
        g = GOES()
        if files is None:
            # S3 mode
            if date is None or product is None:
                raise ValueError(
                    "Please provide a date and product to be able to retrieve data from Amazon S3"
                )
            ds = g.open_amazon_file(date=date, satellite=satellite, product=product, **kwargs)
        else:
            # Local mode
            ds = g.open_local(files, **kwargs)

        # Update history
        history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read GOES data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


class GOES:
    """Helper class for GOES data discovery and lazy grid generation."""

    def __init__(self) -> None:
        self.date: Optional[pd.Timestamp] = None
        self.satellite = "16"
        self.product = "ABI-L2-AODF"
        self.baseurl = f"s3://noaa-goes{self.satellite}/"
        self.url = f"{self.baseurl}"

    def _update_baseurl(self) -> None:
        self.baseurl = f"s3://noaa-goes{self.satellite}/"

    def get_products(self, fs: Any) -> List[str]:
        """Get available products for the satellite."""
        products = [value.rstrip("/").split("/")[-1] for value in fs.ls(self.baseurl)]
        return products

    def date_to_url(self) -> None:
        """Construct the S3 URL path for the given date."""
        if self.date is not None:
            date_url_bit = self.date.strftime("%Y/%j/%H/")
            self.url = f"{self.url}{date_url_bit}"

    def _get_files(self, fs: Any, url: str) -> List[str]:
        """List files in the S3 directory."""
        try:
            files = fs.ls(url)
            if not files:
                raise ValueError
            return files
        except (ValueError, FileNotFoundError):
            print(f"Files not available for product and date at {url}")
            return []

    def _get_closest_date(self, files: List[str]) -> Optional[str]:
        """Find the file closest to the target date."""
        if not files or self.date is None:
            return None
        # Example filename: OR_ABI-L2-AODF-M6_G16_s20230011200000_e20230011209308_c20230011215100.nc
        try:
            file_dates = [
                pd.to_datetime(f.split("_")[-1].split(".")[0][1:], format="%Y%j%H%M%S")
                for f in files
            ]
            nearest_idx = abs(pd.Series(file_dates) - self.date).idxmin()
            return files[nearest_idx]
        except Exception:
            return files[0]  # Fallback to first

    def _product_exists(self, fs: Any, product: str) -> Optional[str]:
        """Verify if the product exists on S3."""
        try:
            products = self.get_products(fs)
            if product not in products:
                raise ValueError
            return product
        except ValueError:
            print(f"Product '{product}' not found for GOES-{self.satellite}")
            return None

    def open_amazon_file(
        self,
        date: Union[pd.Timestamp, str, datetime],
        product: str = "ABI-L2-AODF",
        satellite: str = "16",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open a file from Amazon S3."""
        import s3fs

        self.date = pd.Timestamp(date)
        self.satellite = satellite
        self._update_baseurl()
        fs = s3fs.S3FileSystem(anon=True)
        self.product = self._product_exists(fs, product)
        if not self.product:
            return xr.Dataset()

        self.url = f"{self.baseurl}{self.product}/"
        self.date_to_url()

        files = self._get_files(fs, self.url)
        f = self._get_closest_date(files)
        if not f:
            return xr.Dataset()

        # We pass the S3 URL to Xarray via the driver
        from .drivers import XarrayDriver

        driver = XarrayDriver()
        if not f.startswith("s3://"):
            f = f"s3://{f}"

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = driver.open(f, **kwargs)
        ds = self._get_grid(ds)
        return ds

    def _get_grid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Calculate latitude and longitude lazily from the GOES projection.

        Parameters
        ----------
        ds : xr.Dataset
            Input GOES dataset.

        Returns
        -------
        xr.Dataset
            Dataset with lazy 'latitude' and 'longitude' coordinates.
        """
        import numpy as np
        from pyproj import CRS, Proj

        if "goes_imager_projection" not in ds:
            return ds

        proj_attrs = ds.goes_imager_projection.attrs.copy()
        for k, v in proj_attrs.items():
            if isinstance(v, (np.ndarray, list, tuple)):
                proj_attrs[k] = v[0]

        try:
            crs = CRS.from_cf(proj_attrs)
            proj = Proj(crs)
            ds.attrs["projection"] = crs.to_wkt()
        except Exception:
            return ds

        satellite_height = ds.goes_imager_projection.perspective_point_height

        # Ensure we stay lazy if the dataset is chunked
        x = ds.x
        y = ds.y

        # Define scalar functions for lon/lat to use with vectorize=True
        def _get_lon_scalar(x_rad, y_rad):
            # Proj returns a tuple, we take the first one (longitude)
            lon, _ = proj(x_rad * satellite_height, y_rad * satellite_height, inverse=True)
            return lon if lon < 400 else np.nan

        def _get_lat_scalar(x_rad, y_rad):
            # Proj returns a tuple, we take the second one (latitude)
            _, lat = proj(x_rad * satellite_height, y_rad * satellite_height, inverse=True)
            return lat if lat < 100 else np.nan

        # Use apply_ufunc with vectorize=True and dask='parallelized'
        ds = ds.assign_coords(
            longitude=xr.apply_ufunc(
                _get_lon_scalar,
                x,
                y,
                dask="parallelized",
                output_dtypes=[float],
                vectorize=True,
            ),
            latitude=xr.apply_ufunc(
                _get_lat_scalar,
                x,
                y,
                dask="parallelized",
                output_dtypes=[float],
                vectorize=True,
            ),
        )

        ds = ds.set_coords(["latitude", "longitude"])
        ds.latitude.attrs.update({"units": "degrees_north", "standard_name": "latitude"})
        ds.longitude.attrs.update({"units": "degrees_east", "standard_name": "longitude"})

        return ds

    def open_local(self, files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
        """Open local file(s)."""
        from .drivers import XarrayDriver

        driver = XarrayDriver()
        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = driver.open(files, **kwargs)
        ds = self._get_grid(ds)
        return ds
