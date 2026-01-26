"""GOES Reader"""

import pandas as pd
import s3fs
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("goes")
class GOESReader(GriddedReader):
    def open_dataset(self, date=None, filename=None, satellite="16", product=None, **kwargs):
        """
        Reads GOES data (S3 or local).
        """
        g = GOES()
        if filename is None:
            # S3 mode
            if date is None or product is None:
                raise ValueError("Please provide a date and product to be able to retrieve data from Amazon S3")
            ds = g.open_amazon_file(date=date, satellite=satellite, product=product)
        else:
            # Local mode
            ds = g.open_local(filename)

        return ds


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/sat/goes.py
# -----------------------------------------------------------------------------


class GOES:
    def __init__(self):
        self.date = None
        self.satellite = "16"
        self.product = "ABI-L2-AODF"
        self.baseurl = f"s3://noaa-goes{self.satellite}/"
        self.url = f"{self.baseurl}"
        self.filename = None
        self.fs = None

    def _update_baseurl(self):
        self.baseurl = f"s3://noaa-goes{self.satellite}/"

    def get_products(self):
        # Requires s3fs
        if self.fs is None:
            self._set_s3fs()
        products = [value.split("/")[-1] for value in self.fs.ls(self.baseurl)[:-1]]
        return products

    def date_to_url(self):
        date = pd.Timestamp(self.date)
        date_url_bit = date.strftime("%Y/%j/%H/")
        self.url = f"{self.url}{date_url_bit}"

    def _get_files(self, url=None):
        try:
            files = self.fs.ls(url)
            if len(files) < 1:
                raise ValueError
            else:
                return files
        except ValueError:
            print("Files not available for product and date")
            return []

    def _get_closest_date(self, files=None):
        if files is None:
            files = []
        if not files:
            return None
        file_dates = [pd.to_datetime(f.split("_")[-1][:-4], format="c%Y%j%H%M%S") for f in files]
        date = pd.Timestamp(self.date)
        nearest_date = min(file_dates, key=lambda x: abs(x - date))
        nearest_date_str = nearest_date.strftime("c%Y%j%H%M%S")
        found_file = [f for f in files if nearest_date_str in f][0]
        return found_file

    def _set_s3fs(self):
        self.fs = s3fs.S3FileSystem(anon=True)

    def _product_exists(self, product):
        try:
            if self.fs is None:
                self._set_s3fs()
            products = self.get_products()
            if product not in products:
                raise ValueError
            else:
                return product
        except ValueError:
            print("Product: ", product, "not found")
            return None

    def open_amazon_file(self, date=None, product=None, satellite="16"):
        self.date = pd.Timestamp(date)
        self.satellite = satellite
        self._update_baseurl()
        self._set_s3fs()
        self.product = self._product_exists(product)
        if not self.product:
            return xr.Dataset()

        self.url = f"{self.baseurl}{self.product}/"
        self.date_to_url()

        files = self._get_files(url=self.url)
        f = self._get_closest_date(files=files)
        if not f:
            return xr.Dataset()

        # s3fs open returns a file-like object
        fo = self.fs.open(f)
        out = xr.open_dataset(fo, engine="h5netcdf")
        out = self._get_grid(out)
        return out

    def _get_grid(self, ds):
        from numpy import meshgrid, ndarray
        from pyproj import CRS, Proj

        proj_dict = ds.goes_imager_projection.attrs
        for i in proj_dict.keys():
            if isinstance(proj_dict[i], ndarray):
                proj_dict[i] = proj_dict[i][0]
        crs = CRS.from_cf(proj_dict)
        ds.attrs["projection"] = crs.to_wkt()
        proj = Proj(crs)
        satellite_height = ds.goes_imager_projection.perspective_point_height
        xx, yy = meshgrid(ds.x.values * satellite_height, ds.y.values * satellite_height)
        lon, lat = proj(xx, yy, inverse=True)
        ds["latitude"] = (("y", "x"), lat)
        ds["longitude"] = (("y", "x"), lon)
        ds["longitude"] = ds.longitude.where(ds.longitude < 400).fillna(1e30)
        ds["latitude"] = ds.latitude.where(ds.latitude < 100).fillna(1e30)
        ds = ds.set_coords(["latitude", "longitude"])
        return ds

    def open_local(self, f):
        # Local file
        # We can use FileUtility.get_fs logic if we want to be consistent, but original used direct s3fs for remote
        # For local, assume f is path
        from .drivers import FileUtility

        fs = FileUtility.get_fs(f)
        fo = fs.open(f)
        out = xr.open_dataset(fo, engine="h5netcdf")
        out = self._get_grid(out)
        return out
