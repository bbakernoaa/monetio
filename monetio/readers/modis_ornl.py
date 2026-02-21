"""MODIS ORNL Reader"""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history

try:
    from suds.client import Client

    HAS_SUDS = True
except ImportError:
    HAS_SUDS = False

DEFAULT_WSDL = "https://modis.ornl.gov/cgi-bin/MODIS/soapservice/MODIS_soapservice.wsdl"


@register_reader("modis_ornl")
class MODISORNLReader(GriddedReader):
    """
    Reader for MODIS data from ORNL web service.
    """

    def open_dataset(
        self,
        date: Union[pd.Timestamp, str],
        product: str = "MOD12A2H",
        band: str = "Lai_500m",
        quality_control: Optional[Any] = None,
        latitude: float = 0,
        longitude: float = 0,
        kmAboveBelow: int = 100,
        kmLeftRight: int = 100,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads MODIS data from ORNL.

        Parameters
        ----------
        date : pd.Timestamp or str
            Date to retrieve.
        product : str, optional
            MODIS product.
        band : str, optional
            Product band.
        quality_control : optional
            Quality control filter.
        latitude : float, optional
            Center latitude.
        longitude : float, optional
            Center longitude.
        kmAboveBelow : int, optional
            Kilometers above/below center.
        kmLeftRight : int, optional
            Kilometers left/right center.

        Returns
        -------
        xr.Dataset
            The MODIS ORNL dataset.
        """
        if not HAS_SUDS:
            raise ImportError(
                "Please install a suds client (pip install suds-jurko or suds-community)"
            )

        date = pd.to_datetime(date)
        m = _get_single_retrieval(
            date,
            product=product,
            band=band,
            quality_control=quality_control,
            lat=latitude,
            lon=longitude,
            kmAboveBelow=kmAboveBelow,
            kmLeftRight=kmLeftRight,
        )

        ds = _make_xarray_dataset(m)

        # Update history
        ds = update_history(ds, f"Read MODIS ORNL {product} {band} data.")

        return ds


class MODISData:
    def __init__(self):
        self.server = None
        self.product = None
        self.latitude = None
        self.longitude = None
        self.band = None
        self.nrows = None
        self.ncols = None
        self.cellsize = None
        self.scale = None
        self.units = None
        self.yllcorner = None
        self.xllcorner = None
        self.kmAboveBelow = 0
        self.kmLeftRight = 0
        self.dateStr = []
        self.dateInt = []
        self.data = []
        self.isScaled = False

    def applyScale(self):
        if not self.isScaled:
            self.data = self.data * self.scale
            self.isScaled = True


def _nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def _get_single_retrieval(
    date, product, band, quality_control, lat, lon, kmAboveBelow, kmLeftRight
):
    client = Client(DEFAULT_WSDL)

    # Get available dates
    date_list = client.service.getdates(lat, lon, product)
    available_dates = pd.to_datetime(date_list, format="A%Y%j")

    # Find nearest date
    target_date = _nearest(available_dates, date)
    date_str = target_date.strftime("A%Y%j")

    # Fetch subset
    data = client.service.getsubset(
        lat, lon, product, band, date_str, date_str, kmAboveBelow, kmLeftRight
    )

    m = MODISData()
    m.server = DEFAULT_WSDL
    m.product = product
    m.band = band
    m.latitude = lat
    m.longitude = lon
    m.nrows = int(data.nrows)
    m.ncols = int(data.ncols)
    m.cellsize = data.cellsize
    m.scale = data.scale
    m.units = data.units
    m.yllcorner = data.yllcorner
    m.xllcorner = data.xllcorner
    m.dateInt = [int(target_date.strftime("%Y%j"))]

    # Parse data
    subset_data = data.subset[0].split(",")[5:]
    m.data = np.array([float(x) for x in subset_data]).reshape(1, -1)

    m.applyScale()
    return m


def _make_xarray_dataset(m) -> xr.Dataset:
    # Reshape and flip to match standard orientation if needed
    # The original code did: m.data.reshape(m.ncols, m.nrows, order='C')[::-1, :]
    # which resulted in (x, y) dims.
    grid_data = m.data.reshape(m.nrows, m.ncols)

    ds = xr.Dataset(
        data_vars={m.band: (("y", "x"), grid_data)},
        coords={"time": [pd.to_datetime(str(m.dateInt[0]), format="%Y%j")]},
    )
    ds = ds.expand_dims("time")

    # Add lat/lon
    lon, lat = _get_latlon(m.xllcorner, m.yllcorner, m.cellsize, m.ncols, m.nrows)
    ds = ds.assign_coords(
        latitude=(("y", "x"), lat, {"units": "degrees_north", "standard_name": "latitude"}),
        longitude=(("y", "x"), lon, {"units": "degrees_east", "standard_name": "longitude"}),
    )

    ds[m.band].attrs.update({"units": m.units, "long_name": m.band, "product": m.product})
    ds.attrs["server"] = m.server

    return ds


def _get_latlon(xll, yll, cell_width, nx, ny):
    from pyproj import Proj

    sinu = Proj("+proj=sinu +a=6371007.181 +b=6371007.181 +units=m +R=6371007.181")
    x = np.linspace(xll, xll + cell_width * nx, nx)
    y = np.linspace(yll, yll + cell_width * ny, ny)
    xx, yy = np.meshgrid(x, y)
    lon, lat = sinu(xx, yy, inverse=True)
    # The original code flipped y
    return lon[::-1, :], lat[::-1, :]
