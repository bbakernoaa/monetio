"""NESDIS FRP Reader"""

import datetime
import os
from typing import List, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history

BASE_URL = "https://gsce-dtn.sdstate.edu/index.php/s/e8wPYPOL1bGXk5z/download?path=%2F"


@register_reader("nesdis_frp")
class NESDISFRPReader(GriddedReader):
    """
    Reader for NESDIS Fire Radiative Power (FRP) data on FV3 C384 grid.
    """

    def open_dataset(
        self,
        date: Union[datetime.datetime, str, pd.Timestamp],
        ftype: str = "meanFRP",
        datapath: str = ".",
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS FRP data.

        Parameters
        ----------
        date : datetime.datetime or str
            Date to retrieve.
        ftype : str, optional
            Type of FRP data (e.g., 'meanFRP'). Default is 'meanFRP'.
        datapath : str, optional
            Local path to store downloaded files. Default is '.'.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        xr.Dataset
            The NESDIS FRP dataset.
        """
        date = pd.Timestamp(date)

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        # Download tiles (1-6)
        files = self.download_data(date, ftype=ftype, datapath=datapath)

        das = []
        for i, fname in enumerate(files):
            tile_num = i + 1
            da = self.read_tile(fname, tile=tile_num)
            das.append(da)

        ds = xr.concat(das, dim="tile")
        ds = ds.assign_coords(tile=np.arange(1, 7), time=date).expand_dims("time")
        ds = ds.to_dataset(name=ftype)

        # Update history
        ds = update_history(ds, f"Read NESDIS {ftype} data.")

        return ds

    def download_data(
        self, date: pd.Timestamp, ftype: str = "meanFRP", datapath: str = "."
    ) -> List[str]:
        yyyymmdd = date.strftime("%Y%m%d")
        url_ftype = f"&files={ftype}."

        files = []
        for i in range(1, 7):
            tile = f".FV3C384Grid.tile{i}.bin"
            url = f"{BASE_URL}{yyyymmdd}{url_ftype}{yyyymmdd}{tile}"
            filename = f"{ftype}.{yyyymmdd}.FV3.C384Grid.tile{i}.bin"
            filepath = os.path.join(datapath, filename)

            if not os.path.isfile(filepath):
                fs = FileUtility.get_fs(url)
                fs.get(url, filepath)

            files.append(filepath)

        return files

    def read_tile(
        self, fname: str, tile: int = 1, res: str = "C384", dtype: str = "f4"
    ) -> xr.DataArray:
        from ..util import _import_required

        scipy_io = _import_required("scipy.io")
        FortranFile = scipy_io.FortranFile

        try:
            import fv3grid as fg

            has_fv3grid = True
        except ImportError:
            has_fv3grid = False

        def wrap_longitudes(lon):
            return (lon + 180) % 360 - 180

        with open(fname, "rb") as f:
            w = FortranFile(f)
            a = w.read_reals(dtype=dtype)

        r = int(res[1:])
        s = a.reshape((r, r), order="F")

        if has_fv3grid:
            grid = fg.get_fv3_grid(res=res, tile=tile)
            grid["longitude"] = wrap_longitudes(grid.longitude)
            da = xr.DataArray(s, dims=("x", "y"), coords=grid.coords)
            return da
        else:
            return xr.DataArray(s, dims=("x", "y"))
