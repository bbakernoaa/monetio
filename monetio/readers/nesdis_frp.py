"""NESDIS FRP Reader"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility


@register_reader("nesdis_frp")
class NESDISFRPReader(GriddedReader):
    def open_dataset(self, date, ftype="meanFRP", datapath=".", **kwargs):
        """
        Reads NESDIS FRP data (Download + Binary Read).
        """
        try:
            from scipy.io import FortranFile
        except ImportError:
            raise ImportError("scipy is required to read NESDIS FRP files")

        current = os.getcwd()
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        os.chdir(datapath)

        try:
            files = download_data(date, ftype=ftype)

            das = []
            for i, fname in enumerate(files):
                tile_num = i + 1
                da = read_tile(fname, tile=tile_num)
                das.append(da)

            ds = xr.concat(das, dim="tile")
            ds["tile"] = np.arange(1, 7)

            ds.name = ftype
            ds = ds.to_dataset()

        finally:
            os.chdir(current)

        return ds


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

base_dir = "https://gsce-dtn.sdstate.edu/index.php/s/e8wPYPOL1bGXk5z/download?path=%2F"


def download_data(date, ftype="meanFRP"):
    if isinstance(date, pd.Timestamp):
        yyyymmdd = date.strftime("%Y%m%d")
    else:
        date = pd.Timestamp(date)
        yyyymmdd = date.strftime("%Y%m%d")

    url_ftype = f"&files={ftype}."

    files = []
    for i in range(1, 7):
        tile = f".FV3C384Grid.tile{i}.bin"
        url = f"{base_dir}{yyyymmdd}{url_ftype}{yyyymmdd}{tile}"
        fname = f"{ftype}.{yyyymmdd}.FV3.C384Grid.tile{i}.bin"

        fs = FileUtility.get_fs(url)
        if not os.path.isfile(fname):
            print("Retrieving file:", fname)
            fs.get(url, fname)
        else:
            print("File exists:", fname)

        files.append(fname)

    return files


def read_tile(fname, tile=1, res="C384", dtype="f4"):
    from scipy.io import FortranFile

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
