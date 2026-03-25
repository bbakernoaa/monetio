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
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS FRP data.

        Parameters
        ----------
        date : datetime.datetime, str, or pd.Timestamp
            Date to retrieve.
        ftype : str, optional
            Type of FRP data (e.g., 'meanFRP'). Default is 'meanFRP'.
        datapath : str, optional
            Local path to store downloaded files. Default is '.'.
        lazy : bool, optional
            Whether to read data lazily using Dask, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        xr.Dataset
            The NESDIS FRP dataset.

        Examples
        --------
        >>> reader = NESDISFRPReader()
        >>> ds = reader.open_dataset("2023-01-01", ftype="meanFRP")
        """
        date = pd.Timestamp(date)

        if not os.path.exists(datapath):
            os.makedirs(datapath, exist_ok=True)

        # Download tiles (1-6)
        files = self.download_data(date, ftype=ftype, datapath=datapath)

        das = []
        for i, fname in enumerate(files):
            tile_num = i + 1
            da = self.read_tile(fname, tile=tile_num, lazy=lazy)
            das.append(da)

        ds = xr.concat(das, dim="tile")
        ds = ds.assign_coords(tile=np.arange(1, 7), time=date).expand_dims("time")
        ds = ds.to_dataset(name=ftype)

        # Scientific Hygiene: Coordinate standardization
        if "longitude" in ds.coords:
            ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})
        if "latitude" in ds.coords:
            ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})

        # Update history
        ds = update_history(ds, f"Read NESDIS {ftype} data from {len(files)} tiles.")

        return ds

    def download_data(
        self, date: pd.Timestamp, ftype: str = "meanFRP", datapath: str = "."
    ) -> List[str]:
        """
        Download NESDIS FRP data from the GSCE server.

        Parameters
        ----------
        date : pd.Timestamp
            Date to download.
        ftype : str, optional
            File type (e.g., 'meanFRP'), by default "meanFRP".
        datapath : str, optional
            Local directory to save files, by default ".".

        Returns
        -------
        List[str]
            List of paths to the downloaded files.
        """
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
        self,
        fname: str,
        tile: int = 1,
        res: str = "C384",
        dtype: str = "f4",
        lazy: bool = False,
    ) -> xr.DataArray:
        """
        Read a single NESDIS FRP tile from a binary file.

        Parameters
        ----------
        fname : str
            Path to the binary file.
        tile : int, optional
            Tile number (1-6), by default 1.
        res : str, optional
            Grid resolution, by default "C384".
        dtype : str, optional
            Data type in the binary file, by default "f4".
        lazy : bool, optional
            Whether to use Dask for lazy loading, by default False.

        Returns
        -------
        xr.DataArray
            The tile data with coordinates if fv3grid is available.
        """
        r = int(res[1:])
        shape = (r, r)

        if lazy:
            import dask.array as da
            from dask import delayed

            # Define delayed reader
            load_tile = delayed(_read_binary_tile)(fname, res, dtype)
            data = da.from_delayed(load_tile, shape=shape, dtype=np.dtype(dtype))
        else:
            data = _read_binary_tile(fname, res, dtype)

        # Handle Grid and Coordinates
        try:
            import fv3grid as fg

            grid = fg.get_fv3_grid(res=res, tile=tile)
            # Wrap longitudes to [-180, 180]
            lon = (grid.longitude + 180) % 360 - 180
            lat = grid.latitude
            coords = {"latitude": (("x", "y"), lat), "longitude": (("x", "y"), lon)}
        except ImportError:
            coords = None

        if coords:
            da = xr.DataArray(data, dims=("x", "y"), coords=coords)
        else:
            da = xr.DataArray(data, dims=("x", "y"))

        # Update history on the DataArray if possible
        da = update_history(da, f"Read tile {tile} from {fname} (lazy={lazy}).")

        return da


def _read_binary_tile(fname: str, res: str, dtype: str) -> np.ndarray:
    """
    Core binary reading logic for a single tile.

    Parameters
    ----------
    fname : str
        File path.
    res : str
        Grid resolution (e.g., 'C384').
    dtype : str
        Numpy dtype string.

    Returns
    -------
    np.ndarray
        Reshaped data array.
    """
    from scipy.io import FortranFile

    r = int(res[1:])
    with open(fname, "rb") as f:
        w = FortranFile(f)
        a = w.read_reals(dtype=dtype)

    return a.reshape((r, r), order="F")
