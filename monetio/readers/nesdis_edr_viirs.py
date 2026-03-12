"""NESDIS EDR VIIRS Reader"""

import datetime
import os
from typing import Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history

SERVER = "ftp.star.nesdis.noaa.gov"
BASE_DIR = "/pub/smcd/jhuang/npp.viirs.aerosol.data/edraot550/"


@register_reader("nesdis_edr_viirs")
class NESDISEDRVIIRSReader(GriddedReader):
    """
    Reader for NESDIS EDR VIIRS gridded AOD data.
    Available via FTP.
    """

    def open_dataset(
        self,
        date: Union[datetime.datetime, str, pd.Timestamp],
        resolution: str = "high",
        datapath: str = ".",
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS EDR VIIRS data.

        Parameters
        ----------
        date : datetime.datetime, str, or pd.Timestamp
            Date to retrieve.
        resolution : str, optional
            'high' (0.10 deg) or 'low' (0.25 deg). Default is 'high'.
        datapath : str, optional
            Local path to store downloaded files. Default is '.'.
        lazy : bool, optional
            Whether to read data lazily using Dask, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        xr.Dataset
            The NESDIS EDR VIIRS dataset.
        """
        date = pd.Timestamp(date)

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        # Download
        fname = self.download_data(date, resolution=resolution, datapath=datapath)

        # Unzip
        unzipped_fname = self._unzip_file(fname)

        # Read
        ds = self.read_data(unzipped_fname, date, resolution=resolution, lazy=lazy)

        # Update history
        ds = update_history(ds, "Read NESDIS EDR VIIRS data.")

        return ds

    def download_data(
        self, date: pd.Timestamp, resolution: str = "high", datapath: str = "."
    ) -> str:
        """
        Download NESDIS EDR VIIRS data from FTP.

        Parameters
        ----------
        date : pd.Timestamp
            Date to download.
        resolution : str, optional
            'high' or 'low', by default "high".
        datapath : str, optional
            Local directory to save files, by default ".".

        Returns
        -------
        str
            Path to the downloaded file.
        """
        import ftplib

        year = date.strftime("%Y")
        yyyymmdd = date.strftime("%Y%m%d")

        if resolution in {"high", "h", "0.10"}:
            filename = f"npp_aot550_edr_gridded_0.10_{yyyymmdd}.high.bin.gz"
        else:
            filename = f"npp_aot550_edr_gridded_0.25_{yyyymmdd}.high.bin.gz"

        filepath = os.path.join(datapath, filename)

        if not os.path.isfile(filepath):
            ftp = ftplib.FTP(SERVER)
            ftp.login()
            ftp.cwd(BASE_DIR + year)
            with open(filepath, "wb") as f:
                ftp.retrbinary("RETR " + filename, f.write)
            ftp.quit()

        return filepath

    def _unzip_file(self, fname: str) -> str:
        """
        Unzip .gz file.

        Parameters
        ----------
        fname : str
            Input filename.

        Returns
        -------
        str
            Unzipped filename.
        """
        import gzip
        import shutil

        if not fname.endswith(".gz"):
            return fname

        out_fname = fname[:-3]
        if not os.path.isfile(out_fname):
            with gzip.open(fname, "rb") as f_in:
                with open(out_fname, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return out_fname

    def read_data(
        self, fname: str, date: pd.Timestamp, resolution: str = "high", lazy: bool = False
    ) -> xr.Dataset:
        """
        Read NESDIS EDR VIIRS binary data into an xarray.Dataset.

        Parameters
        ----------
        fname : str
            Path to the binary file.
        date : pd.Timestamp
            The date associated with the data.
        resolution : str, optional
            'high' or 'low', by default "high".
        lazy : bool, optional
            Whether to use Dask for lazy loading, by default False.

        Returns
        -------
        xr.Dataset
            The dataset containing AOD and coordinates.
        """
        if resolution in {"high", "h", "0.10"}:
            nlat, nlon = 1800, 3600
        else:
            nlat, nlon = 720, 1440

        if lazy:
            import dask.array as da
            from dask import delayed

            @delayed
            def load_binary(f):
                data = np.fromfile(f, dtype=np.float32)
                # Binary file contains 2 layers, first is AOD
                return data.reshape(2, nlat, nlon)[0, :, :]

            aot = da.from_delayed(load_binary(fname), shape=(nlat, nlon), dtype=np.float32)
            # Replace invalid values lazily
            aot = da.where(aot < -999, np.nan, aot)
        else:
            f = np.fromfile(fname, dtype=np.float32)
            aot = f.reshape(2, nlat, nlon)[0, :, :]
            aot[aot < -999] = np.nan

        # Generate lat/lon coords (1D for laziness)
        lons = np.linspace(-179.875, 179.875, nlon)
        lats = np.linspace(-89.875, 89.875, nlat)

        ds = xr.Dataset(
            data_vars={"aod_550": (("y", "x"), aot)},
            coords={
                "time": date,
                "latitude": (("y",), lats),
                "longitude": (("x",), lons),
            },
        )

        ds = ds.expand_dims("time")

        # Broadcast to 2D coordinates for backward compatibility with previous reader output
        # Ensuring coordinates are (y, x)
        lat2d, lon2d = xr.broadcast(ds.latitude, ds.longitude)
        ds.coords["latitude"] = lat2d
        ds.coords["longitude"] = lon2d

        # Re-order dimensions to (time, y, x) for consistency
        ds = ds.transpose("time", "y", "x")

        # Metadata
        ds.aod_550.attrs.update(
            {
                "long_name": "Aerosol Optical Thickness at 550nm",
                "units": "1",
                "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol",
            }
        )
        ds.latitude.attrs.update({"units": "degrees_north", "standard_name": "latitude"})
        ds.longitude.attrs.update({"units": "degrees_east", "standard_name": "longitude"})

        ds.attrs["source"] = f"ftp://{SERVER}{BASE_DIR}"

        # Update history
        ds = update_history(ds, "Modernized binary reading via Aero Protocol.")

        return ds
