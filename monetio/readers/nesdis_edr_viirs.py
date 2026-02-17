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
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads NESDIS EDR VIIRS data.

        Parameters
        ----------
        date : datetime.datetime or str
            Date to retrieve.
        resolution : str, optional
            'high' (0.10 deg) or 'low' (0.25 deg). Default is 'high'.
        datapath : str, optional
            Local path to store downloaded files. Default is '.'.
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
        ds = self.read_data(unzipped_fname, date, resolution=resolution)

        # Update history
        ds = update_history(ds, "Read NESDIS EDR VIIRS data.")

        return ds

    def download_data(self, date: pd.Timestamp, resolution: str = "high", datapath: str = ".") -> str:
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

    def read_data(self, fname: str, date: pd.Timestamp, resolution: str = "high") -> xr.Dataset:
        if resolution in {"high", "h", "0.10"}:
            nlat, nlon = 1800, 3600
        else:
            nlat, nlon = 720, 1440

        # Generate lat/lon
        lons = np.linspace(-179.875, 179.875, nlon)
        lats = np.linspace(-89.875, 89.875, nlat)
        lon2d, lat2d = np.meshgrid(lons, lats)

        f = np.fromfile(fname, dtype=np.float32)
        # Binary file contains 2 layers, first is AOD
        aot = f.reshape(2, nlat, nlon)[0, :, :]
        aot[aot < -999] = np.nan

        ds = xr.Dataset(
            data_vars={"aod_550": (("y", "x"), aot)},
            coords={
                "time": [date],
                "latitude": (("y", "x"), lat2d),
                "longitude": (("y", "x"), lon2d),
            },
        )
        ds = ds.expand_dims("time")

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

        return ds
