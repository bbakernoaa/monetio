"""NESDIS EDR VIIRS Reader"""

import os
import xarray as xr
from .base import GriddedReader, register_reader

@register_reader("nesdis_edr_viirs")
class NESDISEDRVIIRSReader(GriddedReader):
    def open_dataset(self,
                     date,
                     resolution="high",
                     datapath=".",
                     **kwargs):
        """
        Reads NESDIS EDR VIIRS data (FTP download).
        """
        current = change_dir(datapath)
        if resolution in {"high", "h"}:
            nlat, nlon = 1800, 3600
            lon, lat = _get_latlons(nlat, nlon)
            fname, date = download_data(date, resolution="high")
        else:
            nlat, nlon = 720, 1440
            lon, lat = _get_latlons(nlat, nlon)
            fname, date = download_data(date, resolution=0.25)

        fname = _unzip_file(fname)
        data = read_data(fname, lat, lon, date)

        # Cleanup? Original code unzips but doesn't delete.

        # Restore dir?
        # change_dir is context manager like
        os.chdir(current)

        return data

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/sat/nesdis_edr_viirs.py
# -----------------------------------------------------------------------------

server = "ftp.star.nesdis.noaa.gov"
base_dir = "/pub/smcd/jhuang/npp.viirs.aerosol.data/edraot550/"

def change_dir(to_path):
    current = os.getcwd()
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    os.chdir(to_path)
    return current

def _get_latlons(nlat, nlon):
    from numpy import linspace, meshgrid
    lon_min = -179.875
    lon_max = -1 * lon_min
    lat_min = -89.875
    lat_max = -1.0 * lat_min
    lons = linspace(lon_min, lon_max, nlon)
    lats = linspace(lat_min, lat_max, nlat)
    lon, lat = meshgrid(lons, lats)
    return lon, lat

def download_data(date, resolution="high"):
    import ftplib
    from pandas import Timestamp

    date = Timestamp(date)
    year = date.strftime("%Y")
    yyyymmdd = date.strftime("%Y%m%d")

    if resolution == "high":
        file = f"npp_aot550_edr_gridded_0.10_{yyyymmdd}.high.bin.gz"
    else:
        file = f"npp_aot550_edr_gridded_0.25_{yyyymmdd}.high.bin.gz"

    ftp = ftplib.FTP(server)
    ftp.login()
    ftp.cwd(base_dir + year)

    if not os.path.isfile(file):
        ftp.retrbinary("RETR " + file, open(file, "wb").write)

    return file, date

def _unzip_file(fname):
    import gzip
    import shutil

    # Pythonic unzip to avoid subprocess gunzip dependency
    out_fname = fname[:-3]
    if not os.path.isfile(out_fname):
        with gzip.open(fname, 'rb') as f_in:
            with open(out_fname, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return out_fname

def read_data(fname, lat, lon, date):
    from numpy import float32, fromfile, nan
    from pandas import to_datetime

    f = fromfile(fname, dtype=float32)
    nlat, nlon = lon.shape
    aot = f.reshape(2, nlat, nlon)[0, :, :].reshape(1, nlat, nlon)
    aot[aot < -999] = nan
    datearr = to_datetime([date])
    da = xr.DataArray(aot, coords=[datearr, range(nlat), range(nlon)], dims=["time", "y", "x"])
    da["latitude"] = (("y", "x"), lat)
    da["longitude"] = (("y", "x"), lon)
    da.attrs["units"] = ""
    da.name = "VIIRS EDR AOD"
    da.attrs["long_name"] = "Aerosol Optical Depth"
    da.attrs["source"] = f"ftp://{server}{base_dir}"
    return da
