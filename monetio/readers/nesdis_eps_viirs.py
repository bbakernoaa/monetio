"""NESDIS EPS VIIRS Reader"""

import os
import xarray as xr
from .base import GriddedReader, register_reader

@register_reader("nesdis_eps_viirs")
class NESDISEPSVIIRSReader(GriddedReader):
    def open_dataset(self,
                     date,
                     datapath=".",
                     **kwargs):
        """
        Reads NESDIS EPS VIIRS data (FTP download).
        """
        current = change_dir(datapath)
        nlat, nlon = 720, 1440
        lon, lat = _get_latlons(nlat, nlon)

        fname, date = download_data(date)
        data = read_data(fname, lat, lon, date)

        data = data.where(data > 0)
        os.chdir(current)
        return data

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/sat/nesdis_eps_viirs.py
# -----------------------------------------------------------------------------

server = "ftp.star.nesdis.noaa.gov"
base_dir = "/pub/smcd/VIIRS_Aerosol/npp.viirs.aerosol.data/epsaot550/"

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
    lats = linspace(lat_max, lat_min, nlat)
    lon, lat = meshgrid(lons, lats)
    return lon, lat

def download_data(date):
    import ftplib
    from pandas import Timestamp

    date = Timestamp(date)
    year = date.strftime("%Y")
    yyyymmdd = date.strftime("%Y%m%d")

    file = f"npp_eaot_ip_gridded_0.25_{yyyymmdd}.high.nc"
    if not os.path.isfile(file):
        ftp = ftplib.FTP(server)
        ftp.login()
        ftp.cwd(base_dir + year)
        ftp.retrbinary("RETR " + file, open(file, "wb").write)
    else:
        print(f"File Already Exists! Reading: {file}")
    return file, date

def read_data(fname, lat, lon, date):
    from pandas import to_datetime

    # We use xr.open_dataset directly as it is netcdf
    f = xr.open_dataset(fname)
    datearr = to_datetime([date])
    da = f["aot_ip_out"]
    da = da.rename({"nlat": "y", "nlon": "x"})
    da["latitude"] = (("y", "x"), lat)
    da["longitude"] = (("y", "x"), lon)
    da = da.expand_dims("time")
    da["time"] = datearr
    da.attrs["units"] = ""
    da.name = "VIIRS EPS AOT"
    da.attrs["long_name"] = "Aerosol Optical Thickness"
    da.attrs["source"] = f"ftp://{server}{base_dir}"
    return da
