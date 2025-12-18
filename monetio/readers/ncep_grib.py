"""NCEP GRIB Reader"""

from glob import glob

import xarray as xr
from numpy import sort

from .base import GriddedReader, register_reader


@register_reader("ncep_grib")
class NCEPGribReader(GriddedReader):
    def open_dataset(self, files, **kwargs):
        """
        Reads NCEP GRIB files using pynio (via cfgrib/xarray logic or custom engine).
        The original code used engine="pynio".
        """

        # Ensure we have engine='pynio' if not specified
        # Note: pynio is often deprecated/hard to install.
        # But we must preserve original behavior.
        if "engine" not in kwargs:
            kwargs["engine"] = "pynio"

        # Also supports open_mfdataset logic
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        return _fix_grib2(ds)


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/ncep_grib.py
# -----------------------------------------------------------------------------


def _fix_grib2(f):
    from numpy import meshgrid

    latitude = f.lat_0.values
    longitude = f.lon_0.values

    # Original logic replaces latitude/longitude with index range?
    # f['latitude'] = range(len(f.latitude))
    # f['longitude'] = range(len(f.longitude))
    # This suggests lat_0/lon_0 were 1D coordinate variables.

    # We rename to y, x
    # Rename lat_0 -> latitude (temporarily?) -> y

    # Let's follow original exactly
    # But f.lat_0 might not exist if opened via cfgrib instead of pynio.
    # But assuming pynio:

    # Original code:
    # latitude = f.lat_0.values
    # longitude = f.lon_0.values
    # f["latitude"] = range(len(f.latitude)) # This fails if f.latitude doesn't exist yet?
    # Actually original code:
    # f['latitude'] = range(len(f.latitude))
    # Wait, 'f.latitude' implies it exists?
    # Or maybe it meant range(len(latitude)) (the local var).
    # Assuming the local var.

    # Renaming
    if "lat_0" in f.coords:
        f = f.rename({"lat_0": "latitude", "lon_0": "longitude"})

    # Now f.latitude exists.
    # The original code reassigns f['latitude'] to indices.
    f["latitude"] = range(len(f.latitude))
    f["longitude"] = range(len(f.longitude))

    f = f.rename({"latitude": "y", "longitude": "x"})

    lon, lat = meshgrid(longitude, latitude)
    f["longitude"] = (("y", "x"), lon)
    f["latitude"] = (("y", "x"), lat)
    f = f.set_coords(["latitude", "longitude"])
    return f
