import monetio.sat.tropomi_l2_no2 as tropomi
import xarray as xr
import numpy as np

# Existing code uses netCDF4.Dataset or h5netcdf.legacyapi.Dataset
# We want to use xr.open_dataset(fname, group=..., engine="h5netcdf")

# Let's look at the functions we need to change.
