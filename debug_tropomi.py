import h5netcdf
import numpy as np
from monetio.util import get_nc_var, get_nc_values
from cftime import num2date
import xarray as xr

fname = 'tests/data/TROPOMI-L2-NO2-20190715.nc'
dso = h5netcdf.File(fname, "r")

try:
    lon_var = get_nc_var(dso, "PRODUCT", "longitude")
    lat_var = get_nc_var(dso, "PRODUCT", "latitude")

    print(f"lon_var: {lon_var}")

    ref_time_var = get_nc_var(dso, "PRODUCT", "time")
    print(f"ref_time_var: {ref_time_var}")

    val = get_nc_values(ref_time_var)
    print(f"ref_time val: {val}")

    dtime_var = get_nc_var(dso, "PRODUCT", "delta_time")
    print(f"dtime_var: {dtime_var}")

    dtime_val = get_nc_values(dtime_var)
    print(f"dtime val shape: {dtime_val.shape}")

    # Now the part that failed in the test
    group_name = "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"
    varname = "latitude_bounds"
    var = get_nc_var(dso, group_name, varname)
    print(f"var: {var}")
    values = get_nc_values(var)
    print(f"values shape: {values.shape}")

finally:
    dso.close()
