import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from monetio.readers.fv3chem import (
    _calc_nemsio_hgt,
    _fix_grib2,
    _fix_time_nemsio,
    _rename_func,
)


def test_fix_time_nemsio_eager():
    # Mock dataset
    time = pd.to_datetime(["2023-01-01 00:00:00"])
    ds = xr.Dataset({"time": (("time",), time)})
    ds = ds.set_coords("time")

    # Single file
    fname = "aqm.t12z.atmf003.nemsio.nc"
    ds_fixed = _fix_time_nemsio(ds.copy(), fname)
    expected = pd.to_datetime("2023-01-01 03:00:00")
    assert ds_fixed.time.values[0] == expected


def test_fix_time_nemsio_lazy():
    # Mock dataset with a data variable that is lazy, since index coordinates aren't easily lazy
    time = pd.to_datetime(["2023-01-01 00:00:00"])
    ds = xr.Dataset(
        {"some_var": (("time",), da.from_array([1.0], chunks=1))},
        coords={"time": (("time",), time)},
    )
    # Even if 'time' is not dask-backed, we can check if it was transformed correctly

    # Single file
    fname = "aqm.t12z.atmf003.nemsio.nc"
    ds_fixed = _fix_time_nemsio(ds, fname)

    expected = pd.to_datetime("2023-01-01 03:00:00")
    assert ds_fixed.time.values[0] == expected
    assert isinstance(ds_fixed.some_var.data, da.Array)


def test_fix_time_nemsio_multi_lazy():
    # Multi-file case
    time = pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:00:00"])
    ds = xr.Dataset(
        {"some_var": (("time",), da.from_array([1.0, 2.0], chunks=1))},
        coords={"time": (("time",), time)},
    )

    fnames = ["aqm.t12z.atmf003.nemsio.nc", "aqm.t12z.atmf006.nemsio.nc"]
    ds_fixed = _fix_time_nemsio(ds, fnames)

    assert ds_fixed.time.values[0] == pd.to_datetime("2023-01-01 03:00:00")
    assert ds_fixed.time.values[1] == pd.to_datetime("2023-01-01 06:00:00")
    assert isinstance(ds_fixed.some_var.data, da.Array)


def test_calc_nemsio_hgt_lazy():
    ds = xr.Dataset(
        {
            "hgtsfc": (("y", "x"), da.from_array(np.ones((2, 2)), chunks=2)),
            "delz": (("z", "y", "x"), da.from_array(np.ones((3, 2, 2)), chunks=(1, 2, 2))),
        }
    )

    hgt = _calc_nemsio_hgt(ds)

    assert isinstance(hgt.data, da.Array)
    # Level 0: sfc + dz[0] = 1 + 1 = 2
    # Level 1: sfc + dz[0] + dz[1] = 1 + 1 + 1 = 3
    # Level 2: sfc + dz[0] + dz[1] + dz[2] = 1 + 1 + 1 + 1 = 4
    np.testing.assert_array_equal(hgt.isel(y=0, x=0).values, [2, 3, 4])


def test_rename_func():
    ds = xr.Dataset(
        {
            "o3midlayer": (("z", "y", "x"), np.zeros((3, 2, 2))),
            "pp25": (("z", "y", "x"), np.zeros((3, 2, 2))),
            "other": (("z", "y", "x"), np.zeros((3, 2, 2))),
        }
    )

    ds_renamed = _rename_func(ds, {"other": "renamed_other"})

    assert "o3" in ds_renamed.data_vars
    assert "o3midlayer" not in ds_renamed.data_vars
    assert "pm25" in ds_renamed.data_vars
    assert "pp25" not in ds_renamed.data_vars
    assert "renamed_other" in ds_renamed.data_vars


def test_fix_grib2_lazy():
    # 1D coordinates
    ds = xr.Dataset(
        {
            "latitude": (("latitude",), [10.0, 20.0]),
            "longitude": (("longitude",), [100.0, 110.0, 120.0]),
            "some_var": (("latitude", "longitude"), da.from_array(np.zeros((2, 3)), chunks=1)),
        }
    )

    ds = ds.set_coords(["latitude", "longitude"])

    ds_fixed = _fix_grib2(ds)

    assert "y" in ds_fixed.dims
    assert "x" in ds_fixed.dims
    assert ds_fixed.latitude.ndim == 2
    assert ds_fixed.longitude.ndim == 2
    # Broadcast of non-dask arrays in xarray results in non-dask arrays
    # unless the input was already dask. 1D coords are usually not dask.
    # But some_var should still be dask.
    assert isinstance(ds_fixed.some_var.data, da.Array)

    # Values check
    assert ds_fixed.latitude.isel(y=0, x=0) == 10.0
    assert ds_fixed.latitude.isel(y=1, x=0) == 20.0
    assert ds_fixed.longitude.isel(y=0, x=1) == 110.0
