import numpy as np
import pandas as pd
import xarray as xr

from monetio.readers.madis import MADISReader


def create_fake_madis(path):
    """Create a fake MADIS NetCDF file."""
    n_obs = 10
    ds = xr.Dataset(
        {
            "latitude": (("recNum",), np.linspace(30, 40, n_obs)),
            "longitude": (("recNum",), np.linspace(-100, -90, n_obs)),
            "observationTime": (("recNum",), np.arange(n_obs) * 3600.0 + 0.5),  # Add sub-second
            "stationId": (("recNum",), [f"S{i}" for i in range(n_obs)]),
            "temperature": (("recNum",), np.random.rand(n_obs) + 290.0),
        },
        coords={"recNum": np.arange(n_obs)},
    )
    ds["observationTime"].attrs["units"] = "seconds since 1970-01-01 00:00:00.0 +0000"
    ds.to_netcdf(path)


def test_madis_eager_lazy(tmp_path):
    fname = str(tmp_path / "test_madis.nc")
    create_fake_madis(fname)

    reader = MADISReader()

    # 1. Eager (NumPy)
    ds_eager = reader.open_dataset(fname, lazy=False, as_xarray=True)
    assert isinstance(ds_eager, xr.Dataset)
    assert not hasattr(ds_eager.temperature.data, "dask")
    assert "time" in ds_eager.coords
    assert ds_eager.time.dtype == "datetime64[ns]"

    # 2. Lazy (Dask)
    ds_lazy = reader.open_dataset(fname, lazy=True, as_xarray=True, chunks={"node": 5})
    assert isinstance(ds_lazy, xr.Dataset)
    assert hasattr(ds_lazy.temperature.data, "dask")

    # Verify values match
    xr.testing.assert_allclose(ds_eager.compute(), ds_lazy.compute())

    # Test DataFrame output
    df_eager = reader.open_dataset(fname, lazy=False, as_xarray=False)
    assert isinstance(df_eager, pd.DataFrame)

    df_lazy = reader.open_dataset(fname, lazy=True, as_xarray=False, chunks={"node": 5})
    # For MADIS, our implementation returns a dask dataframe if it's chunked
    import dask.dataframe as dd

    assert isinstance(df_lazy, dd.DataFrame)

    pd.testing.assert_frame_equal(
        df_eager.sort_values("siteid").reset_index(drop=True),
        df_lazy.compute()[df_eager.columns].sort_values("siteid").reset_index(drop=True),
        check_dtype=False,
    )


def test_sat_utils_conversions():
    from monetio.readers.sat_utils import jpss_time_to_datetime, tai93_to_datetime

    # Test tai93_to_datetime with sub-second precision
    da = xr.DataArray(np.array([0.5, 3600.25]), dims="x")
    res = tai93_to_datetime(da)
    assert res.dtype == "datetime64[ns]"
    assert res.values[0] == np.datetime64("1993-01-01T00:00:00.500000000")
    assert res.values[1] == np.datetime64("1993-01-01T01:00:00.250000000")

    # Test with dask
    da_lazy = da.chunk({"x": 1})
    res_lazy = tai93_to_datetime(da_lazy)
    assert hasattr(res_lazy.data, "dask")
    xr.testing.assert_allclose(res, res_lazy.compute())

    # Test jpss_time_to_datetime
    res_jpss = jpss_time_to_datetime(da, origin="1958-01-01", unit="s")
    assert res_jpss.values[0] == np.datetime64("1958-01-01T00:00:00.500000000")
