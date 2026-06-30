import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.madis import MADISReader


@pytest.fixture
def mock_madis_file(tmp_path):
    fn = tmp_path / "madis_test.nc"

    n_rec = 10
    ds = xr.Dataset(
        {
            "observationTime": (("recNum",), np.arange(n_rec, dtype="float64") * 3600),
            "latitude": (("recNum",), np.linspace(30, 40, n_rec)),
            "longitude": (("recNum",), np.linspace(-100, -90, n_rec)),
            "stationId": (("recNum",), [f"S{i}" for i in range(n_rec)]),
            "temperature": (("recNum",), np.random.rand(n_rec).astype("f4")),
        },
        attrs={"Conventions": "CF-1.6"},
    )
    ds.observationTime.attrs["units"] = "seconds since 1970-01-01 00:00:00.0 +0000"

    ds.to_netcdf(fn)
    return str(fn)


def test_madis_reader_eager_lazy(mock_madis_file):
    reader = MADISReader()

    # Eager (NumPy)
    ds_eager = reader.open_dataset(mock_madis_file, lazy=False, as_xarray=True, expand2d=False)
    assert not hasattr(ds_eager.temperature.data, "dask")
    assert ds_eager.time.dtype == "datetime64[ns]"

    # Lazy (Dask)
    ds_lazy = reader.open_dataset(mock_madis_file, lazy=True, as_xarray=True, expand2d=False)
    assert hasattr(ds_lazy.temperature.data, "dask")
    assert ds_lazy.time.dtype == "datetime64[ns]"

    # Assert identical results (after computing lazy)
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Check 2D expansion
    ds_2d = reader.open_dataset(mock_madis_file, lazy=True, as_xarray=True, expand2d=True)
    assert "node" in ds_2d.dims
    assert "time" in ds_2d.dims
    assert hasattr(ds_2d.temperature.data, "dask")


def test_madis_reader_dataframe(mock_madis_file):
    reader = MADISReader()

    # Eager DataFrame
    df = reader.open_dataset(mock_madis_file, lazy=False, as_xarray=False)
    assert isinstance(df, pd.DataFrame)
    assert "temperature" in df.columns
    assert "time" in df.columns
    assert df.time.dtype == "datetime64[ns]"

    # Lazy DataFrame (Dask)
    ddf = reader.open_dataset(mock_madis_file, lazy=True, as_xarray=False)
    import dask.dataframe as dd

    assert isinstance(ddf, dd.DataFrame)
    assert "temperature" in ddf.columns
    assert ddf.time.dtype == "datetime64[ns]"

    # Assert identical
    pd.testing.assert_frame_equal(df.set_index("node"), ddf.compute().set_index("node"))
