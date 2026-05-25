import dask.array as da
import numpy as np
import xarray as xr

from monetio.readers.ncep_grib import NCEPGribReader, ncep_grib_preprocess


def test_ncep_grib_preprocess():
    # Create dummy dataset with 1D coordinates
    ny, nx = 4, 5
    ds = xr.Dataset(
        {
            "VAR1": (("lat_0", "lon_0"), np.random.rand(ny, nx)),
        },
        coords={
            "lat_0": np.linspace(-90, 90, ny),
            "lon_0": np.linspace(-180, 180, nx),
        },
    )

    ds_out = ncep_grib_preprocess(ds)

    assert "latitude" in ds_out.coords
    assert "longitude" in ds_out.coords
    assert ds_out.latitude.ndim == 2
    assert ds_out.longitude.ndim == 2
    assert ds_out.latitude.shape == (ny, nx)
    assert ds_out.y.size == ny
    assert ds_out.x.size == nx


def test_ncep_grib_eager_lazy_consistency():
    ny, nx = 10, 12
    data = np.random.rand(ny, nx).astype("f4")
    ds_eager = xr.Dataset(
        {
            "VAR1": (("lat_0", "lon_0"), data),
        },
        coords={
            "lat_0": np.linspace(-90, 90, ny),
            "lon_0": np.linspace(-180, 180, nx),
        },
    )

    ds_lazy = ds_eager.chunk({"lat_0": 5, "lon_0": 6})

    out_eager = ncep_grib_preprocess(ds_eager)
    out_lazy = ncep_grib_preprocess(ds_lazy)

    xr.testing.assert_allclose(out_eager, out_lazy.compute())
    assert isinstance(out_lazy.latitude.data, da.Array)


def test_ncep_grib_reader_open(monkeypatch):
    class MockDriver:
        def open(self, files, **kwargs):
            # Just return a simple dataset
            return xr.Dataset({"test": 1})

    reader = NCEPGribReader()
    monkeypatch.setattr(reader, "driver", MockDriver())

    ds = reader.open_dataset("dummy.grib2")
    assert "test" in ds.data_vars
    assert "history" in ds.attrs
