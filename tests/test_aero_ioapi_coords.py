import numpy as np
import pytest
import xarray as xr

from monetio.readers.base import _add_ioapi_latlon


def test_add_ioapi_latlon_eager_lazy_consistency():
    """Verify _add_ioapi_latlon produces identical results for Eager and Lazy backends."""
    # 1. Setup mock IOAPI dataset
    # LAMBERT CONFORMAL example
    proj4 = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"

    ds = xr.Dataset(
        {"data": (("y", "x"), np.random.rand(10, 12))},
        coords={"x": np.arange(12), "y": np.arange(10)},
    )
    ds.attrs.update(
        {
            "XORIG": -2500000.0,
            "YORIG": -1500000.0,
            "XCELL": 12000.0,
            "YCELL": 12000.0,
            "NCOLS": 12,
            "NROWS": 10,
        }
    )

    # 2. Eager execution
    ds_eager = _add_ioapi_latlon(ds.copy(), proj4)

    assert "latitude" in ds_eager.coords
    assert "longitude" in ds_eager.coords
    assert ds_eager.latitude.shape == (10, 12)

    # 3. Lazy execution
    ds_lazy = ds.chunk({"x": 6, "y": 5})
    ds_lazy = _add_ioapi_latlon(ds_lazy, proj4)

    # Verify it is still lazy
    assert hasattr(ds_lazy.latitude.data, "dask")

    # 4. Compare
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # 5. Verify history
    assert "Generated Latitude/Longitude coordinates" in ds_eager.attrs["history"]


def test_add_ioapi_latlon_col_row_dims():
    """Verify it works with COL/ROW dimension names (as in some CMAQ files)."""
    proj4 = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"

    ds = xr.Dataset(
        {"data": (("ROW", "COL"), np.random.rand(5, 5))},
    )
    ds.attrs.update(
        {"XORIG": 0.0, "YORIG": 0.0, "XCELL": 1000.0, "YCELL": 1000.0, "NCOLS": 5, "NROWS": 5}
    )

    ds_out = _add_ioapi_latlon(ds, proj4)
    assert ds_out.latitude.dims == ("ROW", "COL")
    assert ds_out.longitude.dims == ("ROW", "COL")


if __name__ == "__main__":
    pytest.main([__file__])
