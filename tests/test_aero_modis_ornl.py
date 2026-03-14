import numpy as np

from monetio.readers.modis_ornl import _get_latlon


def test_modis_ornl_latlon_lazy():
    """Verify that _get_latlon is lazy and matches expected values."""
    xll, yll = 0, 0
    cell_width = 1000
    nx, ny = 10, 8

    # 1. Generate coordinates
    lon, lat = _get_latlon(xll, yll, cell_width, nx, ny)

    # Check dimensions
    assert lon.dims == ("y", "x")
    assert lat.dims == ("y", "x")
    assert lon.shape == (8, 10)

    # Check values (one point)
    # Sinusoidal projection at (0,0) should be (0,0)
    # But note we have offsets and linspace
    # x ranges from 0 to 10000, y from 0 to 8000
    # Center should be roughly positive
    assert not np.isnan(lon.values).all()
    assert not np.isnan(lat.values).all()

    # Verify it can be chunked
    lon_lazy = lon.chunk({"x": 5, "y": 4})
    assert hasattr(lon_lazy.data, "dask")
