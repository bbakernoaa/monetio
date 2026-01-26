import numpy as np

from monetio.grids import MockArea


def test_mock_area_eager():
    proj_dict = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"
    area_extent = (-2500000, -2000000, 2500000, 2000000)
    nx, ny = 100, 80

    ma = MockArea(proj_dict, area_extent, nx, ny)
    lon, lat = ma.get_lonlats()

    assert lon.shape == (ny, nx)
    assert lat.shape == (ny, nx)
    assert np.all(lon >= -180) and np.all(lon <= 180)
    assert np.all(lat >= -90) and np.all(lat <= 90)


def test_mock_area_lazy():
    proj_dict = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs"
    area_extent = (-2500000, -2000000, 2500000, 2000000)
    nx, ny = 100, 80

    ma = MockArea(proj_dict, area_extent, nx, ny)
    lon_lazy, lat_lazy = ma.get_lonlats_dask()

    # If dask is installed, these should be dask arrays
    try:
        import dask.array as da

        assert isinstance(lon_lazy, da.Array)
        assert isinstance(lat_lazy, da.Array)

        lon, lat = ma.get_lonlats()
        assert np.allclose(lon_lazy.compute(), lon)
        assert np.allclose(lat_lazy.compute(), lat)
    except ImportError:
        # Fallback to eager
        lon, lat = ma.get_lonlats()
        assert np.allclose(lon_lazy, lon)
