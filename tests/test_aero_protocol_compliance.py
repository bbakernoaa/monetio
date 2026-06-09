import numpy as np
import xarray as xr

from monetio.readers.base import _add_ioapi_latlon
from monetio.readers.goes import _add_goes_latlon
from monetio.readers.ncep_grib import ncep_grib_preprocess


def test_add_goes_latlon_aero_protocol():
    """Verify _add_goes_latlon follows the Aero Protocol (Eager vs Lazy identity)."""
    # 1. Setup mock data
    x = np.linspace(-0.1, 0.1, 10).astype(np.float32)
    y = np.linspace(0.1, -0.1, 8).astype(np.float32)
    ds = xr.Dataset(
        {"AOD": (("y", "x"), np.random.rand(8, 10).astype(np.float32))}, coords={"x": x, "y": y}
    )
    proj = xr.DataArray(
        np.int32(0),
        attrs={
            "perspective_point_height": 35786023.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.31414,
            "inverse_flattening": 298.257222103,
            "latitude_of_projection_origin": 0.0,
            "longitude_of_projection_origin": -75.0,
            "sweep_angle_axis": "x",
            "grid_mapping_name": "geostationary",
        },
    )
    ds["goes_imager_projection"] = proj

    # 2. Eager execution
    ds_eager = _add_goes_latlon(ds.copy())

    # 3. Lazy execution
    ds_lazy = ds.chunk({"x": 5, "y": 4})
    ds_lazy_out = _add_goes_latlon(ds_lazy)

    # 4. Assertions
    from dask.array import Array

    assert isinstance(ds_lazy_out.latitude.data, Array)
    assert isinstance(ds_lazy_out.longitude.data, Array)

    xr.testing.assert_allclose(ds_eager.latitude, ds_lazy_out.latitude.compute())
    xr.testing.assert_allclose(ds_eager.longitude, ds_lazy_out.longitude.compute())
    assert (
        "Generated Latitude/Longitude coordinates" in ds_eager.attrs["history"]
        or "Optimized GOES coordinate generation" in ds_eager.attrs["history"]
    )


def test_add_ioapi_latlon_aero_protocol():
    """Verify _add_ioapi_latlon follows the Aero Protocol (Eager vs Lazy identity)."""
    # 1. Setup mock data
    ds = xr.Dataset({"O3": (("y", "x"), np.random.rand(5, 5))})
    ds.attrs.update({"NCOLS": 5, "NROWS": 5, "XORIG": 0, "YORIG": 0, "XCELL": 1000, "YCELL": 1000})
    proj4 = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"

    # 2. Eager execution
    ds_eager = _add_ioapi_latlon(ds.copy(), proj4)

    # 3. Lazy execution
    ds_lazy = ds.chunk({"x": 3, "y": 3})
    ds_lazy_out = _add_ioapi_latlon(ds_lazy, proj4)

    # 4. Assertions
    from dask.array import Array

    assert isinstance(ds_lazy_out.latitude.data, Array)

    xr.testing.assert_allclose(ds_eager.latitude, ds_lazy_out.latitude.compute())
    assert "Generated Latitude/Longitude coordinates" in ds_eager.attrs["history"]


def test_ncep_grib_preprocess_aero_protocol():
    """Verify ncep_grib_preprocess follows the Aero Protocol (Eager vs Lazy identity)."""
    # 1. Setup mock data
    lon = np.linspace(0, 359, 10)
    lat = np.linspace(-90, 90, 5)
    ds = xr.Dataset(
        {"TMP_2maboveground": (("lat", "lon"), np.random.rand(5, 10))},
        coords={"latitude": (("lat",), lat), "longitude": (("lon",), lon)},
    )

    # 2. Eager execution
    ds_eager = ncep_grib_preprocess(ds.copy())

    # 3. Lazy execution
    ds_lazy = ds.chunk({"lat": 3, "lon": 5})
    ds_lazy_out = ncep_grib_preprocess(ds_lazy)

    # 4. Assertions
    from dask.array import Array

    assert isinstance(ds_lazy_out.latitude.data, Array)

    xr.testing.assert_allclose(ds_eager.latitude, ds_lazy_out.latitude.compute())
    assert "Generated 2D latitude/longitude coordinates lazily" in ds_eager.attrs["history"]
