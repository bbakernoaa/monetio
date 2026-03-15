import numpy as np
import pytest
import xarray as xr

from monetio.readers.modis_l2 import modis_l2_preprocess


def create_synthetic_modis_l2():
    """Create a synthetic MODIS L2-like dataset."""
    y_size = 10
    x_size = 5

    # Coordinates
    lat = np.linspace(30, 40, y_size)
    lon = np.linspace(-120, -110, x_size)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Scan Start Time (seconds since 1993-01-01)
    # 2023-01-01 is roughly 30 years after 1993
    seconds_in_30_years = 30 * 365 * 24 * 3600
    scan_time = np.full((y_size, x_size), seconds_in_30_years, dtype=np.float64)

    # Data variables
    aod = np.random.rand(y_size, x_size).astype(np.float32)
    quality = np.random.randint(0, 4, (y_size, x_size)).astype(np.int8)

    ds = xr.Dataset(
        data_vars={
            "AOD_550": (("Cell_Along_Swath", "Cell_Across_Swath"), aod),
            "Quality_Flag": (("Cell_Along_Swath", "Cell_Across_Swath"), quality),
            "Scan_Start_Time": (("Cell_Along_Swath", "Cell_Across_Swath"), scan_time),
            "Latitude": (("Cell_Along_Swath", "Cell_Across_Swath"), lat_2d),
            "Longitude": (("Cell_Along_Swath", "Cell_Across_Swath"), lon_2d),
        }
    )

    return ds


def test_modis_l2_preprocess_eager_lazy():
    """Test modis_l2_preprocess with both Eager and Lazy backends."""
    ds_eager = create_synthetic_modis_l2()

    variable_dict = {
        "AOD_550": {
            "scale": 1.0,
            "minimum": 0.1,
            "maximum": 0.8,
            "quality_flag": 3,  # Mask if >= 3
        },
        "Quality_Flag": {},
    }

    # 1. Run Eager
    res_eager = modis_l2_preprocess(ds_eager, variable_dict=variable_dict)

    # 2. Run Lazy
    ds_lazy = ds_eager.chunk({"Cell_Along_Swath": 5, "Cell_Across_Swath": 5})
    res_lazy = modis_l2_preprocess(ds_lazy, variable_dict=variable_dict)

    # Verify results are identical
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verify standardization
    assert "latitude" in res_eager.coords
    assert "longitude" in res_eager.coords
    assert "time" in res_eager.coords
    assert res_eager.sizes == {"y": 10, "x": 5}

    # Verify transformations
    # Masked by minimum (0.1) or maximum (0.8) or quality flag (>= 3)
    # Check that some values are indeed NaN
    assert res_eager["AOD_550"].isnull().any()

    # Verify history
    assert "history" in res_eager.attrs
    assert "Preprocessed MODIS L2 data via Aero Protocol." in res_eager.attrs["history"]


@pytest.mark.parametrize("lazy", [False, True])
def test_modis_l2_time_calculation(lazy):
    """Verify time calculation from Scan_Start_Time."""
    ds = create_synthetic_modis_l2()
    if lazy:
        ds = ds.chunk({"Cell_Along_Swath": -1})

    res = modis_l2_preprocess(ds)

    # Check time coordinate
    assert "time" in res.coords
    # 1993-01-01 + 30 years (not accounting for leap years perfectly in synthetic data,
    # but enough to check it's a datetime64)
    assert res.time.dtype.kind == "M"  # Datetime
