import dask.array as da
import numpy as np
import xarray as xr

from monetio.util import calc_13_category_usda_soil_type


def test_calc_soil_type_eager_vs_lazy():
    """Verify that calc_13_category_usda_soil_type works identically for NumPy and Dask."""
    # Create some test data representing various categories
    # 1: SAND (si + c * 1.5 < 15.0)
    # 12: CLAY (c >= 40 & sa <= 45 & si < 40)
    # 6: LOAM (c >= 7 & c < 27 & si >= 28 & si < 50 & sa <= 52)

    clay_np = np.array([5.0, 45.0, 20.0, 255.0], dtype=float)
    sand_np = np.array([90.0, 40.0, 40.0, 0.0], dtype=float)
    silt_np = np.array([5.0, 15.0, 40.0, 0.0], dtype=float)

    # Eager execution
    res_eager = calc_13_category_usda_soil_type(clay_np, sand_np, silt_np)
    assert isinstance(res_eager, np.ndarray)

    # Lazy execution
    clay_da = xr.DataArray(da.from_array(clay_np, chunks=2), dims="x")
    sand_da = xr.DataArray(da.from_array(sand_np, chunks=2), dims="x")
    silt_da = xr.DataArray(da.from_array(silt_np, chunks=2), dims="x")

    res_lazy = calc_13_category_usda_soil_type(clay_da, sand_da, silt_da)

    # Ensure it is still lazy
    assert hasattr(res_lazy.data, "dask")
    assert "history" in res_lazy.attrs

    # Verify results match
    res_lazy_computed = res_lazy.compute()
    np.testing.assert_allclose(res_eager, res_lazy_computed.values)

    # Specific value checks
    assert res_eager[0] == 1.0  # SAND
    assert res_eager[1] == 12.0  # CLAY
    assert res_eager[2] == 6.0  # LOAM
    assert res_eager[3] == 0.0  # FillValue 255 -> 0.0


def test_calc_soil_type_dask_identity():
    """Verify that results are identical when using Dask vs NumPy."""
    clay = np.random.rand(10, 10) * 100
    sand = np.random.rand(10, 10) * 100
    silt = np.random.rand(10, 10) * 100

    res_np = calc_13_category_usda_soil_type(clay, sand, silt)

    clay_da = xr.DataArray(clay, dims=("y", "x")).chunk({"x": 5, "y": 5})
    sand_da = xr.DataArray(sand, dims=("y", "x")).chunk({"x": 5, "y": 5})
    silt_da = xr.DataArray(silt, dims=("y", "x")).chunk({"x": 5, "y": 5})

    res_da = calc_13_category_usda_soil_type(clay_da, sand_da, silt_da)

    xr.testing.assert_allclose(
        xr.DataArray(res_np, dims=("y", "x")), res_da.drop_vars("history", errors="ignore")
    )
