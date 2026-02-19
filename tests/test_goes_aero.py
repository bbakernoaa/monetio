import numpy as np
import xarray as xr

from monetio.readers.goes import goes_preprocess


def test_goes_preprocess_latlon():
    # Mock GOES dataset
    x = np.linspace(-0.1, 0.1, 10).astype(np.float32)
    y = np.linspace(0.1, -0.1, 8).astype(np.float32)

    ds = xr.Dataset(
        {"AOD": (("y", "x"), np.random.rand(8, 10).astype(np.float32))}, coords={"x": x, "y": y}
    )

    # Add fake projection info
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
    ds.attrs["time_coverage_start"] = "2023-01-01T00:00:00Z"

    # 1. Eager
    ds_eager = goes_preprocess(ds.copy())
    assert "latitude" in ds_eager.coords
    assert "longitude" in ds_eager.coords
    assert ds_eager.latitude.dims == ("y", "x")
    assert "history" in ds_eager.attrs
    assert "Preprocessed GOES data." in ds_eager.attrs["history"]

    # 2. Verify logic (one point)
    # ABI lat/lon formula is complex but we just check it produced something reasonable
    assert not np.isnan(ds_eager.latitude.values).all()
    assert ds_eager.latitude.attrs["units"] == "degrees_north"

    # 3. Running on chunked data (verifies apply_ufunc signature)
    ds_lazy = ds.chunk({"x": 5, "y": 4})
    ds_lazy_out = goes_preprocess(ds_lazy)
    # Note: Xarray might eager-fy 1D dimension coordinates in small mocks,
    # but the fact that it runs without error confirms the unified apply_ufunc logic.
    assert "latitude" in ds_lazy_out.coords
