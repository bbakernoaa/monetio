import numpy as np
import pytest
import xarray as xr

from monetio.readers.hysplit import check_grid_continuity, fix_grid_continuity


@pytest.fixture
def mock_hysplit_ds():
    """Create a mock HYSPLIT dataset with a gap in the grid."""
    attrs = {
        "llcrnr latitude": 30.0,
        "llcrnr longitude": -100.0,
        "Number Lat Points": 10,
        "Number Lon Points": 10,
        "Latitude Spacing": 1.0,
        "Longitude Spacing": 1.0,
    }

    # Grid with a gap: x=1,2, 4,5 (missing 3)
    x = np.array([1, 2, 4, 5])
    y = np.array([1, 2, 3])

    data = np.random.rand(1, 1, 3, 4)  # time, z, y, x

    ds = xr.Dataset(
        {"CONC": (("time", "z", "y", "x"), data)},
        coords={
            "time": [np.datetime64("2020-01-01")],
            "z": [100],
            "y": y,
            "x": x,
        },
        attrs=attrs,
    )
    return ds


def test_check_grid_continuity(mock_hysplit_ds):
    # Should be False due to gap in x
    assert check_grid_continuity(mock_hysplit_ds) is False

    # Fix it
    ds_fixed = mock_hysplit_ds.reindex(x=[1, 2, 3, 4, 5], fill_value=0)
    assert check_grid_continuity(ds_fixed) is True


def test_fix_grid_continuity_eager_lazy_consistency(mock_hysplit_ds):
    """Verify fix_grid_continuity works identically for NumPy and Dask backends."""
    # Eager (NumPy)
    ds_eager = fix_grid_continuity(mock_hysplit_ds)

    # Lazy (Dask)
    ds_lazy_input = mock_hysplit_ds.chunk({"x": 2})
    ds_lazy = fix_grid_continuity(ds_lazy_input)

    # Check that it's still dask-backed
    assert ds_lazy.CONC.chunks is not None

    # Verify results are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())

    # Verify grid is continuous
    assert ds_eager.x.size == 5
    assert (ds_eager.x.values == np.arange(1, 6)).all()
    assert ds_eager.CONC.sel(x=3).isnull().all() or (ds_eager.CONC.sel(x=3) == 0).all()


def test_fix_grid_continuity_provenance(mock_hysplit_ds):
    ds_fixed = fix_grid_continuity(mock_hysplit_ds)
    assert "Fixed grid continuity" in ds_fixed.attrs["history"]


def test_mass_loading_lazy_consistency():
    from monetio.readers.hysplit import mass_loading

    # Create a 3D dataset (z, y, x)
    z = np.array([100, 200, 300])
    y = np.arange(5)
    x = np.arange(5)
    data = np.random.rand(3, 5, 5)

    da = xr.DataArray(
        data,
        dims=("z", "y", "x"),
        coords={"z": z, "y": y, "x": x},
        name="CONC",
        attrs={"history": "Initial data."},
    )

    # delta = [100, 100, 100]
    delta = np.array([100, 100, 100])

    # Eager
    ml_eager = mass_loading(da, delta=delta)

    # Lazy
    da_lazy = da.chunk({"z": 1})
    ml_lazy = mass_loading(da_lazy, delta=delta)

    assert ml_lazy.chunks is not None
    xr.testing.assert_allclose(ml_eager, ml_lazy.compute())
    assert "Calculated mass loading" in ml_lazy.attrs["history"]


def test_mass_loading_with_deposition():
    from monetio.readers.hysplit import mass_loading

    # z=0 is deposition layer
    z = np.array([0, 100, 200])
    data = np.ones((3, 2, 2))
    da = xr.DataArray(data, dims=("z", "y", "x"), coords={"z": z}, name="CONC")

    # thickness of deposition layer is 0
    delta = np.array([0, 100, 100])

    ml = mass_loading(da, delta=delta)

    # Should only sum z=100 and z=200
    # Expected: 1*100 + 1*100 = 200
    assert (ml == 200).all()
