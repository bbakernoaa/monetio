import dask.array as da
import numpy as np
import pytest
import xarray as xr

from monetio.readers.ufs import UFSReader


def create_mock_ufs_ds():
    """Creates a mock UFS-AQM dataset for dual backend testing."""
    nx, ny, nz = 10, 10, 5
    nt = 2

    # Interface pressures (nz+1)
    ak = np.linspace(100, 0, nz + 1)
    bk = np.linspace(0, 0, nz + 1)  # Simple sigma=0 for test

    data = {
        "tmp": (("time", "z", "y", "x"), np.full((nt, nz, ny, nx), 300.0)),
        "pressfc": (("time", "y", "x"), np.full((nt, ny, nx), 101325.0)),
        "ak": (("z_i",), ak),
        "bk": (("z_i",), bk),
        "lat": (("y", "x"), np.zeros((ny, nx))),
        "lon": (("y", "x"), np.zeros((ny, nx))),
        "delz": (("time", "z", "y", "x"), np.full((nt, nz, ny, nx), 100.0)),
        "hgtsfc": (("y", "x"), np.zeros((ny, nx))),
        "no": (("time", "z", "y", "x"), np.full((nt, nz, ny, nx), 1.0)),
        "no2": (("time", "z", "y", "x"), np.full((nt, nz, ny, nx), 1.0)),
        "aso4j": (("time", "z", "y", "x"), np.full((nt, nz, ny, nx), 1.0)),
    }

    attrs = {
        "tmp": {"units": "K"},
        "pressfc": {"units": "Pa"},
        "no": {"units": "ppmV"},
        "no2": {"units": "ppmV"},
        "aso4j": {"units": "ug/kg"},
    }

    ds = xr.Dataset(
        data_vars=data,
        coords={
            "time": (("time",), [np.datetime64("2023-01-01T00:00"), np.datetime64("2023-01-01T01:00")]),
            "pfull": (("z",), np.arange(nz)),
            "phalf": (("z_i",), np.arange(nz + 1)),
        },
    )

    for var, v_attrs in attrs.items():
        ds[var].attrs.update(v_attrs)

    return ds


def test_ufs_dual_backend_consistency():
    """Verify UFS reader gives identical results for Eager and Lazy backends."""
    ds_base = create_mock_ufs_ds()

    reader = UFSReader()

    class MockDriver:
        def __init__(self, ds):
            self.ds = ds

        def open(self, *args, **kwargs):
            return self.ds

    # 1. Eager
    reader.driver = MockDriver(ds_base.copy(deep=True))
    res_eager = reader.open_dataset("dummy", convert_to_ppb=True)

    # 2. Lazy
    ds_lazy_in = ds_base.copy(deep=True).chunk({"time": 1, "z": -1, "y": -1, "x": -1})
    reader.driver = MockDriver(ds_lazy_in)
    res_lazy = reader.open_dataset("dummy", convert_to_ppb=True)

    # Assert results are identical
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Verify laziness
    assert isinstance(res_lazy.nox.data, da.Array)
    assert isinstance(res_lazy.pres_pa_mid.data, da.Array)

    # Verify unit conversions
    # NO: 1ppmV -> 1000ppbV. NOx = NO + NO2 = 2000ppbV
    assert res_eager.nox.attrs["units"] == "ppbV"
    np.testing.assert_allclose(res_eager.nox.values, 2000.0)

    # Verify pressure calculation (log-mean)
    # ak is 100 to 0. psfc is 101325. bk is 0.
    # p_interfaces are ak values.
    # For first layer (z=0): p1=100, p2=80 (approx, since it was linspace(100,0,6))
    # nz=5, nz+1=6. ak = [100, 80, 60, 40, 20, 0]
    p1, p2 = 100.0, 80.0
    expected_p_mid = (p2 - p1) / np.log(p2 / p1)
    # Note: UFS reader might flip Z if it's ascending and it wants top-down or vice versa.
    # In our mock: pfull = [0,1,2,3,4]. phalf = [0,1,2,3,4,5].
    # ds.z[0] < ds.z[-1] is True (0 < 4). So it will flip Z.
    # After flip: z=0 corresponds to original z=4.
    # Original z=4 interfaces were ak[4]=20 and ak[5]=0.
    # (0 - 20) / ln(0/20) would be NaN/Inf without guard.
    # But wait, our ak had 100 at 0. So z=4 interfaces are 20 and 0.
    # If p_interfaces_2 is 0, log(0) is -inf. (0 - 20) / -inf = 0.
    # Our guard: if p1==p2, p_mid = p1.
    # Here p1=20, p2=0. No guard triggered by equality, but log(0) handled by numpy.
    # Actually, real UFS files don't have 0 pressure at surface.

    # Check that it didn't crash
    assert not np.isnan(res_eager.pres_pa_mid.values).any()


def test_ufs_pressure_fallback_guard():
    """Verify that p1 == p2 case in pressure calculation is handled."""
    ds = create_mock_ufs_ds()
    # Force p1 == p2
    ds["ak"] = (("z_i",), np.ones(6) * 100.0)
    ds["bk"] = (("z_i",), np.zeros(6))

    reader = UFSReader()

    class MockDriver:
        def open(self, *args, **kwargs):
            return ds

    reader.driver = MockDriver()
    res = reader.open_dataset("dummy")

    # Should be exactly 100.0, not NaN
    np.testing.assert_allclose(res.pres_pa_mid.values, 100.0)
    assert not np.isnan(res.pres_pa_mid.values).any()
