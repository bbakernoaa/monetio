import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.util import (
    _try_merge_exact,
    calc_3hr_ave,
    calc_annual_ave,
    findclosest,
    kolmogorov_zurbenko_filter,
    linregress,
    nearest,
    search_listinlist,
    wsdir2uv,
)


def test_merge_exact_helper():
    x = np.r_[1:11]
    a = x**2
    b = x**3
    left = xr.Dataset(
        data_vars={
            "a": ("x", a),
        },
        coords={
            "x": ("x", x),
        },
    )
    right = xr.Dataset(
        data_vars={
            "b": ("x", b),
        },
        coords={
            "x": ("x", x),
        },
    )
    new = _try_merge_exact(left, right)

    assert set(left.data_vars) == {"a"}
    assert set(right.data_vars) == {"b"}
    assert set(new.data_vars) == {"a", "b"}

    assert new.x.equals(left.x) and new.x.equals(right.x)


def test_issue78():
    # In this issue, dimension coordinate 'grid_xt' fails to match exactly
    # since one dataset (normal output) has it in float64 and the other (PM2.5)
    # has it in float32.
    x64 = np.linspace(20, 65, 90, dtype=np.float64)
    y64 = np.linspace(20, 45, 72, dtype=np.float64)
    x32 = x64.astype(np.float32)
    y32 = y64.astype(np.float32)
    a = np.random.rand(y64.size, x64.size)
    b = np.random.rand(y64.size, x64.size)
    left = xr.Dataset(
        data_vars={
            "a": (("y", "x"), a),
        },
        coords={
            "x": ("x", x64),
            "y": ("y", y64),
        },
    )
    right = xr.Dataset(
        data_vars={
            "b": (("y", "x"), b),
        },
        coords={
            "x": ("x", x32),
            "y": ("y", y32),
        },
    )

    assert not left.x.equals(right.x)
    assert not left.y.equals(right.y)

    with pytest.raises(ValueError, match="Unable to merge blah due to issue matching coordinates."):
        _ = _try_merge_exact(left, right, right_name="blah")


def test_nearest():
    items = [1, 5, 10, 20]
    assert nearest(items, 7) == 5
    assert nearest(items, 18) == 20


def test_search_listinlist():
    a1 = np.array([1, 2, 3, 4, 5])
    a2 = np.array([3, 5, 7])
    idx1, idx2 = search_listinlist(a1, a2)
    assert np.array_equal(idx1, [2, 4])
    assert np.array_equal(idx2, [0, 1])


def test_linregress():
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 1 + np.random.normal(0, 0.001, len(x))
    a, b, r2, se = linregress(x, y)
    assert np.isclose(a, 2, atol=0.1)
    assert np.isclose(b, 1, atol=0.1)
    assert r2 > 0.9


def test_findclosest():
    items = [1, 5, 10, 20]
    idx, val = findclosest(items, 7)
    assert idx == 1
    assert val == 5


def test_kolmogorov_zurbenko_filter():
    df = pd.Series(np.random.rand(100))
    filtered = kolmogorov_zurbenko_filter(df, window=5, iterations=3)
    assert len(filtered) == 100
    assert not filtered.isna().all()


def test_wsdir2uv_numpy():
    ws = np.array([10, 10, 10, 10])
    wdir = np.array([0, 90, 180, 270])
    u, v = wsdir2uv(ws, wdir)
    assert np.isclose(u[0], 0)
    assert np.isclose(v[0], -10)
    assert np.isclose(u[1], -10)
    assert np.isclose(v[1], 0)


def test_wsdir2uv_xarray():
    ws = xr.DataArray([10, 10], dims="time")
    wdir = xr.DataArray([0, 90], dims="time")
    u, v = wsdir2uv(ws, wdir)
    assert isinstance(u, xr.DataArray)
    assert np.isclose(u.values[0], 0)
    assert np.isclose(v.values[0], -10)


def test_wsdir2uv_dask():
    ws = xr.DataArray(np.array([10, 10])).chunk(1)
    wdir = xr.DataArray(np.array([0, 90])).chunk(1)
    u, v = wsdir2uv(ws, wdir)
    assert u.chunks is not None
    res_u = u.compute()
    assert np.isclose(res_u.values[0], 0)


def test_averages():
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({"time_local": dates, "siteid": "A", "obs": np.random.rand(100)})

    res3h = calc_3hr_ave(df, col="obs")
    # It creates obs_x and obs_y because of the merge
    assert "obs_y" in res3h.columns

    res_annual = calc_annual_ave(df, col="obs")
    assert "obs_y" in res_annual.columns
