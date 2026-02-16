from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio import raqms

DATA = Path(__file__).parent / "data"

TEST_FP = str(DATA / "uwhyb_06_01_2017_18Z.chem.assim.nc")


def _test_ds(ds):
    # Test _fix_time worked
    assert ds.time.values[0] == pd.Timestamp("2017-06-01 18:00")
    assert set(ds.dims) == {"time", "x", "y", "z"}
    assert "IDATE" not in ds.data_vars
    assert "Times" not in ds.data_vars

    # Test _fix_grid worked
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert np.all(ds.latitude.values[0, :] == ds.latitude.values[0, 0])
    assert np.all(ds.longitude.values[:, 0] == ds.longitude.values[0, 0])
    assert float(ds.longitude.min()) == -180.0  # 1-degree grid
    assert float(ds.longitude.max()) == 179.0
    assert np.all(ds.geop.mean(["time", "x", "y"]) > 0)

    # Test _fix_pres worked
    p_vns = {"surfpres_pa", "dp_pa", "pres_pa_mid"}
    assert p_vns.issubset(ds.variables)
    for vn in p_vns:
        assert ds[vn].units == "Pa"
    assert (ds["pres_pa_mid"].mean(dim=("time", "y", "x")) > 90000).all()
    assert (ds["dp_pa"].mean(dim=("time", "y", "x")) > 1000).all()
    assert 100000 > ds["surfpres_pa"].mean() > 95000

    assert tuple(ds.o3vmr.dims) == ("time", "z", "y", "x")

    # Test conversion of gases to ppbv worked
    assert ds["o3vmr"].units == "ppbv"

    assert "temperature_k" in ds.data_vars


def test_open_dataset():
    ds = raqms.open_dataset(TEST_FP)
    _test_ds(ds)


def test_open_mfdataset():
    ds = raqms.open_mfdataset(TEST_FP)
    _test_ds(ds)


def test_open_dataset_bad():
    with pytest.raises(ValueError, match="^File format "):
        raqms.open_dataset("asdf")


def test_open_mfdataset_bad():
    with pytest.raises(ValueError, match="^File format "):
        raqms.open_mfdataset("asdf")


@pytest.mark.parametrize(
    "fn",
    ["open_dataset", "open_mfdataset"],
)
def test_surf_only(fn):
    ds = getattr(raqms, fn)(TEST_FP, surf_only=True)
    assert set(ds.dims) == {"time", "z", "y", "x"}
    assert tuple(ds.o3vmr.dims) == ("time", "z", "y", "x")
    assert ds.sizes["z"] == 1


def test_raqms_eager_vs_lazy():
    """Verifies RAQMS reader with both Eager and Lazy data."""
    from monetio.readers.raqms import RAQMSReader

    reader = RAQMSReader()

    # 1. Test Eager
    ds_eager = reader.open_dataset(TEST_FP, use_dask=False)
    assert isinstance(ds_eager.o3vmr.data, np.ndarray)

    # 2. Test Lazy
    ds_lazy = reader.open_dataset(TEST_FP, chunks={"time": 1})
    assert hasattr(ds_lazy.o3vmr.data, "dask")

    # 3. Verify results are identical
    # We use compute() on the lazy one and compare with eager
    xr.testing.assert_allclose(ds_eager, ds_lazy.compute())
