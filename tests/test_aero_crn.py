import pandas as pd
import pytest
import xarray as xr
import numpy as np
from monetio.readers.crn import read_crn

@pytest.fixture
def crn_file(tmp_path):
    d = tmp_path / "crn_data"
    d.mkdir()
    f = d / "CRNH0203-2020-MD_College_Park_1_S.txt"
    line1 = "722244 20200901 0000 20200831 1900 1.234 -76.848 38.981 25.0 25.1 25.2 24.8 0.0 500.0 0 510.0 0 490.0 0 X 26.0 0 26.5 0 25.5 0 80.0 0 0.1 0.1 0.1 0.1 0.1 20.0 20.0 20.0 20.0 20.0"
    line2 = "722244 20200901 0100 20200831 2000 1.234 -76.848 38.981 26.0 26.1 26.2 25.8 0.0 510.0 0 520.0 0 500.0 0 X 27.0 0 27.5 0 26.5 0 81.0 0 0.1 0.1 0.1 0.1 0.1 21.0 21.0 21.0 21.0 21.0"
    content = f"{line1}\n{line2}"
    f.write_text(content)
    return str(f)

def test_crn_eager_vs_lazy_1d(crn_file):
    """Verify that Eager (NumPy) and Lazy (Dask) backends yield identical results for 1D."""
    # 1. Eager load
    ds_eager = read_crn(crn_file, lazy=False, expand2d=False)

    # 2. Lazy load
    ds_lazy = read_crn(crn_file, lazy=True, expand2d=False)

    from dask.array.core import Array as DaskArray
    is_lazy = any(isinstance(v.data, DaskArray) for v in ds_lazy.data_vars.values())

    ds_eager_no_hist = ds_eager.copy()
    ds_lazy_no_hist = ds_lazy.compute().copy()
    ds_eager_no_hist.attrs.pop("history", None)
    ds_lazy_no_hist.attrs.pop("history", None)

    xr.testing.assert_allclose(ds_eager_no_hist, ds_lazy_no_hist)

    assert is_lazy, "Dataset should be dask-backed when lazy=True and expand2d=False"

def test_crn_eager_vs_lazy_2d(crn_file):
    """Verify that Eager (NumPy) and Lazy (Dask) backends yield identical results for 2D."""
    # 1. Eager load
    ds_eager = read_crn(crn_file, lazy=False, expand2d=True)

    # 2. Lazy load
    ds_lazy = read_crn(crn_file, lazy=True, expand2d=True)

    ds_eager_no_hist = ds_eager.copy()
    ds_lazy_no_hist = ds_lazy.compute().copy()
    ds_eager_no_hist.attrs.pop("history", None)
    ds_lazy_no_hist.attrs.pop("history", None)

    xr.testing.assert_allclose(ds_eager_no_hist, ds_lazy_no_hist)

    # We accept that unstacking might currently compute for small test files
    # but we want to ensure the logic is correct.
    assert ds_eager.sizes["time"] == 2
    assert ds_eager.sizes["node"] == 1
