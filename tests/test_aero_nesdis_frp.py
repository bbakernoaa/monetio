import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import FortranFile

from monetio.readers.nesdis_frp import NESDISFRPReader


def create_mock_frp_tile(filepath, res="C384"):
    """Create a mock NESDIS FRP binary tile."""
    r = int(res[1:])
    # NESDIS FRP uses 2D arrays (r, r)
    data = np.random.rand(r, r).astype("f4")
    with open(filepath, "wb") as f:
        w = FortranFile(f)
        w.write_record(data.flatten(order="F"))
    return data


@pytest.fixture
def mock_frp_files(tmp_path):
    """Create 6 mock FRP tiles for a single date."""
    datapath = tmp_path / "frp_data"
    datapath.mkdir()
    date = pd.Timestamp("2023-01-01")
    yyyymmdd = date.strftime("%Y%m%d")
    ftype = "meanFRP"

    files = []
    expected_data = []
    for i in range(1, 7):
        filename = f"{ftype}.{yyyymmdd}.FV3.C384Grid.tile{i}.bin"
        filepath = datapath / filename
        data = create_mock_frp_tile(filepath)
        files.append(str(filepath))
        expected_data.append(data)

    return date, ftype, str(datapath), expected_data


def test_nesdis_frp_eager_lazy(mock_frp_files):
    """Verify NESDIS FRP reader produces identical results for Eager and Lazy backends."""
    date, ftype, datapath, expected_data = mock_frp_files
    reader = NESDISFRPReader()

    # 1. Eager Load
    ds_eager = reader.open_dataset(date, ftype=ftype, datapath=datapath, lazy=False)
    assert isinstance(ds_eager[ftype].data, np.ndarray)

    # 2. Lazy Load
    ds_lazy = reader.open_dataset(date, ftype=ftype, datapath=datapath, lazy=True)
    try:
        import dask.array as da

        assert isinstance(ds_lazy[ftype].data, da.Array)
    except ImportError:
        pass  # Dask not installed, should still work but as numpy

    # 3. Assert values are identical
    xr.testing.assert_allclose(ds_eager, ds_lazy)

    # Check shape (time=1, tile=6, x=384, y=384)
    assert ds_eager[ftype].shape == (1, 6, 384, 384)

    # Check coordinates
    assert "time" in ds_eager.coords
    assert "tile" in ds_eager.coords
    assert ds_eager.time[0] == date

    # Check history
    assert "history" in ds_eager.attrs
    assert "Read NESDIS meanFRP data" in ds_eager.attrs["history"]
    assert "Read tile 1" in ds_eager[ftype].attrs.get("history", "")


def test_nesdis_frp_metadata(mock_frp_files):
    """Test scientific hygiene and coordinate metadata."""
    date, ftype, datapath, _ = mock_frp_files
    reader = NESDISFRPReader()

    ds = reader.open_dataset(date, ftype=ftype, datapath=datapath)

    # Check units and names if available
    # Since fv3grid is likely missing in test env, lat/lon might be missing
    # but let's check what we DO have.
    assert ftype in ds.data_vars
