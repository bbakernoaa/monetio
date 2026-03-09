import pytest
import xarray as xr

from monetio.readers.ish_lite import open_dataset


@pytest.fixture
def sample_ish_lite_files(tmp_path):
    """Create two dummy ISH Lite files for testing."""
    file1 = tmp_path / "722158-13897-2023"
    file2 = tmp_path / "722159-13897-2023"

    # year, month, day, hour, temp, dew_pt_temp, press, wdir, ws, sky, precip1, precip6
    data1 = (
        "2023 01 01 00   250  150 10132   270   50    8    0    0\n"
        "2023 01 01 01   260  160 10135   280   60    8    0    0\n"
    )
    data2 = (
        "2023 01 01 00   200  100 10120   180   40    5    0    0\n"
        "2023 01 01 01   210  110 10125   190   45    5    0    0\n"
    )

    file1.write_text(data1)
    file2.write_text(data2)

    return [str(file1), str(file2)]


def test_ish_lite_eager_vs_lazy(sample_ish_lite_files):
    """Verify that Eager (NumPy) and Lazy (Dask) outputs are identical."""
    # Eager
    ds_eager = open_dataset(sample_ish_lite_files, lazy=False)

    # Lazy
    ds_lazy = open_dataset(sample_ish_lite_files, lazy=True)

    # Assert laziness
    assert ds_lazy.temp.chunks is not None

    # Compute and compare
    ds_lazy_computed = ds_lazy.compute()

    xr.testing.assert_allclose(ds_eager, ds_lazy_computed)
    assert "UGRID-1.0" in ds_eager.attrs["Conventions"]
    assert "node" in ds_eager.dims


def test_ish_lite_multi_site_resample(sample_ish_lite_files):
    """Verify that multi-site resampling works correctly by expanding to 2D."""
    # Resample to hourly (should be the same as original since data is hourly)
    ds = open_dataset(sample_ish_lite_files, resample=True, window="h")

    assert ds.temp.dims == ("time", "node")
    assert ds.node.size == 2
    assert ds.time.size == 2

    # Check values for site1 (72215813897) and site2 (72215913897)
    # siteids are sorted alphabetically during unstacking/rename if I'm not mistaken
    # let's verify siteids are correct
    assert "72215813897" in ds.node.values
    assert "72215913897" in ds.node.values
