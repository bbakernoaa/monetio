import pytest
import xarray as xr

from monetio.readers.crn import CRNReader


@pytest.fixture
def dummy_crn_file(tmp_path):
    fn = tmp_path / "CRNH0203-2023-TX_Monahans_6_ENE.txt"
    # HCOLS: WBANNO, UTC_DATE, UTC_TIME, LST_DATE, LST_TIME, CRX_VN, LONGITUDE, LATITUDE, ...
    # 03047 20230101 0000 20221231 1800 1.0 -102.80 31.62 ...
    line = "03047 20230101 0000 20221231 1800 1.0 -102.80 31.62 " + " ".join(["0.0"] * 30) + "\n"
    fn.write_text(line)
    return str(fn)


def test_crn_reader_eager(dummy_crn_file):
    reader = CRNReader()
    # We pass the file directly to avoid build_urls which tries to check existence of remote URLs
    ds = reader.open_dataset(files=dummy_crn_file, as_xarray=True, lazy=False)

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.coords
    assert "node" in ds.coords
    assert ds.sizes["node"] == 1
    # Check if merge worked (from stations.tsv)
    # Monahans is in stations.tsv so it should match WBANNO 03047
    assert "state" in ds.data_vars
    assert ds.state.values[0] == "TX"


def test_crn_reader_lazy(dummy_crn_file):
    pytest.importorskip("dask")
    reader = CRNReader()
    ds = reader.open_dataset(files=dummy_crn_file, as_xarray=True, lazy=True)

    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.coords
    # In xarray, dask-backed variables are identifiable
    assert ds.t_avg.chunks is not None

    # Compute and check
    ds_eager = ds.compute()
    assert ds_eager.state.values[0] == "TX"
    assert ds_eager.sizes["node"] == 1


def test_crn_build_urls():
    # reader = CRNReader()
    # This might fail if it can't reach the server, but it uses FileUtility.get_fs which for https uses fsspec
    # For testing, we might want to mock fs.exists
    pass
