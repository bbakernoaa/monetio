import numpy as np
import pytest
import xarray as xr

from monetio.readers.camx import CAMxReader, camx_preprocess


def mock_camx_dataset():
    """Create a synthetic CAMx dataset for modernization testing."""
    ntime = 2
    nlay = 2
    nrow = 3
    ncol = 4

    # Time flags (YYYYDDD, HHMMSS)
    tflag_data = [
        [[2023001, 0], [2023001, 10000]],  # For Var 1
        [[2023001, 0], [2023001, 10000]],  # For Var 2
    ]
    tflag = xr.DataArray(
        np.array(tflag_data).transpose(1, 0, 2),
        dims=("TSTEP", "VAR", "DATE_TIME"),
        name="TFLAG",
    )

    # Coordinates and data variables
    np.random.seed(42)
    ds = xr.Dataset(
        {
            "O3": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(ntime, nlay, nrow, ncol)),
            "NO": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(ntime, nlay, nrow, ncol)),
            "NO2": (("TSTEP", "LAY", "ROW", "COL"), np.random.rand(ntime, nlay, nrow, ncol)),
            "TFLAG": tflag,
        }
    )

    # IOAPI Grid attributes (Lambert Conformal Example)
    ds.attrs["GDTYP"] = 2
    ds.attrs["P_ALP"] = 30.0
    ds.attrs["P_BET"] = 60.0
    ds.attrs["P_GAM"] = -100.0
    ds.attrs["XCENT"] = -100.0
    ds.attrs["YCENT"] = 40.0
    ds.attrs["XORIG"] = -1000.0
    ds.attrs["YORIG"] = -1000.0
    ds.attrs["XCELL"] = 500.0
    ds.attrs["YCELL"] = 500.0
    ds.attrs["NCOLS"] = ncol
    ds.attrs["NROWS"] = nrow
    ds.attrs["NLAYS"] = nlay

    # Add units
    ds.O3.attrs["units"] = "ppm"
    ds.NO.attrs["units"] = "ppm"
    ds.NO2.attrs["units"] = "ppm"

    return ds


def test_camx_preprocess_eager_lazy_consistency():
    """
    Verify that camx_preprocess produces identical results for Eager (NumPy)
    and Lazy (Dask) input datasets.
    """
    ds_eager = mock_camx_dataset()
    ds_lazy = ds_eager.chunk({"TSTEP": 1})

    res_eager = camx_preprocess(ds_eager, convert_to_ppb=True)
    res_lazy = camx_preprocess(ds_lazy, convert_to_ppb=True)

    # 1. Verify Dask array preservation for lazy path
    assert hasattr(res_lazy.o3.data, "dask")

    # 2. Verify dataset identity/allclose between eager and lazy execution
    xr.testing.assert_allclose(
        res_eager.drop_vars("history", errors="ignore"),
        res_lazy.compute().drop_vars("history", errors="ignore"),
    )

    # 3. Verify unit conversion (ppm to ppbV)
    assert res_eager.o3.attrs["units"] == "ppbV"

    # 4. Verify diagnostic variable calculation (nox = no + no2)
    assert "nox" in res_eager.data_vars
    expected_nox = res_eager.no + res_eager.no2
    xr.testing.assert_allclose(res_eager.nox, expected_nox)


@pytest.mark.parametrize("use_dask", [False, True])
def test_camx_reader_open_dataset(use_dask, tmp_path):
    """
    Test CAMxReader.open_dataset under both Eager (use_dask=False)
    and Lazy (use_dask=True) modes.
    """
    ds_mock = mock_camx_dataset()
    fname = tmp_path / "test_camx_modern.nc"
    ds_mock.to_netcdf(fname, engine="h5netcdf")

    reader = CAMxReader()
    ds = reader.open_dataset(
        str(fname),
        use_dask=use_dask,
        engine="h5netcdf",
        convert_to_ppb=True,
    )

    # Check Lazy vs Eager state
    if use_dask:
        assert ds.o3.chunks is not None
        assert hasattr(ds.o3.data, "dask")
    else:
        assert ds.o3.chunks is None

    # Check standardized dimensions and coordinates
    assert "time" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords
    assert "o3" in ds.data_vars
    assert ds.o3.attrs["units"] == "ppbV"

    # Check history provenance tracking
    assert "history" in ds.attrs
    hist = ds.attrs["history"]
    assert "Read CAMx data" in hist
    assert "Harmonized CAMx dataset" in hist


def test_camx_reader_harmonize():
    """Verify that CAMxReader.harmonize updates dataset and lineage attributes."""
    ds_mock = mock_camx_dataset()
    preprocessed = camx_preprocess(ds_mock)

    reader = CAMxReader()
    harmonized = reader.harmonize(preprocessed)

    assert "Harmonized CAMx dataset" in harmonized.attrs["history"]
    assert "o3" in harmonized.data_vars


if __name__ == "__main__":
    pytest.main([__file__])
