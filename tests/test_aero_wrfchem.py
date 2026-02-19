import numpy as np
import xarray as xr

from monetio.readers.wrfchem import WRFChemReader, wrfchem_preprocess


def make_mock_wrfchem_ds():
    """Create a mock WRF-Chem dataset."""
    nx, ny, nz, nt = 4, 5, 3, 2
    ds = xr.Dataset(
        {
            "OZONE": (("time", "z", "y", "x"), np.random.rand(nt, nz, ny, nx).astype(np.float32)),
            "PM2_5_DRY": (
                ("time", "z", "y", "x"),
                np.random.rand(nt, nz, ny, nx).astype(np.float32),
            ),
            "ALT": (("time", "z", "y", "x"), np.ones((nt, nz, ny, nx), dtype=np.float32) * 0.8),
            "XLAT": (
                ("time", "y", "x"),
                np.broadcast_to(np.arange(ny)[:, None], (nt, ny, nx)).astype(np.float32),
            ),
            "XLONG": (
                ("time", "y", "x"),
                np.broadcast_to(np.arange(nx)[None, :], (nt, ny, nx)).astype(np.float32),
            ),
            "Times": (
                ("time", "DateStrLen"),
                np.array([list("2023-01-01_00:00:00"), list("2023-01-01_01:00:00")], dtype="|S1"),
            ),
        },
        coords={
            "time": np.arange(nt),
            "y": np.arange(ny),
            "x": np.arange(nx),
            "z": np.arange(nz),
        },
    )
    ds.OZONE.attrs["units"] = "ppmv"
    ds.PM2_5_DRY.attrs["units"] = "ug/kg-dryair"
    return ds


def test_wrfchem_preprocess_eager():
    ds = make_mock_wrfchem_ds()
    res = wrfchem_preprocess(ds)

    assert "time" in res.coords
    assert res.time.dtype == "datetime64[ns]"
    assert res.OZONE.attrs["units"] == "ppbV"
    assert res.PM2_5_DRY.attrs["units"] == r"$\mu g m^{-3}$"
    assert "latitude" in res.coords
    assert "longitude" in res.coords
    # Check unit conversion value: 0.8 ALT -> rho = 1.25. pm * 1.25.
    np.testing.assert_allclose(res.PM2_5_DRY.values, ds.PM2_5_DRY.values / 0.8)


def test_wrfchem_preprocess_lazy():
    ds = make_mock_wrfchem_ds().chunk({"time": 1, "z": 1})
    res = wrfchem_preprocess(ds)

    assert res.OZONE.chunks is not None
    assert "time" in res.coords
    assert res.OZONE.attrs["units"] == "ppbV"

    res_computed = res.compute()
    assert res_computed.time.dtype == "datetime64[ns]"
    np.testing.assert_allclose(res_computed.PM2_5_DRY.values, ds.PM2_5_DRY.values / 0.8, rtol=1e-5)


def test_wrfchem_reader_open_dataset():
    ds = make_mock_wrfchem_ds()
    reader = WRFChemReader()

    import unittest.mock as mock

    # We want to test that open_dataset calls driver.open and harmonize.
    # Since driver.open is expected to return the PREPROCESSED dataset,
    # we should mock it to return a preprocessed dataset if we want to check harmonize.
    ds_pre = wrfchem_preprocess(ds)
    with mock.patch("monetio.readers.drivers.XarrayDriver.open", return_value=ds_pre):
        res = reader.open_dataset("dummy.nc")

    assert "history" in res.attrs
    assert "Read WRF-Chem data" in res.attrs["history"]
    assert res.time.dtype == "datetime64[ns]"
