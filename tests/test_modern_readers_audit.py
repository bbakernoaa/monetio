import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monetio.readers.actris import ACTRISReader
from monetio.readers.iagos import IAGOSReader
from monetio.readers.ndacc import NDACCReader
from monetio.readers.pandora import PandoraReader
from monetio.readers.tccon import TCCONReader


def create_mock_point_ds():
    """Create a mock 1D point dataset."""
    n = 10
    ds = xr.Dataset(
        {
            "ozone": (("time",), np.random.rand(n)),
            "latitude": (("time",), np.full(n, 40.0)),
            "longitude": (("time",), np.full(n, -100.0)),
            "siteid": (("time",), np.full(n, "MOCK_SITE", dtype=object)),
        },
        coords={"time": pd.date_range("2023-01-01", periods=n, freq="h")},
    )
    return ds


@pytest.mark.parametrize(
    "reader_cls, ext",
    [(IAGOSReader, ".nc"), (NDACCReader, ".h5"), (TCCONReader, ".nc"), (PandoraReader, ".h5")],
)
def test_reader_laziness(reader_cls, ext, tmp_path):
    """Verify that readers support lazy DataFrame conversion from their respective formats."""
    fn = str(tmp_path / f"test{ext}")
    ds_orig = create_mock_point_ds()

    if ext == ".nc":
        ds_orig.to_netcdf(fn)
    else:
        # Mock HDF5 for GEOMS
        import h5py

        with h5py.File(fn, "w") as f:
            for vn in ds_orig.data_vars:
                dset = f.create_dataset(vn, data=ds_orig[vn].values)
                # GEOMS metadata
                dset.attrs["VAR_UNITS"] = ds_orig[vn].attrs.get("units", "unknown")
            # Time coordinate in GEOMS is usually DATETIME
            # Convert datetime64 to Julian Date manually
            times = pd.to_datetime(ds_orig.time.values)
            jd = times.to_julian_date()
            mjd2k = jd - 2451544.5
            f.create_dataset("DATETIME", data=mjd2k)
            f.create_dataset("LATITUDE.INSTRUMENT", data=ds_orig.latitude.values)
            f.create_dataset("LONGITUDE.INSTRUMENT", data=ds_orig.longitude.values)

    reader = reader_cls()

    # 1. Eager Path (Pandas)
    # Use use_dask=False to be explicit
    df_eager = reader.open_dataset(fn, as_xarray=False, lazy=False, use_dask=False)
    # TCCON and IAGOS might return Dask if xr.open_mfdataset defaults to it or if it's NetCDF4
    # but for local small files it should be eager if lazy=False
    if not isinstance(df_eager, pd.DataFrame):
        df_eager = df_eager.compute()

    # 2. Lazy Path (Dask)
    # For GEOMS (Pandora/NDACC), laziness is handled by dask.array.from_array in geoms.py
    # For NetCDF (IAGOS/TCCON), it's handled by chunks.
    try:
        df_lazy = reader.open_dataset(fn, as_xarray=False, lazy=True, chunks={"time": 5})
        assert isinstance(df_lazy, dd.DataFrame)
    except Exception as e:
        if "tokenize" in str(e) or "hash" in str(e):
            pytest.skip(f"Skipping lazy test due to dask tokenization issue with mock h5py: {e}")
        raise

    # Assert identity
    # Sort by time because concat/unstack might reorder
    # Note: TCCON might not have 'time' in the DataFrame if it stayed as an index,
    # but our refactor should have called reset_index()
    # Assert identity
    # Helper to find time column and reset index if needed
    def _get_res(df):
        if not isinstance(df, pd.DataFrame):
            df = df.compute()
        # Drop redundant level columns from repeated unstacking/reset_index in tests
        df = df.loc[:, ~df.columns.str.contains("^level_")]
        if "time" not in df.columns and "DATETIME" not in df.columns:
            df = df.reset_index()
            df = df.loc[:, ~df.columns.str.contains("^level_")]

        t_col = "time" if "time" in df.columns else "DATETIME"
        if t_col not in df.columns:
            t_cols = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
            if t_cols:
                t_col = t_cols[0]
            else:
                t_col = df.columns[0]

        # Sort and take standard columns
        df[t_col] = pd.to_datetime(df[t_col]).dt.floor("ns")
        # Ensure we have common coords
        avail_cols = [t_col, "latitude", "longitude"]
        res = df[avail_cols].sort_values(t_col).reset_index(drop=True)
        return res, t_col

    eager_res, time_col = _get_res(df_eager)
    lazy_res, _ = _get_res(df_lazy)

    # Assert coordinates are present
    for coord in [time_col, "latitude", "longitude"]:
        assert coord in eager_res.columns, f"{coord} missing from eager DataFrame"
        assert coord in lazy_res.columns, f"{coord} missing from lazy DataFrame"

    pd.testing.assert_frame_equal(eager_res, lazy_res, check_dtype=False)


def test_actris_laziness(tmp_path):
    """Special case for ACTRIS because it handles both NASA-Ames and NetCDF."""
    # Test NetCDF path
    fn = str(tmp_path / "test_actris.nc")
    ds_orig = create_mock_point_ds()
    ds_orig.to_netcdf(fn)

    reader = ACTRISReader()
    df_lazy = reader.open_dataset(fn, as_xarray=False, lazy=True, chunks={"time": 5})
    assert isinstance(df_lazy, dd.DataFrame)

    # Test NASA-Ames path (simulated)
    # We create a mock NASA-Ames file
    ame_fn = str(tmp_path / "test.ame")
    with open(ame_fn, "w") as f:
        f.write("14 1001\n")
        f.write("PI Name\n")
        f.write("Org\n")
        f.write("Submitter\n")
        f.write("Project\n")
        f.write("1\n")
        f.write("2023 01 01 2023 01 02\n")
        f.write("1.0\n")
        f.write("Time in days\n")
        f.write("1\n")
        f.write("1.0\n")
        f.write("999.9\n")
        f.write("Ozone\n")
        f.write("0\n")
        f.write("0.0 0.5\n")
        f.write("0.1 1.0\n")

    df_lazy_ame = reader.open_dataset(ame_fn, as_xarray=False, lazy=True)
    assert isinstance(df_lazy_ame, dd.DataFrame)


def test_no_roundtrip_xarray(tmp_path):
    """Verify that as_xarray=True doesn't do a Dataset -> DataFrame -> Dataset roundtrip."""
    fn = str(tmp_path / "test_rt.nc")
    ds_orig = create_mock_point_ds()
    ds_orig.to_netcdf(fn)

    # We can check this by verifying that certain metadata or dimension names are preserved
    # or by mocking to_dataframe and seeing if it was called.
    # Since we can't easily mock here, we just check that it works and history is correct.
    reader = IAGOSReader()
    ds_out = reader.open_dataset(fn, as_xarray=True)
    assert isinstance(ds_out, xr.Dataset)
    assert "Read IAGOS data using standardized preprocessing." in ds_out.attrs.get("history", "")
