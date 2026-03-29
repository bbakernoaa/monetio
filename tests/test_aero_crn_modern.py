import numpy as np
import pandas as pd
import xarray as xr

from monetio.readers.crn import read_crn
from monetio.readers.time_utils import parse_yyyymmdd_hhmm


def test_aero_crn_read_eager_lazy(tmp_path):
    """
    Verify read_crn logic for both Eager and Lazy backends.
    """
    # 1. Create a mock CRN hourly file
    d = tmp_path / "crn_hourly"
    d.mkdir()
    f = d / "CRNH0203-2023-AL_Fairhope_3_NE.txt"

    # Mock data following CRN hourly format (HCOLS)
    # WBANNO UTC_DATE UTC_TIME LST_DATE LST_TIME CRX_VN LONGITUDE LATITUDE ...
    content = (
        "63893 20230101 1200 20230101 0600 1.000 -87.88 30.54 " + "0.0 " * 30 + "\n"
        "63893 20230101 1300 20230101 0700 1.000 -87.88 30.54 " + "1.0 " * 30 + "\n"
    )
    f.write_text(content)

    # 2. Eager Read
    df_eager = read_crn(str(f))
    assert isinstance(df_eager, pd.DataFrame)
    assert not df_eager.empty
    assert "time" in df_eager.columns
    assert "time_local" in df_eager.columns
    assert df_eager["time"].iloc[0] == pd.Timestamp("2023-01-01 12:00")

    # 3. Verify the parser logic directly (Aero Protocol core logic)
    # Scalar check
    t_scalar = parse_yyyymmdd_hhmm(20230101, 1200)
    assert t_scalar == np.datetime64("2023-01-01T12:00")

    # List check
    t_list = parse_yyyymmdd_hhmm([20230101, 20230101], [1200, 1300])
    assert len(t_list) == 2
    assert t_list[1] == np.datetime64("2023-01-01T13:00")

    # Dask/Xarray check
    import dask.array as da

    yyyymmdd = np.array([20230101, 20230101])
    hhmm = np.array([1200, 1300])

    lazy_time = xr.apply_ufunc(
        parse_yyyymmdd_hhmm,
        xr.DataArray(da.from_array(yyyymmdd, chunks=1)),
        xr.DataArray(da.from_array(hhmm, chunks=1)),
        dask="parallelized",
        output_dtypes=[np.dtype("datetime64[ns]")],
    ).compute()

    np.testing.assert_array_equal(t_list, lazy_time.values)

    # 4. Check provenance via reader
    from monetio.readers.crn import CRNReader

    reader = CRNReader()
    ds_eager = reader.open_dataset(files=str(f), lazy=False, as_xarray=True)
    assert "history" in ds_eager.attrs
    assert "Read CRN" in ds_eager.attrs["history"] or "Merged with CRN" in ds_eager.attrs["history"]
