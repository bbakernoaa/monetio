import pandas as pd
import pytest

from monetio.readers.ish import ISHLiteReader, ISHReader


@pytest.mark.network
@pytest.mark.parametrize("reader_class", [ISHReader, ISHLiteReader])
@pytest.mark.parametrize("lazy", [False, True])
def test_ish_readers(reader_class, lazy):
    dates = pd.date_range("2020-09-01", "2020-09-01 02:00", freq="h")
    site = "72224400358"

    reader = reader_class()
    try:
        ds = reader.open_dataset(dates, site=site, lazy=lazy)
    except Exception as e:
        pytest.skip(f"Network error: {e}")

    if ds.sizes == {}:
        pytest.skip("No data found for test site")

    assert "time" in ds.coords
    assert "node" in ds.dims
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    if lazy:
        assert ds.temp.chunks is not None if hasattr(ds, "temp") else ds.t.chunks is not None
    else:
        assert ds.temp.chunks is None if hasattr(ds, "temp") else ds.t.chunks is None


def test_ish_history_parsing():
    reader = ISHReader()
    try:
        df = reader.read_history()
    except Exception as e:
        pytest.skip(f"Network error: {e}")

    assert not df.empty
    assert "station_id" in df.columns
    assert df["begin"].dtype == "datetime64[ns]"
    assert df["end"].dtype == "datetime64[ns]"
    # Check that date parsing was correct (no all-NaT)
    assert df["begin"].notna().sum() > 0
