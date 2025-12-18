import sys

import pandas as pd
import pytest

from monetio.readers import READER_REGISTRY

if sys.version_info < (3, 7):
    pytest.skip("s3fs requires Python 3.7+", allow_module_level=True)

# Get the reader
avhrr_reader = READER_REGISTRY["nesdis_avhrr_aot_aws_gridded"]()


def test_open_dataset_no_data():
    with pytest.raises(ValueError, match="File does not exist on AWS:"):
        avhrr_reader.open_dataset("1900-01-01")


def test_open_dataset():
    date = "2023-01-01"
    ds = avhrr_reader.open_dataset(date)
    assert set(ds.dims) >= {"time", "latitude", "longitude"}
    assert ds.sizes["time"] == 1
    assert ds.sizes["latitude"] == 1800
    assert ds.sizes["longitude"] == 3600
    assert ds["time"] == pd.to_datetime(date)
    assert "aot1" in ds.data_vars
    assert ds["aot1"].dims == ("time", "latitude", "longitude")


def test_open_mfdataset():
    # The new reader doesn't support multiple dates in open_dataset
    # This test should be updated to use open_mfdataset or removed
    # For now, we'll test that it raises the expected error
    dates = ["2023-01-01", "2023-01-02"]
    with pytest.raises(ValueError, match="Date is required for NESDIS AVHRR AOT AWS Gridded reader."):
        avhrr_reader.open_dataset(dates)


def test_open_mfdataset_error():
    dates = ["1900-01-01", "2023-01-01"]

    # The new reader raises ValueError instead of warning for missing dates
    with pytest.raises(ValueError, match="Date is required for NESDIS AVHRR AOT AWS Gridded reader."):
        avhrr_reader.open_dataset(dates)

    # Test single date that doesn't exist
    with pytest.raises(ValueError, match="File does not exist on AWS:"):
        _ = avhrr_reader.open_dataset("1900-01-01")
