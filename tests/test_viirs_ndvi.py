import sys

import pandas as pd
import pytest

from monetio.sat.nesdis_viirs_ndvi_aws_gridded import open_dataset, open_mfdataset

if sys.version_info < (3, 7):
    pytest.skip("s3fs requires Python 3.7+", allow_module_level=True)


def test_open_dataset_no_data():
    with pytest.raises(ValueError, match="File does not exist on AWS:"):
        open_dataset("1900-01-01")


def test_open_dataset():
    date = "2023-01-01"
    ds = open_dataset(date)
    assert set(ds.dims) >= {"time", "latitude", "longitude"}
    assert ds.sizes["time"] == 1
    assert ds.sizes["latitude"] == 3600
    assert ds.sizes["longitude"] == 7200
    assert ds["time"] == pd.to_datetime(date)
    assert "NDVI" in ds.data_vars
    assert ds["NDVI"].dims == ("time", "latitude", "longitude")


def test_open_mfdataset():
    dates = ["2023-01-01", "2023-01-02"]
    ds = open_mfdataset(dates)
    assert (ds["time"] == pd.DatetimeIndex(dates)).all()


def test_open_mfdataset_error():
    dates = ["1900-01-01", "2023-01-01"]

    with pytest.warns(UserWarning, match="File does not exist on AWS:"):
        ds = open_mfdataset(dates)
        assert ds.sizes["time"] == 1
        assert ds["time"] == pd.to_datetime(dates[-1])

    with pytest.raises(ValueError, match="File does not exist on AWS:"):
        _ = open_mfdataset(dates, error_missing=True)

    with pytest.raises(ValueError, match="Files not available for product and dates"), pytest.warns(
        UserWarning, match="File does not exist on AWS:"
    ):
        _ = open_mfdataset(dates[:1], error_missing=False)
