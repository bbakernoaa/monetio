import pytest

from monetio.models.icap_mme import open_dataset, open_mfdataset


def wrap_network_test(func):
    import functools

    import requests

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
            if isinstance(e, AssertionError):
                raise
            pytest.skip(f"Network or data retrieval error: {e}")

    return wrapper


@wrap_network_test
def test_open_dataset_bad_date():
    with pytest.raises(ValueError, match="File does not exist"):
        open_dataset("1990-08-01")


def test_open_dataset_invalid_param():
    date = "2019-08-01"

    with pytest.raises(ValueError, match="Invalid input for 'product'"):
        open_dataset(date, product="asdf")
        open_mfdataset([date], product="asdf")

    with pytest.raises(ValueError, match="Invalid input for 'data_var'"):
        open_dataset(date, data_var="asdf")
        open_mfdataset([date], data_var="asdf")


@wrap_network_test
@pytest.mark.parametrize(
    "date,product,data_var",
    [
        ("2019-08-01", "MME", "totaldustaod550"),
        ("2024-02-01", "C4", "dustaod550"),
    ],
)
def test_open_dataset(tmp_path, monkeypatch, date, product, data_var):
    ds = open_dataset(date, product=product, data_var=data_var, download=False)
    assert set(ds.dims) == {"time", "lat", "lon"}

    monkeypatch.chdir(tmp_path)
    ds_dl = open_dataset(date, product=product, data_var=data_var, download=True)
    assert len(sorted(tmp_path.glob("*.nc"))) == 1
    assert set(ds_dl.dims) == {"time", "lat", "lon"}

    assert ds_dl.equals(ds)


@wrap_network_test
def test_open_mfdataset(tmp_path, monkeypatch):
    dates = ["2023-08-01", "2023-08-02"]
    product = "C4"
    data_var = "dustaod550"

    ds = open_mfdataset(dates, product=product, data_var=data_var, download=False)
    assert set(ds.dims) == {"time", "lat", "lon"}
    assert ds["dust_aod_mean"].chunks is None, "not Dask-backed"
    assert (
        ~ds.time.to_series().duplicated(keep=False)
    ).sum() == 8, "all overlap except first and last day"

    monkeypatch.chdir(tmp_path)
    ds_dl = open_mfdataset(dates, product=product, data_var=data_var, download=True)
    assert len(sorted(tmp_path.glob("*.nc"))) == 2
    assert set(ds_dl.dims) == {"time", "lat", "lon"}
    assert ds_dl["dust_aod_mean"].chunks is not None

    assert ds_dl.equals(ds)
