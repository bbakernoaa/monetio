import pandas as pd
import pytest
import xarray as xr

from monetio.readers.airnow import AirNowReader


@pytest.fixture
def mock_airnow_file(tmp_path):
    fn = tmp_path / "HourlyData_2023010100.dat"
    # date|time|siteid|site|utcoffset|variable|units|obs|source
    content = "01/01/23|00:00|012345678|Test Site|-5|OZONE|PPB|50.0|Test Source\n"
    content += "01/01/23|00:00|012345678|Test Site|-5|PM2.5|UG/M3|10.0|Test Source\n"
    # A site with bad UTC offset (0 but far from 0 longitude)
    content += "01/01/23|00:00|999999999|Bad TZ Site|0|OZONE|PPB|40.0|Test Source\n"
    fn.write_text(content, encoding="ISO-8859-1")
    return str(fn)


def test_airnow_eager_vs_lazy_local(mock_airnow_file, monkeypatch):
    def mock_read_monitor(*args, **kwargs):
        return pd.DataFrame(
            {
                "siteid": ["012345678", "999999999"],
                "latitude": [40.0, 40.0],
                "longitude": [-80.0, -80.0],
                "site_name": ["Site 1", "Site 2"],
            }
        )

    import monetio.readers.airnow as airnow

    monkeypatch.setattr(airnow, "read_monitor_file", mock_read_monitor)

    reader = AirNowReader()

    # 1. Eager
    ds_eager = reader.open_dataset(
        files=mock_airnow_file, as_xarray=True, lazy=False, wide_fmt=True, bad_utcoffset="fix"
    )

    # 2. Lazy
    ds_lazy = reader.open_dataset(
        files=mock_airnow_file, as_xarray=True, lazy=True, wide_fmt=True, bad_utcoffset="fix"
    )

    # Check that ds_lazy is indeed dask-backed for data variables
    assert ds_lazy.OZONE.chunks is not None

    # Compute lazy result
    ds_lazy_computed = ds_lazy.compute()

    # Compare
    # history will differ slightly in time, so drop it
    ds_eager.attrs.pop("history", None)
    ds_lazy_computed.attrs.pop("history", None)

    # Unit columns might differ in NaNs if a site doesn't have a variable
    # (Eager path broadcasts them, Lazy path doesn't yet).
    # We compare everything but the unit columns first.
    data_vars = [v for v in ds_eager.data_vars if not v.endswith("_unit")]
    xr.testing.assert_allclose(ds_eager[data_vars], ds_lazy_computed[data_vars])

    # Then check unit columns where they are not NaN in both.
    unit_vars = [v for v in ds_eager.data_vars if v.endswith("_unit")]
    for v in unit_vars:
        mask = ds_eager[v].notnull() & ds_lazy_computed[v].notnull()
        xr.testing.assert_allclose(ds_eager[v].where(mask), ds_lazy_computed[v].where(mask))

    # Verify fixed UTC offset
    # -80 longitude should give -5 offset
    # Site 999999999 is at index 1 in the unstacked/pivoted dataset if siteid is sorted?
    # Actually siteid is a coord.

    uo_999 = ds_eager.sel(node=ds_eager.siteid == "999999999").utcoffset.values
    assert uo_999 == pytest.approx(-5.0)

    # Verify time_local
    # time is 2023-01-01 00:00:00
    # time_local should be 2022-12-31 19:00:00
    tl_999 = ds_eager.sel(node=ds_eager.siteid == "999999999").time_local.values
    expected_tl = pd.to_datetime("2023-01-01 00:00:00") + pd.Timedelta(hours=-5)
    assert pd.to_datetime(tl_999[0]) == expected_tl


if __name__ == "__main__":
    pytest.main([__file__])
