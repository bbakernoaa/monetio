import builtins

import numpy as np
import pytest
import xarray as xr


def make_analysis():
    return xr.Dataset(
        {"temperature_2m": (("time", "latitude", "longitude"), np.arange(8).reshape(2, 2, 2))},
        coords={"time": np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[ns]"), "latitude": [10, 0], "longitude": [0, 10]},
        attrs={"source": "fixture"},
    )


def test_catalog_reader_selects_and_preserves_native_coordinates(monkeypatch):
    import dynamical_catalog
    from monetio.readers.dynamical import DynamicalReader

    calls = {}

    def open_dataset(dataset_id, **kwargs):
        calls["id"] = dataset_id
        calls["kwargs"] = kwargs
        return make_analysis()

    monkeypatch.setattr(dynamical_catalog, "open", open_dataset)
    result = DynamicalReader().open_dataset(dataset="noaa-gfs-analysis", variables=["temperature_2m"], sel={"latitude": slice(10, 10)})
    assert calls["id"] == "noaa-gfs-analysis"
    assert list(result.data_vars) == ["temperature_2m"]
    assert list(result.latitude.values) == [10]
    assert result.longitude.dims == ("longitude",)
    assert result.attrs["source"] == "fixture"
    assert "dynamical.org" in result.attrs["history"]


def test_catalog_reader_rejects_missing_dataset_id():
    from monetio.readers.dynamical import DynamicalReader
    with pytest.raises(ValueError, match="dataset"):
        DynamicalReader().open_dataset(dataset="")


def test_catalog_reader_rejects_files_argument():
    from monetio.readers.dynamical import DynamicalReader
    with pytest.raises(ValueError, match="files"):
        DynamicalReader().open_dataset(files="local.zarr", dataset="noaa-gfs-analysis")


def test_catalog_reader_rejects_unknown_variable(monkeypatch):
    import dynamical_catalog
    from monetio.readers.dynamical import DynamicalReader
    monkeypatch.setattr(dynamical_catalog, "open", lambda dataset_id, **kwargs: make_analysis())
    with pytest.raises(KeyError, match="missing"):
        DynamicalReader().open_dataset(dataset="noaa-gfs-analysis", variables=["missing"])


def test_catalog_reader_reports_missing_optional_dependency(monkeypatch):
    from monetio.readers.dynamical import DynamicalReader
    real_import = builtins.__import__
    def blocked_import(name, *args, **kwargs):
        if name == "dynamical_catalog":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match="dynamical-catalog"):
        DynamicalReader().open_dataset(dataset="noaa-gfs-analysis")


def test_catalog_reader_preserves_forecast_dimensions(monkeypatch):
    import dynamical_catalog
    from monetio.readers.dynamical import DynamicalReader

    ds = xr.Dataset(
        {"temperature_2m": (("init_time", "lead_time", "latitude", "longitude"), np.zeros((1, 2, 1, 1)))},
        coords={"init_time": [np.datetime64("2025-01-01")], "lead_time": [0, 6], "latitude": [0], "longitude": [0]},
    )
    monkeypatch.setattr(dynamical_catalog, "open", lambda dataset_id, **kwargs: ds)
    result = DynamicalReader().open_dataset(dataset="noaa-gfs-forecast")
    assert result.temperature_2m.dims == ("init_time", "lead_time", "latitude", "longitude")
    assert list(result.lead_time.values) == [0, 6]


def test_catalog_reader_preserves_ensemble_and_projection_metadata(monkeypatch):
    import dynamical_catalog
    from monetio.readers.dynamical import DynamicalReader

    ds = xr.Dataset(
        {"precipitation": (("member", "y", "x"), np.ones((2, 2, 2)))},
        coords={"member": [0, 1], "x": [100, 200], "y": [300, 400],
                "latitude": (("y", "x"), [[1, 1], [0, 0]]),
                "longitude": (("y", "x"), [[2, 3], [2, 3]])},
        attrs={"crs": "EPSG:4326"},
    )
    monkeypatch.setattr(dynamical_catalog, "open", lambda dataset_id, **kwargs: ds)
    result = DynamicalReader().open_dataset(dataset="noaa-gefs-forecast", variables="precipitation")
    assert result.precipitation.dims == ("member", "y", "x")
    assert result.latitude.dims == ("y", "x")
    assert result.attrs["crs"] == "EPSG:4326"


def test_universal_loader_routes_to_dynamical_reader(monkeypatch):
    import dynamical_catalog
    import monetio

    monkeypatch.setattr(dynamical_catalog, "open", lambda dataset_id, **kwargs: make_analysis())
    result = monetio.load("dynamical", dataset="noaa-gfs-analysis", variables=["temperature_2m"])
    assert list(result.data_vars) == ["temperature_2m"]
    assert "dynamical" in monetio.available_sources()


def test_selection_is_lazy_before_reading_values(monkeypatch):
    import dask.array as da
    import dynamical_catalog
    from monetio.readers.dynamical import DynamicalReader

    ds = make_analysis().chunk({"time": 1, "latitude": 1, "longitude": 1})
    monkeypatch.setattr(dynamical_catalog, "open", lambda dataset_id, **kwargs: ds)
    result = DynamicalReader().open_dataset(
        dataset="noaa-gfs-analysis", variables="temperature_2m", sel={"latitude": 10}
    )
    assert isinstance(result.temperature_2m.data, da.Array)
    assert result.temperature_2m.sizes["time"] == 2
    assert "latitude" not in result.temperature_2m.dims
    assert ds.temperature_2m.chunks is not None
