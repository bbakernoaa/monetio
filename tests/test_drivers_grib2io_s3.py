import pytest
import xarray as xr

from monetio.readers.drivers import FileUtility, XarrayDriver, get_default_storage_options


def test_grib2io_s3_single_file_passes_url_and_storage_options(monkeypatch):
    driver = XarrayDriver()
    captured = {}

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files, fs=None, **kwargs: [files])

    def _fake_open_dataset(file_path, **kwargs):
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", _fake_open_dataset)

    driver.open(
        "s3://noaa-gfs-bdp-pds/example.grib2", engine="grib2io", storage_options={"anon": True}
    )

    assert captured["file_path"] == "s3://noaa-gfs-bdp-pds/example.grib2"
    assert captured["kwargs"]["storage_options"] == {"anon": True}


def test_grib2io_s3_multi_file_passes_urls_and_storage_options(monkeypatch):
    driver = XarrayDriver()
    captured = {}
    files = [
        "s3://noaa-gfs-bdp-pds/example_1.grib2",
        "s3://noaa-gfs-bdp-pds/example_2.grib2",
    ]

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files_in, fs=None, **kwargs: files_in)

    def _fake_open_mfdataset(file_list, **kwargs):
        captured["file_list"] = file_list
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_mfdataset", _fake_open_mfdataset)

    driver.open(files, engine="grib2io", storage_options={"anon": True})

    assert captured["file_list"] == files
    assert captured["kwargs"]["storage_options"] == {"anon": True}
    assert "use_icechunk" in captured["kwargs"]


def test_grib2io_passes_icechunk_backend_kwargs(monkeypatch):
    driver = XarrayDriver()
    captured = {}

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files, fs=None, **kwargs: [files])

    def _fake_open_dataset(file_path, **kwargs):
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", _fake_open_dataset)

    driver.open(
        "s3://noaa-gfs-bdp-pds/example.grib2",
        engine="grib2io",
        use_icechunk=False,
        icechunk_url="./zarr_stores/grib2io_icechunk",
        storage_options={"anon": True},
    )

    # use_icechunk is forwarded to xarray/grib2io backend kwargs
    assert captured["kwargs"]["use_icechunk"] is False
    # icechunk_url is a MONETIO concept stripped before reaching xarray/grib2io
    assert "icechunk_url" not in captured["kwargs"]


def test_grib2io_legacy_icechunk_repo_translates_to_native_kwargs(monkeypatch):
    driver = XarrayDriver()
    captured = {}

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files, fs=None, **kwargs: [files])

    def _fake_call_with_retries(func, *args, **kwargs):
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr("monetio.readers.drivers._call_with_retries", _fake_call_with_retries)

    with pytest.warns(DeprecationWarning, match="icechunk_repo"):
        driver.open(
            "s3://noaa-gfs-bdp-pds/example.grib2",
            engine="grib2io",
            icechunk_repo="./zarr_stores/legacy_repo",
            storage_options={"anon": True},
        )

    # icechunk_repo triggers use_icechunk=True which is forwarded to grib2io
    assert captured["kwargs"]["use_icechunk"] is True
    # icechunk_url/icechunk_repo are MONETIO concepts stripped before reaching xarray/grib2io
    assert "icechunk_url" not in captured["kwargs"]
    assert "icechunk_repo" not in captured["kwargs"]


def test_grib2io_s3_sets_default_storage_options(monkeypatch):
    driver = XarrayDriver()
    captured = {}

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files, fs=None, **kwargs: [files])

    def _fake_open_dataset(file_path, **kwargs):
        captured["file_path"] = file_path
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", _fake_open_dataset)

    path = "s3://noaa-gfs-bdp-pds/example.grib2"
    driver.open(path, engine="grib2io")

    assert captured["kwargs"]["storage_options"] == get_default_storage_options(path)


def test_get_fs_passes_s3_kwargs_through(monkeypatch):
    captured = {}

    def _fake_filesystem(protocol, **kwargs):
        captured["protocol"] = protocol
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("fsspec.filesystem", _fake_filesystem)

    FileUtility.get_fs(
        "s3://noaa-gfs-bdp-pds/example.grib2",
        anon=True,
        config_kwargs={"connect_timeout": 30, "region_name": "us-west-2"},
    )

    assert captured["protocol"] == "s3"
    assert captured["kwargs"]["anon"] is True
    assert captured["kwargs"]["config_kwargs"] == {
        "connect_timeout": 30,
        "region_name": "us-west-2",
    }


def test_grib2_virtual_flag_warns_and_uses_native_path(monkeypatch):
    driver = XarrayDriver()
    captured = {}
    files = [
        "s3://noaa-gfs-bdp-pds/example_1.grib2",
        "s3://noaa-gfs-bdp-pds/example_2.grib2",
    ]

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files_in, fs=None, **kwargs: files_in)
    monkeypatch.setattr(
        "monetio.readers.drivers._select_store",
        lambda file_list, storage_options: (_ for _ in ()).throw(
            AssertionError("_select_store should not be called for native grib2io path")
        ),
    )

    def _fake_open_mfdataset(file_list, **kwargs):
        captured["file_list"] = file_list
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_mfdataset", _fake_open_mfdataset)

    with pytest.warns(DeprecationWarning, match="use_virtualizarr is ignored"):
        driver.open(files, use_virtualizarr=True, engine="grib2io", storage_options={"anon": True})

    assert captured["file_list"] == files
    assert captured["kwargs"]["storage_options"] == {"anon": True}


def test_grib2io_s3_multifile_passes_all_kwargs_to_open_mfdataset(monkeypatch):
    """Verify that grib2io-specific kwargs (max_workers, network_timeout, etc.)
    are passed through to xr.open_mfdataset without modification."""
    driver = XarrayDriver()
    captured = {}

    files = [
        "s3://noaa-gfs-bdp-pds/gfs.20250101/00/atmos/gfs.t00z.pgrb2.0p25.f000",
        "s3://noaa-gfs-bdp-pds/gfs.20250102/00/atmos/gfs.t00z.pgrb2.0p25.f000",
        "s3://noaa-gfs-bdp-pds/gfs.20250103/00/atmos/gfs.t00z.pgrb2.0p25.f000",
    ]

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files_in, fs=None, **kwargs: files_in)

    def _fake_open_mfdataset(file_list, **kwargs):
        captured["file_list"] = file_list
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_mfdataset", _fake_open_mfdataset)

    storage_options = {
        "anon": True,
        "config_kwargs": {
            "connect_timeout": 30,
            "read_timeout": 120,
            "retries": {"max_attempts": 10, "mode": "adaptive"},
        },
    }

    driver.open(
        files,
        engine="grib2io",
        use_icechunk=False,
        storage_options=storage_options,
        filters={"shortName": "TMP", "typeOfFirstFixedSurface": 103},
        max_workers=4,
        network_timeout=300,
        max_concurrent_requests=2,
        chunks={},
    )

    assert captured["file_list"] == files
    assert captured["kwargs"]["engine"] == "grib2io"
    assert captured["kwargs"]["use_icechunk"] is False
    assert captured["kwargs"]["storage_options"] == storage_options
    assert captured["kwargs"]["filters"] == {"shortName": "TMP", "typeOfFirstFixedSurface": 103}
    assert captured["kwargs"]["max_workers"] == 4
    assert captured["kwargs"]["network_timeout"] == 300
    assert captured["kwargs"]["max_concurrent_requests"] == 2
    assert captured["kwargs"]["chunks"] == {}
