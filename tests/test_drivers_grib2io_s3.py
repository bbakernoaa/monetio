import xarray as xr

from monetio.readers.drivers import FileUtility, XarrayDriver


class _FakeFS:
    def __init__(self):
        self.calls = []

    def open(self, path, mode="rb"):
        handle = f"fh::{path}::{mode}"
        self.calls.append((path, mode, handle))
        return handle


def test_grib2io_s3_single_file_uses_file_object(monkeypatch):
    driver = XarrayDriver()
    fake_fs = _FakeFS()
    captured = {}

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files, fs=None, **kwargs: [files])
    monkeypatch.setattr(FileUtility, "get_fs", lambda path, **kwargs: fake_fs)

    def _fake_open_dataset(file_obj, **kwargs):
        captured["file_obj"] = file_obj
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_dataset", _fake_open_dataset)

    driver.open(
        "s3://noaa-gfs-bdp-pds/example.grib2", engine="grib2io", storage_options={"anon": True}
    )

    assert captured["file_obj"] == "fh::s3://noaa-gfs-bdp-pds/example.grib2::rb"
    assert "storage_options" not in captured["kwargs"]


def test_grib2io_s3_multi_file_uses_file_objects(monkeypatch):
    driver = XarrayDriver()
    fake_fs = _FakeFS()
    captured = {}
    files = [
        "s3://noaa-gfs-bdp-pds/example_1.grib2",
        "s3://noaa-gfs-bdp-pds/example_2.grib2",
    ]

    monkeypatch.setattr(FileUtility, "expand_paths", lambda files_in, fs=None, **kwargs: files_in)
    monkeypatch.setattr(FileUtility, "get_fs", lambda path, **kwargs: fake_fs)

    def _fake_open_mfdataset(file_list, **kwargs):
        captured["file_list"] = file_list
        captured["kwargs"] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(xr, "open_mfdataset", _fake_open_mfdataset)

    driver.open(files, engine="grib2io", storage_options={"anon": True})

    assert captured["file_list"] == [
        "fh::s3://noaa-gfs-bdp-pds/example_1.grib2::rb",
        "fh::s3://noaa-gfs-bdp-pds/example_2.grib2::rb",
    ]
    assert "storage_options" not in captured["kwargs"]
