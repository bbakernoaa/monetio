"""``monetio.load`` must forward a reader's own ``source`` keyword.

Readers such as era5/gdas/gfs/hrrr/ish accept ``source=`` in
``open_dataset``; before the positional-only fix, ``load("era5",
source="aws")`` raised ``TypeError: got multiple values for argument
'source'`` because the loader's own parameter swallowed the kwarg.
"""

import pytest

import monetio


class _RecordingReader:
    """Stub reader capturing the kwargs the loader forwards."""

    captured = None

    def open_dataset(self, files=None, **kwargs):
        type(self).captured = {"files": files, **kwargs}
        return "ok"


@pytest.fixture()
def recording_reader(monkeypatch):
    """Route any source name to the recording stub."""
    _RecordingReader.captured = None
    monkeypatch.setattr(monetio._lazy, "get_reader_class", lambda name: _RecordingReader)
    return _RecordingReader


def test_source_kwarg_forwarded_to_reader(recording_reader):
    """A `source` kwarg alongside a positional name belongs to the reader."""
    result = monetio.load("era5", source="aws", variables=["temperature_2m"])
    assert result == "ok"
    assert recording_reader.captured == {
        "files": None,
        "source": "aws",
        "variables": ["temperature_2m"],
    }


def test_legacy_keyword_form_still_works(recording_reader):
    """``load(source="name")`` without a positional name keeps working."""
    result = monetio.load(source="era5", extra=1)
    assert result == "ok"
    assert recording_reader.captured == {"files": None, "extra": 1}


def test_files_still_routed_to_files_parameter(recording_reader):
    """`files=` remains the loader's explicit parameter, not a reader kwarg."""
    monetio.load("cmaq", files="a.nc", source="aws")
    assert recording_reader.captured == {"files": "a.nc", "source": "aws"}


def test_missing_source_raises_typeerror():
    """Neither positional nor keyword source is a clear TypeError."""
    with pytest.raises(TypeError, match="source"):
        monetio.load()


def test_era5_receives_source_through_load():
    """End-to-end (no network): an invalid reader `source` must reach ERA5's
    own validation, proving the kwarg is forwarded instead of colliding."""
    with pytest.raises(ValueError, match="source must be"):
        monetio.load("era5", source="bogus")