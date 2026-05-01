# Feature: virtualizarr-reader-refactor, Property 2: VirtualiZarr Kwargs Forwarding Through Load
"""Property-based test for VirtualiZarr kwargs forwarding through monetio.load().

**Validates: Requirements 3.1, 3.2, 3.3**

For any registered reader name and any combination of VirtualiZarr-related kwargs
(use_virtualizarr, virtualizarr_file, virtualizarr_backend, icechunk_repo), calling
``monetio.load(source, files, **vz_kwargs)`` SHALL forward those kwargs to the reader's
``open_dataset()`` method for GriddedReaders, and SHALL silently ignore them for
PointReaders without raising an error.
"""

import importlib
import unittest.mock as mock

from hypothesis import given, settings
from hypothesis import strategies as st

import monetio
from monetio.readers.base import READER_REGISTRY, GriddedReader, PointReader

# ---------------------------------------------------------------------------
# Ensure readers are registered by importing their modules
# ---------------------------------------------------------------------------

# Import a representative set of GriddedReader and PointReader modules
# so they register themselves in READER_REGISTRY.
_GRIDDED_SOURCES = ["cmaq", "camx", "chimere", "wrfchem", "merra2"]
_POINT_SOURCES = ["airnow", "aqs", "improve", "pams", "crn"]

for _src in _GRIDDED_SOURCES + _POINT_SOURCES:
    if _src not in READER_REGISTRY and _src in monetio._READER_MODULES:
        importlib.import_module(monetio._READER_MODULES[_src], package="monetio")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for VirtualiZarr kwargs — generate random subsets
_vz_kwargs = st.fixed_dictionaries(
    {},
    optional={
        "use_virtualizarr": st.booleans(),
        "virtualizarr_file": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        "virtualizarr_backend": st.sampled_from(["kerchunk", "icechunk"]),
        "icechunk_repo": st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    },
)

# Strategy for selecting a GriddedReader source name
_gridded_source = st.sampled_from(_GRIDDED_SOURCES)

# Strategy for selecting a PointReader source name
_point_source = st.sampled_from(_POINT_SOURCES)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(source=_gridded_source, vz_kwargs=_vz_kwargs)
def test_gridded_reader_receives_vz_kwargs(source, vz_kwargs):
    """GriddedReader.open_dataset() receives all VZ kwargs passed through monetio.load()."""
    reader_cls = READER_REGISTRY[source]
    assert issubclass(reader_cls, GriddedReader), (
        f"{source} should be a GriddedReader subclass"
    )

    with mock.patch.object(reader_cls, "open_dataset", return_value=mock.MagicMock()) as mock_open:
        monetio.load(source, files="dummy.nc", **vz_kwargs)

        mock_open.assert_called_once()
        call_kwargs = mock_open.call_args
        # files should be passed
        assert call_kwargs.kwargs.get("files") == "dummy.nc" or call_kwargs[1].get("files") == "dummy.nc"

        # All VZ kwargs that were provided should arrive at open_dataset
        actual_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        for key, value in vz_kwargs.items():
            assert key in actual_kwargs, (
                f"VZ kwarg '{key}' not forwarded to {source}.open_dataset()"
            )
            assert actual_kwargs[key] == value, (
                f"VZ kwarg '{key}' value mismatch: expected {value!r}, got {actual_kwargs[key]!r}"
            )


@settings(max_examples=100)
@given(source=_point_source, vz_kwargs=_vz_kwargs)
def test_point_reader_accepts_vz_kwargs_without_error(source, vz_kwargs):
    """PointReader.open_dataset() accepts VZ kwargs without raising an error."""
    reader_cls = READER_REGISTRY[source]
    assert issubclass(reader_cls, PointReader), (
        f"{source} should be a PointReader subclass"
    )

    with mock.patch.object(reader_cls, "open_dataset", return_value=mock.MagicMock()) as mock_open:
        # This should NOT raise — VZ kwargs are accepted by PointReader signature
        monetio.load(source, files="dummy.csv", **vz_kwargs)

        mock_open.assert_called_once()
        # VZ kwargs should arrive at open_dataset (they're in the signature)
        actual_kwargs = mock_open.call_args.kwargs if mock_open.call_args.kwargs else mock_open.call_args[1]
        for key, value in vz_kwargs.items():
            assert key in actual_kwargs, (
                f"VZ kwarg '{key}' not accepted by {source}.open_dataset()"
            )


@settings(max_examples=100)
@given(vz_kwargs=_vz_kwargs)
def test_point_reader_does_not_forward_vz_kwargs_to_driver(vz_kwargs):
    """PointReader base class does NOT forward VZ kwargs to PandasDriver.open().

    This tests the base class behavior directly — VZ kwargs are accepted in the
    signature but silently discarded before calling self.driver.open().
    """
    reader = PointReader()

    with mock.patch.object(reader.driver, "open", return_value=mock.MagicMock()) as mock_driver_open:
        with mock.patch.object(reader, "harmonize", side_effect=lambda df: df):
            with mock.patch.object(reader, "to_xarray", return_value=mock.MagicMock()):
                reader.open_dataset(files=["dummy.csv"], **vz_kwargs)

                mock_driver_open.assert_called_once()
                driver_call = mock_driver_open.call_args
                # Combine positional and keyword args for inspection
                driver_kwargs = driver_call.kwargs if driver_call.kwargs else driver_call[1]

                # VZ kwargs must NOT be forwarded to PandasDriver
                vz_keys = {"use_virtualizarr", "virtualizarr_file", "virtualizarr_backend", "icechunk_repo"}
                forwarded_vz = vz_keys & set(driver_kwargs.keys())
                assert not forwarded_vz, (
                    f"VZ kwargs {forwarded_vz} were incorrectly forwarded to PandasDriver"
                )


@settings(max_examples=100)
@given(vz_kwargs=_vz_kwargs)
def test_gridded_reader_forwards_vz_kwargs_to_driver(vz_kwargs):
    """GriddedReader base class forwards VZ kwargs to XarrayDriver.open()."""
    reader = GriddedReader()

    with mock.patch.object(reader.driver, "open", return_value=mock.MagicMock()) as mock_driver_open:
        with mock.patch.object(reader, "harmonize", side_effect=lambda ds: ds):
            reader.open_dataset(files="dummy.nc", **vz_kwargs)

            mock_driver_open.assert_called_once()
            driver_call = mock_driver_open.call_args
            driver_kwargs = driver_call.kwargs if driver_call.kwargs else driver_call[1]

            # All VZ kwargs should be forwarded to XarrayDriver
            for key, value in vz_kwargs.items():
                assert key in driver_kwargs, (
                    f"VZ kwarg '{key}' not forwarded to XarrayDriver"
                )
                assert driver_kwargs[key] == value, (
                    f"VZ kwarg '{key}' value mismatch in driver call: "
                    f"expected {value!r}, got {driver_kwargs[key]!r}"
                )
