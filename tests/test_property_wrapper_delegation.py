# Feature: virtualizarr-reader-refactor, Property 3: Legacy Wrapper Delegation
"""Property-based test for legacy wrapper delegation to reader classes.

**Validates: Requirements 4.1–4.10, 5.2–5.19, 6.2–6.5, 7.2–7.12**

For any legacy wrapper function in monetio/models/, monetio/obs/, monetio/profile/,
or monetio/sat/ that has a corresponding reader in monetio/readers/, calling the
wrapper function with arguments SHALL delegate to the corresponding reader's
``open_dataset()`` method with equivalent arguments.
"""

import unittest.mock as mock
import warnings

import pytest

# ---------------------------------------------------------------------------
# Representative wrapper functions and their expected reader classes.
#
# Each entry is:
#   (wrapper_module_path, wrapper_func_name, reader_module_path, reader_class_name,
#    wrapper_call_kwargs, expected_reader_kwargs)
#
# We test a representative set from each category (models, obs, profile, sat).
# ---------------------------------------------------------------------------

_WRAPPER_DELEGATION_CASES = [
    # --- Models ---
    pytest.param(
        "monetio.models.cmaq",
        "open_dataset",
        "monetio.readers.cmaq",
        "CMAQReader",
        {"fname": "/tmp/cmaq.nc"},
        {"files": "/tmp/cmaq.nc"},
        id="models-cmaq-open_dataset",
    ),
    pytest.param(
        "monetio.models.cmaq",
        "open_mfdataset",
        "monetio.readers.cmaq",
        "CMAQReader",
        {"fname": "/tmp/cmaq*.nc"},
        {"files": "/tmp/cmaq*.nc"},
        id="models-cmaq-open_mfdataset",
    ),
    pytest.param(
        "monetio.models.raqms",
        "open_dataset",
        "monetio.readers.raqms",
        "RAQMSReader",
        {"fname": "/tmp/raqms.nc"},
        {"files": "/tmp/raqms.nc"},
        id="models-raqms-open_dataset",
    ),
    pytest.param(
        "monetio.models.chimere",
        "open_mfdataset",
        "monetio.readers.chimere",
        "ChimereReader",
        {"files": ["/tmp/chimere.nc"], "var_list": None, "surf_only": False},
        {"files": ["/tmp/chimere.nc"], "var_list": None, "surf_only": False},
        id="models-chimere-open_mfdataset",
    ),
    pytest.param(
        "monetio.models.pardump",
        "open_dataset",
        "monetio.readers.pardump",
        "PardumpReader",
        {"fname": "/tmp/pardump.txt"},
        {"files": "/tmp/pardump.txt"},
        id="models-pardump-open_dataset",
    ),
    # --- Obs ---
    pytest.param(
        "monetio.obs.aqs",
        "add_data",
        "monetio.readers.aqs",
        "AQSReader",
        {
            "dates": ["2023-01-01"],
            "param": None,
            "daily": False,
            "network": None,
            "download": False,
            "local": False,
            "wide_fmt": True,
            "n_procs": 1,
            "meta": False,
            "as_xarray": True,
        },
        {
            "dates": ["2023-01-01"],
            "param": None,
            "daily": False,
            "network": None,
            "download": False,
            "local": False,
            "wide_fmt": True,
            "n_procs": 1,
            "meta": False,
            "as_xarray": True,
        },
        id="obs-aqs-add_data",
    ),
    pytest.param(
        "monetio.obs.cems",
        "add_data",
        "monetio.readers.cems",
        "CEMSReader",
        {
            "rdate": "2023-01-01",
            "states": ["md"],
            "download": False,
            "verbose": True,
            "files": None,
            "as_xarray": True,
        },
        {
            "rdate": "2023-01-01",
            "states": ["md"],
            "download": False,
            "verbose": True,
            "files": None,
            "as_xarray": True,
        },
        id="obs-cems-add_data",
    ),
    # --- Profile ---
    pytest.param(
        "monetio.profile.tolnet",
        "open_dataset",
        "monetio.readers.tolnet",
        "TOLNetReader",
        {"fname": "/tmp/tolnet.h5"},
        {"files": "/tmp/tolnet.h5"},
        id="profile-tolnet-open_dataset",
    ),
    pytest.param(
        "monetio.profile.geoms",
        "open_dataset",
        "monetio.readers.geoms",
        "GEOMSReader",
        {"fp": "/tmp/geoms.hdf", "rename_all": True, "squeeze": True},
        {"files": "/tmp/geoms.hdf", "rename_all": True, "squeeze": True},
        id="profile-geoms-open_dataset",
    ),
    # --- Sat ---
    pytest.param(
        "monetio.sat.modis_l2",
        "read_dataset",
        "monetio.readers.modis_l2",
        "MODISL2Reader",
        {"fname": "/tmp/modis.hdf", "variable_dict": {"AOD": "aod"}},
        {"files": "/tmp/modis.hdf", "variable_dict": {"AOD": "aod"}},
        id="sat-modis_l2-read_dataset",
    ),
    pytest.param(
        "monetio.sat.nesdis_edr_viirs",
        "open_dataset",
        "monetio.readers.nesdis_edr_viirs",
        "NESDISEDRVIIRSReader",
        {"date": "2023-01-01", "resolution": "high", "datapath": "."},
        {"dates": "2023-01-01", "resolution": "high", "datapath": "."},
        id="sat-nesdis_edr_viirs-open_dataset",
    ),
]


@pytest.mark.parametrize(
    "wrapper_module,wrapper_func,reader_module,reader_class,call_kwargs,expected_kwargs",
    _WRAPPER_DELEGATION_CASES,
)
def test_wrapper_delegates_to_reader(
    wrapper_module,
    wrapper_func,
    reader_module,
    reader_class,
    call_kwargs,
    expected_kwargs,
):
    """Each legacy wrapper function delegates to the corresponding reader's open_dataset()."""
    import importlib

    # Import the wrapper module and reader module
    wmod = importlib.import_module(wrapper_module)
    rmod = importlib.import_module(reader_module)

    wrapper_fn = getattr(wmod, wrapper_func)
    reader_cls = getattr(rmod, reader_class)

    # Mock the reader's open_dataset to capture the call
    with mock.patch.object(reader_cls, "open_dataset", return_value=mock.MagicMock()) as mock_open:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            wrapper_fn(**call_kwargs)

        mock_open.assert_called_once()
        actual_kwargs = mock_open.call_args.kwargs

        # Verify all expected kwargs were forwarded
        for key, value in expected_kwargs.items():
            assert key in actual_kwargs, (
                f"Expected kwarg '{key}' not found in reader call. "
                f"Got: {list(actual_kwargs.keys())}"
            )
            assert actual_kwargs[key] == value, (
                f"Kwarg '{key}' mismatch: expected {value!r}, got {actual_kwargs[key]!r}"
            )
