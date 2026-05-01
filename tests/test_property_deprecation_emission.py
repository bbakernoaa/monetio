# Feature: virtualizarr-reader-refactor, Property 4: Deprecation Warning Emission
"""Property-based test for deprecation warning emission from legacy wrappers.

**Validates: Requirements 4.11, 5.1, 6.1, 7.1, 9.1, 9.2, 9.3**

For any deprecated legacy wrapper function, calling it SHALL emit exactly one
``DeprecationWarning`` (per function per session) that includes the legacy function
name, the recommended ``monetio.load()`` equivalent, and the target removal version.
"""

import importlib
import unittest.mock as mock
import warnings

import pytest

# ---------------------------------------------------------------------------
# Representative deprecated functions from each category.
#
# Each entry is:
#   (module_path, func_name, reader_module, reader_class,
#    expected_legacy_name, expected_load_fragment, dummy_call_kwargs)
#
# We include the reader info so we can mock open_dataset to prevent real I/O.
# ---------------------------------------------------------------------------

_DEPRECATED_FUNCTIONS = [
    # --- Models ---
    pytest.param(
        "monetio.models.cmaq",
        "open_dataset",
        "monetio.readers.cmaq",
        "CMAQReader",
        "monetio.models.cmaq.open_dataset",
        'monetio.load("cmaq"',
        {"fname": "/tmp/dummy.nc"},
        id="models-cmaq-open_dataset",
    ),
    pytest.param(
        "monetio.models.cmaq",
        "open_mfdataset",
        "monetio.readers.cmaq",
        "CMAQReader",
        "monetio.models.cmaq.open_mfdataset",
        'monetio.load("cmaq"',
        {"fname": "/tmp/dummy.nc"},
        id="models-cmaq-open_mfdataset",
    ),
    pytest.param(
        "monetio.models.raqms",
        "open_dataset",
        "monetio.readers.raqms",
        "RAQMSReader",
        "monetio.models.raqms.open_dataset",
        'monetio.load("raqms"',
        {"fname": "/tmp/dummy.nc"},
        id="models-raqms-open_dataset",
    ),
    pytest.param(
        "monetio.models.chimere",
        "open_mfdataset",
        "monetio.readers.chimere",
        "ChimereReader",
        "monetio.models.chimere.open_mfdataset",
        'monetio.load("chimere"',
        {"files": ["/tmp/dummy.nc"]},
        id="models-chimere-open_mfdataset",
    ),
    pytest.param(
        "monetio.models.hysplit",
        "open_dataset",
        "monetio.readers.hysplit",
        "HYSPLITReader",
        "monetio.models.hysplit.open_dataset",
        'monetio.load("hysplit"',
        {"fname": "/tmp/dummy.nc"},
        id="models-hysplit-open_dataset",
    ),
    pytest.param(
        "monetio.models.pardump",
        "open_dataset",
        "monetio.readers.pardump",
        "PardumpReader",
        "monetio.models.pardump.open_dataset",
        'monetio.load("pardump"',
        {"fname": "/tmp/dummy.txt"},
        id="models-pardump-open_dataset",
    ),
    pytest.param(
        "monetio.models.camx",
        "open_dataset",
        "monetio.readers.camx",
        "CAMxReader",
        "monetio.models.camx.open_dataset",
        'monetio.load("camx"',
        {"fname": "/tmp/dummy.nc"},
        id="models-camx-open_dataset",
    ),
    # --- Obs ---
    pytest.param(
        "monetio.obs.airnow",
        "add_data",
        "monetio.readers.airnow",
        "AirNowReader",
        "monetio.obs.airnow.add_data",
        'monetio.load("airnow"',
        {"dates": ["2023-01-01"]},
        id="obs-airnow-add_data",
    ),
    pytest.param(
        "monetio.obs.aqs",
        "add_data",
        "monetio.readers.aqs",
        "AQSReader",
        "monetio.obs.aqs.add_data",
        'monetio.load("aqs"',
        {"dates": ["2023-01-01"]},
        id="obs-aqs-add_data",
    ),
    pytest.param(
        "monetio.obs.cems",
        "add_data",
        "monetio.readers.cems",
        "CEMSReader",
        "monetio.obs.cems.add_data",
        'monetio.load("cems"',
        {},
        id="obs-cems-add_data",
    ),
    # --- Profile ---
    pytest.param(
        "monetio.profile.tolnet",
        "open_dataset",
        "monetio.readers.tolnet",
        "TOLNetReader",
        "monetio.profile.tolnet.open_dataset",
        'monetio.load("tolnet"',
        {"fname": "/tmp/dummy.h5"},
        id="profile-tolnet-open_dataset",
    ),
    pytest.param(
        "monetio.profile.geoms",
        "open_dataset",
        "monetio.readers.geoms",
        "GEOMSReader",
        "monetio.profile.geoms.open_dataset",
        'monetio.load("geoms"',
        {"fp": "/tmp/dummy.hdf"},
        id="profile-geoms-open_dataset",
    ),
    # --- Sat ---
    pytest.param(
        "monetio.sat.goes",
        "open_dataset",
        "monetio.readers.goes",
        "GOESReader",
        "monetio.sat.goes.open_dataset",
        'monetio.load("goes"',
        {"filename": "/tmp/dummy.nc"},
        id="sat-goes-open_dataset",
    ),
    pytest.param(
        "monetio.sat.modis_l2",
        "read_dataset",
        "monetio.readers.modis_l2",
        "MODISL2Reader",
        "monetio.sat.modis_l2.read_dataset",
        'monetio.load("modis_l2"',
        {"fname": "/tmp/dummy.hdf", "variable_dict": {}},
        id="sat-modis_l2-read_dataset",
    ),
    pytest.param(
        "monetio.sat.nesdis_edr_viirs",
        "open_dataset",
        "monetio.readers.nesdis_edr_viirs",
        "NESDISEDRVIIRSReader",
        "monetio.sat.nesdis_edr_viirs.open_dataset",
        'monetio.load("nesdis_edr_viirs"',
        {"date": "2023-01-01"},
        id="sat-nesdis_edr_viirs-open_dataset",
    ),
]


@pytest.mark.parametrize(
    "wrapper_module,func_name,reader_module,reader_class_name,"
    "expected_legacy_name,expected_load_fragment,call_kwargs",
    _DEPRECATED_FUNCTIONS,
)
def test_deprecation_warning_content(
    wrapper_module,
    func_name,
    reader_module,
    reader_class_name,
    expected_legacy_name,
    expected_load_fragment,
    call_kwargs,
):
    """Calling a deprecated wrapper emits a DeprecationWarning with correct content.

    Verifies:
    - Warning category is DeprecationWarning (Req 9.3)
    - Message contains the legacy function name (Req 9.1)
    - Message contains the monetio.load() equivalent (Req 9.1)
    - Message contains the target removal version (Req 9.1)
    """
    wmod = importlib.import_module(wrapper_module)
    rmod = importlib.import_module(reader_module)
    wrapper_fn = getattr(wmod, func_name)
    reader_cls = getattr(rmod, reader_class_name)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        # Mock the reader's open_dataset to prevent real I/O
        with mock.patch.object(reader_cls, "open_dataset", return_value=mock.MagicMock()):
            try:
                wrapper_fn(**call_kwargs)
            except Exception:
                # Some wrappers may call non-open_dataset methods; ignore errors
                pass

    # Filter to DeprecationWarnings only
    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]

    assert len(dep_warnings) >= 1, (
        f"Expected DeprecationWarning from {wrapper_module}.{func_name}, got {len(dep_warnings)}"
    )

    msg = str(dep_warnings[0].message)

    # Requirement 9.1: Contains legacy function name
    assert expected_legacy_name in msg, (
        f"Warning should contain '{expected_legacy_name}', got: {msg}"
    )

    # Requirement 9.1: Contains monetio.load() equivalent
    assert expected_load_fragment in msg, (
        f"Warning should contain '{expected_load_fragment}', got: {msg}"
    )

    # Requirement 9.1: Contains removal version
    assert "0.4.0" in msg, f"Warning should contain removal version '0.4.0', got: {msg}"

    # Requirement 9.3: Uses DeprecationWarning category
    assert dep_warnings[0].category is DeprecationWarning
