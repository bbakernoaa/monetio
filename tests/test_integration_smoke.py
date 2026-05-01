"""Integration smoke tests for the monetio reader refactor.

**Validates: Requirements 11.1, 11.2, 11.3**

- Verify all reader modules import without error.
- Verify ``monetio.load()`` with each source name doesn't crash on import.
- Verify ``cdump2netcdf.py`` and ``epa_util.py`` have no deprecation infrastructure.
"""

import importlib
import inspect
import unittest.mock as mock

import pytest

import monetio

# ---------------------------------------------------------------------------
# All reader module paths (relative to monetio.readers)
# ---------------------------------------------------------------------------

_READER_MODULES = [
    "monetio.readers.actris",
    "monetio.readers.aeronet",
    "monetio.readers.airnow",
    "monetio.readers.aqs",
    "monetio.readers.base",
    "monetio.readers.camx",
    "monetio.readers.camx_specs",
    "monetio.readers.cems",
    "monetio.readers.chimere",
    "monetio.readers.cmaq",
    "monetio.readers.cmaq_specs",
    "monetio.readers.crn",
    "monetio.readers.drivers",
    "monetio.readers.earlinet",
    "monetio.readers.epa_utils",
    "monetio.readers.eprofile",
    "monetio.readers.geoms",
    "monetio.readers.gfs",
    "monetio.readers.gml_ozonesonde",
    "monetio.readers.goes",
    "monetio.readers.grib2",
    "monetio.readers.hysplit",
    "monetio.readers.hytraj",
    "monetio.readers.iagos",
    "monetio.readers.icap_mme",
    "monetio.readers.icartt",
    "monetio.readers.igra2",
    "monetio.readers.improve",
    "monetio.readers.ish",
    "monetio.readers.ish_lite",
    "monetio.readers.merra2",
    "monetio.readers.modis_l2",
    "monetio.readers.modis_ornl",
    "monetio.readers.mopitt",
    "monetio.readers.mplnet",
    "monetio.readers.nadp",
    "monetio.readers.nasa_modis",
    "monetio.readers.nasa_utils",
    "monetio.readers.ncep_grib",
    "monetio.readers.ndacc",
    "monetio.readers.ndbc",
    "monetio.readers.nesdis_edr_viirs",
    "monetio.readers.nesdis_eps_viirs",
    "monetio.readers.nesdis_frp",
    "monetio.readers.nesdis_viirs_jrr",
    "monetio.readers.omps",
    "monetio.readers.omps_nadir",
    "monetio.readers.openaq",
    "monetio.readers.openaq_aws",
    "monetio.readers.openaq_v2",
    "monetio.readers.pams",
    "monetio.readers.pandora",
    "monetio.readers.pardump",
    "monetio.readers.raqms",
    "monetio.readers.rrfs",
    "monetio.readers.sat_utils",
    "monetio.readers.skynet",
    "monetio.readers.solrad",
    "monetio.readers.surfrad",
    "monetio.readers.tempo",
    "monetio.readers.time_utils",
    "monetio.readers.tolnet",
    "monetio.readers.tropomi",
    "monetio.readers.ufs",
    "monetio.readers.ufs_specs",
    "monetio.readers.umbc_aerosol",
    "monetio.readers.wrfchem",
    "monetio.readers.wrfchem_specs",
]


@pytest.mark.parametrize("module_path", _READER_MODULES)
def test_reader_module_imports_without_error(module_path):
    """Each reader module should import without raising ImportError."""
    mod = importlib.import_module(module_path)
    assert mod is not None


# ---------------------------------------------------------------------------
# All source names from _READER_MODULES in monetio/__init__.py
# ---------------------------------------------------------------------------

_ALL_SOURCE_NAMES = list(monetio._READER_MODULES.keys())


@pytest.mark.parametrize("source_name", _ALL_SOURCE_NAMES)
def test_load_source_resolves_reader(source_name):
    """monetio.load() with each source name should resolve the reader without crash.

    We mock the reader's open_dataset to prevent actual file I/O, but verify
    that the reader class is found and instantiated.
    """
    from monetio.readers.base import READER_REGISTRY

    # Ensure the reader module is imported (lazy loading)
    if source_name not in READER_REGISTRY:
        importlib.import_module(monetio._READER_MODULES[source_name], package="monetio")

    assert source_name in READER_REGISTRY, (
        f"Source '{source_name}' not found in READER_REGISTRY after import"
    )

    reader_cls = READER_REGISTRY[source_name]

    # Mock open_dataset to prevent real I/O
    with mock.patch.object(reader_cls, "open_dataset", return_value=mock.MagicMock()):
        result = monetio.load(source_name, files="dummy.nc")
        assert result is not None


# ---------------------------------------------------------------------------
# Verify cdump2netcdf.py has no deprecation infrastructure
# ---------------------------------------------------------------------------


def test_cdump2netcdf_has_no_deprecation():
    """cdump2netcdf.py should not import or use deprecation infrastructure."""
    mod = importlib.import_module("monetio.models.cdump2netcdf")
    source = inspect.getsource(mod)

    assert "deprecated_wrapper" not in source, (
        "cdump2netcdf.py should not use deprecated_wrapper"
    )
    assert "_deprecation" not in source, (
        "cdump2netcdf.py should not import from _deprecation"
    )
    assert "DeprecationWarning" not in source, (
        "cdump2netcdf.py should not emit DeprecationWarning"
    )


# ---------------------------------------------------------------------------
# Verify epa_util.py has no deprecation infrastructure
# ---------------------------------------------------------------------------


def test_epa_util_has_no_deprecation():
    """epa_util.py should not import or use deprecation infrastructure."""
    mod = importlib.import_module("monetio.obs.epa_util")
    source = inspect.getsource(mod)

    assert "deprecated_wrapper" not in source, (
        "epa_util.py should not use deprecated_wrapper"
    )
    assert "_deprecation" not in source, (
        "epa_util.py should not import from _deprecation"
    )
    assert "DeprecationWarning" not in source, (
        "epa_util.py should not emit DeprecationWarning"
    )


# ---------------------------------------------------------------------------
# Verify READER_REGISTRY completeness
# ---------------------------------------------------------------------------


def test_reader_registry_has_all_sources():
    """READER_REGISTRY should contain all source names from _READER_MODULES."""
    from monetio.readers.base import READER_REGISTRY

    # Force all lazy imports
    for source_name, module_path in monetio._READER_MODULES.items():
        if source_name not in READER_REGISTRY:
            importlib.import_module(module_path, package="monetio")

    missing = set(monetio._READER_MODULES.keys()) - set(READER_REGISTRY.keys())
    assert not missing, (
        f"READER_REGISTRY is missing sources: {missing}"
    )
