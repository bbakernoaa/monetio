"""OMPS Nadir Reader. Deprecated wrapper — use monetio.load('omps_nadir', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.omps_nadir import OMPSNadirReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.omps_nadir.read_OMPS_nm",
    'monetio.load("omps_nadir", files=...)',
)
def read_OMPS_nm(files, **kwargs):
    """Loop to open OMPS nadir mapper L2 files.

    Parameters
    ----------
    files : str or list of str
        Input file paths or URLs.
    **kwargs : dict
        Additional arguments forwarded to ``OMPSNadirReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return OMPSNadirReader().open_dataset(files=files, **kwargs)


@deprecated_wrapper(
    "monetio.sat.omps_nadir.extract_OMPS_nm",
    'monetio.load("omps_nadir", files=...)',
)
def extract_OMPS_nm(fname, **kwargs):
    """Read locally stored OMPS Nadir Mapper L2 file.

    Parameters
    ----------
    fname : str
        Local path to h5 file.
    **kwargs : dict
        Additional arguments forwarded to ``OMPSNadirReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return OMPSNadirReader().open_dataset(files=fname, **kwargs)


@deprecated_wrapper(
    "monetio.sat.omps_nadir.extract_OMPS_nm_opendap",
    'monetio.load("omps_nadir", files=...)',
)
def extract_OMPS_nm_opendap(fname, **kwargs):
    """Read OMPS Nadir Mapper L2 data from OPeNDAP.

    Parameters
    ----------
    fname : str
        URL location of h5 file.
    **kwargs : dict
        Additional arguments forwarded to ``OMPSNadirReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return OMPSNadirReader().open_dataset(files=fname, **kwargs)
