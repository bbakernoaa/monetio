"""NESDIS EPS VIIRS Reader. Deprecated wrapper — use monetio.load('nesdis_eps_viirs', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.nesdis_eps_viirs import NESDISEPSVIIRSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.nesdis_eps_viirs.open_dataset",
    'monetio.load("nesdis_eps_viirs", files=...)',
)
def open_dataset(date, datapath=".", **kwargs):
    """Open NESDIS EPS VIIRS AOT data.

    Parameters
    ----------
    date : str or datetime-like
        Date to retrieve.
    datapath : str
        Path to download/read data.
    **kwargs : dict
        Additional arguments forwarded to ``NESDISEPSVIIRSReader.open_dataset``.

    Returns
    -------
    xarray.DataArray
    """
    return NESDISEPSVIIRSReader().open_dataset(dates=date, datapath=datapath, **kwargs)


@deprecated_wrapper(
    "monetio.sat.nesdis_eps_viirs.open_mfdataset",
    'monetio.load("nesdis_eps_viirs", files=...)',
)
def open_mfdataset(dates, datapath=".", **kwargs):
    """Open multiple NESDIS EPS VIIRS AOT files.

    Parameters
    ----------
    dates : sequence of datetime-like
        Dates to retrieve.
    datapath : str
        Path to download/read data.
    **kwargs : dict
        Additional arguments forwarded to ``NESDISEPSVIIRSReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return NESDISEPSVIIRSReader().open_dataset(dates=dates, datapath=datapath, **kwargs)
