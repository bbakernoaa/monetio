"""NESDIS EDR VIIRS Reader. Deprecated wrapper — use monetio.load('nesdis_edr_viirs', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.nesdis_edr_viirs import NESDISEDRVIIRSReader  # noqa: F401


@deprecated_wrapper(
    "monetio.sat.nesdis_edr_viirs.open_dataset",
    'monetio.load("nesdis_edr_viirs", files=...)',
)
def open_dataset(date, resolution="high", datapath=".", **kwargs):
    """Open NESDIS EDR VIIRS gridded AOD data.

    Parameters
    ----------
    date : str or datetime-like
        Date to retrieve.
    resolution : str
        Resolution ('high' for 0.1 degree, other for 0.25 degree).
    datapath : str
        Path to download/read data.
    **kwargs : dict
        Additional arguments forwarded to ``NESDISEDRVIIRSReader.open_dataset``.

    Returns
    -------
    xarray.DataArray
    """
    return NESDISEDRVIIRSReader().open_dataset(
        dates=date, resolution=resolution, datapath=datapath, **kwargs
    )


@deprecated_wrapper(
    "monetio.sat.nesdis_edr_viirs.open_mfdataset",
    'monetio.load("nesdis_edr_viirs", files=...)',
)
def open_mfdataset(dates, resolution="high", datapath=".", **kwargs):
    """Open multiple NESDIS EDR VIIRS gridded AOD files.

    Parameters
    ----------
    dates : sequence of datetime-like
        Dates to retrieve.
    resolution : str
        Resolution ('high' for 0.1 degree, other for 0.25 degree).
    datapath : str
        Path to download/read data.
    **kwargs : dict
        Additional arguments forwarded to ``NESDISEDRVIIRSReader.open_dataset``.

    Returns
    -------
    xarray.Dataset
    """
    return NESDISEDRVIIRSReader().open_dataset(
        dates=dates, resolution=resolution, datapath=datapath, **kwargs
    )
