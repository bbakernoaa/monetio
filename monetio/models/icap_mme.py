"""ICAP-MME Reader. Deprecated wrapper — use monetio.load('icap_mme', ...) instead."""

from ..readers._deprecation import deprecated_wrapper
from ..readers.icap_mme import ICAPMMEReader  # noqa: F401


@deprecated_wrapper(
    "monetio.models.icap_mme.open_dataset",
    'monetio.load("icap_mme", dates=...)',
)
def open_dataset(
    date,
    product="MMC",
    data_var="dustaod550",
    *,
    download=False,
    verbose=True,
    verify=True,
):
    """Open a single ICAP-MME dataset.

    Parameters
    ----------
    date : str or datetime-like
        The date for which to open the dataset.
    product : {'MMC', 'C4', 'MME'}, optional
    data_var : {'modeaod550', 'dustaod550', 'pm', 'seasaltaod550',
        'smokeaod550', 'totaldustaod550'}, optional
    download : bool, optional
    verbose : bool, optional
    verify : bool, optional

    Returns
    -------
    xarray.Dataset
    """
    return ICAPMMEReader().open_dataset(
        dates=date,
        product=product,
        data_var=data_var,
        download=download,
    )


@deprecated_wrapper(
    "monetio.models.icap_mme.open_mfdataset",
    'monetio.load("icap_mme", dates=...)',
)
def open_mfdataset(
    dates,
    product="MMC",
    data_var="dustaod550",
    *,
    download=False,
    verbose=True,
    verify=True,
):
    """Open multiple ICAP-MME datasets.

    Parameters
    ----------
    dates : iterable of datetime-like
        The dates for which to open the dataset.
    product : {'MMC', 'C4', 'MME'}, optional
    data_var : {'modeaod550', 'dustaod550', 'pm', 'seasaltaod550',
        'smokeaod550', 'totaldustaod550'}, optional
    download : bool, optional
    verbose : bool, optional
    verify : bool, optional

    Returns
    -------
    xarray.Dataset
    """
    return ICAPMMEReader().open_dataset(
        dates=dates,
        product=product,
        data_var=data_var,
        download=download,
    )


# Re-export utility functions for backward compatibility
from ..readers.icap_mme import build_urls, retrieve  # noqa: E402, F401
