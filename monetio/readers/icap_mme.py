"""ICAP-MME Reader"""

from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history

VALID_FILETYPES = ("MMC", "C4", "MME")
VALID_DATA_VARS = (
    "modeaod550",
    "dustaod550",
    "pm",
    "seasaltaod550",
    "smokeaod550",
    "totaldustaod550",
)


@register_reader("icap_mme")
class ICAPMMEReader(GriddedReader):
    """
    Reader for ICAP-MME (International Cooperative for Aerosol Prediction - Multi Model Ensemble) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        product: str = "MMC",
        data_var: str = "dustaod550",
        download: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Retrieve and load ICAP-MME data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs to read. If None, uses `dates` and `product` to discover files.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if `files` is not provided.
        product : str, optional
            ICAP product (e.g., 'MMC', 'C4', 'MME'), by default 'MMC'.
        data_var : str, optional
            Data variable (e.g., 'dustaod550'), by default 'dustaod550'.
        download : bool, optional
            Whether to download files to local directory, by default False.
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The loaded ICAP-MME dataset.

        Examples
        --------
        >>> reader = ICAPMMEReader()
        >>> ds = reader.open_dataset(dates="2024-02-01", product="C4")
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")

            urls, fnames = build_urls(dates, filetype=product, data_var=data_var)
            if download:
                files = []
                for url, fname in zip(urls, fnames):
                    files.append(str(retrieve(url, fname, download=True)))
            else:
                files = urls.tolist()

        # ICAP files are standard NetCDF, often h5netcdf compatible.
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        # Use XarrayDriver to open (Lazy by default)
        ds = self.driver.open(files, **kwargs)

        ds = self.harmonize(ds)

        # Update history
        ds = update_history(ds, "Read ICAP-MME data.")

        return ds


def build_urls(
    dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
    filetype: str = "MMC",
    data_var: str = "dustaod550",
    verbose: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Construct ICAP-MME URLs and filenames for the given dates.

    Parameters
    ----------
    dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
        Dates to build URLs for.
    filetype : str, optional
        ICAP product type (MMC, C4, MME), by default "MMC".
    data_var : str, optional
        Data variable name, by default "dustaod550".
    verbose : bool, optional
        Whether to print status messages, by default True.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (urls, filenames).

    Examples
    --------
    >>> urls, fnames = build_urls("2024-02-01", filetype="C4")
    """
    from collections.abc import Iterable

    if isinstance(dates, Iterable) and not isinstance(dates, str):
        dates = pd.DatetimeIndex(dates)
    else:
        dates = pd.DatetimeIndex([dates])

    urls = []
    fnames = []
    if verbose:
        print("Building ICAP-MME URLs...")
    base_url = "https://usgodae.org/ftp/outgoing/nrl/ICAP-MME/"

    for dt in dates:
        fname = "icap_{}_{}_{}.nc".format(
            dt.strftime(r"%Y%m%d%H"), filetype.upper(), data_var.lower()
        )
        url = base_url + dt.strftime(r"%Y/%Y%m/") + fname
        urls.append(url)
        fnames.append(fname)

    return pd.Series(urls, index=None), pd.Series(fnames, index=None)


def retrieve(
    url: str, fname: str, download: bool = False, verbose: bool = True
) -> Union[str, Path]:
    """
    Retrieve or download an ICAP-MME file.

    Parameters
    ----------
    url : str
        Source URL.
    fname : str
        Target filename.
    download : bool, optional
        Whether to download to a local file, by default False.
    verbose : bool, optional
        Whether to print status, by default True.

    Returns
    -------
    Union[str, Path]
        The path or URL to the file.

    Examples
    --------
    >>> path = retrieve("https://.../file.nc", "file.nc", download=True)
    """
    p = Path(fname).absolute()
    fs = FileUtility.get_fs(url)

    if not download:
        # Return URL for remote opening via fsspec
        return url
    else:
        if not p.is_file():
            if verbose:
                print(f"Downloading {url} to {p.as_posix()}")
            fs.get(url, str(p))
        else:
            if verbose:
                print(f"File Exists: {p.as_posix()}")
        return p
