"""ICAP-MME Reader"""

import pandas as pd
import xarray as xr
from .base import GriddedReader, register_reader

@register_reader("icap_mme")
class ICAPMMEReader(GriddedReader):
    def open_dataset(self,
                     files=None, # if local
                     dates=None, # if downloading
                     product="MMC",
                     data_var="dustaod550",
                     download=False,
                     verbose=True,
                     **kwargs):
        """
        Open ICAP-MME data.
        Supports opening by dates (downloads from FTP/HTTPS) or local files.
        """

        # If files are provided, treat as standard GriddedReader
        if files is not None:
            # We assume the user provided files are correct for the product/var desired
            # But the original code logic is heavily tied to downloading specific products.
            # If files is just a path, we use driver.
            return self.driver.open(files, **kwargs)

        # If dates are provided, we follow the original logic
        if dates is not None:
            if download:
                return open_mfdataset_icap(
                    dates, product=product, data_var=data_var, download=True, verbose=verbose, **kwargs
                )
            else:
                # In-memory opening logic from URL
                return open_mfdataset_icap(
                    dates, product=product, data_var=data_var, download=False, verbose=verbose, **kwargs
                )

        raise ValueError("Must provide 'files' or 'dates'.")

# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/icap_mme.py
# -----------------------------------------------------------------------------

valid_filetypes = ("MMC", "C4", "MME")
valid_data_vars = (
    "modeaod550", "dustaod550", "pm", "seasaltaod550", "smokeaod550", "totaldustaod550",
)

def build_urls(dates, filetype="MMC", data_var="dustaod550", *, verbose=True):
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

def remote_file_exists(file_url, *, verbose=True):
    import requests
    try:
        r = requests.head(file_url)
        if r.status_code == 200:
            return True
        else:
            if verbose:
                print(f"HTTP Error {r.status_code} - {r.reason}")
            return False
    except Exception as e:
        if verbose:
            print(e)
        return False

def retrieve(url, fname, *, download=False, verbose=True):
    from io import BytesIO
    from pathlib import Path
    import requests

    p = Path(fname).absolute()

    if not download:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        return BytesIO(r.content)
    else:
        if not p.is_file():
            if verbose:
                print(f"Downloading {url} to {p.as_posix()}")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(p, "wb") as f:
                f.write(r.content)
        else:
            if verbose:
                print(f"File Exists: {p.as_posix()}")
        return p

def _check_file_url(url, *, verbose=True):
    if not remote_file_exists(url, verbose=verbose):
        raise ValueError(
            f"File does not exist on ICAP HTTPS server: {url}. "
            f"Check {url[:url.index('icap_')]} to see the available "
            "`product` and `data_var`s for this month."
        )

def open_mfdataset_icap(dates, product="MMC", data_var="dustaod550", *, download=False, verbose=True, **kwargs):
    import pandas as pd
    import xarray as xr

    # d = pd.DatetimeIndex(dates) # dates already processed in build_urls if needed, but build_urls handles iterable

    if product.upper() not in valid_filetypes:
        raise ValueError(f"Invalid input for 'product': Valid values are {valid_filetypes}.")

    if data_var.lower() not in valid_data_vars:
        raise ValueError(f"Invalid input for 'data_var': Valid values are {valid_data_vars}.")

    urls, fnames = build_urls(dates, filetype=product, data_var=data_var, verbose=verbose)

    if download is True:
        paths = []
        for url, fname in zip(urls, fnames):
            _check_file_url(url, verbose=verbose)
            paths.append(retrieve(url, fname, download=True, verbose=verbose))

        # Use provided kwargs for open_mfdataset if any
        if 'combine' not in kwargs:
            kwargs['combine'] = 'nested'
        if 'concat_dim' not in kwargs:
            kwargs['concat_dim'] = 'time'

        dset = xr.open_mfdataset(paths, **kwargs)
    else:
        dsets = []
        for url, fname in zip(urls, fnames):
            _check_file_url(url, verbose=verbose)
            o = retrieve(url, fname, download=False, verbose=verbose)
            dsets.append(xr.open_dataset(o))
        dset = xr.concat(dsets, dim="time")

    return dset
