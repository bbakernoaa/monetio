"""ICAP-MME Reader"""

import pandas as pd
import xarray as xr
from .base import GriddedReader, register_reader
from .drivers import FileUtility

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
        if files is not None:
            return self.driver.open(files, **kwargs)

        if dates is not None:
            if download:
                return open_mfdataset_icap(
                    dates, product=product, data_var=data_var, download=True, verbose=verbose, **kwargs
                )
            else:
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
    fs = FileUtility.get_fs(file_url)
    exists = fs.exists(file_url)
    if not exists and verbose:
        print(f"File does not exist: {file_url}")
    return exists

def retrieve(url, fname, *, download=False, verbose=True):
    from io import BytesIO
    from pathlib import Path

    p = Path(fname).absolute()
    fs = FileUtility.get_fs(url)

    if not download:
        # Return BytesIO
        # fs.open returns a file-like object
        # We can read it into BytesIO if needed for compatibility or return fs open object
        # original returned BytesIO(r.content)
        with fs.open(url, "rb") as f:
            return BytesIO(f.read())
    else:
        if not p.is_file():
            if verbose:
                print(f"Downloading {url} to {p.as_posix()}")
            fs.get(url, str(p))
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
