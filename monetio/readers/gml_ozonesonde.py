"""GML Ozonesonde Reader"""

import re
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr

if TYPE_CHECKING:
    pass

from .base import PointReader, register_reader


@register_reader("gml_ozonesonde")
class GMLOzonesondeReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        location: Union[str, List[str]] = None,
        n_procs: int = 1,
        errors: str = "raise",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, Union[pd.DataFrame, xr.Dataset]]:
        """
        Retrieve and load GML Ozonesonde data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs to read. If None, uses `dates` and `location` to discover files.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve.
        location : Union[str, List[str]], optional
            Locations to retrieve (e.g., 'Boulder, Colorado').
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        errors : str, optional
            Whether to 'raise' or 'warn' on read errors, by default 'raise'.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ozonesonde data.
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")

            dates = pd.to_datetime(dates)
            if isinstance(dates, pd.Timestamp):
                dates = pd.DatetimeIndex([dates])

            df_urls = discover_files(location=location)
            # Filter by date
            mask = df_urls["time"].between(dates.min(), dates.max(), inclusive="both")
            urls = df_urls.loc[mask, "url"].tolist()

            if not urls:
                raise RuntimeError(
                    f"No files found for dates {dates.min()} to {dates.max()} "
                    f"at location(s) {location}."
                )
            files = urls

        # Filter out arguments that are not for the reader function
        reader_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["expand2d", "pivot", "wide_fmt"]
        }

        # Use PandasDriver to open files
        # We pass read_100m as the read_method
        df = self.driver.open(files, read_method=read_100m, lazy=lazy, **reader_kwargs)

        if not lazy and n_procs > 1:
            # If dask is installed, we can use it for parallel loading even if as_xarray is False
            # but usually PandasDriver.open(lazy=False) already loaded them sequentially.
            # Actually, if we want parallel sequential load, we'd need to handle it in driver.
            # For now, we follow the ish.py pattern if they use n_procs.
            pass

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Update history and metadata
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read GML Ozonesonde data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history

            # Set variable attributes
            var_attrs = {
                c.name: {"long_name": c.long_name, "units": c.units} for c in COL_INFO_L100
            }
            for var, attrs in var_attrs.items():
                if var in ds.data_vars:
                    ds[var].attrs.update(attrs)

            return ds

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/profile/gml_ozonesonde.py
# -----------------------------------------------------------------------------

TIMEOUT = 15
RETRIES = 5

LOCATIONS = [
    "Boulder, Colorado",
    "Hilo, Hawaii",
    "Huntsville, Alabama",
    "Narragansett, Rhode Island",
    "Pago Pago, American Samoa",
    "San Cristobal, Galapagos",
    "South Pole, Antarctica",
    "Summit, Greenland",
    "Suva, Fiji",
    "Trinidad Head, California",
]

_FILES_L100_CACHE = {location: None for location in LOCATIONS}


def retry(func):
    import time
    from functools import wraps
    from random import random as rand

    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(RETRIES):
            try:
                res = func(*args, **kwargs)
            except (
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
            ):
                time.sleep(0.5 * i**1.5 + rand() * 0.1)
            else:
                break
        else:
            raise RuntimeError(f"{func.__name__} failed after {RETRIES} tries.")
        return res

    return wrapper


def discover_files(location=None, *, n_threads=3, cache=True):
    import itertools
    from multiprocessing.pool import ThreadPool

    base = "https://gml.noaa.gov/aftp/data/ozwv/Ozonesonde"
    if location is None:
        locations = LOCATIONS
    elif isinstance(location, str):
        locations = [location]
    else:
        locations = location
    invalid = set(locations) - set(LOCATIONS)
    if invalid:
        raise ValueError(f"Invalid location(s): {invalid}.")

    @retry
    def get_files(location):
        cached = _FILES_L100_CACHE[location]
        if cached is not None:
            return cached
        url_location = "South Pole, Antartica" if location == "South Pole, Antarctica" else location
        url = f"{base}/{url_location}/100 Meter Average Files/".replace(" ", "%20")
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
        except Exception:
            return []

        data = []
        for m in re.finditer(r'href="([a-z0-9_]+\.l100)"', r.text):
            fn = m.group(1)
            a, b = (3, -1) if fn.startswith("san_cristobal_") else (1, -1)
            t_str = "".join(re.split(r"[_\.]", fn)[a:b])
            try:
                t = pd.to_datetime(t_str, format=r"%Y%m%d%H")
            except ValueError:
                t = np.nan
            data.append((location, t, fn, f"{url}{fn}"))
        return data

    with ThreadPool(processes=min(n_threads, len(locations))) as pool:
        data = list(itertools.chain.from_iterable(pool.imap_unordered(get_files, locations)))
    df = pd.DataFrame(data, columns=["location", "time", "fn", "url"])
    if cache:
        for location in locations:
            _FILES_L100_CACHE[location] = list(
                df[df["location"] == location].itertuples(index=False, name=None)
            )
    return df


def add_data(dates, *, location=None, n_procs=1, errors="raise", **kwargs):
    """
    Reads GML Ozonesonde data.

    Parameters
    ----------
    dates : pd.DatetimeIndex or list of datetime
        Dates to retrieve.
    location : str or list of str, optional
        Locations to retrieve, by default None (all locations).
    n_procs : int, optional
        Number of processors for dask compute, by default 1.
    errors : str, optional
        Whether to 'raise' or 'warn' on read errors, by default 'raise'.
    **kwargs : dict
        Additional arguments passed to open_dataset.

    Returns
    -------
    pd.DataFrame
        The loaded ozonesonde data.
    """
    return GMLOzonesondeReader().open_dataset(
        dates=dates, location=location, n_procs=n_procs, errors=errors, as_xarray=False, **kwargs
    )


class ColInfo(NamedTuple):
    name: str
    long_name: str
    units: str
    na_val: Optional[Union[str, Tuple[str, ...]]]


COL_INFO_L100 = [
    ColInfo("lev", "level", "", None),
    ColInfo("press", "pressure", "hPa", "9999.9"),
    ColInfo("altitude", "altitude", "km", ("99.9", "999.9", "99.999", "999.999")),
    ColInfo("theta", "potential temperature", "K", "9999.9"),
    ColInfo("temp", "air temperature", "degC", "999.9"),
    ColInfo("ftempv", "frost point temperature", "degC", "999.9"),
    ColInfo("rh", "relative humidity", "%", "999"),
    ColInfo("o3_press", "ozone partial pressure", "mPa", "99.90"),
    ColInfo("o3", "ozone mixing ratio", "ppmv", "99.999"),
    ColInfo("o3_int", "integrated ozone below", "atm-cm", "99.9990"),
    ColInfo("ptemp", "pump temperature", "degC", "999.9"),
    ColInfo("o3_nd", "ozone number density", "10^11 cm-3", "999.999"),
    ColInfo(
        "o3_res",
        "estimated total column ozone above",
        "DU",
        ("9999", "99999", "99.999"),
    ),
    ColInfo("o3_uncert", "uncertainty in ozone", "%", ("99999.000", "99.999")),
]

_DATA_BLOCK_START_L100 = """\
Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res  O3 Uncert
 Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU          %
"""
_DATA_BLOCK_START_L100_NO_UNCERT = """\
Level   Press    Alt   Pottp   Temp   FtempV   Hum  Ozone  Ozone   Ozone  Ptemp  O3 # DN O3 Res
 Num     hPa      km     K      C       C       %    mPa    ppmv   atmcm    C   10^11/cc   DU
"""


def read_100m(fp_or_url):
    """
    Reads a GML 100m average file (.l100).

    Parameters
    ----------
    fp_or_url : str
        File path or URL.

    Returns
    -------
    pd.DataFrame
        The loaded data with metadata in attrs.
    """
    if isinstance(fp_or_url, str) and fp_or_url.startswith(("http://", "https://")):

        @retry
        def get_text():
            r = requests.get(fp_or_url, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text

        text = get_text()
    else:
        with open(fp_or_url) as f:
            text = f.read()

    blocks = text.replace("\r", "").split("\n\n")
    nblocks = len(blocks)
    if nblocks == 5:
        meta_block = blocks[3]
        data_block = blocks[4]
    elif nblocks == 2:
        block_lines = blocks[0].splitlines()
        for i, line in enumerate(block_lines):
            if line.startswith(("Station:", "Station: ", "Station  ")):
                break
        else:
            raise ValueError("Expected to find metadata to start with Station")
        meta_block = "\n".join(block_lines[i:])
        data_block = blocks[1]
    else:
        # Some files might have 4 blocks if there is one empty block at the end
        if nblocks > 2 and "Level   Press" in blocks[-1]:
            data_block = blocks[-1]
            meta_block = blocks[-2]
        else:
            raise ValueError(f"Expected 2 or 5 blocks, got {nblocks}")

    meta = {}
    todo = meta_block.splitlines()[::-1]
    on_val_side = ["Background: ", "Flowrate: ", "RH Corr: ", "Sonde Total O3 (SBUV): "]
    while todo:
        line = todo.pop()
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        for key_ish in on_val_side:
            if key_ish in val:
                i = val.index(key_ish)
                meta[key.strip()] = val[:i].strip()
                todo.append(val[i:])
                break
        else:
            meta[key.strip()] = val.strip()

    if data_block.startswith(_DATA_BLOCK_START_L100):
        have_uncert = True
    elif data_block.startswith(_DATA_BLOCK_START_L100_NO_UNCERT):
        have_uncert = False
    else:
        # Try to be more robust, check for the second line as well
        lines = data_block.splitlines()
        if len(lines) > 2 and "hPa" in lines[1] and "km" in lines[1]:
            have_uncert = "Uncert" in lines[0]
        else:
            raise ValueError(
                f"Data block does not start with expected header. Got: {data_block[:100]!r}"
            )

    col_info = COL_INFO_L100[:]
    if not have_uncert:
        col_info = [c for c in col_info if c.name != "o3_uncert"]

    names = [c.name for c in col_info]
    dtype = {c.name: float for c in col_info}
    dtype["lev"] = int
    na_values = {c.name: c.na_val for c in col_info if c.na_val is not None}

    # Robust check for column count
    first_data_line = data_block.splitlines()[2]
    ncols = len(first_data_line.split())
    if ncols != len(names):
        raise ValueError(f"Expected {len(names)} columns in data block, got {ncols}")

    df = pd.read_csv(
        StringIO(data_block),
        skiprows=2,
        header=None,
        delimiter=r"\s+",
        names=names,
        dtype=dtype,
        na_values=na_values,
    )

    df["time"] = pd.Timestamp(f"{meta['Launch Date']} {meta['Launch Time']}").tz_localize(None)
    df["latitude"] = float(meta["Latitude"])
    df["longitude"] = float(meta["Longitude"])
    df["station"] = meta["Station"]
    df["station_height_str"] = meta["Station Height"]
    df["flight_number"] = meta["Flight Number"]

    # Site normalization
    repl = {
        "Boulder, CO": "Boulder, Colorado",
        "Hilo,Hawaii": "Hilo, Hawaii",
        "Huntsville": "Huntsville, Alabama",
        "Huntsville, AL": "Huntsville, Alabama",
        "San Cristobal, Galapagos, Ecuador": "San Cristobal, Galapagos",
        "South Pole": "South Pole, Antarctica",
        "Trinidad Head, CA": "Trinidad Head, California",
    }
    df["siteid"] = df["station"].replace(repl)

    # Metadata for Aero Protocol and existing tests
    df.attrs["ds_attrs"] = meta
    df.attrs["var_attrs"] = {c.name: {"long_name": c.long_name, "units": c.units} for c in col_info}

    return df
