"""GML Ozonesonde Reader"""

import re
import warnings
from io import StringIO
from typing import NamedTuple, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import requests

from .base import PointReader, register_reader


@register_reader("gml_ozonesonde")
class GMLOzonesondeReader(PointReader):
    def open_dataset(self, dates, location=None, n_procs=1, errors="raise", **kwargs):
        """
        Reads GML Ozonesonde data.
        """
        return add_data(dates, location=location, n_procs=n_procs, errors=errors)


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
        except:
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


def add_data(dates, *, location=None, n_procs=1, errors="raise"):
    dates = pd.DatetimeIndex(dates)
    dates_min, dates_max = dates.min(), dates.max()
    df_urls = discover_files(location=location)
    urls = df_urls[df_urls["time"].between(dates_min, dates_max, inclusive="both")]["url"].tolist()
    if not urls:
        raise RuntimeError(f"No files found for dates {dates_min} to {dates_max}.")

    def func(fp_or_url):
        try:
            return read_100m(fp_or_url)
        except Exception as e:
            if errors == "raise":
                raise RuntimeError(f"Failed to read {fp_or_url}") from e
            elif errors == "warn":
                warnings.warn(f"Failed to read {fp_or_url}: {e}")
            return pd.DataFrame()

    dfs = [dask.delayed(func)(url) for url in urls]
    dff = dd.from_delayed(dfs, verify_meta=errors == "raise")
    df = dff.compute(num_workers=n_procs).reset_index()
    df = df[df["time"].between(dates_min, dates_max, inclusive="both")]

    repl = {
        "Boulder, CO": "Boulder, Colorado",
        "Hilo,Hawaii": "Hilo, Hawaii",
        "Huntsville": "Huntsville, Alabama",
        "Huntsville, AL": "Huntsville, Alabama",
        "San Cristobal, Galapagos, Ecuador": "San Cristobal, Galapagos",
        "South Pole": "South Pole, Antarctica",
        "Trinidad Head, CA": "Trinidad Head, California",
    }
    df["station"] = df["station"].replace(repl)
    df = df.rename(columns={"station": "siteid"})
    return df.drop(columns=["index"], errors="ignore").reset_index(drop=True)


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
        raise ValueError(f"Expected 2 or 5 blocks, got {nblocks}")

    meta = {}
    todo = meta_block.splitlines()[::-1]
    on_val_side = ["Background: ", "Flowrate: ", "RH Corr: ", "Sonde Total O3 (SBUV): "]
    while todo:
        line = todo.pop()
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
        raise ValueError("Data block header mismatch")

    col_info = COL_INFO_L100[:]
    if not have_uncert:
        _ = col_info.pop()

    names = [c.name for c in col_info]
    dtype = {c.name: float for c in col_info}
    dtype["lev"] = int
    na_values = {c.name: c.na_val for c in col_info if c.na_val is not None}

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

    return df
