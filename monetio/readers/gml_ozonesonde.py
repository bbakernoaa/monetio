"""GML Ozonesonde Reader"""

import re
from datetime import datetime
from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .drivers import FileUtility

if TYPE_CHECKING:
    import dask.dataframe as dd


@register_reader("gml_ozonesonde")
class GMLOzonesondeReader(PointReader):
    """
    Reader for GML Ozonesonde 100m average files (.l100).
    """

    def open_dataset(
        self,
        files: Union[str, List[str], None] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str, None] = None,
        location: Union[str, List[str], None] = None,
        n_procs: int = 1,
        errors: str = "raise",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
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
            Number of processors for parallel loading (if not lazy), by default 1.
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

            dates_idx = pd.to_datetime(dates)
            if isinstance(dates_idx, pd.Timestamp):
                dates_idx = pd.DatetimeIndex([dates_idx])

            df_urls = discover_files(location=location)
            # Filter by date
            mask = df_urls["time"].between(dates_idx.min(), dates_idx.max(), inclusive="both")
            urls = df_urls.loc[mask, "url"].tolist()

            if not urls:
                raise RuntimeError(
                    f"No files found for dates {dates_idx.min()} to {dates_idx.max()} "
                    f"at location(s) {location}."
                )
            files = urls

        # Filter out arguments that are not for the reader function
        reader_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["expand2d", "pivot", "wide_fmt"]
        }

        # Use PandasDriver to open files
        df = self.driver.open(files, read_method=read_100m, lazy=lazy, **reader_kwargs)

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

    def to_xarray(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], expand2d: bool = True, **kwargs: Any
    ) -> xr.Dataset:
        """
        Convert the DataFrame to an xarray Dataset in UGRID convention.
        For profiles, we often want to keep 'lev' or 'altitude' as coordinates.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.
        expand2d : bool, optional
            Whether to expand to multi-dimensional structure, by default True.
        **kwargs : dict
            Additional arguments passed to the conversion.

        Returns
        -------
        xr.Dataset
            The dataset in UGRID convention.
        """
        # 1. Base conversion to 1D UGRID
        ds = super().to_xarray(df, expand2d=False, **kwargs)

        # 2. Ensure vertical coordinates
        for vcol in ["lev", "press", "altitude"]:
            if vcol in ds.data_vars:
                ds = ds.set_coords(vcol)

        # 3. Handle 2D (or 3D for profiles) expansion
        if expand2d:
            try:
                # We want to unstack to (time, siteid, lev) if possible
                # But since each profile might have different levels,
                # we use 'lev' as a dimension.
                # If there are multiple flights for the same time/site, we might need 'flight_number'
                index_cols = ["time", "siteid"]
                if "lev" in ds.coords:
                    index_cols.append("lev")

                # Check for duplicates in index
                # This is tricky for dask. For now we assume they are unique enough or unstack will handle.
                ds = ds.set_index(node=index_cols).unstack("node")

                # Rename siteid to node if it exists as a dimension
                if "siteid" in ds.dims:
                    ds = ds.rename({"siteid": "node"})

                # Add UGRID mesh metadata if we still have a 'node' dimension
                if "node" in ds.dims:
                    ds.coords["node"] = (("node",), np.arange(ds.sizes["node"]))

            except Exception as e:
                import warnings

                warnings.warn(f"GMLOzonesondeReader.to_xarray expand2d failed: {e}. Returning 1D.")

        return ds


# -----------------------------------------------------------------------------
# Helper functions
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

_FILES_L100_CACHE: Dict[str, Optional[List[Tuple[str, datetime, str, str]]]] = {
    location: None for location in LOCATIONS
}


def retry(func: Any) -> Any:
    """Decorator to retry a function on network errors."""
    import time
    from functools import wraps
    from random import random as rand

    import requests

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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


def discover_files(
    location: Union[str, List[str], None] = None,
    *,
    n_threads: int = 3,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Discover available GML Ozonesonde files.

    Parameters
    ----------
    location : Union[str, List[str]], optional
        Locations to retrieve, by default None (all locations).
    n_threads : int, optional
        Number of threads for parallel discovery, by default 3.
    cache : bool, optional
        Whether to use cached file lists, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with location, time, filename, and URL.
    """
    import itertools
    from multiprocessing.pool import ThreadPool

    import requests

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
    def get_files(loc: str) -> List[Tuple[str, Any, str, str]]:
        cached = _FILES_L100_CACHE[loc]
        if cached is not None:
            return cached
        url_location = "South Pole, Antartica" if loc == "South Pole, Antarctica" else loc
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
            data.append((loc, t, fn, f"{url}{fn}"))
        return data

    with ThreadPool(processes=min(n_threads, len(locations))) as pool:
        results = list(itertools.chain.from_iterable(pool.imap_unordered(get_files, locations)))
    df = pd.DataFrame(results, columns=["location", "time", "fn", "url"])
    if cache:
        for loc in locations:
            _FILES_L100_CACHE[loc] = [
                tuple(x)  # type: ignore
                for x in df[df["location"] == loc].itertuples(index=False, name=None)
            ]
    return df


def add_data(
    dates: Union[pd.DatetimeIndex, List[datetime]],
    *,
    location: Union[str, List[str], None] = None,
    n_procs: int = 1,
    errors: str = "raise",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Reads GML Ozonesonde data.

    Parameters
    ----------
    dates : pd.DatetimeIndex or list of datetime
        Dates to retrieve.
    location : str or list of str, optional
        Locations to retrieve, by default None (all locations).
    n_procs : int, optional
        Number of processors for parallel loading, by default 1.
    errors : str, optional
        Whether to 'raise' or 'warn' on read errors, by default 'raise'.
    **kwargs : dict
        Additional arguments passed to open_dataset.

    Returns
    -------
    pd.DataFrame
        The loaded ozonesonde data.
    """
    return GMLOzonesondeReader().open_dataset(  # type: ignore
        dates=dates,
        location=location,
        n_procs=n_procs,
        errors=errors,
        as_xarray=False,
        **kwargs,
    )


class ColInfo(NamedTuple):
    name: str
    long_name: str
    units: str
    na_val: Optional[Union[str, Tuple[str, ...]]]


COL_INFO_L100 = [
    ColInfo("lev", "level", "", None),
    ColInfo("press", "pressure", "hPa", "9999.9"),
    ColInfo("altitude", "altitude", "km", ("99.9", "99.9", "99.999", "999.999")),
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


def read_100m(fp_or_url: str) -> pd.DataFrame:
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
    fs = FileUtility.get_fs(fp_or_url)
    with fs.open(fp_or_url, mode="rb") as f:
        content = f.read()

    text = content.decode("utf-8", errors="replace")

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
                idx = val.index(key_ish)
                meta[key.strip()] = val[:idx].strip()
                todo.append(val[idx:])
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
    data_lines = data_block.splitlines()
    if len(data_lines) < 3:
        raise ValueError(f"Data block too short: {len(data_lines)} lines")

    first_data_line = data_lines[2]
    ncols = len(first_data_line.split())
    if ncols != len(names):
        raise ValueError(f"Expected {len(names)} columns in data block, got {ncols}")

    df = pd.read_csv(
        StringIO(data_block),
        skiprows=2,
        header=None,
        sep=r"\s+",
        names=names,
        dtype=dtype,
        na_values=na_values,
        engine="python",
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
