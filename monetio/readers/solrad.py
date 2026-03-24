"""SOLRAD Reader"""

import io
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd

# Base headers for SOLRAD
BASE_HEADERS = (
    "year",
    "jday",
    "month",
    "day",
    "hour",
    "minute",
    "dt",
    "zen",
    "ghi",
    "ghi_flag",
    "dni",
    "dni_flag",
    "dhi",
    "dhi_flag",
    "uvb",
    "uvb_flag",
    "uvb_temp",
    "uvb_temp_flag",
)

# Standard headers for remaining columns
STD_HEADERS = ("std_ghi", "std_dni", "std_dhi", "std_uvb")
HEADERS = BASE_HEADERS + STD_HEADERS

# Headers for Madison, WI site (includes DPIR)
DPIR_HEADERS = ("dpir", "dpir_flag", "dpirc", "dpirc_flag", "dpird", "dpird_flag")
MADISON_HEADERS = BASE_HEADERS + DPIR_HEADERS + STD_HEADERS + ("std_dpir", "std_dpirc", "std_dpird")

# Fixed-width columns (base widths from README_SOLRAD.txt)
# year(4), jday(3), month(2), day(2), hour(2), minute(2), dt(6), zen(6)
# then 5 pairs of (value(7), flag(1)) for ghi, dni, dhi, uvb, uvb_temp
# then 4 values of 9 chars for std_ghi, std_dni, std_dhi, std_uvb
WIDTHS = [4, 3, 2, 2, 2, 2, 6, 6] + 5 * [7, 1] + 4 * [9]
# Madison has 8 pairs of (value, flag) because of DPIR
MADISON_WIDTHS = [4, 3, 2, 2, 2, 2, 6, 6] + 8 * [7, 1] + 7 * [9]


# In the files, there is one space between columns.
# To use read_fwf, we can either provide widths that include the space,
# or better, use colspecs.
def get_colspecs(widths):
    colspecs = []
    start = 0
    for w in widths:
        # Each field of width w is followed by 1 space, except possibly the last
        # But actually, the total width of the field including the trailing space is w+1.
        colspecs.append((start, start + w))
        start += w + 1
    return colspecs


COLSPECS = get_colspecs(WIDTHS)
MADISON_COLSPECS = get_colspecs(MADISON_WIDTHS)

# Variable mapping for SOLRAD
VARIABLE_MAP = {
    "zen": "solar_zenith",
    # ghi, dni, dhi are already standard-ish
}


def read_solrad(filename: str, **kwargs: dict) -> pd.DataFrame:
    """
    Read a single SOLRAD file.

    Parameters
    ----------
    filename : str
        The path or URL to the SOLRAD file.
    **kwargs : dict
        Additional arguments passed to pd.read_fwf.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    if "msn" in filename.lower():
        names = MADISON_HEADERS
        colspecs = MADISON_COLSPECS
    else:
        names = HEADERS
        colspecs = COLSPECS

    # Use FileUtility to handle remote files
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        content = f.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        lines = content.splitlines()
        if len(lines) < 3:
            return pd.DataFrame()

        station_name = lines[0].strip()
        metadata_line = lines[1].split()

        latitude = float(metadata_line[0])
        longitude = float(metadata_line[1])
        elevation = float(metadata_line[2])
        # tz_offset = int(metadata_line[3])

        # Data starts from line 2
        data_content = "\n".join(lines[2:])
        df = pd.read_fwf(
            io.StringIO(data_content),
            colspecs=colspecs,
            header=None,
            names=names,
            na_values=-9999.9,
            **kwargs,
        )

    # Add metadata
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["elevation"] = elevation
    df["siteid"] = station_name

    # Create time column
    # Ensure columns are integer and then format to string with zfill
    # We use a custom converter to handle potential issues with read_fwf and NaN in time columns
    def to_str(series, n):
        # Handle cases where column might be floating point because of NaNs, or already string
        # We ensure they are numeric first to avoid '.0' in string conversion
        s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(str)
        return s.str.zfill(n)

    df["time"] = pd.to_datetime(
        to_str(df["year"], 4)
        + to_str(df["month"], 2)
        + to_str(df["day"], 2)
        + to_str(df["hour"], 2)
        + to_str(df["minute"], 2),
        format="%Y%m%d%H%M",
        errors="coerce",
    )

    return df


@register_reader("solrad")
class SOLRADReader(PointReader):
    """
    Reader for NOAA SOLRAD network data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[datetime, List[datetime], pd.DatetimeIndex]] = None,
        sites: Optional[List[str]] = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Open SOLRAD dataset.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs. If None, uses `dates` and `sites` to discover files.
        dates : Union[datetime, List[datetime], pd.DatetimeIndex], optional
            Dates to retrieve if `files` is None.
        sites : List[str], optional
            Site abbreviations (e.g. ['abq', 'bis', 'hnx', 'msn', 'slc', 'sea', 'ste']).
        as_xarray : bool, optional
            If True, returns an xarray.Dataset, by default True.
        lazy : bool, optional
            If True, returns a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[xr.Dataset, pd.DataFrame]
            The loaded dataset.
        """
        if files is None:
            if dates is None or sites is None:
                raise ValueError("Either 'files' or both 'dates' and 'sites' must be provided.")
            files = self.build_urls(dates, sites)

        # Separate driver kwargs from to_xarray/postprocess kwargs
        driver_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["expand2d", "wide_fmt", "pivot", "as_xarray", "lazy"]
        }

        # We use read_solrad as the custom read_method
        df = self.driver.open(files, read_method=read_solrad, lazy=lazy, **driver_kwargs)

        # Post-processing: Harmonize column names
        df = self._postprocess(df)

        # Consistently force object strings
        df = force_object_strings(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)
            # Update history for provenance
            ds = update_history(ds, "Harmonized SOLRAD dataset.")
            return ds

        return df

    def _postprocess(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Harmonize column names.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            Post-processed dataframe.
        """
        # Rename according to VARIABLE_MAP
        df = df.rename(columns=VARIABLE_MAP)

        # Update history for provenance
        df = update_history(df, "Applied variable name mapping for SOLRAD.")

        return df

    def build_urls(
        self,
        dates: Union[datetime, List[datetime], pd.DatetimeIndex],
        sites: List[str],
    ) -> List[str]:
        """
        Discover available URLs for the given dates and sites.

        Parameters
        ----------
        dates : Union[datetime, List[datetime], pd.DatetimeIndex]
            Dates to retrieve.
        sites : List[str]
            Site abbreviations.

        Returns
        -------
        List[str]
            List of URLs.
        """
        baseurl = "https://gml.noaa.gov/aftp/data/radiation/solrad/"

        urls = []
        dates = pd.DatetimeIndex(np.atleast_1d(dates))

        for date in dates:
            for site in sites:
                year = date.year
                # Format: site/year/siteyyjday.dat
                # e.g. abq/2024/abq24001.dat
                fname = f"{site.lower()}{date.strftime('%y%j')}.dat"
                url = f"{baseurl}{site.lower()}/{year}/{fname}"
                urls.append(url)

        return urls
