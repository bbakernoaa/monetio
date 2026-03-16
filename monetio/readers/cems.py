"""CEMS Reader"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd


class CEMS:
    """Legacy CEMS class for backward compatibility."""

    pass


@register_reader("cems")
class CEMSReader(PointReader):
    """
    Reader for Continuous Emissions Monitoring System (CEMS) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        states: Union[str, List[str]] = "md",
        n_procs: int = 1,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs: dict,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load CEMS data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs to read. If None, uses `dates` and `states` to discover files.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve.
        states : Union[str, List[str]], optional
            States to retrieve (e.g., 'md'), by default 'md'.
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded CEMS data.

        Examples
        --------
        >>> reader = CEMSReader()
        >>> ds = reader.open_dataset(dates="2023-01-01", states="md")
        """
        if files is None:
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")

            dates = pd.to_datetime(dates)
            if isinstance(dates, pd.Timestamp):
                dates = pd.DatetimeIndex([dates])

            if isinstance(states, str):
                states = [states]

            # Discovery logic
            files = []
            for dt in dates.to_period("M").to_timestamp().unique():
                for st in states:
                    files.append(build_url(dt, st))

        # Filter out arguments that are not for the reader function
        reader_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["expand2d", "pivot", "wide_fmt"]
        }

        df = self.driver.open(files, read_method=read_cems, lazy=lazy, **reader_kwargs)

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Update history
            ds = update_history(ds, "Read CEMS data.")

            return ds

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/cems_mod.py
# -----------------------------------------------------------------------------


def build_url(date: datetime, state: str) -> str:
    """
    Build CEMS URL for a given date and state.

    Parameters
    ----------
    date : datetime
        The date to retrieve data for.
    state : str
        The state abbreviation (e.g., 'md').

    Returns
    -------
    str
        The constructed URL.
    """
    url = "ftp://newftp.epa.gov/DmDnLoad/emissions/hourly/monthly/"
    url += date.strftime("%Y") + "/"
    fname = date.strftime("%Y") + state.lower() + date.strftime("%m") + ".zip"
    return url + fname


def get_date_fmt(date: str) -> str:
    """
    Determine the date format based on the first entry.

    Parameters
    ----------
    date : str
        The date string to inspect.

    Returns
    -------
    str
        The identified datetime format string.
    """
    temp = date.split("-")
    if len(temp[0]) == 4:
        fmt = "%Y-%m-%d"
    else:
        fmt = "%m-%d-%Y"
    return fmt


def read_cems(efile: str, **kwargs: dict) -> pd.DataFrame:
    """
    Read a single CEMS file.

    Parameters
    ----------
    efile : str
        The path or URL to the CEMS file.
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded CEMS data.
    """
    dftemp = pd.read_csv(efile, sep=",", index_col=False, header=0, **kwargs)
    columns = list(dftemp.columns.values)

    # Standardize column names
    rcolumn = []
    for ccc in columns:
        cl = ccc.lower()
        if "facility" in cl and "name" in cl:
            rcolumn.append("facility_name")
        elif "orispl" in cl:
            rcolumn.append("orispl_code")
        elif "facility" in cl and "id" in cl:
            rcolumn.append("fac_id")
        elif "so2" in cl and "lbs" in cl and "rate" not in cl:
            rcolumn.append("so2_lbs")
        elif "nox" in cl and "lbs" in cl and "rate" not in cl:
            rcolumn.append("nox_lbs")
        elif "co2" in cl and "short" in cl and "tons" in cl:
            rcolumn.append("co2_short_tons")
        elif "date" in cl:
            rcolumn.append("date")
        elif "hour" in cl:
            rcolumn.append("hour")
        elif "lat" in cl:
            rcolumn.append("latitude")
        elif "lon" in cl:
            rcolumn.append("longitude")
        elif "state" in cl:
            rcolumn.append("state_name")
        else:
            rcolumn.append(ccc.strip().lower())
    dftemp.columns = rcolumn

    # Optimized vectorized time construction
    dfmt = get_date_fmt(str(dftemp["date"].iloc[0]))
    dftemp["time_local"] = pd.to_datetime(dftemp["date"], format=dfmt) + pd.to_timedelta(
        dftemp["hour"], unit="h"
    )

    # For backend-agnostic loading, we need a 'time' column (UTC)
    # CEMS data is local time. We set time = time_local for now.
    dftemp["time"] = dftemp["time_local"]

    dftemp = dftemp.drop(columns=["date", "hour", "year"], errors="ignore")

    # siteid construction
    if "orispl_code" in dftemp.columns:
        dftemp["siteid"] = dftemp["orispl_code"].astype(str)

    return dftemp
