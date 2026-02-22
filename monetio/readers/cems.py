"""CEMS Reader"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader


class CEMS:
    """Legacy CEMS class for backward compatibility."""

    pass


@register_reader("cems")
class CEMSReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        states: Union[str, List[str]] = "md",
        n_procs: int = 1,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
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
            for dt in dates.floor("MS").unique():
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
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read CEMS data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history

            return ds

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/cems_mod.py
# -----------------------------------------------------------------------------


def build_url(date, state):
    """
    Build CEMS URL for a given date and state.
    """
    url = "ftp://newftp.epa.gov/DmDnLoad/emissions/hourly/monthly/"
    url += date.strftime("%Y") + "/"
    fname = date.strftime("%Y") + state.lower() + date.strftime("%m") + ".zip"
    return url + fname


def get_date_fmt(date):
    temp = date.split("-")
    if len(temp[0]) == 4:
        fmt = "%Y-%m-%d %H"
    else:
        fmt = "%m-%d-%Y %H"
    return fmt


def read_cems(efile, **kwargs):
    """
    Read a single CEMS file.
    """
    dftemp = pd.read_csv(efile, sep=",", index_col=False, header=0)
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

    dfmt = get_date_fmt(dftemp["date"].iloc[0])
    # Vectorized time construction
    dt_str = dftemp["date"].astype(str) + " " + dftemp["hour"].astype(str)
    dftemp["time"] = pd.to_datetime(dt_str, format=dfmt)
    dftemp = dftemp.rename(columns={"time": "time_local"})
    # For backend-agnostic loading, we need a 'time' column (UTC)
    # CEMS data is local time, and usually doesn't have offset info easily accessible in the file.
    # We set time = time_local for now, or use timezonefinder if we had lat/lon.
    dftemp["time"] = dftemp["time_local"]

    dftemp = dftemp.drop(columns=["date", "hour", "year"], errors="ignore")

    # siteid construction
    if "orispl_code" in dftemp.columns:
        dftemp["siteid"] = dftemp["orispl_code"].astype(str)

    return dftemp
