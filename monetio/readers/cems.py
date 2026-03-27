"""CEMS Reader"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

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

    def build_urls(
        self,
        dates: Optional[Any] = None,
        states: Union[str, List[str]] = "md",
        **kwargs: dict,
    ) -> List[str]:
        """
        Build CEMS URLs.
        """
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
        return files

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
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
        dates : Any, optional
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
            Includes `expand2d`, `pivot`, and `wide_fmt` for Xarray conversion.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded CEMS data. If `as_xarray=True`, units are assigned
            to individual data variables (e.g., 'lbs' for SO2).

        Examples
        --------
        >>> reader = CEMSReader()
        >>> ds = reader.open_dataset(dates="2023-01-01", states="md")
        """
        # Filter out arguments that are not for the reader function
        reader_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["expand2d", "pivot", "wide_fmt", "states"]
        }

        df = super().open_dataset(
            files,
            dates,
            states=states,
            n_procs=n_procs,
            read_method=read_cems,
            as_xarray=False,
            lazy=lazy,
            **reader_kwargs,
        )

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Retrieve unit mapping from dataframe attrs
            # Note: pd.concat and Dask operations can drop attrs.
            unit_map = getattr(df, "attrs", {}).get("unit_mapping", {})

            if not unit_map:
                # Eagerly peek at the first file to recover mapping if missing
                try:
                    from .drivers import FileUtility

                    file_list = FileUtility.expand_paths(files)
                    if file_list:
                        meta_df = read_cems(file_list[0])
                        unit_map = meta_df.attrs.get("unit_mapping", {})
                except Exception as e:
                    import warnings

                    warnings.warn(f"CEMS unit mapping recovery failed: {e}")

            # Apply units to variables
            if unit_map:
                for varname, unit in unit_map.items():
                    if varname in ds.data_vars:
                        ds[varname].attrs["units"] = unit

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
    from ..util import force_object_strings

    dftemp = pd.read_csv(efile, sep=",", index_col=False, header=0, **kwargs)

    # Standardize column names using a mapping
    rename_map = {
        "facility_name": ["facility", "name"],
        "orispl_code": ["orispl"],
        "fac_id": ["facility", "id"],
        "so2_lbs": ["so2", "lbs"],
        "nox_lbs": ["nox", "lbs"],
        "co2_short_tons": ["co2", "short", "tons"],
        "date": ["date"],
        "hour": ["hour"],
        "latitude": ["lat"],
        "longitude": ["lon"],
        "state_name": ["state"],
    }

    new_columns = []
    for col in dftemp.columns:
        cl = col.lower()
        matched = False
        for target, keywords in rename_map.items():
            if all(k in cl for k in keywords):
                # Special case for SO2/NOx to avoid "rate"
                if target in ["so2_lbs", "nox_lbs"] and "rate" in cl:
                    continue
                new_columns.append(target)
                matched = True
                break
        if not matched:
            new_columns.append(cl.strip())

    dftemp.columns = new_columns

    # Capture units mapping for variables
    unit_map = {
        "so2_lbs": "lbs",
        "nox_lbs": "lbs",
        "co2_short_tons": "short tons",
        "gross_load_mw": "MW",
        "steam_load_1000lb_hr": "1000 lb/hr",
    }
    # Store in attributes to be picked up by open_dataset
    dftemp.attrs["unit_mapping"] = {k: v for k, v in unit_map.items() if k in dftemp.columns}

    # Optimized vectorized time construction
    if not dftemp.empty and "date" in dftemp.columns and "hour" in dftemp.columns:
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

    return force_object_strings(dftemp)
