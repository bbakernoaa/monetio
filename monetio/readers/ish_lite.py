"""ISH Lite Reader"""

import datetime
import warnings
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("ish_lite")
class ISHLiteReader(PointReader):
    def open_dataset(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime.datetime], datetime.datetime, str],
        box: List[float] = None,
        country: str = None,
        state: str = None,
        site: str = None,
        resample: bool = False,
        window: str = "h",
        n_procs: int = 1,
        verbose: bool = False,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) Lite data.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to retrieve.
        box : List[float], optional
            Bounding box [latmin, lonmin, latmax, lonmax].
        country : str, optional
            Country code to filter sites.
        state : str, optional
            State code to filter sites.
        site : str, optional
            Specific station ID to filter.
        resample : bool, optional
            Whether to resample data to a regular window, by default False.
        window : str, optional
            Resampling window (e.g., 'h'), by default 'h'.
        n_procs : int, optional
            Number of processors for dask compute, by default 1.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ISH Lite data.
        """
        ish = ISH()
        ish.dates = pd.to_datetime(dates)
        if ish.history is None:
            ish.read_ish_history()
        dfloc = ish.history.copy()

        if box is not None:
            dfloc = ish.subset_sites(latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3])
        elif country is not None:
            dfloc = dfloc.loc[dfloc.ctry == country, :]
        elif state is not None:
            dfloc = dfloc.loc[dfloc.state == state, :]
        elif site is not None:
            dfloc = dfloc.loc[dfloc.station_id == site, :]

        urls = ish.build_urls(sites=dfloc)
        if urls.empty:
            raise ValueError("No data URLs found")

        # Define per-file preprocessing if needed, or just use the reader_func
        # For ISH Lite, we need to handle the fixed-width/space-separated format.
        read_func = read_ish_lite_file

        df = super().open_dataset(
            urls.name.tolist(),
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        # Filtering by time range (exclusive upper bound to match legacy behavior)
        df = df.loc[(df.time >= ish.dates.min()) & (df.time < ish.dates.max())]

        # Merge with location metadata
        # We ensure consistent dtypes for merge
        dfloc["station_id"] = dfloc["station_id"].astype(object)
        if lazy:
            import dask.dataframe as dd

            df = df.assign(siteid=df.siteid.astype(object))
            # Convert dfloc to dask to ensure consistent merging and avoid warnings
            dfloc_dask = dd.from_pandas(dfloc, npartitions=1).assign(
                station_id=lambda x: x.station_id.astype(object)
            )
            df = df.merge(dfloc_dask, how="left", left_on="siteid", right_on="station_id")
        else:
            df["siteid"] = df["siteid"].astype(object)
            df = df.merge(dfloc, how="left", left_on="siteid", right_on="station_id")

        df = df.rename(columns={"ctry": "country"}).drop(columns=["station_id"], errors="ignore")
        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            if resample and ds.sizes.get("time", 0) > 0:
                ds = self._resample_xarray(ds, window=window)

            # Update history
            history = (
                f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH Lite data."
            )
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        # Note: resample is NOT applied to DataFrame here if as_xarray=False
        if resample and not as_xarray:
            if lazy:
                warnings.warn(
                    "ISHLiteReader: Resampling is currently not supported for lazy DataFrames."
                )
            else:
                # Eager path handles resample by checking if empty
                if not df.empty:
                    df = (
                        df.set_index("time")
                        .groupby("siteid")
                        .resample(window)
                        .mean(numeric_only=True)
                        .reset_index()
                    )

        return df

    def _resample_xarray(self, ds: xr.Dataset, window: str) -> xr.Dataset:
        """
        Resample an xarray Dataset while preserving metadata.

        Parameters
        ----------
        ds : xr.Dataset
            Input 1D or 2D dataset.
        window : str
            Resampling window.

        Returns
        -------
        xr.Dataset
            Resampled dataset.
        """
        # 1. Ensure 2D (time, node) for clean resampling
        if "node" in ds.dims and "time" in ds.coords:
            is_2d = "time" in ds.dims and "node" in ds.dims
            if not is_2d:
                from ..util import ds_to_2d

                ds = ds_to_2d(ds)

        if "time" in ds.dims:
            # 2. Identify coordinates and variables to preserve
            coord_names = [c for c in ds.coords if c not in ds.dims]
            ds = ds.reset_coords(coord_names)

            # 3. Resample numeric data variables
            ds_resampled = ds.resample(time=window).mean()

            # 4. Handle non-numeric or static variables that were dropped
            for v in ds.data_vars:
                if v not in ds_resampled.data_vars:
                    # For metadata that is constant over time, we take the first value.
                    if "time" in ds[v].dims:
                        if ds.sizes["time"] > 0:
                            ds_resampled[v] = ds[v].isel(time=0, drop=True)
                    else:
                        ds_resampled[v] = ds[v]

            # 5. Restore coordinates
            to_restore = [c for c in coord_names if c in ds_resampled.data_vars]
            ds_resampled = ds_resampled.set_coords(to_restore)

            return ds_resampled

        return ds


def read_ish_lite_file(fname: str, **kwargs) -> pd.DataFrame:
    """
    Read a single ISH Lite file.

    Parameters
    ----------
    fname : str
        File path or URL.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    columns = [
        "year",
        "month",
        "day",
        "hour",
        "temp",
        "dew_pt_temp",
        "press",
        "wdir",
        "ws",
        "sky_condition",
        "precip_1hr",
        "precip_6hr",
    ]

    # Use FileUtility
    fs = FileUtility.get_fs(str(fname))
    compression = "gzip" if str(fname).endswith(".gz") else None

    storage_options = kwargs.get("storage_options", {})
    if str(fname).startswith("s3://") and not storage_options:
        storage_options = {"anon": True}

    try:
        with fs.open(str(fname), "rb", compression=compression, **storage_options) as f:
            df = pd.read_csv(
                f,
                sep=r"\s+",
                header=None,
                names=columns,
                on_bad_lines="warn",
            )
    except Exception as e:
        warnings.warn(f"ISHLiteReader: Failed to read {fname}. Error: {e}")
        return pd.DataFrame(columns=["time", "siteid"] + columns[4:])

    if df.empty:
        return pd.DataFrame(columns=["time", "siteid"] + columns[4:])

    # Create time column
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["year", "month", "day", "hour"])

    # Extract siteid from filename
    import os

    basename = os.path.basename(str(fname))
    parts = basename.split("-")
    if len(parts) >= 2:
        siteid = parts[0] + parts[1]
    else:
        siteid = "unknown"
    df["siteid"] = siteid

    # Scale values
    for col in ["temp", "dew_pt_temp", "press", "ws", "precip_1hr", "precip_6hr"]:
        if col in df.columns:
            df[col] = df[col] / 10.0

    # Handle missing values
    df = df.replace(-9999, np.nan)
    df = df.replace(-999.9, np.nan)

    return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/ish_lite.py
# -----------------------------------------------------------------------------


class ISH:
    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.dates = None
        self.verbose = False

    def read_ish_history(self, dates=None):
        if dates is None:
            dates = self.dates
        fname = self.history_file

        fs = FileUtility.get_fs(fname)
        try:
            with fs.open(fname, "r") as f:
                self.history = pd.read_csv(
                    f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str}
                )
        except Exception:
            # Fallback for deprecated www1 host
            alt = fname.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
            if alt != fname:
                fs_alt = FileUtility.get_fs(alt)
                with fs_alt.open(alt, "r") as f:
                    self.history = pd.read_csv(
                        f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str}
                    )
                self.history_file = alt
            else:
                raise

        self.history.columns = [i.lower() for i in self.history.columns]
        if dates is not None:
            index1 = (self.history.end >= dates.min()) & (self.history.begin <= dates.max())
            self.history = self.history.loc[index1, :]
        self.history = self.history.dropna(subset=["lat", "lon"])
        self.history.loc[:, "usaf"] = self.history.usaf.astype("str").str.zfill(6)
        self.history.loc[:, "wban"] = self.history.wban.astype("str").str.zfill(5)
        self.history["station_id"] = self.history.usaf + self.history.wban
        self.history.rename(columns={"lat": "latitude", "lon": "longitude"}, inplace=True)

    def subset_sites(self, latmin=32.65, lonmin=-113.3, latmax=34.5, lonmax=-110.4):
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (self.history.longitude <= lonmax)
        dfloc = self.history.loc[latindex & lonindex, :]
        return dfloc

    def build_urls(self, dates=None, sites=None):
        if dates is None:
            dates = self.dates
        if sites is None:
            sites = self.history

        unique_years = pd.to_datetime(dates.year.unique(), format="%Y")
        furls = []
        url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"

        # Assume availability
        for syear in unique_years.strftime("%Y"):
            year_fnames = (
                sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
            )
            for fname in year_fnames:
                furls.append(f"{url}/{syear}/{fname}")

        return pd.Series(furls, name="name").to_frame()
