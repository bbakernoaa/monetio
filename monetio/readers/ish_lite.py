"""ISH Lite Reader"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility


def read_ish_lite_file(fname: str) -> pd.DataFrame:
    """
    Read a single ISH Lite file.

    Parameters
    ----------
    fname : str
        File path or URL.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    from numpy import nan

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
    fs = FileUtility.get_fs(fname)
    compression = "gzip" if fname.endswith(".gz") else None

    with fs.open(fname, "rb", compression=compression) as f:
        df = pd.read_csv(
            f,
            sep=r"\s+",
            header=None,
            names=columns,
        )
    # Create time column manually
    df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df.drop(["year", "month", "day", "hour"], axis=1, inplace=True)

    filename = os.path.basename(fname).split("-")
    siteid = filename[0] + filename[1]
    for col in ["temp", "dew_pt_temp", "press", "ws", "precip_1hr", "precip_6hr"]:
        df[col] /= 10.0
    df["siteid"] = siteid
    df = df.replace(-9999, nan)
    return df


@register_reader("ish_lite")
class ISHLiteReader(PointReader):
    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        box: Optional[List[float]] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        site: Optional[str] = None,
        resample: bool = False,
        window: str = "h",
        n_procs: int = 1,
        verbose: bool = False,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) Lite data .

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to retrieve if files are not provided.
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
            Number of processors for dask compute (if not lazy), by default 1.
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

        if files is None and dates is not None:
            dates = pd.to_datetime(dates)
            if ish.history is None:
                ish.read_ish_history(dates=dates)
            dfloc = ish.history.copy()

            if box is not None:
                dfloc = ish.subset_sites(latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3])
            elif country is not None:
                dfloc = dfloc.loc[dfloc.ctry == country, :]
            elif state is not None:
                dfloc = dfloc.loc[dfloc.state == state, :]
            elif site is not None:
                dfloc = dfloc.loc[dfloc.station_id == site, :]

            urls = ish.build_urls(dates=dates, sites=dfloc)
            if urls.empty:
                raise ValueError("No data URLs found")
            files = urls.name.tolist()

        if not files:
            raise ValueError("Must provide either 'files' or 'dates'.")

        # Use base class to open
        df = super().open_dataset(
            files,
            read_method=read_ish_lite_file,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        # Filtering
        if dates is not None:
            dates = pd.to_datetime(dates)
            # Use exclusive upper bound to match unit test expectations
            df = df.loc[(df.time >= dates.min()) & (df.time < dates.max())]

        df = df.replace(-999.9, np.nan)

        # Merge with metadata
        if ish.history is None:
            ish.read_ish_history()
        dfloc = ish.history.copy()
        dfloc = force_object_strings(dfloc)

        if lazy:
            import dask.dataframe as dd

            df = df.assign(siteid=df.siteid.astype(object))
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

            # Preserve metadata in coordinates during resampling
            meta_coords = [
                "country",
                "state",
                "station name",
                "elev(m)",
                "latitude",
                "longitude",
                "siteid",
            ]
            ds = ds.set_coords([c for c in meta_coords if c in ds.data_vars])

            if resample:
                # Backend-agnostic resampling in xarray
                try:
                    ds = ds.resample(time=window).mean(numeric_only=True)
                except TypeError:
                    ds = ds.resample(time=window).mean()

            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH Lite data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        if resample:
            if not lazy:
                if not df.empty:
                    df = (
                        df.set_index("time")
                        .groupby("siteid")
                        .resample(window)
                        .mean(numeric_only=True)
                        .reset_index()
                    )
                    # Re-join metadata for pandas eager path
                    df = df.merge(
                        dfloc.rename(columns={"ctry": "country"}),
                        how="left",
                        left_on="siteid",
                        right_on="station_id",
                    ).drop(columns=["station_id"], errors="ignore")
            else:
                import warnings

                warnings.warn(
                    "ISHLiteReader: Resampling is currently not supported for lazy DataFrames. "
                    "Convert to xarray (as_xarray=True) for lazy resampling."
                )

        return df


def add_data(
    dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
    box: Optional[List[float]] = None,
    country: Optional[str] = None,
    state: Optional[str] = None,
    site: Optional[str] = None,
    resample: bool = False,
    window: str = "h",
    n_procs: int = 1,
    verbose: bool = False,
    as_xarray: bool = True,
    lazy: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
    """
    Backward-compatible wrapper for ISHLiteReader.open_dataset.
    """
    return ISHLiteReader().open_dataset(
        dates=dates,
        box=box,
        country=country,
        state=state,
        site=site,
        resample=resample,
        window=window,
        n_procs=n_procs,
        verbose=verbose,
        as_xarray=as_xarray,
        lazy=lazy,
        **kwargs,
    )


class ISH:
    """Helper class for ISH Lite data retrieval."""

    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.dates = None
        self.verbose = False

    def read_ish_history(self, dates: Optional[pd.DatetimeIndex] = None):
        """
        Read the ISH history file.

        Parameters
        ----------
        dates : pd.DatetimeIndex, optional
            Dates to filter the history, by default None.
        """
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

    def subset_sites(
        self,
        latmin: float = 32.65,
        lonmin: float = -113.3,
        latmax: float = 34.5,
        lonmax: float = -110.4,
    ) -> pd.DataFrame:
        """
        Subset sites by bounding box.
        """
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (self.history.longitude <= lonmax)
        dfloc = self.history.loc[latindex & lonindex, :]
        return dfloc

    def build_urls(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        sites: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Construct ISH Lite URLs.
        """
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
