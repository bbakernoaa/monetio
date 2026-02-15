"""ISH Lite Reader"""

from datetime import datetime
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
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
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
        df = ish.add_data(
            dates,
            box=box,
            country=country,
            state=state,
            site=site,
            resample=resample,
            window=window,
            n_procs=n_procs,
            verbose=verbose,
            lazy=lazy,
        )

        df = self.harmonize(df)
        if as_xarray:
            ds = self.to_xarray(df)
            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH Lite data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

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

    def read_csv(self, fname):
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

        filename = fname.split("/")[-1].split("-")
        siteid = filename[0] + filename[1]
        for col in ["temp", "dew_pt_temp", "press", "ws", "precip_1hr", "precip_6hr"]:
            df[col] /= 10.0
        df["siteid"] = siteid
        df = df.replace(-9999, nan)
        return df

    def aggregate_files(self, urls, n_procs=1, lazy=False):
        """
        Aggregate multiple ISH Lite files.

        Parameters
        ----------
        urls : pd.DataFrame
            Dataframe with 'name' column containing URLs.
        n_procs : int, optional
            Number of processors for compute, by default 1.
        lazy : bool, optional
            Whether to stay lazy, by default False.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            The aggregated data.
        """
        import dask
        import dask.dataframe as dd

        dfs = [dask.delayed(self.read_csv)(f) for f in urls.name]
        dff = dd.from_delayed(dfs)
        if not lazy:
            return dff.compute(num_workers=n_procs)
        return dff

    def add_data(
        self,
        dates,
        box=None,
        country=None,
        state=None,
        site=None,
        resample=False,
        window="h",
        n_procs=1,
        verbose=False,
        lazy=False,
    ):
        self.dates = pd.to_datetime(dates)
        self.verbose = verbose
        if self.history is None:
            self.read_ish_history()
        dfloc = self.history.copy()

        if box is not None:
            dfloc = self.subset_sites(latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3])
        elif country is not None:
            dfloc = dfloc.loc[dfloc.ctry == country, :]
        elif state is not None:
            dfloc = dfloc.loc[dfloc.state == state, :]
        elif site is not None:
            dfloc = dfloc.loc[dfloc.station_id == site, :]

        urls = self.build_urls(sites=dfloc)
        if urls.empty:
            raise ValueError("No data URLs found")

        df = self.aggregate_files(urls, n_procs=n_procs, lazy=lazy)

        # Use exclusive upper bound to match unit test expectations (e.g. 24 hours for 1 day range)
        df = df.loc[(df.time >= self.dates.min()) & (df.time < self.dates.max())]
        df = df.replace(-999.9, np.nan)

        if resample:
            if not lazy:
                if not df.empty:
                    df = (
                        df.set_index("time").groupby("siteid").resample(window).mean().reset_index()
                    )
            else:
                import warnings

                warnings.warn("ISHLiteReader: Resampling is currently not supported in lazy mode.")

        # Ensure consistent dtypes for merge and to avoid nullable string issues in Pandas 3.0
        def _force_object(df_in):
            for col in df_in.columns:
                if pd.api.types.is_string_dtype(df_in[col]):
                    df_in[col] = df_in[col].astype(object)
            return df_in

        dfloc = _force_object(dfloc)

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

        df = df.rename(columns={"ctry": "country"})
        return df.drop(["station_id"], axis=1)
