"""ISH Lite Reader"""

import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("ish_lite")
class ISHLiteReader(PointReader):
    def open_dataset(
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
        **kwargs,
    ):
        """
        Reads ISH Lite data.
        """
        ish = ISH()
        return ish.add_data(
            dates,
            box=box,
            country=country,
            state=state,
            site=site,
            resample=resample,
            window=window,
            n_procs=n_procs,
            verbose=verbose,
        )


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
                self.history = pd.read_csv(f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str})
        except Exception:
            alt = fname.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
            if alt != fname:
                fs_alt = FileUtility.get_fs(alt)
                with fs_alt.open(alt, "r") as f:
                    self.history = pd.read_csv(f, parse_dates=["BEGIN", "END"], dtype={"USAF": str, "WBAN": str})
                self.history_file = alt
            else:
                raise

        self.history.columns = [i.lower() for i in self.history.columns]
        if dates is not None:
            index1 = (self.history.end >= dates.min()) & (
                self.history.begin <= dates.max()
            )
            self.history = self.history.loc[index1, :]
        self.history = self.history.dropna(subset=["lat", "lon"])
        self.history.loc[:, "usaf"] = self.history.usaf.astype("str").str.zfill(6)
        self.history.loc[:, "wban"] = self.history.wban.astype("str").str.zfill(5)
        self.history["station_id"] = self.history.usaf + self.history.wban
        self.history.rename(
            columns={"lat": "latitude", "lon": "longitude"}, inplace=True
        )

    def subset_sites(self, latmin=32.65, lonmin=-113.3, latmax=34.5, lonmax=-110.4):
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (
            self.history.longitude <= lonmax
        )
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
                sites.usaf.astype(str)
                + "-"
                + sites.wban.astype(str)
                + "-"
                + syear
                + ".gz"
            )
            for fname in year_fnames:
                furls.append(f"{url}/{syear}/{fname}")

        return pd.Series(furls, name="name").to_frame()

    def read_csv(self, fname):
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

    def aggregrate_files(self, urls, n_procs=1):
        dfs = [dask.delayed(self.read_csv)(f) for f in urls.name]
        dff = dd.from_delayed(dfs)
        df = dff.compute(num_workers=n_procs)
        return df

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
    ):
        self.dates = pd.to_datetime(dates)
        self.verbose = verbose
        if self.history is None:
            self.read_ish_history()
        dfloc = self.history.copy()

        if box is not None:
            dfloc = self.subset_sites(
                latmin=box[0], lonmin=box[1], latmax=box[2], lonmax=box[3]
            )
        elif country is not None:
            dfloc = dfloc.loc[dfloc.ctry == country, :]
        elif state is not None:
            dfloc = dfloc.loc[dfloc.state == state, :]
        elif site is not None:
            dfloc = dfloc.loc[dfloc.station_id == site, :]

        urls = self.build_urls(sites=dfloc)
        if urls.empty:
            raise ValueError("No data URLs found")

        df = self.aggregrate_files(urls, n_procs=n_procs)

        # Use exclusive upper bound to match unit test expectations (e.g. 24 hours for 1 day range)
        df = df.loc[(df.time >= self.dates.min()) & (df.time < self.dates.max())]
        df = df.replace(-999.9, np.nan)

        if resample and not df.empty:
            df = (
                df.set_index("time")
                .groupby("siteid")
                .resample(window)
                .mean()
                .reset_index()
            )

        df = pd.merge(
            df, dfloc, how="left", left_on="siteid", right_on="station_id"
        ).rename(columns={"ctry": "country"})
        return df.drop(["station_id"], axis=1)
