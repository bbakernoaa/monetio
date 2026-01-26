"""ISH Reader"""

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("ish")
class ISHReader(PointReader):
    def open_dataset(
        self,
        dates,
        box=None,
        country=None,
        state=None,
        site=None,
        resample=True,
        window="h",
        download=False,
        n_procs=1,
        request_timeout=10,
        request_retries=4,
        verbose=False,
        source="ncdc",
        **kwargs,
    ):
        """
        Reads ISH data.

        source: "ncdc" (default) or "aws".
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
            download=download,
            n_procs=n_procs,
            request_timeout=request_timeout,
            request_retries=request_retries,
            verbose=verbose,
            source=source,
        )


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/ish.py
# -----------------------------------------------------------------------------


class ISH:
    _VAR_INFO = [
        ("varlength", "i2", 4),
        ("station_id", "S11", 11),
        ("date", "i4", 8),
        ("htime", "i2", 4),
        ("source_flag", "S1", 1),
        ("latitude", "float", 6),
        ("longitude", "float", 7),
        ("code", "S5", 5),
        ("elev", "i2", 5),
        ("call_letters", "S5", 5),
        ("qc_process", "S4", 4),
        ("wdir", "i2", 3),
        ("wdir_quality", "S1", 1),
        ("wdir_type", "S1", 1),
        ("ws", "i2", 4),
        ("ws_quality", "S1", 1),
        ("ceiling", "i4", 5),
        ("ceiling_quality", "S1", 1),
        ("ceiling_code", "S1", 1),
        ("ceiling_cavok", "S1", 1),
        ("vsb", "i4", 6),
        ("vsb_quality", "S1", 1),
        ("vsb_variability", "S1", 1),
        ("vsb_variability_quality", "S1", 1),
        ("t", "i2", 5),
        ("t_quality", "S1", 1),
        ("dpt", "i2", 5),
        ("dpt_quality", "S1", 1),
        ("p", "i4", 5),
        ("p_quality", "S1", 1),
    ]
    DTYPES = [(name, dtype) for name, dtype, _ in _VAR_INFO]
    WIDTHS = [width for _, _, width in _VAR_INFO]

    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.df = None
        self.dates = None
        self.verbose = False
        self.source = "ncdc"

    @staticmethod
    def _clean_column(series, missing=9999, multiplier=1):
        series = series.apply(float)
        series[series == missing] = np.nan
        return series // multiplier

    @staticmethod
    def _clean_column_by_name(frame, name, *args, **kwargs):
        frame[name] = ISH._clean_column(frame[name], *args, **kwargs)
        return frame

    @staticmethod
    def _clean(frame):
        if frame.empty:
            for name, _, _ in ISH._VAR_INFO:
                if name not in frame.columns:
                    frame[name] = pd.Series(dtype=object)
            frame["time"] = pd.Series(dtype="datetime64[ns]")
            return frame

        frame["time"] = [pd.Timestamp(f"{date:08}{htime:04}") for date, htime in zip(frame["date"], frame["htime"])]
        frame.drop(["date", "htime"], axis=1, inplace=True)
        frame.set_index("time", drop=True, inplace=True)
        frame = ISH._clean_column_by_name(frame, "wdir", missing=999)
        frame = ISH._clean_column_by_name(frame, "ws", multiplier=10)
        frame = ISH._clean_column_by_name(frame, "ceiling", missing=99999)
        frame = ISH._clean_column_by_name(frame, "vsb", missing=999999)
        frame = ISH._clean_column_by_name(frame, "vsb", missing=99999)
        frame = ISH._clean_column_by_name(frame, "t", multiplier=10, missing=9999)
        frame = ISH._clean_column_by_name(frame, "dpt", multiplier=10, missing=9999)
        frame = ISH._clean_column_by_name(frame, "p", multiplier=10, missing=99999)
        return frame

    @staticmethod
    def _decode_bytes(df):
        if df.empty:
            return df
        bytes_cols = []
        for col in df.columns:
            if df[col].dtype == object:
                non_null = df[col].dropna()
                if not non_null.empty and isinstance(non_null.iloc[0], (bytes, np.bytes_)):
                    bytes_cols.append(col)

        if bytes_cols:
            with pd.option_context("mode.chained_assignment", None):
                for col in bytes_cols:
                    df[col] = df[col].str.decode("utf-8")
        return df

    def read_data_frame(self, url_or_file, *, request_timeout=10, request_retries=4):
        if not request_retries >= 0:
            raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

        if isinstance(url_or_file, str) and url_or_file.startswith("http"):
            url_or_file = url_or_file.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
            url_or_file = url_or_file.replace("/pub/pub/", "/pub/")

            import gzip
            import io

            import requests

            tries = 0
            while tries - 1 < request_retries:
                try:
                    r = requests.get(url_or_file, timeout=request_timeout, stream=True)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    tries += 1
                    if tries - 1 == request_retries:
                        raise RuntimeError(
                            f"Failed to connect to server for URL {url_or_file}. "
                            f"timeout={request_timeout}, retries={request_retries}."
                        ) from e
                else:
                    break

            with gzip.open(io.BytesIO(r.content), "rb") as f:
                frame_as_array = np.genfromtxt(f, delimiter=self.WIDTHS, dtype=self.DTYPES)
        else:
            fs = FileUtility.get_fs(url_or_file)
            compression = "gzip" if url_or_file.endswith(".gz") else None
            with fs.open(url_or_file, "rb", compression=compression) as f:
                frame_as_array = np.genfromtxt(f, delimiter=self.WIDTHS, dtype=self.DTYPES)

        frame = pd.DataFrame.from_records(np.atleast_1d(frame_as_array))
        df = self._clean(frame)
        df.drop(["latitude", "longitude"], axis=1, inplace=True, errors="ignore")

        if self.dates is not None and not df.empty:
            index = (df.index >= self.dates.min()) & (df.index <= self.dates.max())
            df = df.loc[index, :]

        df = ISH._decode_bytes(df)
        df = df.reset_index()

        # Ensure all non-numeric columns are object for dask consistency
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col].dtype) and col != "time":
                df[col] = df[col].astype(object)

        return df

    def read_ish_history(self, dates=None):
        if dates is None:
            dates = self.dates
        fname = self.history_file

        if self.source == "aws":
            fname = "s3://noaa-isd-pds/isd-history.csv"

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

        if self.source == "aws":
            url = "s3://noaa-isd-pds/data"
            for syear in unique_years.strftime("%Y"):
                year_fnames = sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
                for fname in year_fnames:
                    furls.append(f"{url}/{syear}/{fname}")
            return pd.Series(furls, name="name").to_frame()
        else:
            url = "https://www.ncei.noaa.gov/pub/data/noaa"
            all_urls_list = []
            for syear in unique_years.strftime("%Y"):
                try:
                    year_url_df = pd.read_html(f"{url}/{syear}/")[0]
                    if "Name" in year_url_df.columns:
                        names = year_url_df["Name"].iloc[2:-1].to_frame(name="name")
                        all_urls_list.append(f"{url}/{syear}/" + names)
                except:
                    pass
            if all_urls_list:
                all_urls = pd.concat(all_urls_list, ignore_index=True)
            else:
                all_urls = pd.DataFrame(columns=["name"])

            for syear in unique_years.strftime("%Y"):
                year_fnames = sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
                for fname in year_fnames:
                    furls.append(f"{url}/{syear}/{fname}")

            url_series = pd.Series(furls, name="name")
            final_urls = pd.merge(url_series.to_frame(name="name"), all_urls, how="inner")
            return final_urls

    def get_url_file_objs(self, fname):
        import gzip
        import shutil

        import requests

        objs = []
        for iii in fname:
            try:
                r2 = requests.get(iii, stream=True)
                if r2.status_code != 404:
                    temp = iii.split("/")[-1]
                    out_name = "isd." + temp.replace(".gz", "")
                    objs.append(out_name)
                    with open(out_name, "wb") as fid:
                        gzip_file = gzip.GzipFile(fileobj=r2.raw)
                        shutil.copyfileobj(gzip_file, fid)
            except:
                pass
        return objs

    def add_data(
        self,
        dates,
        box=None,
        country=None,
        state=None,
        site=None,
        resample=True,
        window="h",
        download=False,
        n_procs=1,
        request_timeout=10,
        request_retries=4,
        verbose=False,
        source="ncdc",
    ):
        if sum([box is not None, country is not None, state is not None, site is not None]) > 1:
            raise ValueError("Only one of `box`, `country`, `state`, or `site` can be used")
        if not request_retries >= 0:
            raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

        self.dates = pd.to_datetime(dates)
        self.verbose = verbose
        self.source = source

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

        # Robust meta for dask
        meta = None
        for u in urls.name:
            try:
                sample_df = self.read_data_frame(u, request_timeout=request_timeout, request_retries=request_retries)
                if not sample_df.empty:
                    meta = sample_df.iloc[:0].copy()
                    break
            except:
                continue

        if meta is None:
            try:
                sample_df = self.read_data_frame(
                    urls.name.iloc[0],
                    request_timeout=request_timeout,
                    request_retries=request_retries,
                )
                meta = sample_df.iloc[:0].copy()
            except:
                meta = None

        if download:
            objs = self.get_url_file_objs(urls.name)

            def func(fname):
                return self.read_data_frame(fname, request_timeout=request_timeout, request_retries=request_retries)

            dfs = [dask.delayed(func)(f) for f in objs]
            dff = dd.from_delayed(dfs, meta=meta)
            self.df = dff.compute(num_workers=n_procs)
        else:

            def func(url):
                return self.read_data_frame(url, request_timeout=request_timeout, request_retries=request_retries)

            dfs = [dask.delayed(func)(f) for f in urls.name]
            dff = dd.from_delayed(dfs, meta=meta)
            self.df = dff.compute(num_workers=n_procs)

        if resample and not self.df.empty:
            self.df.index = self.df.time
            numeric_cols = self.df.select_dtypes(include=["number"]).columns
            group_cols = ["station_id"]
            self.df = self.df[group_cols + list(numeric_cols)].groupby("station_id").resample(window).mean().reset_index()

        self.df = self.df.merge(dfloc, on="station_id", how="left")
        self.df = self.df.rename(columns={"station_id": "siteid", "ctry": "country"})
        return self.df
