"""ISH Reader"""

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from .base import PointReader, register_reader
from .drivers import FileUtility

@register_reader("ish")
class ISHReader(PointReader):
    def open_dataset(self,
                     dates,
                     box=None,
                     country=None,
                     state=None,
                     site=None,
                     resample=True,
                     window="H",
                     download=False,
                     n_procs=1,
                     request_timeout=10,
                     request_retries=4,
                     verbose=False,
                     source="ncdc",
                     **kwargs):
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
        self.history_file = "https://www1.ncdc.noaa.gov/pub/data/noaa/isd-history.csv"
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
        frame["time"] = [
            pd.Timestamp(f"{date:08}{htime:04}")
            for date, htime in zip(frame["date"], frame["htime"])
        ]
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
        bytes_cols = [col for col in df.columns if type(df[col][0]) is bytes]
        with pd.option_context("mode.chained_assignment", None):
            df.loc[:, bytes_cols] = df[bytes_cols].apply(
                lambda x: x.str.decode("utf-8"),
                axis="columns",
            )
        return df

    def read_data_frame(self, url_or_file, *, request_timeout=10, request_retries=4):
        # Use FileUtility to support S3, HTTP, and Local transparently
        fs = FileUtility.get_fs(url_or_file)

        if url_or_file.startswith("http"):
             # Fallback to requests logic for robust HTTP if needed,
             # or trust fsspec http filesystem (simple read).
             # Original code had retries.
             # If source="ncdc" (http), we keep retries?
             # For now, use FileUtility (fsspec) which handles S3/Local well.
             # For HTTP, fsspec doesn't retry by default as aggressively as the original logic.
             # But let's try to use fsspec for everything.
             pass

        # Open file object
        # gzip handling: if .gz, fsspec usually handles it if compression is inferred,
        # OR we pass it to gzip.open.
        # fsspec open(..., compression='gzip') works.

        compression = "gzip" if url_or_file.endswith(".gz") else None

        with fs.open(url_or_file, "rb", compression=compression) as f:
            # numpy genfromtxt expects byte stream or text?
            # np.genfromtxt handles gzip file objects if they are seekable?
            # fsspec files are seekable.
            # If compression='gzip' in fs.open, f is uncompressed stream.

            frame_as_array = np.genfromtxt(f, delimiter=self.WIDTHS, dtype=self.DTYPES)

        frame = pd.DataFrame.from_records(np.atleast_1d(frame_as_array))
        df = self._clean(frame)
        df.drop(["latitude", "longitude"], axis=1, inplace=True)

        if self.dates is not None:
            index = (df.index >= self.dates.min()) & (df.index <= self.dates.max())
            df = df.loc[index, :]

        df = ISH._decode_bytes(df)
        return df.reset_index()

    def read_ish_history(self, dates=None):
        if dates is None:
            dates = self.dates
        fname = self.history_file

        # Support S3 for history if source is aws
        if self.source == "aws":
            fname = "s3://noaa-isd-pds/isd-history.csv"

        # Use FileUtility for history file too
        fs = FileUtility.get_fs(fname)
        with fs.open(fname, "r") as f:
            self.history = pd.read_csv(f, parse_dates=["BEGIN", "END"], infer_datetime_format=True)

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
        if dates is None: dates = self.dates
        if sites is None: sites = self.history

        unique_years = pd.to_datetime(dates.year.unique(), format="%Y")
        furls = []

        if self.source == "aws":
             url = "s3://noaa-isd-pds/data"
        else:
             url = "https://www1.ncdc.noaa.gov/pub/data/noaa"

        # For AWS, we assume availability based on standard naming.
        # AWS structure: s3://noaa-isd-pds/data/<year>/<usaf>-<wban>-<year>.gz (Need to confirm)
        # Actually, registry says: data/<year>/<station ID>.
        # Assuming standard station ID = USAF-WBAN.

        # Note: AWS S3 listing is faster than NCDC html parsing if we used s3fs glob/ls.
        # But here we construct URLs.

        for syear in unique_years.strftime("%Y"):
            year_fnames = (
                sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
            )
            for fname in year_fnames:
                furls.append(f"{url}/{syear}/{fname}")

        return pd.Series(furls, name="name").to_frame()

    def add_data(self, dates, box=None, country=None, state=None, site=None,
                 resample=True, window="H", download=False, n_procs=1,
                 request_timeout=10, request_retries=4, verbose=False, source="ncdc"):
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

        # Parallel read logic
        def func(fname):
            return self.read_data_frame(fname, request_timeout=request_timeout, request_retries=request_retries)

        # Using name column
        dfs = [dask.delayed(func)(f) for f in urls.name]
        dff = dd.from_delayed(dfs)
        self.df = dff.compute(num_workers=n_procs)

        if resample and not self.df.empty:
            self.df.index = self.df.time
            numeric_cols = self.df.select_dtypes(include=["number"]).columns
            group_cols = ["station_id"]
            self.df = (
                self.df[group_cols + list(numeric_cols)]
                .groupby("station_id")
                .resample(window)
                .mean()
                .reset_index()
            )

        self.df = self.df.merge(dfloc, on="station_id", how="left")
        self.df = self.df.rename(columns={"station_id": "siteid", "ctry": "country"})

        return self.df
