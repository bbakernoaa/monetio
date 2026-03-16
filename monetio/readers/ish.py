"""Reader for NOAA Integrated Surface Hourly (ISH; also known as ISD, Integrated Surface Data).

https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import normalize_pandas_freq
from .core import PointReader
from .drivers import PandasDriver


class ISHReader(PointReader):
    """Reader for NOAA Integrated Surface Hourly (ISH) data."""

    _VAR_INFO = [
        # name, dtype, width
        ("varlength", "i2", 4),
        ("station_id", "S11", 11),
        ("date", "i4", 8),
        ("htime", "i2", 4),
        ("source_flag", "S1", 1),
        ("latitude_orig", "float", 6),
        ("longitude_orig", "float", 7),
        ("code", "S5", 5),
        ("elevation_orig", "i2", 5),
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
        self.history_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.base_url = "https://www.ncei.noaa.gov/pub/data/noaa"
        self.s3_bucket = "s3://noaa-isd-pds"
        self._history = None

    def read_history(self) -> pd.DataFrame:
        """Read the ISH history file.

        Returns
        -------
        pd.DataFrame
            The history DataFrame.
        """
        if self._history is None:
            # Try S3 first for speed if fsspec/s3fs available, else HTTPS
            try:
                import fsspec

                fs = fsspec.filesystem("s3", anon=True)
                with fs.open(f"{self.s3_bucket}/isd-history.csv") as f:
                    df = pd.read_csv(f, dtype={"USAF": str, "WBAN": str})
            except Exception:
                df = pd.read_csv(self.history_url, dtype={"USAF": str, "WBAN": str})

            df.columns = [i.lower() for i in df.columns]
            # Fix date parsing: ensure format is specified or handled robustly
            df["begin"] = pd.to_datetime(df["begin"], format="%Y%m%d", errors="coerce").astype(
                "datetime64[ns]"
            )
            df["end"] = pd.to_datetime(df["end"], format="%Y%m%d", errors="coerce").astype(
                "datetime64[ns]"
            )

            df.rename(
                columns={"lat": "latitude", "lon": "longitude", "elev(m)": "elevation"},
                inplace=True,
            )
            df = df.dropna(subset=["latitude", "longitude"]).copy()

            df["usaf"] = df.usaf.astype(str).str.zfill(6)
            df["wban"] = df.wban.astype(str).str.zfill(5)
            df["station_id"] = df.usaf + df.wban
            self._history = df

        return self._history

    def build_urls(
        self,
        dates: pd.DatetimeIndex,
        site: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        box: Optional[List[float]] = None,
    ) -> List[str]:
        """Build URLs for the requested data.

        Parameters
        ----------
        dates : pd.DatetimeIndex
            Dates to retrieve.
        site : Optional[str], optional
            Station ID.
        state : Optional[str], optional
            State code.
        country : Optional[str], optional
            Country code.
        box : Optional[List[float]], optional
            Bounding box [latmin, lonmin, latmax, lonmax].

        Returns
        -------
        List[str]
            List of URLs.
        """
        history = self.read_history()
        subset = history.copy()

        if site:
            subset = subset.loc[subset.station_id == site]
        elif state:
            subset = subset.loc[subset.state == state]
        elif country:
            subset = subset.loc[subset.ctry == country]
        elif box:
            latmin, lonmin, latmax, lonmax = box
            subset = subset.loc[
                (subset.latitude >= latmin)
                & (subset.latitude <= latmax)
                & (subset.longitude >= lonmin)
                & (subset.longitude <= lonmax)
            ]

        # Filter by date range
        subset = subset.loc[(subset.end >= dates.min()) & (subset.begin <= dates.max())]

        if subset.empty:
            return []

        years = dates.year.unique()
        urls = []
        for _, row in subset.iterrows():
            for year in years:
                if (row.begin.year <= year) and (row.end.year >= year):
                    # Prioritize S3
                    urls.append(f"{self.s3_bucket}/{year}/{row.usaf}-{row.wban}-{year}.gz")

        return urls

    def read_file(self, url: str, **kwargs) -> pd.DataFrame:
        """Read a single ISH file.

        Parameters
        ----------
        url : str
            URL or path to the file.
        **kwargs
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            The data.
        """
        import gzip
        import io

        import fsspec

        # Convert s3:// to https:// if s3 fails or if we want to be safe
        if url.startswith("s3://"):
            http_url = url.replace("s3://noaa-isd-pds", "https://www.ncei.noaa.gov/pub/data/noaa")
        else:
            http_url = url

        request_timeout = kwargs.get("request_timeout", 10)
        request_retries = kwargs.get("request_retries", 4)

        try:
            fs = fsspec.open(url, mode="rb", anon=True)
            with fs as f:
                with gzip.open(f, "rb") as gz:
                    data = np.genfromtxt(gz, delimiter=self.WIDTHS, dtype=self.DTYPES)
        except Exception:
            # Fallback to requests
            import requests

            tries = 0
            while tries <= request_retries:
                try:
                    r = requests.get(http_url, timeout=request_timeout)
                    r.raise_for_status()
                    break
                except requests.exceptions.RequestException as re:
                    tries += 1
                    if tries > request_retries:
                        raise RuntimeError(
                            f"Failed to connect to server for URL {http_url}. "
                            f"timeout={request_timeout}, retries={request_retries}."
                        ) from re

            with gzip.open(io.BytesIO(r.content), "rb") as f:
                data = np.genfromtxt(f, delimiter=self.WIDTHS, dtype=self.DTYPES)

        df = pd.DataFrame.from_records(np.atleast_1d(data))
        df = self._clean(df)

        # Add siteid if not present
        if "station_id" in df.columns:
            if isinstance(df["station_id"].iloc[0], bytes):
                df["siteid"] = df["station_id"].str.decode("utf-8")
            else:
                df["siteid"] = df["station_id"].astype(str)

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the ISH DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The raw DataFrame.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame.
        """
        # Vectorized time parsing
        df["time"] = pd.to_datetime(
            df["date"].astype(str) + df["htime"].astype(str).str.zfill(4),
            format="%Y%m%d%H%M",
            errors="coerce",
        )

        # Clean columns
        cols_to_clean = {
            "wdir": {"missing": 999, "multiplier": 1},
            "ws": {"missing": 9999, "multiplier": 10},
            "ceiling": {"missing": 99999, "multiplier": 1},
            "vsb": {"missing": 999999, "multiplier": 1},
            "t": {"missing": 9999, "multiplier": 10},
            "dpt": {"missing": 9999, "multiplier": 10},
            "p": {"missing": 99999, "multiplier": 10},
        }

        for col, params in cols_to_clean.items():
            if col in df.columns:
                val = df[col].astype(float)
                val[val == params["missing"]] = np.nan
                df[col] = val / params["multiplier"]

        # Decode byte strings
        for col in df.columns:
            if df[col].dtype == object and len(df) > 0:
                if isinstance(df[col].iloc[0], bytes):
                    df[col] = df[col].str.decode("utf-8")

        return df.drop(
            columns=["date", "htime", "latitude_orig", "longitude_orig", "elevation_orig"],
            errors="ignore",
        )

    def open_dataset(
        self,
        dates: Union[pd.DatetimeIndex, str, List[str]],
        site: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        box: Optional[List[float]] = None,
        resample: bool = True,
        window: str = "h",
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Open ISH data as an xarray Dataset.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, str, List[str]]
            Dates to retrieve.
        site : Optional[str], optional
            Station ID.
        state : Optional[str], optional
            State code.
        country : Optional[str], optional
            Country code.
        box : Optional[List[float]], optional
            Bounding box [latmin, lonmin, latmax, lonmax].
        resample : bool, optional
            Whether to resample the data.
        window : str, optional
            Resampling window.
        lazy : bool, optional
            Whether to load the data lazily.
        **kwargs
            Additional arguments.

        Returns
        -------
        xr.Dataset
            The data.
        """
        if sum([box is not None, country is not None, state is not None, site is not None]) > 1:
            raise ValueError("Only one of `box`, `country`, `state`, or `site` can be used")

        request_retries = kwargs.get("request_retries", 4)
        if request_retries < 0:
            raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

        if isinstance(dates, (str, list)):
            dates = pd.to_datetime(dates)
            if isinstance(dates, pd.Timestamp):
                dates = pd.DatetimeIndex([dates])

        urls = self.build_urls(dates, site=site, state=state, country=country, box=box)
        if not urls:
            return xr.Dataset()

        df = PandasDriver.open_dataset(urls, self.read_file, lazy=lazy, **kwargs)

        # Post-process
        def _post_process(df_in):
            if df_in.empty:
                return df_in

            # Filter by date
            df_in = df_in.loc[(df_in.time >= dates.min()) & (df_in.time <= dates.max())]

            if resample:
                window_norm = normalize_pandas_freq(window)
                df_in = (
                    df_in.groupby("siteid")
                    .resample(window_norm, on="time")
                    .mean(numeric_only=True)
                    .reset_index()
                )

            # Add metadata from history
            history = self.read_history()
            df_in = df_in.merge(
                history[["station_id", "latitude", "longitude", "elevation", "ctry", "state"]],
                left_on="siteid",
                right_on="station_id",
                how="left",
            )
            df_in.rename(columns={"ctry": "country"}, inplace=True)
            return df_in

        if lazy:
            df = df.map_partitions(_post_process)
        else:
            df = _post_process(df)

        ds = self.to_xarray(df)
        self.update_history(ds, "Loaded ISH data.")
        return ds


class ISHLiteReader(ISHReader):
    """Reader for NOAA Integrated Surface Hourly (ISH) Lite data."""

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
        self.s3_bucket = "s3://noaa-isd-pds/isd-lite"

    def read_file(self, url: str, **kwargs) -> pd.DataFrame:
        """Read a single ISH Lite file.

        Parameters
        ----------
        url : str
            URL or path to the file.
        **kwargs
            Additional arguments.

        Returns
        -------
        pd.DataFrame
            The data.
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

        if url.startswith("s3://"):
            http_url = url.replace(
                "s3://noaa-isd-pds/isd-lite", "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
            )
        else:
            http_url = url

        try:
            # Handle .gz automatically if using fsspec with compression='infer' or 'gzip'
            df = pd.read_csv(
                url, sep=r"\s+", header=None, names=columns, storage_options={"anon": True}
            )
        except Exception:
            # Fallback
            df = pd.read_csv(http_url, sep=r"\s+", header=None, names=columns, compression="gzip")

        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]].assign(minute=0, second=0))

        # Site ID from filename
        import os

        filename = os.path.basename(url).split("-")
        siteid = filename[0] + filename[1]
        df["siteid"] = siteid

        # Scaling
        df["temp"] /= 10.0
        df["dew_pt_temp"] /= 10.0
        df["press"] /= 10.0
        df["ws"] /= 10.0
        df["precip_1hr"] /= 10.0
        df["precip_6hr"] /= 10.0

        df = df.replace(-999.9, np.nan).replace(-9999, np.nan)

        return df.drop(columns=["year", "month", "day", "hour"])

    def open_dataset(
        self,
        dates: Union[pd.DatetimeIndex, str, List[str]],
        site: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        box: Optional[List[float]] = None,
        resample: bool = False,
        window: str = "h",
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """Open ISH Lite data as an xarray Dataset."""
        # ISH Lite uses different URL structure sometimes, but build_urls from ISHReader
        # is adaptable if we change self.s3_bucket.

        if sum([box is not None, country is not None, state is not None, site is not None]) > 1:
            raise ValueError("Only one of `box`, `country`, `state`, or `site` can be used")

        request_retries = kwargs.get("request_retries", 4)
        if request_retries < 0:
            raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

        if isinstance(dates, (str, list)):
            dates = pd.to_datetime(dates)
            if isinstance(dates, pd.Timestamp):
                dates = pd.DatetimeIndex([dates])

        urls = self.build_urls(dates, site=site, state=state, country=country, box=box)
        if not urls:
            return xr.Dataset()

        df = PandasDriver.open_dataset(urls, self.read_file, lazy=lazy, **kwargs)

        def _post_process(df_in):
            if df_in.empty:
                return df_in

            df_in = df_in.loc[(df_in.time >= dates.min()) & (df_in.time <= dates.max())]

            if resample:
                window_norm = normalize_pandas_freq(window)
                df_in = (
                    df_in.groupby("siteid")
                    .resample(window_norm, on="time")
                    .mean(numeric_only=True)
                    .reset_index()
                )

            history = self.read_history()
            df_in = df_in.merge(
                history[["station_id", "latitude", "longitude", "elevation", "ctry", "state"]],
                left_on="siteid",
                right_on="station_id",
                how="left",
            )
            df_in.rename(columns={"ctry": "country"}, inplace=True)
            return df_in

        if lazy:
            df = df.map_partitions(_post_process)
        else:
            df = _post_process(df)

        ds = self.to_xarray(df)
        self.update_history(ds, "Loaded ISH Lite data.")
        return ds
