"""ISH Reader"""

import gzip
import io
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader
from .drivers import FileUtility

VAR_INFO = [
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
DTYPES = [(name, dtype) for name, dtype, _ in VAR_INFO]
WIDTHS = [width for _, _, width in VAR_INFO]


def read_ish_file(
    filename: str,
    *,
    dates: Optional[pd.DatetimeIndex] = None,
    request_timeout: int = 10,
    request_retries: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a single ISH (Integrated Surface Hourly) file.

    Parameters
    ----------
    filename : str
        File path or URL.
    dates : pd.DatetimeIndex, optional
        Dates to filter the data, by default None.
    request_timeout : int, optional
        Timeout for HTTP requests in seconds, by default 10.
    request_retries : int, optional
        Number of retries for HTTP requests, by default 4.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        The loaded and cleaned data.
    """
    if not request_retries >= 0:
        raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

    if isinstance(filename, str) and filename.startswith("http"):
        filename = filename.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
        filename = filename.replace("/pub/pub/", "/pub/")

        import requests

        tries = 0
        while tries - 1 < request_retries:
            try:
                r = requests.get(filename, timeout=request_timeout, stream=True)
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                tries += 1
                if tries - 1 == request_retries:
                    raise RuntimeError(
                        f"Failed to connect to server for URL {filename}. "
                        f"timeout={request_timeout}, retries={request_retries}."
                    ) from e
            else:
                break

        with gzip.open(io.BytesIO(r.content), "rb") as f:
            frame_as_array = np.genfromtxt(f, delimiter=WIDTHS, dtype=DTYPES)
    else:
        fs = FileUtility.get_fs(filename)
        compression = "gzip" if filename.endswith(".gz") else None
        with fs.open(filename, "rb", compression=compression) as f:
            frame_as_array = np.genfromtxt(f, delimiter=WIDTHS, dtype=DTYPES)

    df = pd.DataFrame.from_records(np.atleast_1d(frame_as_array))
    df = _clean_ish(df)
    df.drop(["latitude", "longitude"], axis=1, inplace=True, errors="ignore")

    if dates is not None and not df.empty:
        index = (df.time >= dates.min()) & (df.time <= dates.max())
        df = df.loc[index, :]

    df = _decode_ish_bytes(df)

    # Ensure all non-numeric columns are object for dask consistency
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype) and col != "time":
            df[col] = df[col].astype(object)

    return df


def _clean_ish(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal cleaning logic for ISH data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ISH data.

    Returns
    -------
    pd.DataFrame
        Cleaned ISH data.
    """
    if df.empty:
        for name, _, _ in VAR_INFO:
            if name not in df.columns:
                df[name] = pd.Series(dtype=object)
        df["time"] = pd.Series(dtype="datetime64[ns]")
        return df

    # Vectorized time construction
    df["time"] = pd.to_datetime(
        df["date"].astype(str).str.zfill(8) + df["htime"].astype(str).str.zfill(4),
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    df.drop(["date", "htime"], axis=1, inplace=True)

    # Clean columns
    df = _clean_column(df, "wdir", missing=999)
    df = _clean_column(df, "ws", multiplier=10)
    df = _clean_column(df, "ceiling", missing=99999)
    # vsb appears twice in original code, likely to handle different missing values
    df = _clean_column(df, "vsb", missing=999999)
    df = _clean_column(df, "vsb", missing=99999)
    df = _clean_column(df, _col="t", multiplier=10, missing=9999)
    df = _clean_column(df, _col="dpt", multiplier=10, missing=9999)
    df = _clean_column(df, _col="p", multiplier=10, missing=99999)

    return df


def _clean_column(
    df: pd.DataFrame, _col: str, missing: float = 9999, multiplier: float = 1
) -> pd.DataFrame:
    """
    Helper to clean numeric columns in ISH.

    Parameters
    ----------
    df : pd.DataFrame
        ISH data.
    _col : str
        Column name to clean.
    missing : float, optional
        Value representing missing data, by default 9999.
    multiplier : float, optional
        Scaling factor, by default 1.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column.
    """
    if _col in df.columns:
        series = df[_col].astype(float)
        series = series.where(series != missing, np.nan)
        df[_col] = series / multiplier
    return df


def _decode_ish_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode byte columns in ISH data.

    Parameters
    ----------
    df : pd.DataFrame
        ISH data.

    Returns
    -------
    pd.DataFrame
        Dataframe with decoded byte strings.
    """
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


@register_reader("ish")
class ISHReader(PointReader):
    """
    Reader for ISH (Integrated Surface Hourly) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[pd.DatetimeIndex, List[datetime], datetime, str]] = None,
        box: Optional[List[float]] = None,
        country: Optional[str] = None,
        state: Optional[str] = None,
        site: Optional[str] = None,
        resample: bool = True,
        window: str = "h",
        download: bool = False,
        n_procs: int = 1,
        request_timeout: int = 10,
        request_retries: int = 4,
        verbose: bool = False,
        source: str = "aws",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) data following the Aero Protocol.

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
            Whether to resample data to a regular window, by default True.
        window : str, optional
            Resampling window (e.g., 'h'), by default 'h'.
        download : bool, optional
            Whether to download files (if source is ncdc), by default False.
        n_procs : int, optional
            Number of processors for dask compute (if not lazy), by default 1.
        request_timeout : int, optional
            Timeout for HTTP requests in seconds, by default 10.
        request_retries : int, optional
            Number of retries for HTTP requests, by default 4.
        verbose : bool, optional
            Whether to print verbose output, by default False.
        source : str, optional
            Data source: 'ncdc' or 'aws', by default 'ncdc'.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ISH data.
        """
        if sum([box is not None, country is not None, state is not None, site is not None]) > 1:
            raise ValueError("Only one of `box`, `country`, `state`, or `site` can be used")
        if not request_retries >= 0:
            raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

        ish = ISH()
        ish.source = source

        if files is None and dates is not None:
            dates = pd.to_datetime(dates)
            if isinstance(dates, pd.Timestamp):
                dates = pd.DatetimeIndex([dates])

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

            if download:
                files = ish.get_url_file_objs(urls.name)
            else:
                files = urls.name.tolist()

        if not files:
            raise ValueError("Must provide either 'files' or 'dates'.")

        # Use base class to open via PandasDriver
        df = super().open_dataset(
            files,
            read_method=read_ish_file,
            as_xarray=False,
            lazy=lazy,
            dates=dates,
            request_timeout=request_timeout,
            request_retries=request_retries,
            **kwargs,
        )

        # Merge with metadata
        if ish.history is None:
            ish.read_ish_history(dates=dates)
        dfloc = ish.history.copy()

        if lazy:
            import dask.dataframe as dd

            df = df.assign(station_id=df.station_id.astype(object))
            dfloc_dask = dd.from_pandas(dfloc, npartitions=1).assign(
                station_id=lambda x: x.station_id.astype(object)
            )
            df = df.merge(dfloc_dask, how="left", on="station_id")
        else:
            df["station_id"] = df["station_id"].astype(object)
            df = df.merge(dfloc, how="left", on="station_id")

        df = df.rename(columns={"station_id": "siteid", "ctry": "country"})

        df = self.harmonize(df)

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Preserve metadata in coordinates
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
                # Ensure monotonic for resampling
                if "time" in ds.coords:
                    ds = ds.sortby("time")
                # Backend-agnostic resampling in xarray
                try:
                    ds = ds.resample(time=window).mean(numeric_only=True)
                except TypeError:
                    ds = ds.resample(time=window).mean()

            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH data."
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
                    # Re-join metadata for pandas eager path.
                    meta_cols = dfloc.columns.tolist()
                    if "ctry" in meta_cols:
                        meta_cols.remove("ctry")
                        meta_cols.append("country")
                    cols_to_drop = [c for c in df.columns if c in meta_cols and c != "siteid"]
                    df = df.drop(columns=cols_to_drop, errors="ignore")

                    df = df.merge(
                        dfloc.rename(columns={"ctry": "country"}),
                        how="left",
                        left_on="siteid",
                        right_on="station_id",
                    ).drop(columns=["station_id"], errors="ignore")
            else:
                import warnings

                warnings.warn(
                    "ISHReader: Resampling is currently not supported for lazy DataFrames. "
                    "Convert to xarray (as_xarray=True) for lazy resampling."
                )

        return df


class ISH:
    """Helper class for ISH data retrieval."""

    _VAR_INFO = VAR_INFO
    DTYPES = DTYPES
    WIDTHS = WIDTHS

    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.df = None
        self.dates = None
        self.verbose = False
        self.source = "aws"

    def read_ish_history(self, dates: Optional[pd.DatetimeIndex] = None):
        """
        Read the ISH history file.
        """
        if dates is None:
            dates = self.dates
        fname = self.history_file

        if self.source == "aws":
            # For AWS, we prefer the S3 copy of the history file.
            fname = "s3://noaa-isd-pds/isd-history.csv"

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
        """Subset sites."""
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (self.history.longitude <= lonmax)
        dfloc = self.history.loc[latindex & lonindex, :]
        return dfloc

    def build_urls(
        self,
        dates: Optional[pd.DatetimeIndex] = None,
        sites: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Construct ISH URLs."""
        if dates is None:
            dates = self.dates
        if sites is None:
            sites = self.history

        if dates is None or sites is None:
            return pd.DataFrame(columns=["name"])

        unique_years = pd.to_datetime(dates.year.unique(), format="%Y")
        furls = []

        if self.source == "aws":
            # Directly construct S3 URLs for the requested sites and years.
            # AWS S3 structure: s3://noaa-isd-pds/data/<year>/<usaf>-<wban>-<year>.gz
            base_s3_url = "s3://noaa-isd-pds/data"
            for syear in unique_years.strftime("%Y"):
                # vectorized construction for each year
                year_furls = (
                    base_s3_url
                    + "/"
                    + syear
                    + "/"
                    + sites.usaf.astype(str).str.zfill(6)
                    + "-"
                    + sites.wban.astype(str).str.zfill(5)
                    + "-"
                    + syear
                    + ".gz"
                )
                furls.extend(year_furls.tolist())
            return pd.Series(furls, name="name").drop_duplicates().to_frame()
        else:
            url = "https://www.ncei.noaa.gov/pub/data/noaa"
            all_urls_list = []
            for syear in unique_years.strftime("%Y"):
                try:
                    year_url_df = pd.read_html(f"{url}/{syear}/")[0]
                    if "Name" in year_url_df.columns:
                        names = year_url_df["Name"].iloc[2:-1].to_frame(name="name")
                        all_urls_list.append(f"{url}/{syear}/" + names)
                except Exception:
                    pass
            if all_urls_list:
                all_urls = pd.concat(all_urls_list, ignore_index=True)
            else:
                all_urls = pd.DataFrame(columns=["name"])

            for syear in unique_years.strftime("%Y"):
                year_fnames = (
                    sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
                )
                for fname in year_fnames:
                    furls.append(f"{url}/{syear}/{fname}")

            url_series = pd.Series(furls, name="name")
            final_urls = pd.merge(url_series.to_frame(name="name"), all_urls, how="inner")
            return final_urls

    def get_url_file_objs(self, fname: List[str]) -> List[str]:
        """Download ISH files."""
        import gzip
        import shutil

        objs = []
        for iii in fname:
            try:
                temp = iii.split("/")[-1]
                out_name = "isd." + temp.replace(".gz", "")

                if iii.startswith("s3://"):
                    fs = FileUtility.get_fs(iii)
                    with fs.open(iii, "rb") as f_in:
                        with open(out_name, "wb") as f_out:
                            if iii.endswith(".gz"):
                                with gzip.GzipFile(fileobj=f_in) as gz:
                                    shutil.copyfileobj(gz, f_out)
                            else:
                                shutil.copyfileobj(f_in, f_out)
                    objs.append(out_name)
                else:
                    import requests

                    r2 = requests.get(iii, stream=True)
                    if r2.status_code != 404:
                        objs.append(out_name)
                        with open(out_name, "wb") as fid:
                            gzip_file = gzip.GzipFile(fileobj=r2.raw)
                            shutil.copyfileobj(gzip_file, fid)
            except Exception:
                pass
        return objs

    def read_data_frame(
        self, url_or_file: str, *, request_timeout: int = 10, request_retries: int = 4
    ) -> pd.DataFrame:
        """Redirect to read_ish_file."""
        return read_ish_file(
            url_or_file,
            dates=self.dates,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )

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
        source="aws",
        lazy=False,
    ):
        """Backward-compatible method."""
        self.dates = pd.to_datetime(dates)
        if isinstance(self.dates, pd.Timestamp):
            self.dates = pd.DatetimeIndex([self.dates])
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
        import dask
        import dask.dataframe as dd

        def func(u):
            return read_ish_file(
                u,
                dates=self.dates,
                request_timeout=request_timeout,
                request_retries=request_retries,
            )

        if download:
            objs = self.get_url_file_objs(urls.name)
            dfs = [dask.delayed(func)(f) for f in objs]
        else:
            dfs = [dask.delayed(func)(f) for f in urls.name]
        self.df = dd.from_delayed(dfs)
        if not lazy:
            self.df = self.df.compute(num_workers=n_procs)
        if resample and not lazy and not self.df.empty:
            self.df.index = self.df.time
            self.df = (
                self.df.groupby("station_id").resample(window).mean(numeric_only=True).reset_index()
            )
        self.df = self.df.merge(dfloc, on="station_id", how="left")
        self.df = self.df.rename(columns={"station_id": "siteid", "ctry": "country"})
        return self.df
