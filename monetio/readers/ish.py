"""ISH Reader"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("ish")
class ISHReader(PointReader):
    def open_dataset(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
        box: List[float] = None,
        country: str = None,
        state: str = None,
        site: str = None,
        resample: bool = True,
        window: str = "h",
        download: bool = False,
        n_procs: int = 1,
        request_timeout: int = 10,
        request_retries: int = 4,
        verbose: bool = False,
        source: str = "ncdc",
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load ISH (Integrated Surface Hourly) data.

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
            Whether to resample data to a regular window, by default True.
        window : str, optional
            Resampling window (e.g., 'h'), by default 'h'.
        download : bool, optional
            Whether to download files (if source is ncdc), by default False.
        n_procs : int, optional
            Number of processors for dask compute, by default 1.
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

        ish = ISH()
        ish.dates = pd.to_datetime(dates)
        ish.source = source
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

        if download:
            files = ish.get_url_file_objs(urls.name)
        else:
            files = urls.name.tolist()

        from functools import partial

        read_func = partial(
            read_ish_file,
            dates=ish.dates,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )

        # Load data using PandasDriver via super()
        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        if not lazy and hasattr(df, "compute"):
            df = df.compute(num_workers=n_procs)

        # Backend-agnostic Metadata Merge
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        if resample and not as_xarray:
            if is_dask:
                import warnings

                warnings.warn(
                    "ISHReader: Resampling for Dask DataFrames (as_xarray=False) "
                    "is not yet implemented. Returning long-format lazy data."
                )
            elif not df.empty:
                df.index = df.time
                numeric_cols = df.select_dtypes(include=["number"]).columns
                group_cols = ["station_id"]
                df = (
                    df[group_cols + list(numeric_cols)]
                    .groupby("station_id")
                    .resample(window)
                    .mean()
                    .reset_index()
                )

        # Backend-agnostic Metadata Merge
        is_dask = False
        try:
            import dask.dataframe as dd

            if isinstance(df, dd.DataFrame):
                is_dask = True
        except ImportError:
            pass

        if is_dask:
            dfloc_dask = dd.from_pandas(dfloc, npartitions=1)
            # Ensure dtypes match and strip for merge
            df["station_id"] = df["station_id"].astype(str).str.strip()
            dfloc_dask["station_id"] = dfloc_dask["station_id"].astype(str).str.strip()
            df = df.merge(dfloc_dask, on="station_id", how="left")
        else:
            df["station_id"] = df["station_id"].astype(str).str.strip()
            dfloc["station_id"] = dfloc["station_id"].astype(str).str.strip()
            df = df.merge(dfloc, on="station_id", how="left")

        df = df.rename(columns={"station_id": "siteid", "ctry": "country"})
        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df)

            if resample:
                # Xarray resampling is lazy-friendly.
                # If we have a 'time' dimension (2D case), we can resample directly.
                if "time" in ds.dims:
                    ds = ds.resample(time=window).mean()
                elif "node" in ds.dims and "time" in ds.coords:
                    # 1D long format: Resample per node (site) to avoid mixing data.
                    # This remains lazy for both NumPy and Dask backends.
                    def _resample_node(x):
                        return x.swap_dims({"node": "time"}).resample(time=window).mean()

                    # Use groupby().map() for lazy per-site resampling.
                    group_labels = None
                    if "siteid" in dfloc.columns:
                        group_labels = dfloc["siteid"].unique()
                    elif "station_id" in dfloc.columns:
                        group_labels = dfloc["station_id"].unique()

                    if group_labels is not None and len(group_labels) == 1:
                        # Single site: avoid expensive groupby
                        ds = ds.swap_dims({"node": "time"}).resample(time=window).mean()
                    elif "siteid" in ds.coords or "siteid" in ds.data_vars:
                        # Use standard groupby. Note: for dask-backed arrays,
                        # this might trigger compute of siteid if not careful,
                        # or fail if not using flox.
                        # For now, we use standard groupby which is portable.
                        ds = ds.groupby("siteid").map(_resample_node)
                    else:
                        # Fallback if no siteid, but usually ISH has it.
                        ds = ds.swap_dims({"node": "time"}).resample(time=window).mean()

                    if "node" in ds.dims and "time" in ds.dims:
                        # If it became 2D or kept dimensions, we ensure it matches expectation.
                        pass

            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ISH data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/ish.py
# -----------------------------------------------------------------------------


def read_ish_file(
    url_or_file: str,
    *,
    dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
    request_timeout: int = 10,
    request_retries: int = 4,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a single ISH (Integrated Surface Hourly) file.

    Parameters
    ----------
    url_or_file : str
        URL or local file path to the ISH data.
    dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
        Specific dates to keep in the resulting DataFrame.
    request_timeout : int, optional
        Timeout for HTTP requests in seconds, by default 10.
    request_retries : int, optional
        Number of retries for HTTP requests, by default 4.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        The cleaned ISH data.
    """
    if not request_retries >= 0:
        raise ValueError(f"`request_retries` must be >= 0, got {request_retries!r}")

    if isinstance(url_or_file, str) and url_or_file.startswith("http"):
        url_or_file = url_or_file.replace("www1.ncdc.noaa.gov", "www.ncei.noaa.gov")
        url_or_file = url_or_file.replace("/pub/pub/", "/pub/")

        import gzip
        import io
        import warnings

        import requests

        tries = 0
        while tries - 1 < request_retries:
            try:
                r = requests.get(url_or_file, timeout=request_timeout, stream=True)
                r.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    warnings.warn(f"File not found: {url_or_file}. Skipping.")
                    return pd.DataFrame()
                raise
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
            frame_as_array = np.genfromtxt(f, delimiter=_ISH_WIDTHS, dtype=_ISH_DTYPES)
    else:
        fs = FileUtility.get_fs(url_or_file)
        compression = "gzip" if url_or_file.endswith(".gz") else None
        with fs.open(url_or_file, "rb", compression=compression) as f:
            frame_as_array = np.genfromtxt(f, delimiter=_ISH_WIDTHS, dtype=_ISH_DTYPES)

    frame = pd.DataFrame.from_records(np.atleast_1d(frame_as_array))
    df = _clean_ish(frame)
    df.drop(["latitude", "longitude"], axis=1, inplace=True, errors="ignore")

    if dates is not None and not df.empty:
        dates = pd.to_datetime(dates)
        index = (df.time >= dates.min()) & (df.time <= dates.max())
        df = df.loc[index, :]

    df = _decode_ish_bytes(df)

    # Ensure all non-numeric columns are object for dask consistency
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col].dtype) and col != "time":
            df[col] = df[col].astype(object)

    return df


def _clean_ish_column(
    series: pd.Series, missing: float = 9999.0, multiplier: float = 1.0
) -> pd.Series:
    """Clean a single ISH column."""
    series = pd.to_numeric(series, errors="coerce")
    series = series.where(series != missing, np.nan)
    return series / multiplier


def _clean_ish(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean the ISH DataFrame."""
    if frame.empty:
        for name, _, _ in _ISH_VAR_INFO:
            if name not in frame.columns:
                frame[name] = pd.Series(dtype=object)
        frame["time"] = pd.Series(dtype="datetime64[ns]")
        return frame

    # Vectorized time construction
    dt_str = frame["date"].astype(str) + frame["htime"].astype(str).str.zfill(4)
    frame["time"] = pd.to_datetime(dt_str, format="%Y%m%d%H%M")

    frame.drop(["date", "htime"], axis=1, inplace=True)

    frame["wdir"] = _clean_ish_column(frame["wdir"], missing=999.0)
    frame["ws"] = _clean_ish_column(frame["ws"], multiplier=10.0)
    frame["ceiling"] = _clean_ish_column(frame["ceiling"], missing=99999.0)
    frame["vsb"] = _clean_ish_column(frame["vsb"], missing=999999.0)
    # Note: original had two cleans for vsb with different missing values.
    # We apply them sequentially if needed, but usually 999999 is the one.
    frame["vsb"] = frame["vsb"].where(frame["vsb"] != 99999.0, np.nan)

    frame["t"] = _clean_ish_column(frame["t"], multiplier=10.0, missing=9999.0)
    frame["dpt"] = _clean_ish_column(frame["dpt"], multiplier=10.0, missing=9999.0)
    frame["p"] = _clean_ish_column(frame["p"], multiplier=10.0, missing=99999.0)

    return frame


def _decode_ish_bytes(df: pd.DataFrame) -> pd.DataFrame:
    """Decode byte columns in ISH DataFrame."""
    if df.empty:
        return df
    for col in df.columns:
        if df[col].dtype.kind == "S":
            df[col] = df[col].str.decode("utf-8")
        elif df[col].dtype == object:
            # Handle mixed or object-wrapped bytes
            non_null = df[col].dropna()
            if not non_null.empty and isinstance(non_null.iloc[0], (bytes, np.bytes_)):
                df[col] = df[col].str.decode("utf-8")
    return df


_ISH_VAR_INFO = [
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
_ISH_DTYPES = [(name, dtype) for name, dtype, _ in _ISH_VAR_INFO]
_ISH_WIDTHS = [width for _, _, width in _ISH_VAR_INFO]


class ISH:
    DTYPES = _ISH_DTYPES
    WIDTHS = _ISH_WIDTHS

    def __init__(self):
        self.history_file = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
        self.history = None
        self.df = None
        self.dates = None
        self.verbose = False
        self.source = "ncdc"

    def read_data_frame(
        self, url_or_file: str, *, request_timeout: int = 10, request_retries: int = 4
    ) -> pd.DataFrame:
        """Read a single ISH data frame."""
        return read_ish_file(
            url_or_file,
            dates=self.dates,
            request_timeout=request_timeout,
            request_retries=request_retries,
        )

    def read_ish_history(
        self, dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None
    ):
        """
        Read the ISH history file and filter by dates.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
            Dates to filter the history by.
        """
        if dates is None:
            dates = self.dates
        fname = self.history_file

        if self.source == "aws":
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
            dates = pd.to_datetime(dates)
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
        Subset sites based on a bounding box.

        Parameters
        ----------
        latmin : float, optional
        lonmin : float, optional
        latmax : float, optional
        lonmax : float, optional

        Returns
        -------
        pd.DataFrame
            The subset of site metadata.
        """
        latindex = (self.history.latitude >= latmin) & (self.history.latitude <= latmax)
        lonindex = (self.history.longitude >= lonmin) & (self.history.longitude <= lonmax)
        dfloc = self.history.loc[latindex & lonindex, :]
        return dfloc

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        sites: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Build URLs for ISH data files.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
        sites : pd.DataFrame, optional

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the URLs in the 'name' column.
        """
        if dates is None:
            dates = self.dates
        if sites is None:
            sites = self.history

        unique_years = pd.to_datetime(dates.year.unique(), format="%Y")
        furls = []

        if self.source == "aws":
            url = "s3://noaa-isd-pds/data"
            for syear in unique_years.strftime("%Y"):
                year_fnames = (
                    sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
                )
                for fname in year_fnames:
                    furls.append(f"{url}/{syear}/{fname}")
            return pd.Series(furls, name="name").to_frame()
        else:
            url = "https://www.ncei.noaa.gov/pub/data/noaa"
            for syear in unique_years.strftime("%Y"):
                year_fnames = (
                    sites.usaf.astype(str) + "-" + sites.wban.astype(str) + "-" + syear + ".gz"
                )
                for fname in year_fnames:
                    furls.append(f"{url}/{syear}/{fname}")

            return pd.Series(furls, name="name").to_frame()

    def get_url_file_objs(self, fname: List[str]) -> List[str]:
        """
        Download ISH files and return local filenames.

        Parameters
        ----------
        fname : List[str]
            List of URLs to download.

        Returns
        -------
        List[str]
            List of local filenames.
        """
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
            except Exception:
                pass
        return objs

    def add_data(
        self,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
        box: List[float] = None,
        country: str = None,
        state: str = None,
        site: str = None,
        resample: bool = True,
        window: str = "h",
        download: bool = False,
        n_procs: int = 1,
        request_timeout: int = 10,
        request_retries: int = 4,
        verbose: bool = False,
        source: str = "ncdc",
        lazy: bool = False,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """Legacy method to add ISH data."""
        return ISHReader().open_dataset(
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
            as_xarray=False,
            lazy=lazy,
        )
