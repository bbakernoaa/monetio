"""AERONET Reader following the Aero Protocol."""

import warnings
from datetime import datetime, timezone
from functools import lru_cache, partial
from io import BytesIO
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import dask.dataframe as dd
import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("aeronet")
class AERONETReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]] = None,
        dates: Union[pd.DatetimeIndex, List[datetime], datetime, str] = None,
        product: str = "AOD15",
        inv_type: str = None,
        latlonbox: List[float] = None,
        siteid: str = None,
        daily: bool = False,
        lunar: bool = False,
        freq: str = None,
        detect_dust: bool = False,
        interp_to_aod_values: Union[List[float], np.ndarray] = None,
        n_procs: int = 1,
        as_xarray: bool = True,
        lazy: bool = False,
        retries: int = 5,
        backoff_factor: float = 2.0,
        n_chunks: Optional[int] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
                Retrieve and load AERONET data following the Aero Protocol.

                Parameters
                ----------
                files : Union[str, List[str]], optional
                    File path, list of paths, or glob pattern.
                dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
                    Dates to retrieve if files are not provided.
                product : str, optional
                    AERONET product (e.g., 'AOD15', 'SDA20'), by default "AOD15".
                inv_type : str, optional
                    Inversion type (e.g., 'ALM15', 'HYB20'), by default None.
                latlonbox : List[float], optional
                    Bounding box [latmin, lonmin, latmax, lonmax], by default None.
                siteid : str, optional
                    Specific AERONET site ID, by default None.
                daily : bool, optional
                    Whether to load daily averages instead of all points, by default False.
                lunar : bool, optional
                    Whether to include lunar data, by default False.
                freq : str, optional
                    Resampling frequency (e.g., '1H'), by default None.
                detect_dust : bool, optional
                    Whether to add a 'dust' column based on AOD/Angstrom, by default False.
                interp_to_aod_values : Union[List[float], np.ndarray], optional
                    Wavelengths (nm) to interpolate AOD to, by default None.
                n_procs : int, optional
                    Number of processors for parallel loading (non-lazy), by default 1.
                as_xarray : bool, optional
                    Whether to return an xarray.Dataset, by default True.
                lazy : bool, optional
                    Whether to return a dask-backed object, by default False.
                retries : int, optional
                    Number of retries for network calls, by default 5.
                backoff_factor : float, optional
                    Backoff factor for exponential retries, by default 1.0.
                **kwargs : dict
                    Additional arguments passed to the driver.

                Returns
        -------
                Union[pd.DataFrame, xr.Dataset]
                    The loaded AERONET data.

                Examples
                --------
                >>> from monetio.readers.aeronet import AERONETReader
                >>> reader = AERONETReader()
                >>> ds = reader.open_dataset(dates='2021-08-01', siteid='Mauna_Loa')
        """
        if files is None:
            if dates is None:
                # Default to today (use naive to avoid pd.date_range issues)
                now = datetime.now(timezone.utc)
                start = datetime(now.year, now.month, now.day)
                dates = pd.date_range(start=start, end=now.replace(tzinfo=None), freq="h")

            # Throttling mitigation:
            # By default, we request the whole range to minimize calls to NASA.
            # However, if we are in parallel (n_procs > 1) or lazy (lazy=True),
            # we split into a small number of chunks (defaulting to 8, < 10)
            # to ensure robust retrieval without timeouts or hitting rate limits hard.
            if n_chunks is None and (n_procs > 1 or lazy):
                n_chunks = 8

            # Construct URLs from dates
            files = build_urls(
                dates,
                product=product,
                inv_type=inv_type,
                daily=daily,
                lunar=lunar,
                siteid=siteid,
                latlonbox=latlonbox,
                n_chunks=n_chunks,
                **kwargs,
            )

        if not files:
            if dates is not None:
                if as_xarray:
                    return xr.Dataset()
                raise Exception("valid query but no data found")
            raise ValueError("Must provide either 'files' or 'dates'.")

        # Define per-file preprocessing
        storage_options = kwargs.get("storage_options", {})
        read_func = partial(
            read_aeronet_csv,
            inv_type=inv_type,
            interp_to_aod_values=interp_to_aod_values,
            detect_dust=detect_dust,
            storage_options=storage_options,
            retries=retries,
            backoff_factor=backoff_factor,
            **kwargs,
        )

        # Use base class to open
        # If n_procs > 1 and not lazy, we use dask to parallelize the load then compute
        use_dask = lazy or (n_procs > 1)
        df = super().open_dataset(
            files,
            read_method=read_func,
            as_xarray=False,
            lazy=use_dask,
            **kwargs,
        )

        if not lazy and use_dask:
            try:
                import dask.dataframe as dd

                if isinstance(df, dd.DataFrame):
                    df = df.compute(num_workers=n_procs)
            except ImportError:
                pass

        if not lazy and df.empty:
            raise Exception("valid query but no data found")

        # Post-processing (Freq resampling)
        if freq is not None and not lazy:
            # We can only resample eagerly here to avoid hidden compute in dask
            if not df.empty:
                df = (
                    df.set_index("time")
                    .groupby("siteid")
                    .resample(freq, include_groups=False)
                    .mean(numeric_only=True)
                    .reset_index()
                )

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Update history
            history = (
                f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}: Read AERONET data."
            )
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history

            return ds

        return df

    def harmonize(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Standardize column names and types.
        """
        # Force string columns to object for Pandas 3.0 compatibility
        df = force_object_strings(df)
        return df


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _get_robust_session(retries: int = 5, backoff_factor: float = 2.0):
    """
    Create a requests session with retries and a standard User-Agent.

    Parameters
    ----------
    retries : int, optional
        Number of retries, by default 5.
    backoff_factor : float, optional
        Backoff factor for retries, by default 2.0.

    Returns
    -------
    requests.Session
        A robust requests session.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
    )
    return session


@lru_cache(1)
def get_valid_sites(retries: int = 5, backoff_factor: float = 2.0) -> pd.DataFrame:
    """
    Fetch valid AERONET sites from NASA.

    Parameters
    ----------
    retries : int, optional
        Number of retries for network call, by default 5.
    backoff_factor : float, optional
        Backoff factor for retries, by default 2.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with valid sites and their locations.

    Examples
    --------
    >>> sites = get_valid_sites()
    """
    try:
        session = _get_robust_session(retries=retries, backoff_factor=backoff_factor)
        url = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"
        response = session.get(url, timeout=120)
        response.raise_for_status()

        df = pd.read_csv(
            BytesIO(response.content),
            skiprows=1,
        ).rename(
            columns={
                "Site_Name": "siteid",
                "Longitude(decimal_degrees)": "longitude",
                "Latitude(decimal_degrees)": "latitude",
                "Elevation(meters)": "elevation",
            },
        )
    except Exception as e:
        # Check if it's a connection error
        import requests

        if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            raise

        warnings.warn(f"Getting valid sites failed: {e}. Site validation will be skipped.")
        # Return empty with correct columns to avoid AttributeError in legacy code
        return pd.DataFrame(columns=["siteid", "longitude", "latitude", "elevation"])
    return df


def build_urls(
    dates: Union[pd.DatetimeIndex, List[datetime], datetime, str],
    product: str = "AOD15",
    *,
    inv_type: Optional[str] = None,
    daily: bool = False,
    lunar: bool = False,
    siteid: Optional[str] = None,
    latlonbox: Optional[List[float]] = None,
    split_by_day: bool = False,
    n_chunks: Optional[int] = None,
    **kwargs,
) -> List[str]:
    """
    Construct AERONET URLs.

    Parameters
    ----------
    dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
        Dates to build URLs for.
    product : str, optional
        AERONET product, by default "AOD15".
    inv_type : Optional[str], optional
        Inversion type, by default None.
    daily : bool, optional
        Whether to request daily averages, by default False.
    lunar : bool, optional
        Whether to request lunar data, by default False.
    siteid : Optional[str], optional
        Specific site ID, by default None.
    latlonbox : Optional[List[float]], optional
        Bounding box, by default None.
    split_by_day : bool, optional
        Whether to generate one URL per day, by default False.

    Returns
    -------
    List[str]
        List of AERONET download URLs.

    Examples
    --------
    >>> urls = build_urls('2021-08-01', siteid='Mauna_Loa')
    """
    dates = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))
    if dates.empty:
        return []

    if split_by_day or n_chunks is not None:
        min_date = dates.min()
        max_date = dates.max()

        if n_chunks is not None:
            # Generate N roughly equal chunks
            # Ensure at least 1 chunk
            n_chunks = max(1, n_chunks)
            if (max_date - min_date).total_seconds() == 0:
                time_list = [min_date, max_date]
            else:
                # Use linspace for dates
                # Convert to unix timestamps for easier division
                t_start = min_date.timestamp()
                t_end = max_date.timestamp()
                t_list = np.linspace(t_start, t_end, n_chunks + 1)
                time_list = [pd.to_datetime(t, unit="s", utc=True) for t in t_list]
        else:
            # Generate daily URLs
            time_bounds = pd.date_range(start=min_date.floor("D"), end=max_date.ceil("D"), freq="D")

            # Clip bounds to the actual requested range
            time_list = time_bounds.tolist()
            if not time_list or time_list[0] < min_date:
                if not time_list:
                    time_list = [min_date, max_date]
                else:
                    time_list[0] = min_date
            if time_list[-1] > max_date:
                time_list[-1] = max_date
            elif time_list[-1] < max_date:
                time_list.append(max_date)

        # Ensure unique and sorted
        time_list = sorted(list(set(time_list)))

        if len(time_list) < 2:
            return [
                _build_single_url(
                    min_date,
                    max_date,
                    product=product,
                    inv_type=inv_type,
                    daily=daily,
                    lunar=lunar,
                    siteid=siteid,
                    latlonbox=latlonbox,
                    **kwargs,
                )
            ]

        urls = []
        for i in range(len(time_list) - 1):
            urls.append(
                _build_single_url(
                    time_list[i],
                    time_list[i + 1],
                    product=product,
                    inv_type=inv_type,
                    daily=daily,
                    lunar=lunar,
                    siteid=siteid,
                    latlonbox=latlonbox,
                    **kwargs,
                )
            )
        return urls
    else:
        return [
            _build_single_url(
                dates.min(),
                dates.max(),
                product=product,
                inv_type=inv_type,
                daily=daily,
                lunar=lunar,
                siteid=siteid,
                latlonbox=latlonbox,
                **kwargs,
            )
        ]


def _build_single_url(d1, d2, product, inv_type, daily, lunar, siteid, latlonbox, **kwargs):
    """Internal helper to build a single URL."""
    sy, sm, sd, sh = d1.strftime(r"%Y"), d1.strftime(r"%m"), d1.strftime(r"%d"), d1.strftime(r"%H")
    ey, em, ed, eh = d2.strftime(r"%Y"), d2.strftime(r"%m"), d2.strftime(r"%d"), d2.strftime(r"%H")
    dates_ = f"year={sy}&month={sm}&day={sd}&hour={sh}&year2={ey}&month2={em}&day2={ed}&hour2={eh}"

    valid_prod_noninv = (
        "AOD10",
        "AOD15",
        "AOD20",
        "SDA10",
        "SDA15",
        "SDA20",
        "TOT10",
        "TOT15",
        "TOT20",
    )
    valid_prod_inv = ("SIZ", "RIN", "CAD", "VOL", "TAB", "AOD", "SSA", "ASY", "FRC", "LID", "FLX")
    valid_inv_type = ("ALM15", "ALM20", "HYB15", "HYB20")

    product = product.upper()
    if inv_type is None:
        if product not in valid_prod_noninv:
            raise ValueError(f"invalid product {product!r}")
        base_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?"
        inv_type_ = ""
        product_ = f"&{product}=1"
    elif inv_type in valid_inv_type:
        if product not in valid_prod_inv:
            raise ValueError(f"invalid product {product!r}")
        base_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?"
        inv_type_ = f"&{inv_type}=1"
        product_ = f"&product={product}"
    else:
        raise ValueError(f"invalid inv type: {inv_type!r}")

    avg_ = f"&AVG={20 if daily else 10}"
    lunar_ = f"&lunar_merge={1 if lunar else 0}"

    if siteid is not None:
        # Restore validation for test_add_data_bad_siteid
        retries = kwargs.get("retries", 5)
        valid_sites = get_valid_sites(retries=retries)
        if not valid_sites.empty and siteid not in valid_sites.siteid.values:
            raise ValueError(f"invalid site {siteid!r}")
        loc_ = f"&site={siteid}"
    elif latlonbox is not None:
        lat1, lon1, lat2, lon2 = map(str, map(float, latlonbox))
        loc_ = f"&lat1={lat1}&lat2={lat2}&lon1={lon1}&lon2={lon2}"
    else:
        loc_ = ""

    return f"{base_url}{dates_}{product_}{avg_}{lunar_}{inv_type_}{loc_}&if_no_html=1"


def read_aeronet_csv(
    fn: str,
    *,
    inv_type: Optional[str] = None,
    interp_to_aod_values: Optional[Union[List[float], np.ndarray]] = None,
    detect_dust: bool = False,
    storage_options: Optional[dict] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a single AERONET file or URL.

    Parameters
    ----------
    fn : str
        File path or URL.
    inv_type : Optional[str], optional
        Inversion type, by default None.
    interp_to_aod_values : Optional[Union[List[float], np.ndarray]], optional
        Wavelengths to interpolate to, by default None.
    detect_dust : bool, optional
        Whether to detect dust, by default False.
    storage_options : Optional[dict], optional
        fsspec storage options, by default None.

    Returns
    -------
    pd.DataFrame
        Loaded AERONET data.

    Examples
    --------
    >>> df = read_aeronet_csv('path/to/file.txt')
    """
    # Robust fetch for HTTP(S) URLs
    if str(fn).startswith("http"):
        try:
            retries = kwargs.get("retries", 5)
            backoff_factor = kwargs.get("backoff_factor", 2.0)
            session = _get_robust_session(retries=retries, backoff_factor=backoff_factor)

            # Jitter to avoid thundering herd on throttled servers
            if kwargs.get("n_procs", 1) > 1 or kwargs.get("lazy", False):
                import random
                import time

                time.sleep(random.uniform(0, 5))

            response = session.get(str(fn), timeout=120)
            response.raise_for_status()
            source = BytesIO(response.content)
        except Exception as e:
            import requests

            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                raise

            warnings.warn(f"Failed to fetch {fn}: {e}")
            return pd.DataFrame()
    else:
        source = fn

    # Determine skiprows and check for errors
    try:
        if isinstance(source, BytesIO):
            source.seek(0)
            header_lines = [
                source.readline().decode("utf-8", errors="replace").strip() for _ in range(10)
            ]
            source.seek(0)
        else:
            fs = FileUtility.get_fs(str(fn))
            with fs.open(str(fn), mode="rb") as f:
                header_lines = [
                    f.readline().decode("utf-8", errors="replace").strip() for _ in range(10)
                ]
    except Exception as e:
        warnings.warn(f"Failed to read header of {fn}: {e}")
        return pd.DataFrame()

    header_text = "\n".join(header_lines)
    if "<html>" in header_text:
        # Invalid query
        return pd.DataFrame()

    if len([line for line in header_lines if line]) < 2:
        # Might be "valid query but no data found"
        return pd.DataFrame()

    is_inv = "Inversion" in header_text or inv_type is not None
    skiprows = 5 if not is_inv else 6

    try:
        df = pd.read_csv(
            source,
            engine="python",
            header="infer",
            skiprows=skiprows,
            na_values=-999,
            storage_options=storage_options,
        )
    except Exception as e:
        warnings.warn(f"Error parsing CSV from {fn}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.rename(columns=str.lower).copy()

    # Handle time
    date_col = [c for c in df.columns if "date(" in c]
    time_col = [c for c in df.columns if "time(" in c]
    if date_col and time_col:
        df["time"] = pd.to_datetime(
            df[date_col[0]] + " " + df[time_col[0]], format=r"%d:%m:%Y %H:%M:%S", errors="coerce"
        )
        df = df.drop(columns=[date_col[0], time_col[0]])

    # Standard names
    df = df.rename(
        columns={
            "aeronet_site": "siteid",
            "aeronet_aeronet_site": "siteid",
            "site_latitude(degrees)": "latitude",
            "site_longitude(degrees)": "longitude",
            "site_elevation(m)": "elevation",
            "latitude(degrees)": "latitude",
            "longitude(degrees)": "longitude",
            "elevation(m)": "elevation",
        }
    )

    # Apply Aero Protocol Scientific Hygiene
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude", "longitude"])
    df = df.dropna(axis=1, how="all")

    if hasattr(df, "attrs"):
        df.attrs["info"] = header_text

    # Dust detect
    if detect_dust:
        df = _dust_detect(df)

    # Interpolate
    if interp_to_aod_values is not None:
        df = _calc_new_aod_values(df, interp_to_aod_values)

    return df.copy()


def _dust_detect(df: pd.DataFrame) -> pd.DataFrame:
    """Detect dust based on AOD and Angstrom exponent."""
    if "aod_1020nm" in df.columns and "440-870_angstrom_exponent" in df.columns:
        df["dust"] = (df["aod_1020nm"] > 0.3) & (df["440-870_angstrom_exponent"] < 0.6)
    return df


def _calc_new_aod_values(df: pd.DataFrame, new_wv: Union[List[float], np.ndarray]) -> pd.DataFrame:
    """Interpolate AOD to new wavelengths."""

    def _tspack_aod_interp(row, new_wv):
        try:
            import pytspack
        except ImportError:
            # Re-raise as RuntimeError to match expected behavior in tests
            raise RuntimeError("You must install pytspack before using this function.")

        aod_columns = [c for c in row.index if c.startswith("aod_") and c.endswith("nm")]
        aods = row[aod_columns]
        wv = [float(c.replace("aod_", "").replace("nm", "")) for c in aod_columns]

        a = pd.DataFrame({"aod": aods, "wv": wv}).dropna()
        a = a.sort_values(by="wv")
        if len(a) < 2:
            return new_wv * np.nan
        else:
            try:
                try:
                    # New API
                    interp = pytspack.TsPack().interpolate(a.wv.values, a.aod.values)
                    return interp(new_wv)
                except AttributeError:
                    # Old API
                    x, y, yp, sigma = pytspack.tspsi(a.wv.values, a.aod.values)
                    return pytspack.hval(new_wv, x, y, yp, sigma)
            except RuntimeError as e:
                raise RuntimeError(f"pytspack is installed but not usable: {e}")

    new_wv = np.asarray(new_wv)
    # Copy to avoid fragmentation warning when adding many columns
    df = df.copy()
    out = df.apply(_tspack_aod_interp, axis=1, result_type="expand", new_wv=new_wv)
    names = "aod_" + pd.Series(new_wv.astype(int).astype(str)) + "nm"
    out.columns = names.values

    dup_names = list(set(df.columns) & set(out.columns))
    if dup_names:
        suff = "_orig"
        warnings.warn(
            f"Renaming duplicate AOD columns {dup_names} by adding suffix '{suff}'.",
            stacklevel=2,
        )
        for name in dup_names:
            df = df.rename(columns={name: f"{name}{suff}"})
            # Also rename exact wavelengths if they exist
            wl = name[4:-2]
            ename = f"exact_wavelengths_of_aod(um)_{wl}nm"
            if ename in df.columns:
                df = df.rename(columns={ename: f"{ename}{suff}"})

    return pd.concat([df, out], axis=1)


class AERONET:
    """Legacy AERONET class for backward compatibility."""

    _valid_prod_noninv = (
        "AOD10",
        "AOD15",
        "AOD20",
        "SDA10",
        "SDA15",
        "SDA20",
        "TOT10",
        "TOT15",
        "TOT20",
    )
    _valid_prod_inv = (
        "SIZ",
        "RIN",
        "CAD",
        "VOL",
        "TAB",
        "AOD",
        "SSA",
        "ASY",
        "FRC",
        "LID",
        "FLX",
    )
    _valid_inv_type = ("ALM15", "ALM20", "HYB15", "HYB20")

    def __init__(self):
        self.url = None
        self.df = None
        self.dates = None
        self.prod = None
        self.inv_type = None
        self.daily = None
        self.lunar = None
        self.latlonbox = None
        self.siteid = None
        self.new_aod_values = None

    def build_url(self):
        """Build the AERONET URL."""
        assert self.dates is not None, "required parameter"
        assert self.prod is not None, "required parameter"
        assert self.daily in {10, 20}, "required parameter"

        self.url = _build_single_url(
            self.dates.min(),
            self.dates.max(),
            product=self.prod,
            inv_type=self.inv_type,
            daily=(self.daily == 20),
            lunar=(self.lunar == 1),
            siteid=self.siteid,
            latlonbox=self.latlonbox,
        )

    def read_aeronet(self):
        """Read the AERONET data."""
        self.df = read_aeronet_csv(
            self.url,
            inv_type=self.inv_type,
            interp_to_aod_values=self.new_aod_values,
        )
        if self.df.empty:
            # Matches old behavior for some tests
            raise Exception("valid query but no data found")

    def add_data(self, **kwargs):
        """Add data (legacy)."""
        # Determine if we should return xarray or dataframe
        # In legacy mode, if not specified, return dataframe
        as_xarray = kwargs.get("as_xarray", False)
        kwargs["as_xarray"] = False
        df = AERONETReader().open_dataset(**kwargs)
        if df.empty:
            raise Exception("valid query but no data found")
        self.df = df
        if as_xarray:
            return AERONETReader().to_xarray(df)
        return self.df

    def dust_detect(self):
        """Detect dust."""
        if self.df is not None:
            self.df = _dust_detect(self.df)

    def calc_new_aod_values(self):
        """Calculate new AOD values."""
        if self.df is not None and self.new_aod_values is not None:
            self.df = _calc_new_aod_values(self.df, self.new_aod_values)


def _parallel_aeronet_call(**kwargs):
    """Legacy parallel call."""
    # This remains for backward compatibility
    return AERONETReader().open_dataset(as_xarray=False, **kwargs)
