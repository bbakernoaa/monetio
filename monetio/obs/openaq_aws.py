"""OpenAQ archive data on AWS.

https://openaq.org/

https://registry.opendata.aws/openaq/

https://docs.openaq.org/docs/accessing-openaq-archive-data
"""

import logging
import warnings
from pathlib import Path
from time import perf_counter

import pandas as pd

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def read(fp):
    """Read OpenAQ archive data from a file-like object.

    Parameters
    ----------
    fp : str or path-like or file-like
        OpenAQ archive data, suitable for passing to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        OpenAQ archive data.
    """

    df = pd.read_csv(
        fp,
        dtype={
            0: str,  # location_id
            1: str,  # sensor_id or sensors_id ??
            2: str,  # location
            3: str,  # datetime
            4: float,  # lat
            5: float,  # lon
            6: str,  # parameter
            7: str,  # unit or units ??
            8: float,  # value
        },
        parse_dates=["datetime"],
    )

    # Normalize to web API column names
    if "sensors_id" in df.columns:
        df = df.rename(columns={"sensors_id": "sensor_id"})
    if "units" in df.columns:
        df = df.rename(columns={"units": "unit"})
    df = df.rename(
        columns={
            "location_id": "siteid",
            "datetime": "time",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # Convert to UTC, non-localized
    df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    return df


def _maybe_to_list(x):
    """Convert non-None scalar to singleton list, or return original."""
    if x is not None and pd.api.types.is_scalar(x):
        return [x]
    else:
        return x


def _cache_site_days():
    """Discover all available site-days and save to CSV.

    Returns
    -------
    None
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)

    # TODO: could try multi-thread by month (12 threads) and start/ending digit of location ID
    glb = "openaq-data-archive/records/csv.gz/locationid=*/year=*/month=*/location-*-*.csv.gz"
    tic = perf_counter()
    paths = fs.glob(glb)
    print(f"found {len(paths)} site-days " f"in {pd.Timedelta(seconds=perf_counter() - tic)}")

    df = pd.DataFrame({"path": paths})
    df["filename"] = df["path"].str.rsplit("/", n=1, expand=True)[1]

    ext = ".csv.gz"
    assert df["filename"].str.endswith(ext).all()
    df[["siteid", "date"]] = (
        df["filename"].str.slice(None, -len(ext)).str.rsplit("-", expand=True)[[1, 2]]
    )

    df[["siteid", "date"]].to_csv(
        HERE / "openaq-data-archive_site-days.csv.gz",
        index=False,
    )


def get_paths(dates, *, siteid=None, country=None, provider=None):
    """
    Parameters
    ----------
    dates : pd.DatetimeIndex
    """
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)

    location_ids = _maybe_to_list(siteid)
    providers = _maybe_to_list(provider)
    countries = _maybe_to_list(country)

    if location_ids is None and providers is None and countries is None:
        warnings.warn(
            "location ID(s) not provided; using all locations, which may be quite slow",
            stacklevel=2,
        )
        location_ids = ["*"]

    unique_dates = dates.floor("D").unique()

    print("discovering paths...")
    paths = []

    if location_ids is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "locationid={loc}/year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for loc in location_ids:
                glb = tpl.format(loc=loc, date=date)
                if "*" in glb:
                    loc_date_paths = fs.glob(glb)
                    logger.debug(f"found {len(loc_date_paths)} path(s) for glob='{glb}'")
                    paths.extend(loc_date_paths)
                else:
                    if fs.exists(glb):
                        logger.debug(f"path exists: {glb}")
                        paths.append(glb)
                    else:
                        logger.debug(f"path does not exist: {glb}")

    if providers is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "provider={prvdr}/country={cntry}/locationid={loc}/"
            "year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for prvdr in providers:
                glb = tpl.format(prvdr=prvdr.lower(), cntry="*", loc="*", date=date)
                prvdr_date_paths = fs.glob(glb)
                logger.debug(f"found {len(prvdr_date_paths)} path(s) for glob='{glb}'")
                paths.extend(prvdr_date_paths)

    if countries is not None:
        tpl = (
            "openaq-data-archive/records/csv.gz/"
            "country={cntry}/locationid={loc}/"
            "year={date:%Y}/month={date:%m}/"
            "location-{loc}-{date:%Y%m%d}.csv.gz"
        )
        for date in unique_dates:
            for cntry in countries:
                glb = tpl.format(cntry=cntry.lower(), loc="*", date=date)
                print(glb)
                cntry_date_paths = fs.glob(glb)
                logger.debug(f"found {len(cntry_date_paths)} path(s) for glob='{glb}'")
                paths.extend(cntry_date_paths)

    return sorted(set(paths))


def get_locations(*, country=None, provider=None):
    """Get location IDs corresponding to country/countries OR provider(s).

    Default: all locations.

    Returns
    -------
    list of str
    """
    import re

    import s3fs

    if country is not None and provider is not None:
        raise ValueError("cannot specify both `country` and `provider`")

    print("discovering locations...")
    fs = s3fs.S3FileSystem(anon=True)
    if country is not None:
        if pd.api.types.is_scalar(country):
            countries = [country]
        else:
            countries = country

        paths = []
        for cntry in countries:
            cntry_paths = fs.find(
                # TODO: these paths (csv.gz/country=) are no longer there
                f"openaq-data-archive/records/csv.gz/country={cntry.lower()}/",
                withdirs=True,
                maxdepth=1,
            )
            paths.extend(cntry_paths)

        locs = []
        for p in paths:
            m = re.fullmatch(
                r"openaq-data-archive/records/csv\.gz/country=([a-z]{2}|\-\-|99)/"
                r"locationid=([0-9]+)",
                p,
            )
            if m is not None:
                locs.append(m.group(2))

    elif provider is not None:
        if pd.api.types.is_scalar(provider):
            providers = [provider]
        else:
            providers = provider

        paths = []
        for prvdr in providers:
            prvdr_paths = fs.find(
                f"openaq-data-archive/records/csv.gz/provider={prvdr.lower()}/",
                withdirs=True,
                maxdepth=2,
            )
            paths.extend(prvdr_paths)

        locs = []
        for p in paths:
            m = re.fullmatch(
                r"openaq-data-archive/records/csv\.gz/provider=([a-z0-9\-]+)/"
                r"country=([a-z]{2}|\-\-|99)/locationid=([0-9]+)",
                p,
            )
            if m is not None:
                locs.append(m.group(3))

    else:  # All locs
        paths = fs.find("openaq-data-archive/records/csv.gz/", withdirs=True, maxdepth=1)
        locs = []
        for p in paths:
            m = re.fullmatch(r"openaq-data-archive/records/csv\.gz/locationid=([0-9]+)", p)
            if m is not None:
                locs.append(m.group(1))

    if not locs:
        warnings.warn(f"no locations found for country={country!r} provider={provider!r}")

    return sorted(locs)


def _build_urls(dates, sites, *, protocol="s3"):
    """Naively build URLs for OpenAQ archive data on AWS
    for the given `sites` (location IDs) and dates.

    "Naively" meaning not checking if the files actually exist.

    Parameters
    ----------
    dates : datetime-like or array-like of datetime-like
        Desired dates.
    sites : str or list of str
        Sites (OpenAQ location IDs).
    protocol : {'s3', 'http', 'https'}
        URL protocol to use.

    Returns
    -------
    list of str
    """
    dates = pd.to_datetime(dates)
    if pd.api.types.is_scalar(dates):
        dates = pd.DatetimeIndex([dates])

    if pd.api.types.is_scalar(sites):
        sites = [sites]

    if protocol.lower() == "s3":
        pref = "s3://openaq-data-archive"
    elif protocol.lower() in {"http", "https"}:
        pref = f"{protocol.lower()}://openaq-data-archive.s3.amazonaws.com"
    else:
        raise ValueError(f"protocol: {protocol!r}")

    _urls = []
    for site in sites:
        for date in dates:
            _urls.append(
                f"{pref}/records/csv.gz/"
                f"locationid={site}/year={date:%Y}/month={date:%m}/"
                f"location-{site}-{date:%Y%m%d}.csv.gz"
            )

    return _urls


def _maybe_read(fp):
    """Try to :func:`read` a file, returning an empty DataFrame if it doesn't exist."""
    try:
        return read(fp)
    except FileNotFoundError:
        logger.info(f"file not found: {fp}")
        return pd.DataFrame(
            columns=[
                "siteid",
                "sensor_id",
                "location",
                "time",
                "latitude",
                "longitude",
                "parameter",
                "unit",
                "value",
            ]
        )


def add_data(
    dates,
    *,
    siteid=None,
    country=None,
    provider=None,
    n_procs=1,
):
    """Add OpenAQ data AWS Open Data.

    Parameters
    ----------
    dates : datetime-like or array-like of datetime-like
        Desired dates (the archive data is stored in daily files, per location).
    siteid : str or int or list, optional
        OpenAQ location ID(s) to include.
        For example, from :func:`get_locations`.
    country : str or list of str, optional
        Country or countries to include. 2-character ISO country codes.
    provider : str or list of str, optional
        Data provider(s) to include.

    Returns
    -------
    pd.DataFrame
        OpenAQ archive data.
    """
    import dask.dataframe as dd

    dates = pd.to_datetime(dates)
    if pd.api.types.is_scalar(dates):
        dates = pd.DatetimeIndex([dates])
    dates = dates.dropna()
    if dates.empty:
        raise ValueError("must provide at least one datetime-like")

    paths = get_paths(dates, siteid=siteid, country=country, provider=provider)
    print(f"found {len(paths)}")
    uris = [f"s3://{p}" for p in paths]

    meta = [
        ("siteid", str),
        ("sensor_id", str),
        ("location", str),
        ("time", "datetime64[ns]"),
        ("latitude", float),
        ("longitude", float),
        ("parameter", str),
        ("unit", str),
        ("value", float),
    ]
    print("reading...")
    df = dd.from_map(read, uris, meta=meta).compute(num_workers=n_procs)

    return df.reset_index(drop=True)
