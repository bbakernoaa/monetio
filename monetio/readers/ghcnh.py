"""NOAA Global Historical Climatology Network hourly (GHCNh) reader.

GHCNh is the next-generation hourly/synoptic surface dataset replacing the
Integrated Surface Dataset (ISD/ISH). Period-of-record and per-year station
files are published on the AWS Open Data bucket ``noaa-ghcnh-pds`` (anonymous
access, us-east-1) in PSV and Parquet layouts:

    s3://noaa-ghcnh-pds/hourly/access/by-year/{YYYY}/parquet/GHCNh_{ID}_{YYYY}.parquet
    s3://noaa-ghcnh-pds/hourly/access/by-year/{YYYY}/psv/GHCNh_{ID}_{YYYY}.psv
    s3://noaa-ghcnh-pds/hourly/access/by-station/GHCNh_{ID}_por.psv

Station metadata (id, lat, lon, elevation, state, country) comes from
``hourly/doc/ghcnh-station-list.csv``. The archive updates daily with a lag
of a few days; per-variable quality flags follow the ``*_Quality_Code``
convention (blank = passed QC).

References
----------
- https://registry.opendata.aws/noaa-ghcnh/
- https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd

logger = logging.getLogger(__name__)

BUCKET = "noaa-ghcnh-pds"
STATION_LIST_URL = f"s3://{BUCKET}/hourly/doc/ghcnh-station-list.csv"

# Default evaluation variables (GHCNh native names are already descriptive).
DEFAULT_VARIABLES = (
    "temperature",
    "dew_point_temperature",
    "sea_level_pressure",
    "station_level_pressure",
    "wind_speed",
    "wind_direction",
    "relative_humidity",
)

# Physical units per GHCNh documentation (values are stored unscaled).
VARIABLE_UNITS = {
    "temperature": "degC",
    "dew_point_temperature": "degC",
    "sea_level_pressure": "hPa",
    "station_level_pressure": "hPa",
    "wind_speed": "m/s",
    "wind_direction": "degrees",
    "relative_humidity": "%",
    "precipitation": "mm",
}

_META_COLS = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]


def _to_datetime_index(dates) -> pd.DatetimeIndex:
    """Normalize a date/sequence/slice argument to a DatetimeIndex."""
    if isinstance(dates, slice):
        start = pd.Timestamp(dates.start)
        stop = pd.Timestamp(dates.stop)
        return pd.date_range(start, stop, freq="D")
    if isinstance(dates, (str, datetime, pd.Timestamp)):
        return pd.DatetimeIndex([pd.Timestamp(dates)])
    return pd.DatetimeIndex(pd.to_datetime(list(dates)))


def read_station_list(**storage_options) -> pd.DataFrame:
    """Read the GHCNh station list (id, position, state, country)."""
    from .drivers import FileUtility

    fs = FileUtility.get_fs(STATION_LIST_URL, storage_options=storage_options or None)
    with fs.open(STATION_LIST_URL, "rb") as f:
        df = pd.read_csv(f, dtype=str)
    df = df.rename(
        columns={
            "GHCN_ID": "siteid",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "ELEVATION": "elevation",
            "STATE": "state",
            "ISO_CODE": "country",
            "NAME": "station_name",
        }
    )
    for col in ("latitude", "longitude", "elevation"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def select_stations(
    stations: pd.DataFrame,
    *,
    siteid: str | list[str] | None = None,
    box: list[float] | None = None,
    country: str | list[str] | None = None,
    state: str | list[str] | None = None,
) -> pd.DataFrame:
    """Subset the station list by ids, bounding box, country, or state.

    ``box`` is ``[latmin, lonmin, latmax, lonmax]`` (same order as the ISH
    readers). Filters combine with AND; at least one must be given.
    """
    if siteid is None and box is None and country is None and state is None:
        raise ValueError("Provide at least one of siteid/box/country/state to select stations.")
    out = stations
    if siteid is not None:
        ids = [siteid] if isinstance(siteid, str) else list(siteid)
        out = out[out.siteid.isin(ids)]
    if box is not None:
        latmin, lonmin, latmax, lonmax = box
        out = out[
            (out.latitude >= latmin)
            & (out.latitude <= latmax)
            & (out.longitude >= lonmin)
            & (out.longitude <= lonmax)
        ]
    if country is not None:
        codes = [country] if isinstance(country, str) else list(country)
        out = out[out.country.isin([c.upper() for c in codes])]
    if state is not None:
        codes = [state] if isinstance(state, str) else list(state)
        out = out[out.state.isin([s.upper() for s in codes])]
    return out


def build_urls(
    dates: pd.DatetimeIndex,
    sites: pd.DataFrame | list[str],
    *,
    layout: str = "parquet",
) -> list[str]:
    """Per-station-year object URLs for the requested dates.

    ``layout`` selects ``parquet`` (compact, column-subsettable; default) or
    ``psv``. Missing station-years are filtered by the caller's tolerance
    policy at read time, not here (listing every candidate would be slow).
    """
    if layout not in ("parquet", "psv"):
        raise ValueError(f"layout must be 'parquet' or 'psv', got {layout!r}")
    ids = sites.siteid.tolist() if isinstance(sites, pd.DataFrame) else list(sites)
    years = sorted({int(y) for y in dates.year})
    ext = "parquet" if layout == "parquet" else "psv"
    return [
        f"s3://{BUCKET}/hourly/access/by-year/{year}/{layout}/GHCNh_{sid}_{year}.{ext}"
        for sid in ids
        for year in years
    ]


def read_ghcnh_file(
    fname: str,
    *,
    variables: list[str] | None = None,
    quality_filter: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Read one GHCNh station-year file into a tidy point DataFrame.

    Columns: ``time``, ``siteid``, ``latitude``, ``longitude``, ``elevation``
    plus the requested variables. When ``quality_filter`` is true, values
    with a non-blank ``{var}_Quality_Code`` are masked to NaN (flagged
    records are excluded but their timestamps remain, so downstream pairing
    counts stay honest).
    """
    from .drivers import FileUtility

    fs = FileUtility.get_fs(fname, storage_options=kwargs.get("storage_options"))
    variables = list(variables) if variables is not None else list(DEFAULT_VARIABLES)
    qc_cols = [f"{v}_Quality_Code" for v in variables]
    columns = _META_COLS + variables + qc_cols

    if fname.endswith(".parquet"):
        import pyarrow.parquet as pq

        with fs.open(fname, "rb") as f:
            schema = pq.read_schema(f)
        available = [c for c in columns if c in schema.names]
        with fs.open(fname, "rb") as f:
            df = pd.read_parquet(f, columns=available)
    else:
        with fs.open(fname, "rb") as f:
            header = f.readline().decode().strip().split("|")
        available = [c for c in columns if c in header]
        with fs.open(fname, "rb") as f:
            df = pd.read_csv(f, sep="|", usecols=available)

    if df.empty:
        return df

    df = df.rename(
        columns={
            "STATION": "siteid",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
            "ELEVATION": "elevation",
            "DATE": "time",
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    for col in ("latitude", "longitude", "elevation"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if quality_filter:
        for var in variables:
            qc = f"{var}_Quality_Code"
            if var in df and qc in df:
                flagged = df[qc].notna() & (df[qc].astype(str).str.strip() != "")
                df.loc[flagged, var] = np.nan
    df = df.drop(columns=[c for c in qc_cols if c in df])
    for var in variables:
        if var in df:
            df[var] = pd.to_numeric(df[var], errors="coerce")
    return df


@register_reader("ghcnh")
class GHCNhReader(PointReader):
    """Reader for the GHCN hourly archive on AWS Open Data (anonymous)."""

    def open_dataset(
        self,
        files: str | list[str] | None = None,
        dates: pd.DatetimeIndex | list[datetime] | datetime | str | slice | None = None,
        siteid: str | list[str] | None = None,
        box: list[float] | None = None,
        country: str | list[str] | None = None,
        state: str | list[str] | None = None,
        variables: list[str] | None = None,
        layout: str = "parquet",
        quality_filter: bool = True,
        wide_fmt: bool = True,
        n_procs: int = 1,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> pd.DataFrame | xr.Dataset | dd.DataFrame:
        """Load GHCNh station observations.

        Parameters
        ----------
        files : str or list of str, optional
            Explicit station-year object URLs or local paths. When omitted,
            URLs are built from ``dates`` + station selection.
        dates : date-like or sequence or slice
            Requested period; per-year files are fetched for each touched
            year and trimmed to ``[min(dates), max(dates) + 1 day)``.
        siteid, box, country, state
            Station selection (at least one required without ``files``).
            ``box`` order: ``[latmin, lonmin, latmax, lonmax]``.
        variables : list of str, optional
            GHCNh variable names (default: temperature, dew point, both
            pressures, wind speed/direction, relative humidity).
        layout : {"parquet", "psv"}
            Remote file layout; parquet (default) is far more compact.
        quality_filter : bool
            Mask values whose per-variable quality code is non-blank.
        wide_fmt : bool
            Return the 2-D (time, node) UGRID layout (default) vs 1-D records.
        as_xarray : bool
            Return xarray (default) or the tidy DataFrame.
        """
        if files is None:
            if dates is None:
                raise ValueError("Provide 'dates' (with station selection) or explicit 'files'.")
            idx = _to_datetime_index(dates)
            stations = read_station_list()
            sites = select_stations(stations, siteid=siteid, box=box, country=country, state=state)
            if sites.empty:
                raise ValueError("Station selection matched no GHCNh stations.")
            files = build_urls(idx, sites, layout=layout)
        else:
            idx = _to_datetime_index(dates) if dates is not None else None
            files = [files] if isinstance(files, str) else list(files)

        from .drivers import FileUtility

        fs = FileUtility.get_fs(files[0], storage_options=kwargs.get("storage_options"))
        available = []
        missing = []
        for path in files:
            if fs.exists(path):
                available.append(path)
            else:
                missing.append(path)
        if missing:
            logger.info(
                "GHCNh: %d requested station-year files not published (skipped): %s",
                len(missing),
                missing[:5],
            )
        if not available:
            raise ValueError("No GHCNh files found for the requested stations/period.")

        frames = []
        for path in available:
            frame = read_ghcnh_file(
                path, variables=variables, quality_filter=quality_filter, **kwargs
            )
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise ValueError("GHCNh files contained no records for the requested variables.")
        df = pd.concat(frames, ignore_index=True)

        if idx is not None:
            start = idx.min().normalize()
            end = idx.max().normalize() + pd.Timedelta(days=1)
            df = df[(df.time >= start) & (df.time < end)]
        df = df.sort_values(["siteid", "time"]).reset_index(drop=True)
        df = self.harmonize(df)
        df = update_history(
            df, f"Read GHCNh ({len(available)} station-year files; layout={layout})."
        )

        if not as_xarray:
            return df

        ds = self.to_xarray(
            df,
            expand2d=wide_fmt,
            **{k: v for k, v in kwargs.items() if k not in ("storage_options", "expand2d")},
        )
        for var, unit in VARIABLE_UNITS.items():
            if var in ds:
                ds[var].attrs.setdefault("units", unit)
        ds.attrs["ghcnh_quality_filter"] = bool(quality_filter)
        return update_history(ds, "Read GHCNh station observations.")
