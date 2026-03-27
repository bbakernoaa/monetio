"""SURFRAD Reader"""

import io
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd

SURFRAD_COLUMNS = [
    "year",
    "jday",
    "month",
    "day",
    "hour",
    "minute",
    "dt",
    "zen",
    "dw_solar",
    "dw_solar_flag",
    "uw_solar",
    "uw_solar_flag",
    "direct_n",
    "direct_n_flag",
    "diffuse",
    "diffuse_flag",
    "dw_ir",
    "dw_ir_flag",
    "dw_casetemp",
    "dw_casetemp_flag",
    "dw_dometemp",
    "dw_dometemp_flag",
    "uw_ir",
    "uw_ir_flag",
    "uw_casetemp",
    "uw_casetemp_flag",
    "uw_dometemp",
    "uw_dometemp_flag",
    "uvb",
    "uvb_flag",
    "par",
    "par_flag",
    "netsolar",
    "netsolar_flag",
    "netir",
    "netir_flag",
    "totalnet",
    "totalnet_flag",
    "temp",
    "temp_flag",
    "rh",
    "rh_flag",
    "windspd",
    "windspd_flag",
    "winddir",
    "winddir_flag",
    "pressure",
    "pressure_flag",
]

# Dictionary mapping surfrad variables to standard names
VARIABLE_MAP = {
    "temp": "air_temperature",
    "rh": "relative_humidity",
    "windspd": "wind_speed",
    "winddir": "wind_direction",
    "pressure": "surface_pressure",
    "dw_solar": "ghi",
    "direct_n": "dni",
    "diffuse": "dhi",
}


def read_surfrad(filename: str, **kwargs: dict) -> pd.DataFrame:
    """
    Read a single SURFRAD file.

    Parameters
    ----------
    filename : str
        The path or URL to the SURFRAD file.
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    # Use FileUtility to handle remote files
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        # Read first two lines for metadata
        # Some files might have encoding issues, so we read as bytes and decode
        content = f.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        lines = content.splitlines()
        if len(lines) < 3:
            return pd.DataFrame()

        station_name = lines[0].strip()
        metadata_line = lines[1].split()

        latitude = float(metadata_line[0])
        longitude = float(metadata_line[1])
        elevation = float(metadata_line[2])
        # version = int(metadata_line[-1]) # not used currently

        # Data starts from line 2
        data_content = "\n".join(lines[2:])
        df = pd.read_csv(
            io.StringIO(data_content),
            sep=r"\s+",
            names=SURFRAD_COLUMNS,
            header=None,
            na_values=[-9999.9, -999.9, -99.9],
            **kwargs,
        )

    # Add metadata as columns (will be coordinates in xarray)
    df["latitude"] = latitude
    df["longitude"] = longitude
    df["elevation"] = elevation
    df["siteid"] = station_name

    # Create time column
    # year, jday, hour, minute
    # jday is 1-indexed
    df["time"] = pd.to_datetime(
        df["year"].astype(str)
        + df["jday"].astype(str).str.zfill(3)
        + df["hour"].astype(str).str.zfill(2)
        + df["minute"].astype(str).str.zfill(2),
        format="%Y%j%H%M",
        errors="coerce",
    )

    return df


@register_reader("surfrad")
class SURFRADReader(PointReader):
    """
    Reader for Surface Radiation Budget Network (SURFRAD) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        sites: Optional[List[str]] = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Open SURFRAD dataset.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs. If None, uses `dates` and `sites` to discover files.
        dates : Any, optional
            Dates to retrieve if `files` is None.
        sites : List[str], optional
            Site abbreviations (e.g. ['tbl', 'bon', 'fpk', 'gwn', 'psu', 'sxf', 'dra']).
        as_xarray : bool, optional
            If True, returns an xarray.Dataset, by default True.
        lazy : bool, optional
            If True, returns a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[xr.Dataset, pd.DataFrame]
            The loaded dataset.
        """
        if files is None:
            if dates is None or sites is None:
                raise ValueError("Either 'files' or both 'dates' and 'sites' must be provided.")
            files = self.build_urls(dates, sites)

        # Separate driver kwargs from to_xarray/postprocess kwargs
        driver_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["expand2d", "wide_fmt", "pivot", "as_xarray", "lazy"]
        }

        # We use read_surfrad as the custom read_method via super()
        df = super().open_dataset(
            files,
            dates,
            sites=sites,
            read_method=read_surfrad,
            as_xarray=False,
            lazy=lazy,
            **driver_kwargs,
        )

        # Post-processing: Harmonize column names
        df = self._postprocess(df)

        # Consistently force object strings
        df = force_object_strings(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)
            # Update history for provenance
            ds = update_history(ds, "Harmonized SURFRAD dataset.")
            return ds

        return df

    def _postprocess(
        self, df: Union[pd.DataFrame, "dd.DataFrame"]
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Harmonize column names.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            Post-processed dataframe.
        """
        # Rename according to VARIABLE_MAP
        df = df.rename(columns=VARIABLE_MAP)

        # Update history for provenance
        df = update_history(df, "Applied variable name mapping for SURFRAD.")

        return df

    def build_urls(
        self,
        dates: Union[datetime, List[datetime], pd.DatetimeIndex],
        sites: List[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Discover available URLs for the given dates and sites.

        Parameters
        ----------
        dates : Union[datetime, List[datetime], pd.DatetimeIndex]
            Dates to retrieve.
        sites : List[str]
            Site abbreviations.

        Returns
        -------
        List[str]
            List of URLs.
        """
        if sites is None:
            raise ValueError("Must specify 'sites' to build URLs for SURFRAD.")

        baseurl = "https://gml.noaa.gov/aftp/data/radiation/surfrad/"

        # Site mapping from abbreviation to directory name
        site_map = {
            "tbl": "Table_Mountain_CO",
            "bon": "Bondville_IL",
            "fpk": "Fort_Peck_MT",
            "gwn": "Goodwin_Creek_MS",
            "psu": "Penn_State_PA",
            "sxf": "Sioux_Falls_SD",
            "dra": "Desert_Rock_NV",
        }

        urls = []
        dates = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))

        for date in dates:
            for site in sites:
                site_dir = site_map.get(site.lower(), site)
                year = date.year
                # Format: site_dir/year/site_yyjday.dat
                # e.g. Bondville_IL/2024/bon24001.dat
                fname = f"{site.lower()}{date.strftime('%y%j')}.dat"
                url = f"{baseurl}{site_dir}/{year}/{fname}"
                urls.append(url)

        return urls
