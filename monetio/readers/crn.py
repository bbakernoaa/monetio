"""CRN Reader"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility

if TYPE_CHECKING:
    import dask.dataframe as dd

HCOLS = [
    "WBANNO",
    "UTC_DATE",
    "UTC_TIME",
    "LST_DATE",
    "LST_TIME",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "T_CALC",
    "T_AVG",
    "T_MAX",
    "T_MIN",
    "P_CALC",
    "SOLARAD",
    "SOLARAD_FLAG",
    "SOLARAD_MAX",
    "SOLARAD_MAX_FLAG",
    "SOLARAD_MIN",
    "SOLARAD_MIN_FLAG",
    "SUR_TEMP_TYPE",
    "SUR_TEMP",
    "SUR_TEMP_FLAG",
    "SUR_TEMP_MAX",
    "SUR_TEMP_MAX_FLAG",
    "SUR_TEMP_MIN",
    "SUR_TEMP_MIN_FLAG",
    "RH_AVG",
    "RH_AVG_FLAG",
    "SOIL_MOISTURE_5",
    "SOIL_MOISTURE_10",
    "SOIL_MOISTURE_20",
    "SOIL_MOISTURE_50",
    "SOIL_MOISTURE_100",
    "SOIL_TEMP_5",
    "SOIL_TEMP_10",
    "SOIL_TEMP_20",
    "SOIL_TEMP_50",
    "SOIL_TEMP_100",
]

DCOLS = [
    "WBANNO",
    "LST_DATE",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "T_MAX",
    "T_MIN",
    "T_MEAN",
    "T_AVG",
    "P_CALC",
    "SOLARAD",
    "SUR_TEMP_TYPE",
    "SUR_TEMP_MAX",
    "SUR_TEMP_MAX_2",
    "SUR_TEMP_MIN",
    "SUR_TEMP_AVG",
    "RH_MAX",
    "RH_MIN",
    "RH_AVG",
    "SOIL_MOISTURE_5",
    "SOIL_MOISTURE_10",
    "SOIL_MOISTURE_20",
    "SOIL_MOISTURE_50",
    "SOIL_MOISTURE_100",
    "SOIL_TEMP_5",
    "SOIL_TEMP_10",
    "SOIL_TEMP_20",
    "SOIL_TEMP_50",
    "SOIL_TEMP_100",
]

SHCOLS = [
    "WBANNO",
    "UTC_DATE",
    "UTC_TIME",
    "LST_DATE",
    "LST_TIME",
    "CRX_VN",
    "LONGITUDE",
    "LATITUDE",
    "T_MEAN",
    "P_CALC",
    "SOLARAD",
    "SOLARAD_FLAG",
    "SUR_TEMP_AVG",
    "SUR_TEMP_TYPE",
    "SUR_TEMP_FLAG",
    "RH_AVG",
    "RH_FLAG",
    "SOIL_MOISTURE_5",
    "SOIL_TEMP_5",
    "WETNESS",
    "WET_FLAG",
    "WIND",
    "WIND_FLAG",
]


def read_crn(filename: str, **kwargs) -> pd.DataFrame:
    """
    Read a single CRN file.

    Parameters
    ----------
    filename : str
        The path or URL to the CRN file.
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    nanvals = [-99999, -9999.0]
    if "CRND0103" in filename:
        cols = DCOLS
        is_daily = True
    elif "CRNS0101" in filename:
        cols = SHCOLS
        is_daily = False
    else:
        cols = HCOLS
        is_daily = False

    # Use FileUtility to handle remote files
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        df = pd.read_csv(
            f,
            sep=r"\s+",
            names=cols,
            na_values=nanvals,
            index_col=False,
            **kwargs,
        )

    # Manual date parsing for compatibility with Pandas 3.0
    if not is_daily:
        if "UTC_DATE" in df.columns and "UTC_TIME" in df.columns:
            df["time"] = pd.to_datetime(
                df["UTC_DATE"].astype(str) + df["UTC_TIME"].astype(str).str.zfill(4),
                format="%Y%m%d%H%M",
                errors="coerce",
            )
        if "LST_DATE" in df.columns and "LST_TIME" in df.columns:
            df["time_local"] = pd.to_datetime(
                df["LST_DATE"].astype(str) + df["LST_TIME"].astype(str).str.zfill(4),
                format="%Y%m%d%H%M",
                errors="coerce",
            )
    else:
        if "LST_DATE" in df.columns:
            df["time_local"] = pd.to_datetime(
                df["LST_DATE"].astype(str), format="%Y%m%d", errors="coerce"
            )

    return df


@register_reader("crn")
class CRNReader(PointReader):
    """
    Reader for US Climate Reference Network (USCRN) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[datetime, List[datetime], pd.DatetimeIndex]] = None,
        daily: bool = False,
        sub_hourly: bool = False,
        download: bool = False,
        latlonbox: Optional[List[float]] = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Open CRN dataset.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs. If None, uses `dates` to discover files.
        dates : Union[datetime, List[datetime], pd.DatetimeIndex], optional
            Dates to retrieve if `files` is None.
        daily : bool, optional
            If True, retrieves daily data, by default False.
        sub_hourly : bool, optional
            If True, retrieves sub-hourly (5-min) data, by default False.
        download : bool, optional
            If True, downloads files locally, by default False.
        latlonbox : List[float], optional
            Bounding box [lat_min, lon_min, lat_max, lon_max].
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
            if dates is None:
                raise ValueError("Either 'files' or 'dates' must be provided.")
            files, _ = self.build_urls(
                dates, daily=daily, sub_hourly=sub_hourly, latlonbox=latlonbox
            )

        if download:
            files = self.retrieve(files)

        # We use read_crn as the custom read_method
        df = self.driver.open(files, read_method=read_crn, lazy=lazy, **kwargs)

        # Post-processing: Merge with monitor info and fix columns
        df = self._postprocess(df, latlonbox=latlonbox)

        # Consistently force object strings
        df = force_object_strings(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)
            # Update history for provenance
            history = (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                "Merged with CRN station metadata and harmonized."
            )
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df

    def _postprocess(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], latlonbox: Optional[List[float]] = None
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Merge with station metadata and harmonize column names.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.
        latlonbox : List[float], optional
            Bounding box [lat_min, lon_min, lat_max, lon_max].

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            Post-processed dataframe.
        """
        monitors = self.get_monitor_df(latlonbox=latlonbox)
        # Rename WBAN to WBANNO to match data files
        monitors = monitors.rename(columns={"WBAN": "WBANNO"})
        # Ensure siteid/WBANNO is string and padded to 5 digits
        monitors["WBANNO"] = monitors["WBANNO"].astype(str).str.zfill(5)

        # Identify backend
        try:
            import dask.dataframe as dd

            is_dask = isinstance(df, dd.DataFrame)
        except ImportError:
            is_dask = False

        df["WBANNO"] = df["WBANNO"].astype(str).str.zfill(5)
        # Merge (unified logic for both backends)
        df = df.merge(monitors, how="left", on=["WBANNO", "LATITUDE", "LONGITUDE"])

        # Handle time conversion if needed
        if "time" not in df.columns:
            if "time_local" in df.columns and "GMT_OFFSET" in df.columns:
                if is_dask:
                    df["time"] = df["time_local"] + dd.to_timedelta(df["GMT_OFFSET"], unit="h")
                else:
                    df["time"] = df["time_local"] + pd.to_timedelta(df["GMT_OFFSET"], unit="h")
            elif "time_local" in df.columns:
                # Fallback for daily data if GMT_OFFSET is missing
                df["time"] = df["time_local"]

        df = df.rename(columns={"WBANNO": "siteid"})
        # Lowercase all columns
        df.columns = [c.lower() for c in df.columns]

        return df

    def get_monitor_df(self, latlonbox: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Load the CRN station metadata.

        Parameters
        ----------
        latlonbox : List[float], optional
            Bounding box [lat_min, lon_min, lat_max, lon_max].

        Returns
        -------
        pd.DataFrame
            Station metadata.
        """
        import monetio

        path = os.path.join(os.path.dirname(monetio.__file__), "data", "stations.tsv")
        try:
            mdf = pd.read_csv(path, delimiter="\t")
            # Filter for USCRN
            mdf = mdf.loc[mdf["NETWORK"] == "USCRN"].copy()
        except Exception:
            # Fallback if file missing
            mdf = pd.DataFrame(
                columns=[
                    "STATE",
                    "LOCATION",
                    "VECTOR",
                    "WBAN",
                    "LATITUDE",
                    "LONGITUDE",
                    "NETWORK",
                ]
            )

        if latlonbox is not None:
            con = (
                (mdf.LATITUDE >= latlonbox[0])
                & (mdf.LATITUDE <= latlonbox[2])
                & (mdf.LONGITUDE >= latlonbox[1])
                & (mdf.LONGITUDE <= latlonbox[3])
            )
            mdf = mdf.loc[con].copy()

        return mdf

    def build_urls(
        self,
        dates: Union[datetime, List[datetime], pd.DatetimeIndex],
        daily: bool = False,
        sub_hourly: bool = False,
        latlonbox: Optional[List[float]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Discover available URLs for the given dates and monitors.

        Parameters
        ----------
        dates : Union[datetime, List[datetime], pd.DatetimeIndex]
            Dates to retrieve.
        daily : bool, optional
            If True, retrieves daily data, by default False.
        sub_hourly : bool, optional
            If True, retrieves sub-hourly (5-min) data, by default False.
        latlonbox : List[float], optional
            Bounding box [lat_min, lon_min, lat_max, lon_max].

        Returns
        -------
        Tuple[List[str], List[str]]
            List of URLs and filenames.
        """
        baseurl = "https://www1.ncdc.noaa.gov/pub/data/uscrn/products/"
        monitors = self.get_monitor_df(latlonbox=latlonbox)
        years = pd.DatetimeIndex(np.atleast_1d(dates)).year.unique().astype(str)

        urls = []
        fnames = []

        for _, row in monitors.iterrows():
            for y in years:
                state = row["STATE"]
                site = row["LOCATION"].replace(" ", "_")
                vector = row["VECTOR"].replace(" ", "_")

                if daily:
                    beginning = f"{baseurl}daily01/{y}/"
                    fname_prefix = "CRND0103-"
                elif sub_hourly:
                    beginning = f"{baseurl}subhourly01/{y}/"
                    fname_prefix = "CRNS0101-05-"
                else:
                    beginning = f"{baseurl}hourly02/{y}/"
                    fname_prefix = "CRNH0203-"

                rest = f"{y}-{state}_{site}_{vector}.txt"
                url = f"{beginning}{fname_prefix}{rest}"
                fname = f"{fname_prefix}{rest}"

                fs = FileUtility.get_fs(url)
                try:
                    if fs.exists(url):
                        urls.append(url)
                        fnames.append(fname)
                except Exception:
                    pass

        return urls, fnames

    def retrieve(self, urls: Union[str, List[str]]) -> List[str]:
        """
        Download files locally if they don't exist.

        Parameters
        ----------
        urls : Union[str, List[str]]
            List of URLs to download.

        Returns
        -------
        List[str]
            List of local filenames.
        """
        if isinstance(urls, str):
            urls = [urls]

        local_files = []
        for url in urls:
            fname = os.path.basename(url)
            if not os.path.isfile(fname):
                fs = FileUtility.get_fs(url)
                try:
                    fs.get(url, fname)
                except Exception as e:
                    print(f"Failed to retrieve {url}: {e}")
                    continue
            local_files.append(fname)
        return local_files
