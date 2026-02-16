"""NADP Reader"""

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from ..util import force_object_strings
from .base import PointReader, register_reader
from .drivers import FileUtility

if TYPE_CHECKING:
    import dask.dataframe as dd

# Meta URLs
META_URLS = {
    "ntn": "https://bit.ly/2sPMvaO",
    "mdn": "https://bit.ly/2Lq6kgq",
    "airmon": "https://bit.ly/2xMlgTW",
    "amon": "https://bit.ly/2sJmkCg",
    "amnet": "https://bit.ly/2sJmkCg",
}


def read_nadp(filename: str, network: str = "ntn", **kwargs) -> pd.DataFrame:
    """
    Read a single NADP file.

    Parameters
    ----------
    filename : str
        The path or URL to the NADP file.
    network : str, optional
        The NADP network (ntn, mdn, amon, airmon, amnet), by default "ntn".
    **kwargs : dict
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    network = network.lower()
    if network == "ntn":
        parse_dates = [2, 3]
        rename_cols = {"dateon": "time", "dateoff": "time_off"}
    elif network == "mdn":
        parse_dates = [1, 2]
        rename_cols = {"dateon": "time", "dateoff": "time_off"}
    elif network in ["airmon", "amon", "amnet"]:
        parse_dates = [2, 3]
        if network == "airmon":
            rename_cols = {"dateon": "time", "dateoff": "time_off"}
        else:
            rename_cols = {"startdate": "time", "enddate": "time_off"}
    else:
        parse_dates = False
        rename_cols = {}

    # Use FileUtility to handle remote files
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        df = pd.read_csv(f, parse_dates=parse_dates, **kwargs)

    df.columns = [i.lower() for i in df.columns]
    df = df.rename(columns=rename_cols)

    # Apply network-specific cleaning
    if network == "ntn":
        for col in ["mg", "br", "so4", "cl", "no3", "nh4", "k", "na", "ca"]:
            flag = "flag" + col
            if flag in df.columns:
                df.loc[(df[flag] == "<") | (df[col] < 0), col] = np.nan
    elif network == "mdn":
        if "qr" in df.columns:
            df.loc[df.qr == "C", ["rgppt", "svol", "subppt", "hgconc", "hgdep"]] = np.nan
    elif network == "airmon":
        if "qrcode" in df.columns:
            cols = [
                "subppt",
                "pptnws",
                "pptbel",
                "svol",
                "ca",
                "mg",
                "k",
                "na",
                "nh4",
                "no3",
                "cl",
                "so4",
                "po4",
                "phlab",
                "phfield",
                "conduclab",
                "conducfield",
            ]
            df.loc[df.qrcode == "C", cols] = np.nan
    elif network in ["amon", "amnet"]:
        if "qr" in df.columns:
            df.loc[df.qr == "C", ["airvol", "conc"]] = np.nan

    return df


@register_reader("nadp")
class NADPReader(PointReader):
    """
    Reader for National Atmospheric Deposition Program (NADP) data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Union[datetime, List[datetime], pd.DatetimeIndex]] = None,
        network: str = "NTN",
        siteid: Optional[str] = None,
        weekly: bool = True,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Open NADP dataset.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File paths or URLs. If None, uses `network`, `siteid`, and `weekly` to build URL.
        dates : Union[datetime, List[datetime], pd.DatetimeIndex], optional
            Dates to filter data.
        network : str, optional
            NADP network (NTN, MDN, AMON, AIRMON, AMNET), by default "NTN".
        siteid : str, optional
            Specific site ID to retrieve.
        weekly : bool, optional
            Whether to retrieve weekly data (if applicable), by default True.
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
            files = self.build_url(network=network, siteid=siteid, weekly=weekly)

        # We use read_nadp as the custom read_method
        def _reader(f, **inner_kwargs):
            return read_nadp(f, network=network, **inner_kwargs)

        df = self.driver.open(files, read_method=_reader, lazy=lazy, **kwargs)

        # Filter by dates if provided
        if dates is not None:
            dates = pd.DatetimeIndex(np.atleast_1d(dates))
            df = df.loc[(df.time >= dates.min()) & (df.time_off <= dates.max())]

        # Post-processing: Merge with monitor info
        df = self._postprocess(df, network=network)

        # Consistently force object strings
        df = force_object_strings(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)
            # Update history for provenance
            history = (
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
                f"Merged with NADP ({network}) station metadata and harmonized."
            )
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df

    def _postprocess(
        self, df: Union[pd.DataFrame, "dd.DataFrame"], network: str
    ) -> Union[pd.DataFrame, "dd.DataFrame"]:
        """
        Merge with station metadata and harmonize column names.

        Parameters
        ----------
        df : Union[pd.DataFrame, dd.DataFrame]
            Input dataframe.
        network : str
            NADP network name.

        Returns
        -------
        Union[pd.DataFrame, dd.DataFrame]
            Post-processed dataframe.
        """
        meta = self.get_monitor_df(network=network)

        # Merge (unified logic for both backends)
        df = df.merge(meta, on="siteid", how="left")

        # Original code dropped NaNs in latitude/longitude in NADP.read_*
        # PointReader.harmonize also does this.
        df = self.harmonize(df)

        return df

    def get_monitor_df(self, network: str) -> pd.DataFrame:
        """
        Load the NADP station metadata for a specific network.

        Parameters
        ----------
        network : str
            NADP network name.

        Returns
        -------
        pd.DataFrame
            Station metadata.
        """
        network = network.lower()
        url = META_URLS.get(network)
        if url is None:
            return pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        try:
            meta = pd.read_csv(url)
        except Exception:
            return pd.DataFrame(columns=["siteid", "latitude", "longitude"])

        meta.columns = [i.lower() for i in meta.columns]
        if "startdate" in meta.columns:
            meta = meta.drop(["startdate", "stopdate"], axis=1, errors="ignore")

        return meta

    def build_url(self, network: str = "NTN", siteid: Optional[str] = None, weekly: bool = True):
        """
        Build URL for NADP data.

        Parameters
        ----------
        network : str, optional
            NADP network (NTN, MDN, AMON, AIRMON, AMNET), by default "NTN".
        siteid : str, optional
            Specific site ID to retrieve.
        weekly : bool, optional
            Whether to retrieve weekly data, by default True.

        Returns
        -------
        str
            The URL to the data file.
        """
        baseurl = "http://nadp.slh.wisc.edu/datalib/"
        site_part = (siteid.upper() + "-") if siteid is not None else ""
        network = network.lower()

        if network == "amnet":
            return "http://nadp.slh.wisc.edu/datalib/AMNet/AMNet-All.zip"
        elif network == "amon":
            return "http://nadp.slh.wisc.edu/dataLib/AMoN/csv/all-ave.csv"
        elif network == "airmon":
            return "http://nadp.slh.wisc.edu/datalib/AIRMoN/AIRMoN-ALL.csv"
        else:
            if weekly:
                return f"{baseurl}{network}/weekly/{site_part}{network.upper()}-All-w.csv"
            else:
                return f"{baseurl}{network}/annual/{site_part}{network.upper()}-All-a.csv"
