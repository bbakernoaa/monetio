"""PAMS Reader"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, List, Union

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    import dask.dataframe as dd

from .base import PointReader, register_reader
from .drivers import FileUtility


@register_reader("pams")
class PAMSReader(PointReader):
    def open_dataset(
        self,
        files: Union[str, List[str]],
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, xr.Dataset, "dd.DataFrame"]:
        """
        Retrieve and load PAMS data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File paths or URLs to read.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded PAMS data.
        """
        # Filter out arguments that are not for the reader function
        reader_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["expand2d", "pivot", "wide_fmt"]
        }

        df = self.driver.open(files, read_method=read_pams, lazy=lazy, **reader_kwargs)

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Update history
            history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read PAMS data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history

            return ds

        return df


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/obs/pams.py
# -----------------------------------------------------------------------------


def read_pams(filename):
    """
    Read a single PAMS JSON file.

    Parameters
    ----------
    filename : str
        File path or URL.

    Returns
    -------
    pd.DataFrame
        The loaded data.
    """
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        jsonf = json.load(f)

    dataf = jsonf.get("Data", [])
    data = pd.DataFrame.from_dict(dataf)

    if data.empty:
        return data

    data["siteid"] = (
        data.state_code.astype(str).str.zfill(2)
        + data.county_code.astype(str).str.zfill(3)
        + data.site_number.astype(str).str.zfill(4)
    )

    data["time"] = pd.to_datetime(data["date_gmt"] + " " + data["time_gmt"])
    data["time_local"] = pd.to_datetime(data["date_local"] + " " + data["time_local"])

    data = data.rename(
        columns={
            "sample_measurement": "obs",
            "units_of_measure": "units",
            "units_of_measure_code": "unit_code",
        }
    )

    cols_to_drop = [
        "state_code",
        "county_code",
        "site_number",
        "datum",
        "qualifier",
        "uncertainty",
        "county",
        "state",
        "date_of_last_change",
        "date_local",
        "time_local",
        "date_gmt",
        "time_gmt",
        "poc",
        "unit_code",
        "sample_duration_code",
        "method_code",
    ]
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

    # Standardize units
    repl = {
        "Parts per billion Carbon": "ppbC",
        "Parts per billion": "ppb",
        "Parts per million": "ppm",
    }
    data["units"] = data["units"].replace(repl)

    return data
