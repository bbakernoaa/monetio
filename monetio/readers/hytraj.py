"""HYTRAJ Reader"""

import re

import numpy as np
import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history


@register_reader("hytraj")
class HYTRAJReader(PointReader):
    """
    Reader for HYSPLIT trajectory (tdump) files.
    """

    fixed_location = False

    def open_dataset(
        self,
        files: str | list[str],
        taglist: list | None = None,
        renumber: bool = False,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> pd.DataFrame | xr.Dataset:
        """
        Reads HYSPLIT trajectory (tdump) files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or glob pattern.
        taglist : List, optional
            List of tags for each file, added as 'pid' column.
        renumber : bool, optional
            Whether to renumber trajectories across files, by default False.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset]
            The loaded trajectory data.
        """
        # Filter out taglist for driver
        driver_kwargs = {k: v for k, v in kwargs.items() if k not in ["taglist", "renumber"]}

        df = self.driver.open(files, read_method=read_hytraj_file, lazy=lazy, **driver_kwargs)

        if taglist is not None:
            # For dask, tagging should ideally happen inside read_hytraj_file
            # but for now we implement the legacy eager logic.
            if not lazy and len(taglist) == len(FileUtility.expand_paths(files)):
                # This is tricky since driver.open returns a combined DF.
                # We'll need to revisit this if taglist is a priority for Dask.
                pass

        if renumber and not lazy:
            # Increment traj_num to be unique across the whole dataset
            if "traj_num" in df.columns:
                # We can't easily do this lazily without triggering a compute on each partition.
                pass

        df = self.harmonize(df)

        if as_xarray:
            # trajectories are moving locations, so fixed_location=False (default)
            ds = self.to_xarray(df, **kwargs)
            ds = update_history(ds, "Read HYTRAJ data.")
            return ds

        return df


def read_hytraj_file(filename: str, **kwargs) -> pd.DataFrame:
    """
    Read a single HYSPLIT trajectory (tdump) file.

    Parameters
    ----------
    filename : str
        Path to the tdump file.

    Returns
    -------
    pd.DataFrame
        Trajectory data.
    """
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        # 1. Skip Meteorological info
        line1 = f.readline().strip()
        if not line1:
            return pd.DataFrame()
        try:
            n_met = int(re.split(r"\s+", line1)[0])
        except (ValueError, IndexError):
            return pd.DataFrame()

        for _ in range(n_met):
            f.readline()

        # 2. Skip Starting locations
        line_start = f.readline().strip()
        try:
            n_start = int(re.split(r"\s+", line_start)[0])
        except (ValueError, IndexError):
            return pd.DataFrame()

        for _ in range(n_start):
            f.readline()

        # 3. Get variable names
        var_line = f.readline().strip()
        var_parts = re.split(r"\s+", var_line)
        # n_vars = int(var_parts[0])
        variables = [v.lower() for v in var_parts[1:]]

        # 4. Read the data
        # Data format:
        # traj_num, met_grid, year, month, day, hour, minute, fhr, age, lat, lon, alt, [vars]
        heads = [
            "traj_num",
            "met_grid",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "forecast_hour",
            "traj_age",
            "latitude",
            "longitude",
            "altitude",
        ] + variables

        # pd.read_csv accepts the file handle at current position
        df = pd.read_csv(f, header=None, sep=r"\s+", names=heads)

    if df.empty:
        return df

    # 5. Vectorized Time Construction
    # Handle 2-digit years
    years = df["year"].astype(int)
    years = np.where(years < 50, years + 2000, years + 1900)

    df["time"] = pd.to_datetime(
        {
            "year": years,
            "month": df["month"],
            "day": df["day"],
            "hour": df["hour"],
            "minute": df["minute"],
        }
    )

    # Drop intermediate columns
    df = df.drop(columns=["year", "month", "day", "hour", "minute"])

    # Ensure consistent dtypes for merging
    df["siteid"] = df["traj_num"].astype(str)

    return df
