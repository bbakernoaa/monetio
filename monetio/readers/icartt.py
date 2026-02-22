"""ICARTT Reader ."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .drivers import FileUtility

if TYPE_CHECKING:
    import dask.dataframe as dd


def read_icartt(filename: str, **kwargs) -> pd.DataFrame:
    """Reads a single ICARTT file into a pandas DataFrame.

    Parameters
    ----------
    filename : str
        Path to the ICARTT file.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        Data from the ICARTT file with time and metadata.
    """
    fs = FileUtility.get_fs(filename)
    with fs.open(filename, "r") as f:
        header_lines = []
        line1 = f.readline()
        if not line1:
            return pd.DataFrame()
        header_lines.append(line1.strip())
        try:
            n_header = int(line1.split(",")[0])
        except (ValueError, IndexError):
            return pd.DataFrame()

        for _ in range(n_header - 1):
            line = f.readline()
            if not line:
                break
            header_lines.append(line.strip())

    if len(header_lines) < 13:
        return pd.DataFrame()

    # Line 7: Dates (index 6) (YYYY, MM, DD, YYYY, MM, DD)
    try:
        date_line = [int(x) for x in header_lines[6].split(",")]
        date_valid = datetime.datetime(date_line[0], date_line[1], date_line[2])
    except (ValueError, IndexError):
        date_valid = datetime.datetime(1970, 1, 1)

    # Line 10: Number of dependent variables (index 9)
    try:
        n_vars = int(header_lines[9])
    except (ValueError, IndexError):
        n_vars = 0

    # Line 11: Scales (index 10)
    try:
        scales = [float(x) for x in header_lines[10].split(",")]
    except (ValueError, IndexError):
        scales = [1.0] * n_vars

    # Line 12: Missing values (index 11)
    try:
        missing_values = [x.strip() for x in header_lines[11].split(",")]
    except (ValueError, IndexError):
        missing_values = ["-9999"] * n_vars

    # Variable names
    # Independent variable (IVAR) on line 9 (index 8)
    ivar_line = header_lines[8].split(",")
    ivar_name = ivar_line[0].strip()
    var_names = [ivar_name]

    # Dependent variables (DVAR) on lines 13 (index 12) onwards
    for i in range(n_vars):
        try:
            vname = header_lines[12 + i].split(",")[0].strip()
            var_names.append(vname)
        except IndexError:
            var_names.append(f"var_{i}")

    # Data part
    # We use n_header because line numbers are 1-based and we want to skip n_header lines
    df = pd.read_csv(filename, skiprows=n_header, names=var_names, sep=",", skipinitialspace=True)

    # Apply scales and handle missing values for dependent variables
    for i, (scale, miss) in enumerate(zip(scales, missing_values)):
        if i + 1 >= len(var_names):
            break
        col = var_names[i + 1]
        try:
            miss_val = float(miss)
            # Use a small tolerance for floating point comparison
            mask = np.isclose(df[col].astype(float), miss_val)
            df.loc[mask, col] = np.nan
        except (ValueError, TypeError):
            df.loc[df[col].astype(str) == miss, col] = np.nan

        df[col] = df[col].astype(float) * scale

    # Time conversion
    # Assume IVAR is seconds from date_valid
    if "time" in ivar_name.lower() or "sec" in ivar_line[1].lower():
        df["time"] = date_valid + pd.to_timedelta(df[ivar_name], unit="s")

    return df


@register_reader("icartt")
class ICARTTReader(PointReader):
    """ICARTT Data Reader."""

    fixed_location = False

    def open_dataset(
        self,
        files: str | list[str],
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset | pd.DataFrame | dd.DataFrame:
        """Retrieve and load ICARTT data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        as_xarray : bool, optional
            Whether to return an xarray.Dataset, by default True.
        lazy : bool, optional
            Whether to return a dask-backed object, by default False.
        **kwargs : dict
            Additional arguments passed to the reader and driver.

        Returns
        -------
        Union[pd.DataFrame, xr.Dataset, dd.DataFrame]
            The loaded ICARTT data.
        """
        df = self.driver.open(files, read_method=read_icartt, lazy=lazy, **kwargs)

        df = self.harmonize(df)

        if as_xarray:
            ds = self.to_xarray(df, **kwargs)

            # Add global metadata from the first file
            file_list = FileUtility.expand_paths(files)
            if file_list:
                try:
                    meta = self._get_metadata(file_list[0])
                    ds.attrs.update(meta)
                except Exception:
                    pass

            # Update history
            history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read ICARTT data."
            if "history" in ds.attrs:
                ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
            else:
                ds.attrs["history"] = history
            return ds

        return df

    def _get_metadata(self, filename: str) -> dict:
        """Extract global metadata from the ICARTT header."""
        meta = {}
        fs = FileUtility.get_fs(filename)
        with fs.open(filename, "r") as f:
            f.readline()  # Line 1: n_header, format
            meta["PI"] = f.readline().strip()  # Line 2
            meta["organization"] = f.readline().strip()  # Line 3
            meta["source"] = f.readline().strip()  # Line 4
            meta["mission"] = f.readline().strip()  # Line 5
        return meta

    def harmonize(self, df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame | dd.DataFrame:
        """Standardize column names for coordinates."""
        # Common ICARTT names for lat/lon
        rename_dict = {}
        for col in df.columns:
            lcol = col.lower()
            if "latitude" in lcol and col != "latitude":
                rename_dict[col] = "latitude"
            if "longitude" in lcol and col != "longitude":
                rename_dict[col] = "longitude"
            if "siteid" in lcol and col != "siteid":
                rename_dict[col] = "siteid"

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return super().harmonize(df)
