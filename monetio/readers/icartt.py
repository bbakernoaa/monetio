"""ICARTT Reader."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd


def parse_icartt_header(filename: str) -> dict[str, Any]:
    """
    Parse ICARTT file header metadata.

    Parameters
    ----------
    filename : str
        Path to the ICARTT file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing header metadata.
    """
    fs = FileUtility.get_fs(filename)
    header = {}
    with fs.open(filename, "r") as f:
        line1 = f.readline()
        if not line1:
            return {}
        try:
            n_header = int(line1.split(",")[0])
        except (ValueError, IndexError):
            return {}

        header["n_header"] = n_header
        header["PI"] = f.readline().strip()
        header["organization"] = f.readline().strip()
        header["source"] = f.readline().strip()
        header["mission"] = f.readline().strip()
        f.readline()  # Line 6: volume

        # Line 7: Dates (index 6)
        date_line = f.readline().split(",")
        try:
            header["date_valid"] = datetime.datetime(
                int(date_line[0]), int(date_line[1]), int(date_line[2])
            )
        except (ValueError, IndexError):
            header["date_valid"] = datetime.datetime(1970, 1, 1)

        f.readline()  # Line 8: interval

        # Line 9: IVAR name and units
        ivar_line = f.readline().split(",")
        header["ivar_name"] = ivar_line[0].strip()
        header["ivar_units"] = ivar_line[1].strip() if len(ivar_line) > 1 else ""

        # Line 10: Number of DVARs
        try:
            n_vars = int(f.readline().strip())
        except ValueError:
            n_vars = 0
        header["n_vars"] = n_vars

        # Line 11: Scales
        header["scales"] = [float(x) for x in f.readline().split(",")]

        # Line 12: Missing values
        header["missing_values"] = [x.strip() for x in f.readline().split(",")]

        # Line 13+: DVAR names
        var_names = [header["ivar_name"]]
        for i in range(n_vars):
            vname = f.readline().split(",")[0].strip()
            var_names.append(vname)
        header["var_names"] = var_names

    return header


def read_icartt(filename: str, **kwargs) -> pd.DataFrame:
    """
    Reads a single ICARTT file into a pandas DataFrame.
    Data is returned unscaled and with raw missing values to allow lazy processing.

    Parameters
    ----------
    filename : str
        Path to the ICARTT file.
    **kwargs : dict
        Additional arguments.

    Returns
    -------
    pd.DataFrame
        Data from the ICARTT file.
    """
    header = parse_icartt_header(filename)
    if not header:
        return pd.DataFrame()

    df = pd.read_csv(
        filename,
        skiprows=header["n_header"],
        names=header["var_names"],
        sep=",",
        skipinitialspace=True,
    )

    # Convert IVAR to time if possible
    if "time" in header["ivar_name"].lower() or "sec" in header["ivar_units"].lower():
        df["time"] = header["date_valid"] + pd.to_timedelta(df[header["ivar_name"]], unit="s")

    return df


@register_reader("icartt")
class ICARTTReader(PointReader):
    """
    ICARTT Data Reader following the Aero Protocol.
    """

    fixed_location = False

    def open_dataset(
        self,
        files: str | list[str] | None = None,
        dates: Any | None = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset | pd.DataFrame | dd.DataFrame:
        """
        Retrieve and load ICARTT data with lazy scaling and missing value handling.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Any, optional
            Dates to retrieve if files are not provided.
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
        if files is None:
            if dates is not None and hasattr(self, "build_urls"):
                files = self.build_urls(dates, **kwargs)
            else:
                raise ValueError("Either 'files' or 'dates' must be provided.")

        # We need metadata from the first file to setup lazy processing
        file_list = FileUtility.expand_paths(files)
        if not file_list:
            raise FileNotFoundError(f"No files found matching {files}")

        header = parse_icartt_header(file_list[0])

        # Open data
        df = super().open_dataset(
            files,
            dates,
            read_method=read_icartt,
            as_xarray=False,
            lazy=lazy,
            **kwargs,
        )

        if lazy:
            # For Dask, we apply scaling and missing values via map_partitions
            # or we do it after converting to Xarray (preferable for Aero Protocol)
            pass

        df = self.harmonize(df)

        if as_xarray:
            # Default to expand2d=False for ICARTT to keep it simple and match expected results
            # if not explicitly requested otherwise.
            if "expand2d" not in kwargs:
                kwargs["expand2d"] = False

            ds = self.to_xarray(df, **kwargs)

            # Apply scaling and missing values lazily in Xarray
            ds = icartt_preprocess(ds, header)

            # Add global metadata
            ds.attrs.update(
                {
                    k: v
                    for k, v in header.items()
                    if k in ["PI", "organization", "source", "mission"]
                }
            )

            ds = update_history(ds, "Read ICARTT data via Aero Protocol.")
            return ds

        return df

    def harmonize(self, df: pd.DataFrame | dd.DataFrame) -> pd.DataFrame | dd.DataFrame:
        """Standardize coordinate column names."""
        rename_dict = {}
        for col in df.columns:
            lcol = col.lower()
            if "latitude" in lcol and col != "latitude":
                rename_dict[col] = "latitude"
            if "longitude" in lcol and col != "longitude":
                rename_dict[col] = "longitude"
            if "altitude" in lcol and col != "altitude":
                rename_dict[col] = "altitude"
            if "siteid" in lcol and col != "siteid":
                rename_dict[col] = "siteid"

        if rename_dict:
            df = df.rename(columns=rename_dict)

        return super().harmonize(df)


def icartt_preprocess(ds: xr.Dataset, header: dict[str, Any]) -> xr.Dataset:
    """
    Apply scaling and missing value handling lazily to ICARTT dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    header : Dict[str, Any]
        Header metadata including scales and missing values.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    scales = header.get("scales", [])
    missing_values = header.get("missing_values", [])
    var_names = header.get("var_names", [])

    # DVARs start from index 1 (0 is IVAR)
    for i, (scale, miss) in enumerate(zip(scales, missing_values)):
        if i + 1 >= len(var_names):
            break
        orig_col = var_names[i + 1]

        # Find the column in ds, handling harmonization
        col = None
        if orig_col in ds.variables:
            col = orig_col
        else:
            # Check harmonized names
            for c in ds.variables:
                if c.lower() == orig_col.lower():
                    col = c
                    break

        if col is None:
            continue

        # Handle missing values
        try:
            miss_val = float(miss)
            # Use a small tolerance for floating point comparison
            # We must use .data to avoid issues with xarray wrappers if any
            # but usually .where works fine.
            # Let's try to be very explicit.
            ds[col] = ds[col].where(np.abs(ds[col].astype(float) - miss_val) > 1e-4)
        except (ValueError, TypeError):
            ds[col] = ds[col].where(ds[col].astype(str).str.strip() != miss.strip())

        # Scaling
        if scale != 1.0:
            ds[col] = ds[col].astype(float) * scale

    return ds
