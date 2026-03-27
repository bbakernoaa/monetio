"""IAGOS Reader."""

from __future__ import annotations

import datetime
import os
from typing import Any, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import xarray as xr

from .base import PointReader, register_reader
from .sat_utils import update_history

if TYPE_CHECKING:
    import dask.dataframe as dd


@register_reader("iagos")
class IAGOSReader(PointReader):
    """
    IAGOS Data Reader following the Aero Protocol.
    Supports both local files and retrieval via IAGOS API (placeholder).
    """

    fixed_location = False

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        as_xarray: bool = True,
        lazy: bool = False,
        **kwargs,
    ) -> xr.Dataset | pd.DataFrame | dd.DataFrame:
        """
        Retrieve and load IAGOS data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str], optional
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
            The loaded IAGOS data.
        """
        # IAGOS data is NetCDF. We use XarrayDriver via super() for robustness.
        ds = super().open_dataset(
            files,
            dates,
            **kwargs,
        )

        if not as_xarray:
            return ds.to_dataframe().reset_index()

        ds = update_history(ds, "Read IAGOS data via Aero Protocol.")
        return ds

    def build_urls(
        self,
        dates: pd.DatetimeIndex | list[datetime.datetime] | datetime.datetime | str,
        **kwargs,
    ) -> list[str]:
        """
        Construct IAGOS URLs.
        Note: IAGOS usually requires registration. This implementation
        documents the required credentials for API access.

        Parameters
        ----------
        dates : Union[pd.DatetimeIndex, List[datetime], datetime, str]
            Dates to build URLs for.
        api_key : str, optional
            IAGOS API key. Can also be set via IAGOS_API_KEY environment variable.

        Returns
        -------
        List[str]
            List of matching IAGOS URLs.
        """
        api_key = kwargs.get("api_key") or os.environ.get("IAGOS_API_KEY")
        if not api_key:
            # We cannot build URLs without an API key or a public mirror.
            # For now, return empty and warn the user.
            import warnings

            warnings.warn(
                "IAGOS retrieval requires an API key. Please provide 'api_key' or set IAGOS_API_KEY env var."
            )
            return []

        # Implementation of IAGOS API retrieval would go here.
        # Example: https://services.iagos-data.fr/prod/v2.0/download?api_key=...
        return []

    def harmonize(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Standardize IAGOS variable names and units.

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset.

        Returns
        -------
        xr.Dataset
            Harmonized dataset.
        """
        # Common IAGOS to MONETIO mappings
        rename_dict = {
            "lon": "longitude",
            "lat": "latitude",
            "baro_alt": "altitude",
            "air_temp": "temperature",
            "o3": "ozone",
            "co": "carbon_monoxide",
            "h2o": "water_vapor",
            "no": "nitrogen_monoxide",
            "nox": "nitrogen_oxides",
            "gps_lat": "latitude",
            "gps_lon": "longitude",
            "gps_alt": "altitude",
        }

        # Filter rename_dict to only include variables present in ds
        actual_rename = {}
        for k, v in rename_dict.items():
            if k in ds.variables and v not in ds.variables:
                actual_rename[k] = v

        if actual_rename:
            ds = ds.rename(actual_rename)

        # Ensure coordinates are set correctly for PointReader expectations
        for coord in ["time", "latitude", "longitude", "altitude"]:
            if coord in ds.variables and coord not in ds.coords:
                ds = ds.set_coords(coord)

        # Standard units and metadata
        if "ozone" in ds.variables:
            # Check if units are already set, only override if sure
            if ds["ozone"].attrs.get("units") in ["ppb", "ppbv", "1e-9"]:
                ds["ozone"].attrs["units"] = "ppb"
                ds["ozone"].attrs["standard_name"] = "mole_fraction_of_ozone_in_air"

        if "carbon_monoxide" in ds.variables:
            if ds["carbon_monoxide"].attrs.get("units") in ["ppb", "ppbv", "1e-9"]:
                ds["carbon_monoxide"].attrs["units"] = "ppb"
                ds["carbon_monoxide"].attrs["standard_name"] = (
                    "mole_fraction_of_carbon_monoxide_in_air"
                )

        return ds
