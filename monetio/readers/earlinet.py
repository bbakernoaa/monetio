"""EARLINET Reader"""

import re
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .actris import get_ebas_catalog
from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("earlinet")
class EARLINETReader(GriddedReader):
    """
    Reader for EARLINET (European Aerosol Research Lidar Network) NetCDF data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        siteid: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Retrieve and load EARLINET data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path, list of paths, or glob pattern.
        dates : Any, optional
            Dates to retrieve if files are not provided.
        siteid : str, optional
            Specific station ID to filter.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The loaded EARLINET data.
        """
        # Default to combined dimensions if not specified
        kwargs.setdefault("combine", "by_coords")

        # Use XarrayDriver (via GriddedReader) to open via super()
        ds = super().open_dataset(
            files, dates, siteid=siteid, preprocess=earlinet_preprocess, **kwargs
        )

        return ds

    def build_urls(
        self,
        dates: Union[pd.DatetimeIndex, List[Any], Any],
        siteid: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Construct EARLINET THREDDS URLs from EBAS catalog.
        """
        dates = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))
        if dates.empty:
            return []

        catalog_xml = get_ebas_catalog()
        if not catalog_xml:
            return []

        # Find EARLINET datasets in the catalog
        pattern = r'<dataset name="([^"]+)" ID="([^"]+)" urlPath="([^"]+)"'
        matches = re.findall(pattern, catalog_xml)

        urls = []
        base_url = "https://thredds.nilu.no/thredds/dodsC/"

        requested_min = dates.min()
        requested_max = dates.max()

        for name, dataset_id, url_path in matches:
            # ONLY include EARLINET
            if "earlinet" not in name.lower():
                continue

            # Check siteid
            if siteid and siteid not in name:
                continue

            # Check dates
            # EARLINET filename format usually contains a date
            # Example: DE0001R.20230101000000...earlinet...nc
            parts = name.split(".")
            if len(parts) < 2:
                continue

            try:
                # Try to parse date from the second part
                start_date = pd.to_datetime(parts[1], format="%Y%m%d%H%M%S", errors="coerce")
                if pd.isna(start_date):
                    # Try YYYYMMDD
                    start_date = pd.to_datetime(parts[1][:8], format="%Y%m%d", errors="coerce")
            except Exception:
                continue

            if pd.isna(start_date):
                continue

            # Check for overlap (Assume 1 day for EARLINET files if duration not clear)
            # Most EARLINET files in EBAS are per-measurement or daily
            end_date = start_date + pd.Timedelta(days=1)

            if start_date <= requested_max and end_date >= requested_min:
                urls.append(f"{base_url}{url_path}")

        return urls


def earlinet_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess EARLINET dataset: standardize coordinates and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input EARLINET dataset.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset.
    """
    # 1. Standardize Time
    # EARLINET NetCDF files are usually CF compliant and xarray handles them.

    # 2. Rename Dimensions/Coordinates
    # altitude is usually a coordinate/dimension already named 'altitude' or 'height'.
    # If it's something else, we can rename it.
    if "height" in ds.dims and "altitude" not in ds.dims:
        ds = ds.rename({"height": "altitude"})

    # 3. Coordinate handling
    # Ensure latitude and longitude are coordinates
    coord_vars = ["latitude", "longitude", "altitude", "time", "wavelength"]
    actual_coords = [v for v in coord_vars if v in ds.variables]
    if actual_coords:
        ds = ds.set_coords(actual_coords)

    # 4. Standard attributes
    if "altitude" in ds.coords:
        ds["altitude"].attrs.update({"units": "m", "standard_name": "altitude", "positive": "up"})

    if "latitude" in ds.coords:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})

    if "longitude" in ds.coords:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # Update history
    ds = update_history(ds, "Preprocessed EARLINET data.")

    return ds
