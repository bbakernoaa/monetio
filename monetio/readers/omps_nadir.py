"""OMPS Nadir Reader"""

from typing import List, Optional, Union

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import jpss_time_to_datetime, standardize_satellite_coords, update_history


@register_reader("omps_nadir")
class OMPSNadirReader(GriddedReader):
    """
    Reader for OMPS (Ozone Mapping and Profiler Suite) Nadir products.
    Supports NOAA JPSS (V8TOZ, NP SDR/GEO) and NASA (NMTO3) products.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        product: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads OMPS Nadir data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        product : str, optional
            OMPS product type:
            - 'v8toz': NOAA Total Ozone EDR (default if V8TOZ in name)
            - 'nmto3_l2': NASA Nadir Mapper Total Ozone L2
            - 'nmto3_l3': NASA Nadir Mapper Total Ozone L3
            - 'tc_sdr': NOAA Total Column SDR (requires matching GEO files)
            - 'np_sdr': NOAA Nadir Profiler SDR (requires matching GEO files)
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The OMPS dataset.
        """
        if product is None:
            # Try to infer product from file names
            first_file = files[0] if isinstance(files, list) else files
            if "V8TOZ" in first_file:
                product = "v8toz"
            elif "NMTO3" in first_file:
                if "L3" in first_file:
                    product = "nmto3_l3"
                else:
                    product = "nmto3_l2"
            elif "SOMTC" in first_file:
                product = "tc_sdr"
            elif "SONPS" in first_file:
                product = "np_sdr"
            else:
                product = "v8toz"  # Default

        if "preprocess" not in kwargs:
            from functools import partial

            kwargs["preprocess"] = partial(omps_nadir_preprocess, product=product)

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, f"Read OMPS Nadir {product} data.")

        return ds


def omps_nadir_preprocess(ds: xr.Dataset, product: str = "v8toz") -> xr.Dataset:
    """
    Preprocess OMPS Nadir dataset lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    product : str, optional
        Product type, by default "v8toz".

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    if product == "v8toz":
        ds = _preprocess_v8toz(ds)
    elif product in ["nmto3_l2", "nmto3_l3"]:
        # Use existing logic from omps.py
        from .omps import omps_preprocess

        ds = omps_preprocess(ds, product=product)
        return ds
    elif product in ["tc_sdr", "np_sdr"]:
        ds = _preprocess_sdr(ds)

    # Standardize coordinates and dimensions
    ds = standardize_satellite_coords(ds)

    return ds


def _preprocess_v8toz(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess NOAA V8TOZ Total Ozone EDR."""
    mapping = {
        "Latitude": "latitude",
        "Longitude": "longitude",
        "ScanTime": "time_raw",
        "ColumnAmountO3": "ozone_column",
        "AerosolIndex": "aerosol_index",
        "So2Index": "so2_index",
        "CloudFraction": "cloud_fraction",
        "QualityFlag": "quality_flag",
    }

    # Rename variables if they exist
    rename_dict = {old: new for old, new in mapping.items() if old in ds.variables}
    if rename_dict:
        ds = ds.rename(rename_dict)

    # Handle Time (Lazy)
    if "time_raw" in ds.variables:
        ds["time"] = jpss_time_to_datetime(ds["time_raw"])
        ds = ds.set_coords("time")

    return ds


def _preprocess_sdr(ds: xr.Dataset) -> xr.Dataset:
    """Preprocess NOAA SDR (Sensor Data Record)."""
    # SDRs usually have data in groups
    mapping = {
        "All_Data/OMPS-TC-SDR_All/Radiance": "radiance",
        "All_Data/OMPS-NP-SDR_All/Radiance": "radiance",
        "All_Data/OMPS-TC-GEO_All/Latitude": "latitude",
        "All_Data/OMPS-TC-GEO_All/Longitude": "longitude",
        "All_Data/OMPS-NP-GEO_All/Latitude": "latitude",
        "All_Data/OMPS-NP-GEO_All/Longitude": "longitude",
    }
    rename_dict = {old: new for old, new in mapping.items() if old in ds.variables}
    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds
