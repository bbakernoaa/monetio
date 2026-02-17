"""TEMPO Reader"""

from typing import Dict, List, Optional, Union

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("tempo")
class TEMPOReader(GriddedReader):
    """
    Reader for TEMPO (Tropospheric Emissions: Monitoring of Pollution) L2 data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        variable_dict: Optional[Dict] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads TEMPO data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        variable_dict : dict, optional
            Dictionary mapping variable names to processing options (scale, minimum, maximum).
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The TEMPO dataset.
        """
        if "preprocess" not in kwargs:
            from functools import partial

            kwargs["preprocess"] = partial(tempo_preprocess, variable_dict=variable_dict)

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read TEMPO L2 data.")

        return ds


def tempo_preprocess(ds: xr.Dataset, variable_dict: Optional[Dict] = None) -> xr.Dataset:
    """
    Preprocess TEMPO dataset: standardize coordinates and apply quality flags.
    """
    # 1. Map variables from groups to root if needed
    # TEMPO uses PRODUCT, GEOLOCATION, SUPPORT_DATA groups
    # If opened with xarray directly, they might be prefixed or not present depending on group choice.
    # We assume standard xarray-compatible naming or we manually map them.

    mapping = {
        "geolocation/longitude": "longitude",
        "geolocation/latitude": "latitude",
        "geolocation/time": "time",
        "product/vertical_column_troposphere": "no2_column_trop",
        # Add more as needed or use variable_dict
    }

    for old, new in mapping.items():
        if old in ds.variables:
            ds = ds.rename({old: new})

    # 2. Standardize dimensions and coordinates
    # TEMPO uses (x, y) where x is along-track and y is cross-track
    ds = standardize_satellite_coords(ds, lat_name="latitude", lon_name="longitude")

    # 3. Handle Time if needed (TEMPO 'time' is usually seconds since a reference)
    # If xarray decoded it correctly, we are good.

    # 4. Apply variable-specific processing from variable_dict
    if variable_dict:
        for var, options in variable_dict.items():
            if var in ds.variables:
                if "scale" in options:
                    ds[var] = ds[var] * options["scale"]
                if "minimum" in options:
                    ds[var] = ds[var].where(ds[var] >= options["minimum"])
                if "maximum" in options:
                    ds[var] = ds[var].where(ds[var] <= options["maximum"])
                if "quality_flag_max" in options and "main_data_quality_flag" in ds.variables:
                    ds[var] = ds[var].where(
                        ds["main_data_quality_flag"] <= options["quality_flag_max"]
                    )

    return ds
