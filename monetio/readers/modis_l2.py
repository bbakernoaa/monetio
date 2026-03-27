"""MODIS L2 Swath Reader"""

from typing import Any, List, Optional, Union

import pandas as pd
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("modis_l2")
class MODISL2Reader(GriddedReader):
    """
    Reader for MODIS L2 swath data.
    """

    def open_dataset(
        self,
        files: Optional[Union[str, List[str]]] = None,
        dates: Optional[Any] = None,
        variable_dict: dict = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads MODIS L2 swath data.

        Parameters
        ----------
        files : Union[str, List[str]], optional
            File path(s) or URL(s).
        dates : Any, optional
            Dates to retrieve if files are not provided.
        variable_dict : dict, optional
            Dictionary of variables to read with metadata (scale, minimum, maximum, quality_flag).
        **kwargs : dict
            Additional arguments passed to the reader.

        Returns
        -------
        xr.Dataset
            The processed MODIS dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = lambda ds: modis_l2_preprocess(ds, variable_dict=variable_dict)

        # If variable_dict is provided, we can try to only load those variables
        # plus the required coordinates (Latitude, Longitude, Scan_Start_Time)
        if variable_dict is not None and "drop_variables" not in kwargs:
            # We don't know all variables in the file without opening it,
            # but we can specify which ones we WANT if the engine supports it.
            # Most xarray engines don't have an 'only_variables' but have 'drop_variables'.
            # For now, we'll load everything and filter in preprocess,
            # or the user can pass drop_variables.
            pass

        ds = super().open_dataset(files, dates, variable_dict=variable_dict, **kwargs)

        # Filter to requested variables if variable_dict is provided
        if variable_dict is not None:
            # Keep variables in variable_dict
            vars_to_keep = list(variable_dict.keys())

            # Only filter data_vars, preserve ALL coordinates
            available_vars = [v for v in vars_to_keep if v in ds.data_vars]
            ds = ds[available_vars]

        # Update history
        ds = update_history(ds, "Read MODIS L2 data.")

        return ds


def modis_l2_preprocess(ds: xr.Dataset, variable_dict: dict = None) -> xr.Dataset:
    """
    Preprocess MODIS L2 dataset: standardize coords, apply scaling/clipping, and quality flags.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    variable_dict : dict, optional
        Metadata for variables.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize dimensions and coordinates
    # MODIS L2 uses 'Cell_Along_Swath' and 'Cell_Across_Swath'
    ds = standardize_satellite_coords(
        ds,
        y_dim=["Cell_Along_Swath", "Rows", "scanline"],
        x_dim=["Cell_Across_Swath", "Columns", "ground_pixel"],
    )

    # 2. Add time coordinate if Scan_Start_Time is present
    if "Scan_Start_Time" in ds.variables and "time" not in ds.coords:
        # Seconds since 1993-01-01 00:00:00 UTC
        epoch_1993 = pd.Timestamp("1993-01-01", tz="UTC")

        def _calc_time(s):
            return (epoch_1993.to_datetime64() + (s * 1e9).astype("timedelta64[ns]")).astype(
                "datetime64[ns]"
            )

        ds["time"] = xr.apply_ufunc(
            _calc_time,
            ds["Scan_Start_Time"],
            dask="parallelized",
            output_dtypes=["datetime64[ns]"],
        )
        ds = ds.set_coords("time")

    # 3. Apply variable_dict transformations (scale, minimum, maximum)
    if variable_dict:
        for varname, meta in variable_dict.items():
            if varname in ds.variables:
                if "scale" in meta:
                    ds[varname] = ds[varname] * meta["scale"]
                if "minimum" in meta:
                    ds[varname] = ds[varname].where(ds[varname] >= meta["minimum"])
                if "maximum" in meta:
                    ds[varname] = ds[varname].where(ds[varname] <= meta["maximum"])

                if "quality_flag" in meta:
                    # Store info for masking in next step
                    ds.attrs["_quality_flag_var"] = varname
                    ds.attrs["_quality_flag_thresh"] = meta["quality_flag"]

    # 4. Apply Quality Flag masking
    if "_quality_flag_var" in ds.attrs:
        q_var = ds.attrs["_quality_flag_var"]
        q_thresh = ds.attrs["_quality_flag_thresh"]
        quality_flag = ds[q_var]
        for varname in ds.data_vars:
            if varname != q_var:
                ds[varname] = ds[varname].where(quality_flag < q_thresh)

        # Clean up temporary attrs
        del ds.attrs["_quality_flag_var"]
        del ds.attrs["_quality_flag_thresh"]

    ds = update_history(ds, "Preprocessed MODIS L2 data via Aero Protocol.")

    return ds
