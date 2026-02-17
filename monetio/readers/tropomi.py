"""TROPOMI Reader"""

import datetime
import warnings
from typing import List, Optional, Union

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords


@register_reader("tropomi")
class TROPOMIReader(GriddedReader):
    """
    Reader for TROPOMI L2 (Sentinel-5P) data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        group: Optional[Union[str, List[str]]] = "PRODUCT",
        calculate_pressure: bool = True,
        qa_threshold: Optional[float] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads TROPOMI data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        group : str or list of str, optional
            The NetCDF group(s) to open. If a list is provided, groups will be merged.
            Common TROPOMI groups include:
            - "PRODUCT" (default)
            - "PRODUCT/SUPPORT_DATA/DETAILED_RESULTS"
            - "PRODUCT/SUPPORT_DATA/INPUT_DATA"
            - "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"
        calculate_pressure : bool, optional
            Whether to calculate pressure levels if necessary variables are found,
            by default True.
        qa_threshold : float, optional
            If provided, mask data where 'qa_value' is less than this threshold.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The TROPOMI dataset.

        Examples
        --------
        Open standard NO2 product with support data for pressure:
        >>> reader = TROPOMIReader()
        >>> ds = reader.open_dataset(
        ...     files="S5P_OFFL_L2_NO2_*.nc",
        ...     group=[
        ...         "PRODUCT",
        ...         "PRODUCT/SUPPORT_DATA/INPUT_DATA",
        ...         "PRODUCT/SUPPORT_DATA/GEOLOCATIONS"
        ...     ],
        ...     qa_threshold=0.75
        ... )
        """
        if "preprocess" not in kwargs:
            from functools import partial

            kwargs["preprocess"] = partial(
                tropomi_preprocess, calculate_pressure=calculate_pressure, qa_threshold=qa_threshold
            )

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        if group is None or isinstance(group, str):
            groups = [group] if group else [None]
        else:
            groups = group

        dsets = []
        for g in groups:
            # We copy kwargs to avoid modifying the original dict in the loop
            g_kwargs = kwargs.copy()
            g_kwargs["group"] = g
            try:
                ds_g = super().open_dataset(files, **g_kwargs)
                dsets.append(ds_g)
            except Exception as e:
                warnings.warn(f"Could not open group {g}: {e}")

        if not dsets:
            raise RuntimeError("No groups could be opened.")

        # Merge groups
        # We use compat='override' or 'no_conflicts' because coordinates should be identical
        ds = xr.merge(dsets, compat="no_conflicts")

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read TROPOMI data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def tropomi_preprocess(
    ds: xr.Dataset, calculate_pressure: bool = True, qa_threshold: Optional[float] = None
) -> xr.Dataset:
    """
    Preprocess TROPOMI dataset: standardize coordinates, handle time, and
    optionally calculate pressure and apply quality flags.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset from a single file/group (or merged groups).
    calculate_pressure : bool, optional
        Whether to calculate pressure levels.
    qa_threshold : float, optional
        Quality value threshold for masking.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize dimensions and coordinates
    # TROPOMI uses 'scanline' and 'ground_pixel'
    ds = standardize_satellite_coords(ds, lat_name="latitude", lon_name="longitude")

    # 2. Handle Time
    if "time" in ds.coords and "delta_time" in ds.data_vars:
        ref_time = ds.coords["time"]
        delta_time = ds.data_vars["delta_time"]
        if "y" in delta_time.dims:
            scan_time = ref_time + delta_time.astype("timedelta64[ms]")
            ds = ds.assign_coords(time=scan_time)

    # 3. Calculate Pressure (Lazy) - must happen before dim rename if it depends on 'y'
    if calculate_pressure:
        ds = _add_pressure_levels(ds)

    if "time" in ds.coords and "time" not in ds.dims:
        if ds.coords["time"].dims == ("y",):
            ds = ds.swap_dims({"y": "time"})

    # Ensure all data variables have 'time' dimension if it exists and they have 'y'
    if "time" in ds.dims:
        for var in ds.data_vars:
            if "y" in ds[var].dims:
                ds[var] = ds[var].rename({"y": "time"})

    # 5. Quality Flagging (Lazy)
    if qa_threshold is not None and "qa_value" in ds.data_vars:
        # Mask all data variables where qa_value < threshold
        # We exclude coordinates and the qa_value itself from masking
        qa = ds["qa_value"]
        for var in ds.data_vars:
            if var != "qa_value":
                ds[var] = ds[var].where(qa >= qa_threshold)

    return ds


def _add_pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate mid-layer and interface pressure levels for TROPOMI lazily.
    Supports NO2/HCHO (TM5) and CO style pressure definitions.
    """
    # NO2/HCHO style (TM5 constant a, b)
    if all(v in ds.data_vars for v in ["tm5_constant_a", "tm5_constant_b", "surface_pressure"]):
        a = ds["tm5_constant_a"]
        b = ds["tm5_constant_b"]
        ps = ds["surface_pressure"]

        # Interface pressure: p_int = a + b * ps
        p_int = a + b * ps
        # p_int now has dims (z, vertices, y, x) or similar

        # Mid-layer pressure: (p_bottom + p_top) / 2
        p_mid = p_int.mean(dim="vertices") if "vertices" in p_int.dims else p_int.mean(dim="v")

        ds["pres_pa_mid"] = p_mid.assign_attrs({"units": "Pa", "long_name": "mid-layer pressure"})

        # If tm5_tropopause_layer_index is present, calculate tropopause pressure
        if "tm5_tropopause_layer_index" in ds.data_vars:
            itrop = ds["tm5_tropopause_layer_index"]
            try:
                # Ensure itrop is within bounds and integer
                itrop_valid = itrop.where((itrop >= 0) & (itrop < ds.sizes["z"]), 0).astype(int)

                if hasattr(itrop_valid.data, "dask"):
                    # Dask path: pick from z dimension lazily
                    def _index_3d(arr, idx):
                        return np.take_along_axis(arr, idx[np.newaxis, ...], axis=0).squeeze(axis=0)

                    ds["troppres"] = xr.apply_ufunc(
                        _index_3d,
                        p_mid.chunk({"z": -1}),
                        itrop_valid,
                        input_core_dims=[["z"], []],
                        output_core_dims=[[]],
                        dask="parallelized",
                        output_dtypes=[p_mid.dtype],
                    )
                else:
                    # Eager path: use standard Xarray indexing
                    ds["troppres"] = p_mid.isel(z=itrop_valid)
                    if "z" in ds["troppres"].coords:
                        ds["troppres"] = ds["troppres"].drop_vars("z")
                ds["troppres"].attrs.update({"units": "Pa", "long_name": "tropopause pressure"})
            except Exception as e:
                warnings.warn(f"Could not calculate tropopause pressure: {e}")

    # CO style
    elif "pressure_levels" in ds.data_vars:
        p_int = ds["pressure_levels"]
        # Interface pressure is directly provided
        # Mid-layer pressure is average of adjacent interfaces
        # We can use rolling mean if the dimension order is correct
        p_mid = p_int.rolling(z=2).mean().dropna("z")
        ds["pres_pa_mid"] = p_mid.assign_attrs({"units": "Pa", "long_name": "mid-layer pressure"})

    return ds
