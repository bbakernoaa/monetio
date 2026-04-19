"""UFS-AQM Reader"""

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .base import (
    GriddedReader,
    _convert_ugkg_to_ugm3,
    add_lazy_diagnostic,
    register_reader,
)
from .sat_utils import apply_lazy_conversion, update_history
from .ufs_specs import DIAGNOSTICS


@register_reader("ufs")
class UFSReader(GriddedReader):
    """
    Reader for UFS-AQM model output files.
    """

    def open_dataset(
        self,
        files: str | list[str],
        convert_to_ppb: bool = True,
        mech: str = "cb6r3_ae6_aq",
        var_list: list[str] | None = None,
        fname_pm25: str | list[str] | None = None,
        surf_only: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads UFS-AQM netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        convert_to_ppb : bool, optional
            Convert gas species from ppmV to ppbV, by default True.
        mech : str, optional
            Mechanism name for species sums, by default "cb6r3_ae6_aq".
        var_list : List[str], optional
            List of variables to include, by default None.
        fname_pm25 : Union[str, List[str]], optional
            Optional separate PM2.5 files to merge, by default None.
        surf_only : bool, optional
            Whether to only keep surface data, by default False.
        **kwargs : Any
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The processed UFS-AQM dataset.

        Examples
        --------
        >>> reader = UFSReader()
        >>> ds = reader.open_dataset("aqm.t12z.dyn.f*.nc", surf_only=True)
        """
        # Prepare kwargs
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        # 1. Open dataset
        ds = self.driver.open(files, **kwargs)

        # 2. Merge PM25 file if present
        if fname_pm25 is not None:
            ds_pm25 = self.driver.open(fname_pm25, **kwargs)
            ds_pm25 = ds_pm25.drop_vars(["lat", "lon", "pfull"], errors="ignore")
            ds_pm25.attrs = {}
            from monetio.util import _try_merge_exact

            ds = _try_merge_exact(ds, ds_pm25, right_name="PM2.5")

        # 3. Standardize Names
        rename_dict = {
            "grid_yt": "y",
            "grid_xt": "x",
            "pfull": "z",
            "phalf": "z_i",
            "lon": "longitude",
            "lat": "latitude",
            "tmp": "temperature_k",
            "pressfc": "surfpres_pa",
            "dpres": "dp_pa",
            "hgtsfc": "surfalt_m",
            "delz": "dz_m",
        }
        # Only rename what exists
        actual_rename = {k: v for k, v in rename_dict.items() if k in ds.variables or k in ds.dims}
        if actual_rename:
            ds = ds.rename(actual_rename)

        # 4. Calculations (Lazy)
        if "surfpres_pa" in ds and "ak" in ds and "bk" in ds:
            ds["pres_pa_mid"] = _calc_pressure(ds)

        # 5. Coordinate Sorting and Height calculation
        if "z" in ds.coords and ds.z.size > 1:
            # UFS is usually top-down, we want bottom-up
            is_ascending = ds.z[0] < ds.z[-1]
            if is_ascending:
                ds = ds.isel(z=slice(None, None, -1))
                if "dz_m" in ds:
                    with xr.set_options(keep_attrs=True):
                        ds["dz_m"] = ds["dz_m"] * -1.0
        if "z_i" in ds.coords and ds.z_i.size > 1:
            is_ascending = ds.z_i[0] < ds.z_i[-1]
            if is_ascending:
                ds = ds.isel(z_i=slice(None, None, -1))

        if not surf_only and "dz_m" in ds and "surfalt_m" in ds:
            ds["alt_msl_m_full"] = _calc_hgt(ds)

        # 6. Dimensions and Coordinates cleanup
        if "latitude" in ds.data_vars and "time" in ds["latitude"].dims:
            ds["latitude"] = ds["latitude"].isel(time=0)
        if "longitude" in ds.data_vars and "time" in ds["longitude"].dims:
            ds["longitude"] = ds["longitude"].isel(time=0)

        coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
        ds = ds.set_coords(coords)

        if surf_only and "z" in ds.dims:
            ds = ds.isel(z=[0])

        # 7. Unit conversion (Lazy)
        if convert_to_ppb:
            from .base import _convert_to_ppb

            ds = _convert_to_ppb(ds)

        # ug/kg to ug/m3
        ds = _convert_ugkg_to_ugm3(ds)

        # 8. Add all lazy diagnostics
        for name, spec in DIAGNOSTICS.items():
            ds = add_lazy_diagnostic(ds, name, spec)

        # 9. Time fix (Backend-agnostic, avoids eager .indexes)
        if "time" in ds.coords and ds.time.dtype == object:
            # Handle potential CFTimeIndex lazily
            def _to_dt64(t_arr):
                # Try to convert cftime to string then to datetime64
                return pd.to_datetime([str(t) for t in t_arr.ravel()]).values.reshape(t_arr.shape)

            ds["time"] = apply_lazy_conversion(ds["time"], _to_dt64, "datetime64[ns]")
            ds = update_history(ds, "Converted time from cftime to datetime64[ns] lazily.")

        # 10. Subset variables if requested
        if var_list is not None:
            # We must keep coordinates and some essentials
            essentials = [
                "latitude",
                "longitude",
                "time",
                "z",
                "z_i",
                "pres_pa_mid",
                "temperature_k",
            ]
            # Only keep requested variables and essentials.
            # add_lazy_diagnostic adds variables to the dataset,
            # so they will be kept if they are in var_list.
            to_keep = set(var_list) | set(essentials)
            available = [v for v in ds.variables if v in to_keep]
            ds = ds[available]

        # 11. Scientific Hygiene
        for var in ds.variables:
            for attr, val in ds[var].attrs.items():
                if isinstance(val, str):
                    ds[var].attrs[attr] = val.strip()

        # Update history
        ds = update_history(ds, "Read UFS-AQM data.")

        return ds


def dict_species_sums(mech: str) -> dict[str, list]:
    """
    Returns species groups for sums based on mechanism.
    (Deprecated: Use ufs_specs.py for new code).
    """
    from .ufs_specs import ACCUMULATION, AITKEN, COARSE

    if mech == "cb6r3_ae6_aq":
        sum_dict = {}
        sum_dict["accumulation"] = ACCUMULATION
        sum_dict["aitken"] = AITKEN
        sum_dict["coarse"] = COARSE
        sum_dict["noy_gas"] = DIAGNOSTICS["noy_gas"].variables
        sum_dict["noy_gas_weight"] = DIAGNOSTICS["noy_gas"].weights
        sum_dict["noy_aer"] = DIAGNOSTICS["noy_aer"].variables
        sum_dict["nox"] = DIAGNOSTICS["nox"].variables
        sum_dict["pm25_cl"] = DIAGNOSTICS["pm25_cl"].variables
        sum_dict["pm25_cl_weight"] = DIAGNOSTICS["pm25_cl"].weights
        sum_dict["pm25_ec"] = DIAGNOSTICS["pm25_ec"].variables
        sum_dict["pm25_ec_weight"] = DIAGNOSTICS["pm25_ec"].weights
        sum_dict["pm25_na"] = DIAGNOSTICS["pm25_na"].variables
        sum_dict["pm25_na_weight"] = DIAGNOSTICS["pm25_na"].weights
        sum_dict["pm25_ca"] = DIAGNOSTICS["pm25_ca"].variables
        sum_dict["pm25_ca_weight"] = DIAGNOSTICS["pm25_ca"].weights
        sum_dict["pm25_nh4"] = DIAGNOSTICS["pm25_nh4"].variables
        sum_dict["pm25_nh4_weight"] = DIAGNOSTICS["pm25_nh4"].weights
        sum_dict["pm25_no3"] = DIAGNOSTICS["pm25_no3"].variables
        sum_dict["pm25_no3_weight"] = DIAGNOSTICS["pm25_no3"].weights
        sum_dict["pm25_so4"] = DIAGNOSTICS["pm25_so4"].variables
        sum_dict["pm25_so4_weight"] = DIAGNOSTICS["pm25_so4"].weights
        sum_dict["pm25_om"] = DIAGNOSTICS["pm25_om"].variables

        return sum_dict
    else:
        raise NotImplementedError(f"Mechanism {mech} not supported")


def add_lazy_pm25(ds: xr.Dataset, dict_sum: dict | None = None) -> xr.Dataset:
    """Legacy wrapper for PM25 diagnostic."""
    return add_lazy_diagnostic(ds, "PM25", DIAGNOSTICS["PM25"])


def add_lazy_pm10(ds: xr.Dataset, dict_sum: dict | None = None) -> xr.Dataset:
    """Legacy wrapper for PM10 diagnostic."""
    return add_lazy_diagnostic(ds, "PM10", DIAGNOSTICS["PM10"])


def add_lazy_noy_g(ds: xr.Dataset, dict_sum: dict | None = None) -> xr.Dataset:
    """Legacy wrapper for noy_gas diagnostic."""
    return add_lazy_diagnostic(ds, "noy_gas", DIAGNOSTICS["noy_gas"])


def add_lazy_noy_a(ds: xr.Dataset, dict_sum: dict | None = None) -> xr.Dataset:
    """Legacy wrapper for noy_aer diagnostic."""
    return add_lazy_diagnostic(ds, "noy_aer", DIAGNOSTICS["noy_aer"])


def add_lazy_nox(ds: xr.Dataset, dict_sum: dict | None = None) -> xr.Dataset:
    """Legacy wrapper for nox diagnostic."""
    return add_lazy_diagnostic(ds, "nox", DIAGNOSTICS["nox"])


def _calc_pressure(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate mid-layer pressure from hybrid coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing ak, bk, and surfpres_pa.

    Returns
    -------
    xr.DataArray
        Pressure at layer mid-points in Pa.
    """
    psfc = ds.surfpres_pa
    ak = ds.ak
    bk = ds.bk

    if ak.size == ds.sizes.get("z", 0) + 1 or ak.size == ds.sizes.get("z_i", 0):
        # We have interface values
        p_interfaces_1 = ak[:-1] + psfc * bk[:-1]
        p_interfaces_2 = ak[1:] + psfc * bk[1:]
        # Logarithmic interpolation for mid-points
        # Guard against zero or negative pressure
        with xr.set_options(keep_attrs=True):
            # p_mid = (p2 - p1) / ln(p2/p1)
            # Use where to handle p1 == p2 to avoid division by zero
            p_mid = xr.where(
                p_interfaces_1 == p_interfaces_2,
                p_interfaces_1,
                (p_interfaces_2 - p_interfaces_1) / np.log(p_interfaces_2 / p_interfaces_1),
            )
    else:
        # Fallback if ak/bk are already mid-points
        p_mid = ak + psfc * bk

    # The resulting dimension should be 'z' (midpoints)
    if "z_i" in p_mid.dims:
        p_mid = p_mid.rename({"z_i": "z"})

    p_mid.attrs.update({"units": "Pa", "long_name": "Pressure at layer mid-points"})
    return p_mid


def _calc_hgt(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate altitude MSL lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing dz_m and surfalt_m.

    Returns
    -------
    xr.DataArray
        Altitude above MSL in meters.
    """
    # Assuming dz_m is positive upwards and we already handled the flip
    alt = ds.dz_m.cumsum(dim="z") + ds.surfalt_m
    alt.attrs.update({"units": "m", "long_name": "Altitude above MSL"})
    return alt
