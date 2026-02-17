"""UFS-AQM Reader"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("ufs")
class UFSReader(GriddedReader):
    """
    Reader for UFS-AQM NetCDF files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        convert_to_ppb: bool = True,
        mech: str = "cb6r3_ae6_aq",
        var_list: Optional[List[str]] = None,
        fname_pm25: Optional[Union[str, List[str]]] = None,
        surf_only: bool = False,
        **kwargs: Any,
    ) -> xr.Dataset:
        """
        Reads UFS-AQM NetCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) to read.
        convert_to_ppb : bool, optional
            Whether to convert gas species from ppmv to ppbv, by default True.
        mech : str, optional
            Chemical mechanism name, by default "cb6r3_ae6_aq".
        var_list : List[str], optional
            List of variables to keep. If None, all variables are kept.
        fname_pm25 : Union[str, List[str]], optional
            Optional separate PM2.5 file(s) to merge.
        surf_only : bool, optional
            Whether to only load the surface level, by default False.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open (e.g., chunks).

        Returns
        -------
        xr.Dataset
            The loaded UFS-AQM dataset.
        """
        dict_sum = dict_species_sums(mech=mech)

        list_calc_sum = []
        list_remove_extra_only = []

        # Prepare kwargs for mfdataset
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"

        # Handling var_list logic
        if var_list is not None:
            var_list_orig = var_list.copy()
            list_remove_extra = []

            for var_sum in [
                "PM25",
                "PM10",
                "noy_gas",
                "noy_aer",
                "nox",
                "pm25_cl",
                "pm25_ec",
                "pm25_ca",
                "pm25_na",
                "pm25_nh4",
                "pm25_no3",
                "pm25_so4",
                "pm25_om",
            ]:
                if var_sum in var_list:
                    if var_sum == "PM25" or var_sum == "PM10":
                        comps = dict_sum["aitken"] + dict_sum["accumulation"] + dict_sum["coarse"]
                        var_list.extend(comps)
                        list_remove_extra.extend(comps)
                    else:
                        var_list.extend(dict_sum[var_sum])
                        list_remove_extra.extend(dict_sum[var_sum])

                    if var_sum in var_list:
                        var_list.remove(var_sum)
                    list_calc_sum.append(var_sum)

            # Append other needed species for calculations
            needed = [
                "lat",
                "lon",
                "phalf",
                "tmp",
                "pressfc",
                "dpres",
                "hgtsfc",
                "delz",
                "ak",
                "bk",
            ]
            var_list.extend(needed)

            # Remove standard names if present (we'll add them later)
            for vn in [
                "temperature_k",
                "surfpres_pa",
                "dp_pa",
                "surfalt_m",
                "dz_m",
                "pres_pa_mid",
            ]:
                if vn in var_list:
                    var_list.remove(vn)

            var_list = list(dict.fromkeys(var_list))
            list_remove_extra = list(dict.fromkeys(list_remove_extra))
            list_remove_extra_only = list(set(list_remove_extra) - set(var_list_orig))

            # Remove internal pm25 vars if present
            pm25_vars = [
                "PM25_TOT",
                "PM25_TOT_NSOM",
                "PM25_EC",
                "PM25_NH4",
                "PM25_NO3",
                "PM25_SO4",
                "PM25_OC",
                "PM25_OM",
            ]
            for v in pm25_vars:
                if v in var_list:
                    var_list.remove(v)
        else:
            list_calc_sum = [
                "PM25",
                "PM10",
                "noy_gas",
                "noy_aer",
                "nox",
                "pm25_cl",
                "pm25_ec",
                "pm25_ca",
                "pm25_na",
                "pm25_nh4",
                "pm25_no3",
                "pm25_so4",
                "pm25_om",
            ]

        # Open dataset
        dset = self.driver.open(files, **kwargs)

        # Subset if var_list
        if var_list is not None:
            available = [v for v in var_list if v in dset.variables]
            dset = dset[available]

        # Merge PM25 file if present
        if fname_pm25 is not None:
            dset_pm25 = self.driver.open(fname_pm25, **kwargs)
            dset_pm25 = dset_pm25.drop_vars(["lat", "lon", "pfull"], errors="ignore")
            # Clear attrs to avoid conflicts during merge
            dset_pm25.attrs = {}
            from monetio.util import _try_merge_exact

            dset = _try_merge_exact(dset, dset_pm25, right_name="PM2.5")

        # Standardize Naming
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
        rename_dict = {
            k: v for k, v in rename_dict.items() if k in dset.variables or k in dset.dims
        }
        dset = dset.rename(rename_dict)

        # Calculations
        if "surfpres_pa" in dset and "ak" in dset and "bk" in dset:
            dset["pres_pa_mid"] = _calc_pressure(dset)

        # Resort z (Restore original sorting logic but keep it efficient)
        if "z" in dset.coords:
            z_vals = dset.z.values
            if z_vals.size > 1 and np.all(np.diff(z_vals) > 0):
                dset = dset.isel(z=slice(None, None, -1))
                if "dz_m" in dset:
                    dset["dz_m"] = dset["dz_m"] * -1.0
        if "z_i" in dset.coords:
            zi_vals = dset.z_i.values
            if zi_vals.size > 1 and np.all(np.diff(zi_vals) > 0):
                dset = dset.isel(z_i=slice(None, None, -1))

        if not surf_only and "dz_m" in dset and "surfalt_m" in dset:
            dset["alt_msl_m_full"] = _calc_hgt(dset)

        # Set standard coordinates
        if "latitude" in dset.data_vars and "time" in dset["latitude"].dims:
            dset["latitude"] = dset["latitude"].isel(time=0, drop=True)
        if "longitude" in dset.data_vars and "time" in dset["longitude"].dims:
            dset["longitude"] = dset["longitude"].isel(time=0, drop=True)

        coords_to_set = [c for c in ["latitude", "longitude"] if c in dset.data_vars]
        dset = dset.set_coords(coords_to_set)

        if surf_only and "z" in dset.dims:
            # Maintain dimension as requested in code review
            dset = dset.isel(z=[0])

        # Unit conversions
        if convert_to_ppb:
            for var in dset.data_vars:
                if "units" in dset[var].attrs and "ppmv" in dset[var].attrs["units"]:
                    # In-place modification of DataArray attributes
                    dset[var] = dset[var] * 1000.0
                    dset[var].attrs["units"] = "ppbv"

        # Conversion to ug/m3 for aerosols if requested/needed
        for var in dset.data_vars:
            if "units" in dset[var].attrs and "ug/kg" in dset[var].attrs["units"]:
                if "pres_pa_mid" in dset and "temperature_k" in dset:
                    # Ideal gas law conversion: ug/kg * rho = ug/m3
                    # rho = P / (R * T)
                    dset[var] = dset[var] * dset["pres_pa_mid"] / dset["temperature_k"] / 287.05
                    dset[var].attrs["units"] = "ug m-3"

        # Lazy diagnostics
        if "PM25" in list_calc_sum:
            dset = add_lazy_pm25(dset, dict_sum)
        if "PM10" in list_calc_sum:
            dset = add_lazy_pm10(dset, dict_sum)
        if "noy_gas" in list_calc_sum:
            dset = add_lazy_noy_g(dset, dict_sum)
        if "noy_aer" in list_calc_sum:
            dset = add_lazy_noy_a(dset, dict_sum)
        if "nox" in list_calc_sum:
            dset = add_lazy_nox(dset, dict_sum)

        # Time fix (Restore original time fixing logic to match baseline dtypes)
        try:
            dset["time"] = dset.indexes["time"].to_datetimeindex(unsafe=True)
        except Exception:
            pass

        # Clean up extra variables
        if var_list is not None and bool(list_remove_extra_only):
            dset = dset.drop_vars(list_remove_extra_only, errors="ignore")

        # Update history
        history = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read UFS-AQM data."
        if "history" in dset.attrs:
            dset.attrs["history"] = f"{dset.attrs['history']}\n{history}"
        else:
            dset.attrs["history"] = history

        return dset


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def dict_species_sums(mech: str) -> Dict[str, List[str]]:
    """
    Get the species mapping for chemical mechanisms.

    Parameters
    ----------
    mech : str
        Chemical mechanism name.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of aggregate species to individual components.
    """
    if mech == "cb6r3_ae6_aq":
        sum_dict = {}
        sum_dict.update(
            {
                "accumulation": [
                    "aso4j",
                    "ano3j",
                    "anh4j",
                    "anaj",
                    "aclj",
                    "aecj",
                    "aothrj",
                    "afej",
                    "asij",
                    "atij",
                    "acaj",
                    "amgj",
                    "amnj",
                    "aalj",
                    "akj",
                    "alvpo1j",
                    "asvpo1j",
                    "asvpo2j",
                    "asvpo3j",
                    "aivpo1j",
                    "axyl1j",
                    "axyl2j",
                    "axyl3j",
                    "atol1j",
                    "atol2j",
                    "atol3j",
                    "abnz1j",
                    "abnz2j",
                    "abnz3j",
                    "aiso1j",
                    "aiso2j",
                    "aiso3j",
                    "atrp1j",
                    "atrp2j",
                    "asqtj",
                    "aalk1j",
                    "aalk2j",
                    "apah1j",
                    "apah2j",
                    "apah3j",
                    "aorgcj",
                    "aolgbj",
                    "aolgaj",
                    "alvoo1j",
                    "alvoo2j",
                    "asvoo1j",
                    "asvoo2j",
                    "asvoo3j",
                    "apcsoj",
                ]
            }
        )
        sum_dict.update(
            {
                "aitken": [
                    "aso4i",
                    "ano3i",
                    "anh4i",
                    "anai",
                    "acli",
                    "aeci",
                    "aothri",
                    "alvpo1i",
                    "asvpo1i",
                    "asvpo2i",
                    "alvoo1i",
                    "alvoo2i",
                    "asvoo1i",
                    "asvoo2i",
                ]
            }
        )
        sum_dict.update(
            {"coarse": ["asoil", "acors", "aseacat", "aclk", "aso4k", "ano3k", "anh4k"]}
        )
        sum_dict.update(
            {
                "noy_gas": [
                    "no",
                    "no2",
                    "no3",
                    "n2o5",
                    "hono",
                    "hno3",
                    "pna",
                    "cron",
                    "clno2",
                    "pan",
                    "panx",
                    "opan",
                    "ntr1",
                    "ntr2",
                    "intr",
                ],
                "noy_gas_weight": [1.0] * 15,  # Default weights
            }
        )
        # Update specific weights for NOy
        # n2o5 has 2 nitrogen atoms
        sum_dict["noy_gas_weight"][3] = 2.0

        sum_dict.update({"noy_aer": ["ano3i", "ano3j", "ano3k"]})
        sum_dict.update({"nox": ["no", "no2"]})
        sum_dict.update({"pm25_cl": ["acli", "aclj", "aclk"], "pm25_cl_weight": [1.0, 1.0, 0.2]})
        sum_dict.update({"pm25_ec": ["aeci", "aecj"], "pm25_ec_weight": [1.0, 1.0]})
        sum_dict.update(
            {
                "pm25_na": ["anai", "anaj", "aseacat", "asoil", "acors"],
                "pm25_na_weight": [1.0, 1.0, 0.16746, 0.01252, 0.00046],  # 0.2 * ratios
            }
        )
        # Ratio values from original code's multiplication
        sum_dict.update(
            {
                "pm25_ca": ["acaj", "aseacat", "asoil", "acors"],
                "pm25_ca_weight": [1.0, 0.0064, 0.01676, 0.01124],
            }
        )
        sum_dict.update(
            {"pm25_nh4": ["anh4i", "anh4j", "anh4k"], "pm25_nh4_weight": [1.0, 1.0, 0.2]}
        )
        sum_dict.update(
            {"pm25_no3": ["ano3i", "ano3j", "ano3k"], "pm25_no3_weight": [1.0, 1.0, 0.2]}
        )
        sum_dict.update(
            {"pm25_so4": ["aso4i", "aso4j", "aso4k"], "pm25_so4_weight": [1.0, 1.0, 0.2]}
        )
        sum_dict.update({"pm25_om": sum_dict["aitken"] + sum_dict["accumulation"]})

        return sum_dict
    else:
        raise NotImplementedError(f"Mechanism '{mech}' not supported.")


def _calc_pressure(dset: xr.Dataset) -> xr.DataArray:
    """
    Calculate mid-layer pressure from surface pressure and hybrid coordinates.

    Parameters
    ----------
    dset : xr.Dataset
        Dataset containing surfpres_pa, ak, and bk.

    Returns
    -------
    xr.DataArray
        Calculated mid-layer pressure.
    """
    psfc = dset.surfpres_pa
    ak = dset.ak
    bk = dset.bk

    # Hybrid coordinate calculation
    # P_interface = ak + bk * psfc
    # Robustly handle dimension names for interfaces vs mid-points
    z_dim = "z" if "z" in dset.dims else "pfull"

    # If ak/bk still have old names, they might not match renamed dset.dims
    ak_zdim = ak.dims[0]

    if ak.sizes[ak_zdim] == dset.sizes.get(z_dim, 0) + 1:
        p_int1 = (
            ak.isel({ak_zdim: slice(0, -1)}).drop_vars(ak_zdim)
            + bk.isel({ak_zdim: slice(0, -1)}).drop_vars(ak_zdim) * psfc
        )
        p_int2 = (
            ak.isel({ak_zdim: slice(1, None)}).drop_vars(ak_zdim)
            + bk.isel({ak_zdim: slice(1, None)}).drop_vars(ak_zdim) * psfc
        )
    else:
        # Fallback
        p_int1 = ak + bk * psfc
        p_int2 = p_int1

    # Logarithmic interpolation for mid-layer pressure
    eps = 1e-10
    p_mid = (p_int2 - p_int1) / np.log(np.maximum(p_int2, eps) / np.maximum(p_int1, eps))

    # Rename vertical dimension from interface to mid-point if needed
    if ak_zdim in p_mid.dims and ak_zdim != z_dim:
        p_mid = p_mid.rename({ak_zdim: z_dim})
        # If z is a coordinate in dset, we should ensure the coordinates match or are dropped
        if z_dim in dset.coords:
            p_mid = p_mid.assign_coords({z_dim: dset.coords[z_dim]})

    p_mid.name = "pres_pa_mid"
    p_mid.attrs["units"] = "Pa"
    return p_mid


def _calc_hgt(dset: xr.Dataset) -> xr.DataArray:
    """
    Calculate altitude above MSL from layer thickness and surface altitude.

    Parameters
    ----------
    dset : xr.Dataset
        Dataset containing dz_m and surfalt_m.

    Returns
    -------
    xr.DataArray
        Calculated altitude.
    """
    z_dim = "z" if "z" in dset.dims else "pfull"
    # Cumsum along z (assuming z starts from surface)
    z_alt = dset.dz_m.cumsum(dim=z_dim) + dset.surfalt_m
    z_alt.name = "alt_msl_m_full"
    z_alt.attrs["units"] = "m"
    return z_alt


def add_multiple_lazy2(
    dset: xr.Dataset, variables: List[str], weights: Optional[List[float]] = None
) -> xr.DataArray:
    """
    Sum multiple variables lazily with optional weights.

    Parameters
    ----------
    dset : xr.Dataset
        Input dataset.
    variables : List[str]
        Variables to sum.
    weights : List[float], optional
        Weights for each variable.

    Returns
    -------
    xr.DataArray
        Lazy sum of variables.
    """
    subset = dset[variables]
    if weights is not None:
        for i, var in enumerate(variables):
            subset[var] = subset[var] * weights[i]

    return subset.to_array(dim="variable").sum("variable")


# Lazy Adders
def add_lazy_pm25(d: xr.Dataset, dict_sum: Dict[str, List[str]]) -> xr.Dataset:
    """Add PM2.5 to dataset lazily."""
    allvars = dict_sum["aitken"] + dict_sum["accumulation"] + dict_sum["coarse"]
    weights = [1.0] * (len(dict_sum["aitken"]) + len(dict_sum["accumulation"])) + [0.2] * len(
        dict_sum["coarse"]
    )

    available = [v for v in allvars if v in d.data_vars]
    avail_weights = [weights[allvars.index(v)] for v in available]

    if available:
        d["PM25"] = add_multiple_lazy2(d, available, weights=avail_weights)
        d["PM25"].attrs.update({"long_name": "PM2.5", "units": "ug m-3"})
    return d


def add_lazy_pm10(d: xr.Dataset, dict_sum: Dict[str, List[str]]) -> xr.Dataset:
    """Add PM10 to dataset lazily."""
    allvars = dict_sum["aitken"] + dict_sum["accumulation"] + dict_sum["coarse"]
    available = [v for v in allvars if v in d.data_vars]

    if available:
        d["PM10"] = add_multiple_lazy2(d, available)
        d["PM10"].attrs.update({"long_name": "PM10", "units": "ug m-3"})
    return d


def add_lazy_noy_g(d: xr.Dataset, dict_sum: Dict[str, List[str]]) -> xr.Dataset:
    """Add NOy gas to dataset lazily."""
    allvars = dict_sum["noy_gas"]
    weights = dict_sum.get("noy_gas_weight", [1.0] * len(allvars))
    available = [v for v in allvars if v in d.data_vars]
    avail_weights = [weights[allvars.index(v)] for v in available]

    if available:
        d["noy_gas"] = add_multiple_lazy2(d, available, weights=avail_weights)
        d["noy_gas"].attrs.update({"long_name": "Reactive Nitrogen Gas", "units": "ppbv"})
    return d


def add_lazy_noy_a(d: xr.Dataset, dict_sum: Dict[str, List[str]]) -> xr.Dataset:
    """Add NOy aerosol to dataset lazily."""
    allvars = dict_sum["noy_aer"]
    available = [v for v in allvars if v in d.data_vars]

    if available:
        d["noy_aer"] = add_multiple_lazy2(d, available)
        d["noy_aer"].attrs.update({"long_name": "Reactive Nitrogen Aerosol", "units": "ug m-3"})
    return d


def add_lazy_nox(d: xr.Dataset, dict_sum: Dict[str, List[str]]) -> xr.Dataset:
    """Add NOx to dataset lazily."""
    allvars = dict_sum["nox"]
    available = [v for v in allvars if v in d.data_vars]

    if available:
        d["nox"] = add_multiple_lazy2(d, available)
        d["nox"].attrs.update({"long_name": "Nitrogen Oxides", "units": "ppbv"})
    return d
