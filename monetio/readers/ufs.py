"""UFS-AQM Reader"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("ufs")
class UFSReader(GriddedReader):
    """
    Reader for UFS-AQM model output files.
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
        dict_sum = dict_species_sums(mech=mech)

        list_calc_sum = []
        list_remove_extra_only = []

        # Prepare kwargs
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

            # Append other needed species
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

            # Remove standard names if present
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

            # Remove pm25 vars if present
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
        ds = self.driver.open(files, **kwargs)

        # Subset if var_list
        if var_list is not None:
            # Only keep available vars
            available = [v for v in var_list if v in ds.variables]
            ds = ds[available]

        # Merge PM25 file if present
        if fname_pm25 is not None:
            ds_pm25 = self.driver.open(fname_pm25, **kwargs)
            ds_pm25 = ds_pm25.drop_vars(["lat", "lon", "pfull"], errors="ignore")
            ds_pm25.attrs = {}
            from monetio.util import _try_merge_exact

            ds = _try_merge_exact(ds, ds_pm25, right_name="PM2.5")

        # Standardize
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

        # Calculations
        if "surfpres_pa" in ds and "ak" in ds and "bk" in ds:
            ds["pres_pa_mid"] = _calc_pressure(ds)

        # Resort z (Lazy)
        if "z" in ds.coords:
            if ds.z.size > 1:
                is_ascending = ds.z[0] < ds.z[-1]
                if is_ascending:
                    ds = ds.isel(z=slice(None, None, -1))
                    if "dz_m" in ds:
                        ds["dz_m"] = ds["dz_m"] * -1.0
        if "z_i" in ds.coords:
            if ds.z_i.size > 1:
                is_ascending = ds.z_i[0] < ds.z_i[-1]
                if is_ascending:
                    ds = ds.isel(z_i=slice(None, None, -1))

        if not surf_only and "dz_m" in ds and "surfalt_m" in ds:
            ds["alt_msl_m_full"] = _calc_hgt(ds)

        if "latitude" in ds.data_vars and "time" in ds["latitude"].dims:
            ds["latitude"] = ds["latitude"].isel(time=0)
        if "longitude" in ds.data_vars and "time" in ds["longitude"].dims:
            ds["longitude"] = ds["longitude"].isel(time=0)

        coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
        ds = ds.set_coords(coords)

        if surf_only and "z" in ds.dims:
            ds = ds.isel(z=0).expand_dims("z", axis=1)

        # Unit conversion (Lazy)
        ds = _convert_units(ds, convert_to_ppb=convert_to_ppb)

        # Lazy diagnostics
        ds = _add_all_diagnostics(ds, list_calc_sum, dict_sum)

        # Time fix (Avoid eager .indexes)
        if "time" in ds.coords:
            if ds.indexes["time"].__class__.__name__ == "CFTimeIndex":
                ds["time"] = ds.indexes["time"].to_datetimeindex()

        if var_list is not None and bool(list_remove_extra_only):
            ds = ds.drop_vars(list_remove_extra_only, errors="ignore")

        # Scientific Hygiene
        for var in ds.variables:
            for attr, val in ds[var].attrs.items():
                if isinstance(val, str):
                    ds[var].attrs[attr] = val.strip()

        # Update history
        ds = update_history(ds, "Read UFS-AQM data.")

        return ds


def dict_species_sums(mech: str) -> Dict[str, List]:
    """
    Returns species groups for sums based on mechanism.

    Parameters
    ----------
    mech : str
        Mechanism name.

    Returns
    -------
    Dict[str, List]
        Dictionary of species groups.

    Examples
    --------
    >>> groups = dict_species_sums("cb6r3_ae6_aq")
    """
    if mech == "cb6r3_ae6_aq":
        sum_dict = {}
        sum_dict["accumulation"] = [
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
        sum_dict["aitken"] = [
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
        sum_dict["coarse"] = ["asoil", "acors", "aseacat", "aclk", "aso4k", "ano3k", "anh4k"]
        sum_dict["noy_gas"] = [
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
        ]
        sum_dict["noy_gas_weight"] = [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        sum_dict["noy_aer"] = ["ano3i", "ano3j", "ano3k"]
        sum_dict["nox"] = ["no", "no2"]
        sum_dict["pm25_cl"] = ["acli", "aclj", "aclk"]
        sum_dict["pm25_cl_weight"] = [1, 1, 0.2]
        sum_dict["pm25_ec"] = ["aeci", "aecj"]
        sum_dict["pm25_ec_weight"] = [1, 1]
        sum_dict["pm25_na"] = ["anai", "anaj", "aseacat", "asoil", "acors"]
        sum_dict["pm25_na_weight"] = [1, 1, 0.2 * 0.8373, 0.2 * 0.0626, 0.2 * 0.0023]
        sum_dict["pm25_ca"] = ["acaj", "aseacat", "asoil", "acors"]
        sum_dict["pm25_ca_weight"] = [1, 0.2 * 0.0320, 0.2 * 0.0838, 0.2 * 0.0562]
        sum_dict["pm25_nh4"] = ["anh4i", "anh4j", "anh4k"]
        sum_dict["pm25_nh4_weight"] = [1, 1, 0.2]
        sum_dict["pm25_no3"] = ["ano3i", "ano3j", "ano3k"]
        sum_dict["pm25_no3_weight"] = [1, 1, 0.2]
        sum_dict["pm25_so4"] = ["aso4i", "aso4j", "aso4k"]
        sum_dict["pm25_so4_weight"] = [1, 1, 0.2]
        sum_dict["pm25_om"] = sum_dict["accumulation"][15:] + sum_dict["aitken"][7:]

        return sum_dict
    else:
        raise NotImplementedError(f"Mechanism {mech} not supported")


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

    Examples
    --------
    >>> p_mid = _calc_pressure(ds)
    """
    psfc = ds.surfpres_pa
    ak = ds.ak
    bk = ds.bk

    if ak.size == ds.sizes["z"] + 1:
        p_interfaces_1 = ak[:-1] + psfc * bk[:-1]
        p_interfaces_2 = ak[1:] + psfc * bk[1:]
    else:
        # Fallback
        p_interfaces_1 = ak + psfc * bk
        p_interfaces_2 = p_interfaces_1

    p_mid = (p_interfaces_2 - p_interfaces_1) / np.log(p_interfaces_2 / p_interfaces_1)

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

    Examples
    --------
    >>> alt = _calc_hgt(ds)
    """
    # Assuming dz_m is positive upwards or we already flipped it
    alt = ds.dz_m.cumsum(dim="z") + ds.surfalt_m
    alt.attrs.update({"units": "m", "long_name": "Altitude above MSL"})
    return alt


def _convert_units(ds: xr.Dataset, convert_to_ppb: bool = True) -> xr.Dataset:
    """
    Convert units lazily.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    convert_to_ppb : bool, optional
        Whether to convert ppmV to ppbv, by default True.

    Returns
    -------
    xr.Dataset
        Dataset with converted units.

    Examples
    --------
    >>> ds = _convert_units(ds)
    """
    converted = False
    for i in ds.data_vars:
        if "units" in ds[i].attrs:
            units = ds[i].attrs["units"].lower()
            if convert_to_ppb and "ppmv" in units:
                ds[i] = ds[i] * 1000.0
                ds[i].attrs["units"] = "ppbv"
                converted = True
            if "ug/kg" in units:
                if "pres_pa_mid" in ds and "temperature_k" in ds:
                    # Density rho = P / (R * T)
                    # ug/m3 = ug/kg * rho = ug/kg * P / (R * T)
                    ds[i] = ds[i] * ds["pres_pa_mid"] / (ds["temperature_k"] * 287.05535)
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"
                    converted = True

    if converted:
        ds = update_history(ds, "Converted units.")
    return ds


def _add_all_diagnostics(ds: xr.Dataset, list_calc_sum: List[str], dict_sum: Dict) -> xr.Dataset:
    """
    Adds all requested lazy diagnostics.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    list_calc_sum : List[str]
        List of diagnostic names to calculate.
    dict_sum : Dict
        Dictionary of species groups and weights.

    Returns
    -------
    xr.Dataset
        Dataset with added diagnostics.

    Examples
    --------
    >>> ds = _add_all_diagnostics(ds, ["PM25"], dict_sum)
    """
    if "PM25" in list_calc_sum:
        ds = add_lazy_pm25(ds, dict_sum)
    if "PM10" in list_calc_sum:
        ds = add_lazy_pm10(ds, dict_sum)
    if "noy_gas" in list_calc_sum:
        ds = add_lazy_noy_g(ds, dict_sum)
    if "noy_aer" in list_calc_sum:
        ds = add_lazy_noy_a(ds, dict_sum)
    if "nox" in list_calc_sum:
        ds = add_lazy_nox(ds, dict_sum)
    # Add other common ones
    for var in [
        "pm25_cl",
        "pm25_ec",
        "pm25_ca",
        "pm25_na",
        "pm25_nh4",
        "pm25_no3",
        "pm25_so4",
        "pm25_om",
    ]:
        if var in list_calc_sum:
            ds = add_lazy_diagnostic_generic(ds, var, dict_sum)

    return ds


def add_multiple_lazy2(
    ds: xr.Dataset, variables: List[str], weights: Optional[List[float]] = None
) -> Optional[xr.DataArray]:
    """
    Sums multiple variables lazily with optional weights.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    variables : List[str]
        List of variable names to sum.
    weights : List[float], optional
        List of weights for each variable, by default None.

    Returns
    -------
    Optional[xr.DataArray]
        The lazy sum DataArray, or None if no variables are available.

    Examples
    --------
    >>> res = add_multiple_lazy2(ds, ["var1", "var2"], weights=[1.0, 0.5])
    """
    available = [v for v in variables if v in ds.data_vars]
    if not available:
        return None

    if weights is not None:
        weight_map = dict(zip(variables, weights))
        new = ds[available[0]] * weight_map[available[0]]
        for i in range(1, len(available)):
            new = new + ds[available[i]] * weight_map[available[i]]
    else:
        new = ds[available[0]]
        for i in range(1, len(available)):
            new = new + ds[available[i]]
    return new


def add_lazy_pm25(ds: xr.Dataset, dict_sum: Dict) -> xr.Dataset:
    """
    Adds PM2.5 lazy diagnostic.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with PM25 added.

    Examples
    --------
    >>> ds = add_lazy_pm25(ds, dict_sum)
    """
    vars_to_sum = dict_sum["aitken"] + dict_sum["accumulation"] + dict_sum["coarse"]
    weights = [1.0] * (len(dict_sum["aitken"]) + len(dict_sum["accumulation"])) + [0.2] * len(
        dict_sum["coarse"]
    )
    res = add_multiple_lazy2(ds, vars_to_sum, weights=weights)
    if res is not None:
        ds["PM25"] = res.assign_attrs({"name": "PM2.5", "units": r"$\mu g m^{-3}$"})
    return ds


def add_lazy_pm10(ds: xr.Dataset, dict_sum: Dict) -> xr.Dataset:
    """
    Adds PM10 lazy diagnostic.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with PM10 added.

    Examples
    --------
    >>> ds = add_lazy_pm10(ds, dict_sum)
    """
    vars_to_sum = dict_sum["aitken"] + dict_sum["accumulation"] + dict_sum["coarse"]
    res = add_multiple_lazy2(ds, vars_to_sum)
    if res is not None:
        ds["PM10"] = res.assign_attrs({"name": "PM10", "units": r"$\mu g m^{-3}$"})
    return ds


def add_lazy_noy_g(ds: xr.Dataset, dict_sum: Dict) -> xr.Dataset:
    """
    Adds NOy gas lazy diagnostic.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with noy_gas added.

    Examples
    --------
    >>> ds = add_lazy_noy_g(ds, dict_sum)
    """
    res = add_multiple_lazy2(ds, dict_sum["noy_gas"], weights=dict_sum["noy_gas_weight"])
    if res is not None:
        ds["noy_gas"] = res.assign_attrs({"name": "NOy gas", "units": "ppbv"})
    return ds


def add_lazy_noy_a(ds: xr.Dataset, dict_sum: Dict) -> xr.Dataset:
    """
    Adds NOy aerosol lazy diagnostic.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with noy_aer added.

    Examples
    --------
    >>> ds = add_lazy_noy_a(ds, dict_sum)
    """
    res = add_multiple_lazy2(ds, dict_sum["noy_aer"])
    if res is not None:
        ds["noy_aer"] = res.assign_attrs({"name": "NOy aerosol", "units": r"$\mu g m^{-3}$"})
    return ds


def add_lazy_nox(ds: xr.Dataset, dict_sum: Dict) -> xr.Dataset:
    """
    Adds NOx lazy diagnostic.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with nox added.

    Examples
    --------
    >>> ds = add_lazy_nox(ds, dict_sum)
    """
    res = add_multiple_lazy2(ds, dict_sum["nox"])
    if res is not None:
        ds["nox"] = res.assign_attrs({"name": "NOx", "units": "ppbv"})
    return ds


def add_lazy_diagnostic_generic(ds: xr.Dataset, name: str, dict_sum: Dict) -> xr.Dataset:
    """
    Generic lazy diagnostic adder.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    name : str
        Name of the diagnostic.
    dict_sum : Dict
        Species dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with added diagnostic.

    Examples
    --------
    >>> ds = add_lazy_diagnostic_generic(ds, "pm25_so4", dict_sum)
    """
    vars_to_sum = dict_sum.get(name)
    weights = dict_sum.get(f"{name}_weight")
    if vars_to_sum:
        res = add_multiple_lazy2(ds, vars_to_sum, weights=weights)
        if res is not None:
            ds[name] = res.assign_attrs({"name": name})
    return ds
