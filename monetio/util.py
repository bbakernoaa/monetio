import datetime
from typing import Union

import numpy as np
import xarray as xr


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


def search_listinlist(array1, array2):
    import numpy as np

    # Find common elements and their indices
    # This vectorizes the search using np.isin which is significantly faster
    # than iterating over the intersection set and calling np.where in a loop.
    # It also correctly handles multidimensional arrays by flattening logic
    # implicit in np.isin for the check, but np.where acts on original shape.
    # We return the indices along the first dimension, matching original behavior.

    mask1 = np.isin(array1, array2)
    index1 = np.where(mask1)[0]

    mask2 = np.isin(array2, array1)
    index2 = np.where(mask2)[0]

    return np.sort(np.int32(index1)), np.sort(np.int32(index2))


def linregress(x, y):
    import numpy as np
    import statsmodels.api as sm

    xx = sm.add_constant(x)
    model = sm.OLS(y, xx)
    fit = model.fit()
    b, a = fit.params[0], fit.params[1]
    rsquared = fit.rsquared
    std_err = np.sqrt(fit.mse_resid)
    return a, b, rsquared, std_err


def findclosest(list, value):
    a = min((abs(x - value), x, i) for i, x in enumerate(list))
    return a[2], a[1]


def _force_forder(x):
    """
    Converts arrays x to fortran order. Returns
    a tuple in the form (x, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


def kolmogorov_zurbenko_filter(df, window, iterations):
    import pandas as pd

    """KZ filter implementation
        series is a pandas series
        window is the filter window m in the units of the data (m = 2q+1)
        iterations is the number of times the moving average is evaluated
        """
    z = df.copy()
    for i in range(iterations):
        z = pd.rolling_mean(z, window=window, min_periods=1, center=True)
    return z


def wsdir2uv(ws, wdir):
    from numpy import cos, pi, sin

    u = -ws * sin(wdir * pi / 180.0)
    v = -ws * cos(wdir * pi / 180.0)
    return u, v


def long_to_wide(df):
    """
    Convert a long-format DataFrame (or Dask DataFrame) to wide format.

    Parameters
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        The input DataFrame in long format, containing 'time', 'siteid', 'variable', 'obs', and 'units'.

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        The DataFrame in wide format.
    """
    try:
        import dask.dataframe as dd

        is_dask = isinstance(df, dd.DataFrame)
    except ImportError:
        is_dask = False

    if is_dask:
        # Dask doesn't support multi-index pivot_table well and requires categories.
        # To remain lazy, we should avoid computing here if possible.
        # However, many parts of MONETIO expect a wide DataFrame before Xarray conversion.
        # For now, we keep the compute but make it explicit that it's a bottleneck.
        import warnings

        warnings.warn(
            "long_to_wide: Computing dask dataframe to perform pivot_table. "
            "Consider using as_xarray=True with lazy=True to avoid this.",
            UserWarning,
        )
        df = df.compute()

    # Pivot the data
    w = df.pivot_table(values="obs", index=["time", "siteid"], columns="variable").reset_index()

    # Add units (columns)
    # We do this in a vectorized way to be faster
    if not w.empty:
        # Get unique variable/unit pairs
        units_map = df[["variable", "units"]].drop_duplicates()
        # If there are multiple units for one variable, we take the first
        units_map = units_map.drop_duplicates(subset=["variable"])
        for _, row in units_map.iterrows():
            w[f"{row.variable}_unit"] = row.units

    # Get site info to add, allowing for possible time variation
    # We drop 'variable', 'obs', 'units' which are handled by the pivot/units_map
    site_info = df.drop(columns=["variable", "obs", "units"], errors="ignore").drop_duplicates()

    return w.merge(site_info, on=["time", "siteid"], how="left")


def calc_8hr_rolling_max(df, col=None, window=None):
    df.index = df.time_local
    df_rolling = (
        df.groupby("siteid")[col]
        .rolling(window, center=True, win_type="boxcar")
        .mean()
        .reset_index()
        .dropna()
    )
    df_rolling_max = (
        df_rolling.groupby("siteid").resample("D", on="time_local").max().reset_index(drop=True)
    )
    df = df.reset_index(drop=True)
    return df.merge(df_rolling_max, on=["siteid", "time_local"])


def calc_24hr_ave(df, col=None):
    df.index = df.time_local
    df_24hr_ave = df.groupby("siteid")[col].resample("D").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])


def calc_3hr_ave(df, col=None):
    df.index = df.time_local
    df_3hr_ave = df.groupby("siteid")[col].resample("3H").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_3hr_ave, on=["siteid", "time_local"])


def calc_annual_ave(df, col=None):
    df.index = df.time_local
    df_annual_ave = df.groupby("siteid")[col].resample("A").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_annual_ave, on=["siteid", "time_local"])


def get_giorgi_region_bounds(index=None, acronym=None):
    import pandas as pd

    i = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    acro = [
        "NAU",
        "SAU",
        "AMZ",
        "SSA",
        "CAM",
        "WNA",
        "CNA",
        "ENA",
        "ALA",
        "GRL",
        "MED",
        "NEU",
        "WAF",
        "EAF",
        "SAF",
        "SAH",
        "SEA",
        "EAS",
        "SAS",
        "CAS",
        "TIB",
        "NAS",
    ]
    lonmax = [
        155,
        155,
        -34,
        -40,
        -83,
        -103,
        -85,
        -60,
        -103,
        -10,
        40,
        40,
        22,
        52,
        52,
        65,
        155,
        145,
        100,
        75,
        100,
        180,
    ]
    lonmin = [
        110,
        110,
        -82,
        -76,
        -116,
        -130,
        -103,
        -85,
        -170,
        -103,
        -10,
        -10,
        -20,
        22,
        -10,
        -20,
        95,
        100,
        65,
        40,
        75,
        40,
    ]
    latmax = [
        -11,
        -28,
        12,
        -20,
        30,
        60,
        50,
        50,
        72,
        85,
        48,
        75,
        18,
        18,
        -12,
        30,
        20,
        50,
        30,
        50,
        50,
        70,
    ]
    latmin = [
        -28,
        -45,
        -20,
        -56,
        10,
        30,
        30,
        25,
        60,
        50,
        30,
        48,
        -12,
        -12,
        -35,
        18,
        -11,
        20,
        5,
        30,
        30,
        50,
    ]
    df = pd.DataFrame(
        {
            "latmin": latmin,
            "lonmin": lonmin,
            "latmax": latmax,
            "lonmax": lonmax,
            "acronym": acro,
        },
        index=i,
    )
    try:
        if index is None and acronym is None:
            print("either index or acronym needs to be supplied")
            print("look here https://web.northeastern.edu/sds/web/demsos/images_002/subregions.jpg")
            raise ValueError
        elif index is not None:
            return df.loc[df.index == index].values.flatten()
        else:
            return df.loc[df.acronym == acronym.upper()].values.flatten()
    except ValueError:
        exit


def get_giorgi_region_df(df):
    df.loc[:, "GIORGI_INDEX"] = None
    df.loc[:, "GIORGI_ACRO"] = None
    for i in range(22):
        latmin, lonmin, latmax, lonmax, acro = get_giorgi_region_bounds(index=int(i + 1))
        con = (
            (df.longitude <= lonmax)
            & (df.longitude >= lonmin)
            & (df.latitude <= latmax)
            & (df.latitude >= latmin)
        )
        df.loc[con, "GIORGI_INDEX"] = i + 1
        df.loc[con, "GIORGI_ACRO"] = acro
    return df


def calc_13_category_usda_soil_type(
    clay: Union[xr.DataArray, np.ndarray],
    sand: Union[xr.DataArray, np.ndarray],
    silt: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """Calculate the 13 category USDA soil type from clay, sand, and silt percentages.

    The categories are:
    0  -- WATER
    1  -- SAND
    2  -- LOAMY SAND
    3  -- SANDY LOAM
    4  -- SILT LOAM
    5  -- SILT
    6  -- LOAM
    7  -- SANDY CLAY LOAM
    8  -- SILTY CLAY LOAM
    9  -- CLAY LOAM
    10 -- SANDY CLAY
    11 -- SILTY CLAY
    12 -- CLAY

    Parameters
    ----------
    clay : xarray.DataArray or numpy.ndarray
        Percentage of clay (0-100).
    sand : xarray.DataArray or numpy.ndarray
        Percentage of sand (0-100).
    silt : xarray.DataArray or numpy.ndarray
        Percentage of silt (0-100).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        The 13-category USDA soil type.
    """

    def _logic(c, sa, si):
        # We use the reverse order of the original assignments to ensure correct priority
        # in np.select (first matching condition wins).
        condlist = [
            (c >= 40) & (sa <= 45) & (si < 40) & (c != 255),  # 12: CLAY
            (c >= 40) & (si >= 40) & (c != 255),  # 11: SILTY CLAY
            (c >= 35) & (sa > 45) & (c != 255),  # 10: SANDY CLAY
            (c >= 27) & (c < 40.0) & (sa > 20) & (sa <= 45) & (c != 255),  # 9: CLAY LOAM
            (c >= 27) & (c < 40.0) & (sa > 40) & (c != 255),  # 8: SILTY CLAY LOAM
            (c >= 20) & (c < 35) & (si < 28) & (sa > 45) & (c != 255),  # 7: SANDY CLAY LOAM
            (c >= 7) & (c < 27) & (si >= 28) & (si < 50) & (sa <= 52) & (c != 255),  # 6: LOAM
            (si >= 80) & (c < 12) & (c != 255),  # 5: SILT
            ((si >= 50) & (c >= 12) & (c < 27) & (c != 255))
            | ((si >= 50) & (si < 80) & (c < 12) & (c != 255)),  # 4: SILT LOAM
            ((c >= 7.0) & (c < 20) & (sa > 52) & (si + 2 * c >= 30) & (c != 255))
            | ((c < 7) & (si < 50) & (si + 2 * c >= 30) & (c != 255)),  # 3: SANDY LOAM
            (si + 1.5 * c >= 15.0) & (si + 1.5 * c < 30) & (c != 255),  # 2: LOAMY SAND
            (si + c * 1.5 < 15.0) & (c != 255),  # 1: SAND
        ]
        choicelist = [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        return np.select(condlist, choicelist, default=0.0)

    result = xr.apply_ufunc(
        _logic,
        clay,
        sand,
        silt,
        dask="parallelized",
        output_dtypes=[float],
    )

    if isinstance(result, xr.DataArray):
        history = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            "Calculated USDA soil type using Aero Protocol."
        )
        if "history" in result.attrs:
            result.attrs["history"] = f"{result.attrs['history']}\n{history}"
        else:
            result.attrs["history"] = history

    return result


_module_install_names = {
    # module: GH, PyPI, conda-forge
    "pyhdf": ("fhs/pyhdf", "pyhdf", "pyhdf"),
}


def _install_message(mod_name):
    if mod_name not in _module_install_names:
        return ""

    gh, pypi_name, cf_name = _module_install_names[mod_name]
    cf_ = f"Try installing from conda-forge using `conda install -c conda-forge {cf_name}`."

    return f"{cf_}"


def _import_required(mod_name: str):
    from importlib import import_module

    try:
        return import_module(mod_name)
    except ImportError as e:
        raise RuntimeError(
            f"importing required module '{mod_name}' failed. {_install_message(mod_name)}"
        ) from e


def get_nc_attrs(nc_obj):
    """Safe retrieval of attributes from both netCDF4 and h5netcdf."""
    if hasattr(nc_obj, "ncattrs"):
        return {a: nc_obj.getncattr(a) for a in nc_obj.ncattrs()}
    elif hasattr(nc_obj, "attrs"):
        return dict(nc_obj.attrs)
    return getattr(nc_obj, "__dict__", {})


def get_nc_var(dso, group_path, varname):
    """Safe retrieval of a variable from nested groups in both netCDF4 and h5netcdf."""
    # Handle list-like group_path
    if isinstance(group_path, list):
        group_path = group_path[0]

    if not group_path or group_path == "/":
        return dso.variables[varname]

    # Try direct access if supported (h5netcdf)
    # h5netcdf.legacyapi.Dataset and Group support this.
    full_path = f"/{group_path.strip('/')}/{varname}"
    try:
        # Check if we are dealing with h5netcdf
        if hasattr(dso, "_h5group") or "h5netcdf" in str(type(dso)):
            return dso[full_path]
    except (KeyError, TypeError, AttributeError):
        pass

    # Fallback to nested navigation (netCDF4)
    obj = dso
    for part in group_path.strip("/").split("/"):
        if part:
            if hasattr(obj, "groups") and part in obj.groups:
                obj = obj.groups[part]
            else:
                # Try accessing as an item (works for some objects)
                try:
                    obj = obj[part]
                except Exception:
                    # Last resort, try variables if part is the varname but it shouldn't be
                    if part == varname:
                        return obj.variables[varname]
                    raise
    return obj.variables[varname]


def get_nc_values(nc_var):
    """Safe retrieval of masked and scaled values from both netCDF4 and h5netcdf."""
    import numpy as np

    values = nc_var[:].squeeze()
    if not isinstance(values, np.ma.MaskedArray):
        # Handle manual masking/scaling for h5netcdf
        attrs = get_nc_attrs(nc_var)
        fill_value = attrs.get("_FillValue", attrs.get("missing_value"))
        if fill_value is not None:
            # Handle possible array-like fill_value
            if hasattr(fill_value, "__iter__") and not isinstance(fill_value, (str, bytes)):
                fill_value = fill_value[0]
            # Use masked_values for float precision tolerance
            values = np.ma.masked_values(values, fill_value, atol=1e-5, copy=False)

        scale_factor = attrs.get("scale_factor")
        add_offset = attrs.get("add_offset")
        if scale_factor is not None or add_offset is not None:
            sf = float(scale_factor) if scale_factor is not None else 1.0
            ao = float(add_offset) if add_offset is not None else 0.0
            values = values * sf + ao
    else:
        # NetCDF4 already masked
        # Sometimes netCDF4 doesn't mask INF values if they were meant to be FillValue
        # but the attribute didn't match perfectly.
        pass
    return values


def force_object_strings(df):
    """
    Force string columns to 'object' dtype to avoid nullable string issues in Pandas/Dask.

    Parameters
    ----------
    df : Union[pd.DataFrame, dd.DataFrame]
        Input dataframe.

    Returns
    -------
    Union[pd.DataFrame, dd.DataFrame]
        Dataframe with string columns cast to object.
    """
    import pandas as pd

    try:
        import dask.dataframe as dd

        is_dask = isinstance(df, dd.DataFrame)
    except ImportError:
        is_dask = False

    if is_dask:
        # For Dask, we use assign to ensure metadata is updated
        # and we explicitly cast to object.
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df = df.assign(**{col: df[col].astype(object)})
        return df
    else:
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(object)
        return df


def _try_merge_exact(left, right, *, right_name=None):
    """For two ``xr.Dataset``s, try ``left.merge(right, compat="equals", join="exact")``.
    If it fails, print informative debugging messages and re-raise.
    Otherwise, return the result.
    """
    import warnings

    if right_name is None:
        right_name = " "
    else:
        right_name = f" {right_name.strip()} "

    try:
        left = left.merge(right, compat="equals", join="exact")
    except ValueError as e:
        # Try to print more debug info
        import re

        name = r"(?P<name>[a-zA-Z0-9_]*)"

        m = None
        for regex in [
            rf"not equal along these coordinates \(dimensions\): '{name}'",
            # Older message, used up to at least 0.21.1:
            rf"indexes along dimension '{name}' are not equal",
        ]:
            m = re.search(regex, str(e))
            if m is not None:
                break
        if m is None:
            warnings.warn(
                f"Unexpected Exception message (expected to match {regex!r}): {e}",
                stacklevel=2,
            )
            raise
        else:
            vn = m.groupdict()["name"]
            print(f"self {vn!r}: dtype={left[vn].dtype}")
            print(left[vn])
            print(f"other {vn!r}: dtype={right[vn].dtype}")
            print(right[vn])
            raise ValueError(
                f"Unable to merge{right_name}due to issue matching coordinates. "
                "See debug messages above the traceback."
            ) from e
    else:
        return left
