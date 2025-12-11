import warnings
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr


def nearest(items: Any, pivot: float) -> Any:
    """Finds the item in the iterable closest to the pivot value.

    Args:
        items: An iterable of numbers.
        pivot: The value to compare against.

    Returns:
        The item in `items` closest to `pivot`.
    """
    return min(items, key=lambda x: abs(x - pivot))


def search_listinlist(array1: np.ndarray, array2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds indices of elements common to both arrays.

    Args:
        array1: First input array.
        array2: Second input array.

    Returns:
        A tuple containing two arrays:
        - Sorted indices in `array1` where elements are present in `array2`.
        - Sorted indices in `array2` where elements are present in `array1`.
    """
    # find intersections
    s1 = set(array1.flatten())
    s2 = set(array2.flatten())

    inter = list(s1.intersection(s2))

    # find the indexes in array1
    index1 = np.where(np.isin(array1, inter))[0]
    index2 = np.where(np.isin(array2, inter))[0]

    return np.sort(np.int32(index1)), np.sort(np.int32(index2))


def linregress(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Performs a linear regression.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        A tuple (slope, intercept, r_squared, std_err).
    """
    # Uses scipy.stats.linregress which is a dependency
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value**2, std_err


def findclosest(list_in: List[float], value: float) -> Tuple[int, float]:
    """Finds the index and value in a list closest to a given value.

    Args:
        list_in: A list of values.
        value: The target value.

    Returns:
        A tuple (index, closest_value).
    """
    a = min((abs(x - value), x, i) for i, x in enumerate(list_in))
    return a[2], a[1]


def _force_forder(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Converts array x to Fortran order.

    Args:
        x: Input array.

    Returns:
        A tuple (x_fortran, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


def kolmogorov_zurbenko_filter(
    df: Union[pd.DataFrame, pd.Series], window: int, iterations: int
) -> Union[pd.DataFrame, pd.Series]:
    """KZ filter implementation.

    Args:
        df: Input pandas DataFrame or Series.
        window: The filter window m in the units of the data (m = 2q+1).
        iterations: The number of times the moving average is evaluated.

    Returns:
        Filtered DataFrame or Series.
    """
    z = df.copy()
    for i in range(iterations):
        z = z.rolling(window=window, min_periods=1, center=True).mean()
    return z


def wsdir2uv(ws: Any, wdir: Any) -> Tuple[Any, Any]:
    """Converts wind speed and direction to U and V components.

    Args:
        ws: Wind speed.
        wdir: Wind direction in degrees.

    Returns:
        A tuple (u, v).
    """
    u = -ws * np.sin(wdir * np.pi / 180.0)
    v = -ws * np.cos(wdir * np.pi / 180.0)
    return u, v


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a long-format DataFrame to wide format.

    Args:
        df: Input DataFrame with columns 'time', 'siteid', 'variable', 'obs', 'units'.

    Returns:
        Wide-format DataFrame merged with site info.
    """
    w = df.pivot_table(values="obs", index=["time", "siteid"], columns="variable").reset_index()

    # Add units (columns)
    for name, group in df.groupby("variable"):
        units = group.units.unique().tolist()
        if len(units) > 1:
            print(f"warning: non-unique units found, {units!r}, taking first")
        w[f"{name}_unit"] = units[0]

    # Get site info to add, allowing for possible time variation
    site_info = df.drop(["variable", "obs", "units"], axis=1).drop_duplicates()

    return w.merge(site_info, on=["time", "siteid"], how="left")


def calc_8hr_rolling_max(
    df: pd.DataFrame, col: str = None, window: int = None
) -> pd.DataFrame:
    """Calculates 8-hour rolling max.

    Args:
        df: Input DataFrame.
        col: Column to calculate on.
        window: Window size.

    Returns:
        DataFrame with 8-hour rolling max.
    """
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


def calc_24hr_ave(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Calculates 24-hour average.

    Args:
        df: Input DataFrame.
        col: Column to calculate on.

    Returns:
        DataFrame with 24-hour average.
    """
    df.index = df.time_local
    df_24hr_ave = df.groupby("siteid")[col].resample("D").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_24hr_ave, on=["siteid", "time_local"])


def calc_3hr_ave(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Calculates 3-hour average.

    Args:
        df: Input DataFrame.
        col: Column to calculate on.

    Returns:
        DataFrame with 3-hour average.
    """
    df.index = df.time_local
    df_3hr_ave = df.groupby("siteid")[col].resample("3H").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_3hr_ave, on=["siteid", "time_local"])


def calc_annual_ave(df: pd.DataFrame, col: str = None) -> pd.DataFrame:
    """Calculates annual average.

    Args:
        df: Input DataFrame.
        col: Column to calculate on.

    Returns:
        DataFrame with annual average.
    """
    df.index = df.time_local
    df_annual_ave = df.groupby("siteid")[col].resample("A").mean().reset_index()
    df = df.reset_index(drop=True)
    return df.merge(df_annual_ave, on=["siteid", "time_local"])


def get_giorgi_region_bounds(
    index: Optional[int] = None, acronym: Optional[str] = None
) -> np.ndarray:
    """Gets the bounds for a Giorgi region.

    Args:
        index: Region index (1-22).
        acronym: Region acronym (e.g., 'NAU').

    Returns:
        Array of bounds [latmin, lonmin, latmax, lonmax].

    Raises:
        ValueError: If neither index nor acronym is provided, or if they are invalid.
    """
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
        {"latmin": latmin, "lonmin": lonmin, "latmax": latmax, "lonmax": lonmax, "acronym": acro},
        index=i,
    )

    if index is None and acronym is None:
        raise ValueError(
            "Either index or acronym needs to be supplied. "
            "See https://web.northeastern.edu/sds/web/demsos/images_002/subregions.jpg"
        )
    elif index is not None:
        result = df.loc[df.index == index].values.flatten()
    else:
        result = df.loc[df.acronym == acronym.upper()].values.flatten()

    if result.size == 0:
         raise ValueError(f"Region not found for index={index} or acronym={acronym}")

    return result


def get_giorgi_region_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Giorgi region index and acronym to the DataFrame based on lat/lon.

    Args:
        df: Input DataFrame with 'latitude' and 'longitude' (or 'lat'/'lon').

    Returns:
        DataFrame with new columns 'GIORGI_INDEX' and 'GIORGI_ACRO'.
    """
    df.loc[:, "GIORGI_INDEX"] = None
    df.loc[:, "GIORGI_ACRO"] = None
    for i in range(22):
        latmin, lonmin, latmax, lonmax, acro = get_giorgi_region_bounds(index=int(i + 1))
        # Ensure we use correct column names if they differ
        # Assuming lat/lon exists as per usage
        if "longitude" in df.columns:
            lon = df.longitude
            lat = df.latitude
        else:
             # Fallback or assume user ensures columns exist
            lon = df.lon
            lat = df.lat

        con = (
            (lon <= lonmax)
            & (lon >= lonmin)
            & (lat <= latmax)
            & (lat >= latmin)
        )
        df.loc[con, "GIORGI_INDEX"] = i + 1
        df.loc[con, "GIORGI_ACRO"] = acro
    return df


def calc_13_category_usda_soil_type(
    clay: np.ndarray, sand: np.ndarray, silt: np.ndarray
) -> np.ndarray:
    """Calculate the 13 category usda soil type from the clay sand and silt

    0 -- WATER
    1 -- SAND
    2 -- LOAMY SAND
    3 -- SANDY LOAM
    4 -- SILT LOAM
    5 -- SILT
    6 -- LOAM
    7 -- SANDY CLAY LOAM
    8 -- SILTY CLAY LOAM
    9 -- CLAY LOAM
    10 --SANDY CLAY
    11 --SILY CLAY
    12 --CLAY

    Args:
        clay: Clay content array.
        sand: Sand content array.
        silt: Silt content array.

    Returns:
        Array of soil types.
    """
    stype = np.zeros(clay.shape)
    # Using np.where to simplify boolean indexing

    # 1. SAND
    mask = (silt + clay * 1.5 < 15.0) & (clay != 255)
    stype[mask] = 1.0

    # 2. LOAMY SAND
    mask = (silt + 1.5 * clay >= 15.0) & (silt + 1.5 * clay < 30) & (clay != 255)
    stype[mask] = 2.0

    # 3. SANDY LOAM
    mask1 = (clay >= 7.0) & (clay < 20) & (sand > 52) & (silt + 2 * clay >= 30) & (clay != 255)
    mask2 = (clay < 7) & (silt < 50) & (silt + 2 * clay >= 30) & (clay != 255)
    stype[mask1 | mask2] = 3.0

    # 4. SILT LOAM
    mask1 = (silt >= 50) & (clay >= 12) & (clay < 27) & (clay != 255)
    mask2 = (silt >= 50) & (silt < 80) & (clay < 12) & (clay != 255)
    stype[mask1 | mask2] = 4.0

    # 5. SILT
    mask = (silt >= 80) & (clay < 12) & (clay != 255)
    stype[mask] = 5.0

    # 6. LOAM
    mask = (clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52) & (clay != 255)
    stype[mask] = 6.0

    # 7. SANDY CLAY LOAM
    mask = (clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45) & (clay != 255)
    stype[mask] = 7.0

    # 8. SILTY CLAY LOAM
    mask = (clay >= 27) & (clay < 40.0) & (sand > 40) & (clay != 255)
    stype[mask] = 8.0

    # 9. CLAY LOAM
    mask = (clay >= 27) & (clay < 40.0) & (sand > 20) & (sand <= 45) & (clay != 255)
    stype[mask] = 9.0

    # 10. SANDY CLAY
    mask = (clay >= 35) & (sand > 45) & (clay != 255)
    stype[mask] = 10.0

    # 11. SILTY CLAY
    mask = (clay >= 40) & (silt >= 40) & (clay != 255)
    stype[mask] = 11.0

    # 12. CLAY
    mask = (clay >= 40) & (sand <= 45) & (silt < 40) & (clay != 255)
    stype[mask] = 12.0

    return stype


_module_install_names = {
    # module: GH, PyPI, conda-forge
    "pyhdf": ("fhs/pyhdf", "pyhdf", "pyhdf"),
}


def _install_message(mod_name: str) -> str:
    if mod_name not in _module_install_names:
        return ""

    gh, pypi_name, cf_name = _module_install_names[mod_name]
    cf_ = f"Try installing from conda-forge using `conda install -c conda-forge {cf_name}`."

    return f"{cf_}"


def _import_required(mod_name: str) -> Any:
    from importlib import import_module

    try:
        return import_module(mod_name)
    except ImportError as e:
        raise RuntimeError(
            f"importing required module '{mod_name}' failed. {_install_message(mod_name)}"
        ) from e


def _try_merge_exact(
    left: xr.Dataset, right: xr.Dataset, *, right_name: Optional[str] = None
) -> xr.Dataset:
    """For two ``xr.Dataset``s, try ``left.merge(right, compat="equals", join="exact")``.
    If it fails, print informative debugging messages and re-raise.
    Otherwise, return the result.
    """
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
                f"Unexpected Exception message (expected to match {regex!r}): {e}", stacklevel=2
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
