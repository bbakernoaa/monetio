"""CMAQ File Reader"""

import xarray as xr
from numpy import array, concatenate, ones
from pandas import Series, to_datetime

from monetio.grids import get_ioapi_pyresample_area_def, grid_from_dataset

from .base import GriddedReader, register_reader


@register_reader("cmaq")
class CMAQReader(GriddedReader):
    def open_dataset(
        self, files, earth_radius=6370000, convert_to_ppb=True, drop_duplicates=False, **kwargs
    ):
        """
        Reads CMAQ netCDF files.
        """
        # 1. Open the dataset using standard xarray (Lazy loading)

        # We ensure standard CMAQ combination logic is present
        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "TSTEP"

        # Use cmaq_preprocess to add lazy diagnostic variables
        kwargs["preprocess"] = cmaq_preprocess

        ds = self.driver.open(files, **kwargs)

        # 2. Pre-processing specific to CMAQ (Global)

        # get the grid information
        grid = grid_from_dataset(ds, earth_radius=earth_radius)
        area_def = get_ioapi_pyresample_area_def(ds, grid)

        # assign attributes for dataset and all DataArrays
        ds = ds.assign_attrs({"proj4_srs": grid})
        for i in ds.variables:
            ds[i] = ds[i].assign_attrs({"proj4_srs": grid})
            for j in ds[i].attrs:
                # Strip whitespace from attributes
                if isinstance(ds[i].attrs[j], str):
                    ds[i].attrs[j] = ds[i].attrs[j].strip()

        # get the times
        if "TFLAG" in ds.variables or "TFLAG" in ds.coords:
            ds = _get_times(ds, drop_duplicates=drop_duplicates)

        # get the lat lon
        ds = _get_latlon(ds, area_def)

        # rename dimensions
        ds = ds.rename({"COL": "x", "ROW": "y", "LAY": "z"})

        # convert all gas species to ppbv
        if convert_to_ppb:
            for i in ds.variables:
                if "units" in ds[i].attrs:
                    if "ppmV" in ds[i].attrs["units"]:
                        ds[i] *= 1000.0
                        ds[i].attrs["units"] = "ppbV"

        # convert 'micrograms to \mu g'
        for i in ds.variables:
            if "units" in ds[i].attrs:
                if "micrograms" in ds[i].attrs["units"]:
                    ds[i].attrs["units"] = r"$\mu g m^{-3}$"

        # 3. Harmonize (Standardize names)
        ds = self.harmonize(ds)

        return ds

    def harmonize(self, ds):
        # Placeholder for future harmonization logic
        return ds


# -----------------------------------------------------------------------------
# Helper functions ported from monetio/models/cmaq.py
# -----------------------------------------------------------------------------


def cmaq_preprocess(ds):
    """
    Preprocess function to add lazy diagnostic variables.
    Can be passed to xarray.open_mfdataset.
    """
    ds = add_lazy_pm25(ds)
    ds = add_lazy_pm10(ds)
    ds = add_lazy_pm_course(ds)
    ds = add_lazy_clf(ds)
    ds = add_lazy_naf(ds)
    ds = add_lazy_caf(ds)
    ds = add_lazy_noy(ds)
    ds = add_lazy_nox(ds)
    ds = add_lazy_no3f(ds)
    ds = add_lazy_nh4f(ds)
    ds = add_lazy_so4f(ds)
    ds = add_lazy_rh(ds)
    return ds


def can_do(index):
    if index.max():
        return True
    else:
        return False


def _get_times(d, drop_duplicates):
    idims = len(d.TFLAG.dims)
    if idims == 2:
        tflag1 = Series(d["TFLAG"][:, 0]).astype(str).str.zfill(7)
        tflag2 = Series(d["TFLAG"][:, 1]).astype(str).str.zfill(6)
    else:
        tflag1 = Series(d["TFLAG"][:, 0, 0]).astype(str).str.zfill(7)
        tflag2 = Series(d["TFLAG"][:, 0, 1]).astype(str).str.zfill(6)
    date = to_datetime([i + j for i, j in zip(tflag1, tflag2)], format="%Y%j%H%M%S")
    if drop_duplicates:
        indexdates = Series(date).drop_duplicates(keep="last").index.values
        d = d.isel(TSTEP=indexdates)
        d["TSTEP"] = date[indexdates]
    else:
        d["TSTEP"] = date
    return d.rename({"TSTEP": "time"})


def _get_latlon(dset, area):
    lon, lat = area.get_lonlats()
    dset["longitude"] = xr.DataArray(lon[::-1, :], dims=["ROW", "COL"])
    dset["latitude"] = xr.DataArray(lat[::-1, :], dims=["ROW", "COL"])
    dset = dset.assign_coords(longitude=dset.longitude, latitude=dset.latitude)
    return dset


def _get_keys(d):
    keys = Series([i for i in d.data_vars.keys()])
    return keys


def add_multiple_lazy(dset, variables, weights=None):
    if weights is None:
        weights = ones(len(variables))
    else:
        weights = weights.values
    variables = variables.values
    new = dset[variables[0]].copy() * weights[0]
    for i, j in zip(variables[1:], weights[1:]):
        new = new + dset[i] * j
    return new


# Variable lists
accumulation = array(
    [
        "AALJ",
        "AALK1J",
        "AALK2J",
        "ABNZ1J",
        "ABNZ2J",
        "ABNZ3J",
        "ACAJ",
        "ACLJ",
        "AECJ",
        "AFEJ",
        "AISO1J",
        "AISO2J",
        "AISO3J",
        "AKJ",
        "AMGJ",
        "AMNJ",
        "ANAJ",
        "ANH4J",
        "ANO3J",
        "AOLGAJ",
        "AOLGBJ",
        "AORGCJ",
        "AOTHRJ",
        "APAH1J",
        "APAH2J",
        "APAH3J",
        "APNCOMJ",
        "APOCJ",
        "ASIJ",
        "ASO4J",
        "ASQTJ",
        "ATIJ",
        "ATOL1J",
        "ATOL2J",
        "ATOL3J",
        "ATRP1J",
        "ATRP2J",
        "AXYL1J",
        "AXYL2J",
        "AXYL3J",
        "AORGAJ",
        "AORGPAJ",
        "AORGBJ",
    ]
)
aitken = array(
    [
        "ACLI",
        "AECI",
        "ANAI",
        "ANH4I",
        "ANO3I",
        "AOTHRI",
        "APNCOMI",
        "APOCI",
        "ASO4I",
        "AORGAI",
        "AORGPAI",
        "AORGBI",
    ]
)
coarse = array(["ACLK", "ACORS", "ANH4K", "ANO3K", "ASEACAT", "ASO4K", "ASOIL"])
noy_gas = array(
    [
        "NO",
        "NO2",
        "NO3",
        "N2O5",
        "HONO",
        "HNO3",
        "PAN",
        "PANX",
        "PNA",
        "NTR",
        "CRON",
        "CRN2",
        "CRNO",
        "CRPX",
        "OPAN",
    ]
)


# Diagnostic Additions
def add_lazy_pm25(d):
    keys = _get_keys(d)
    allvars = Series(concatenate([aitken, accumulation, coarse]))
    weights = Series(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
            0.2,
        ]
    )
    if "PM25_TOT" in keys.to_list():
        d["PM25"] = d["PM25_TOT"]
    else:
        index = allvars.isin(keys)
        if can_do(index):
            newkeys = allvars.loc[index]
            newweights = weights.loc[index]
            d["PM25"] = add_multiple_lazy(d, newkeys, weights=newweights)
            d["PM25"] = d["PM25"].assign_attrs(
                {"units": r"$\mu g m^{-3}$", "name": "PM2.5", "long_name": "PM2.5"}
            )
    return d


def add_lazy_pm10(d):
    keys = _get_keys(d)
    allvars = Series(concatenate([aitken, accumulation, coarse]))
    if "PMC_TOT" in keys.to_list():
        d["PM10"] = d["PMC_TOT"]
    else:
        index = allvars.isin(keys)
        if can_do(index):
            newkeys = allvars.loc[index]
            d["PM10"] = add_multiple_lazy(d, newkeys)
            d["PM10"] = d["PM10"].assign_attrs(
                {
                    "units": r"$\mu g m^{-3}$",
                    "name": "PM10",
                    "long_name": "Particulate Matter < 10 microns",
                }
            )
    return d


def add_lazy_pm_course(d):
    keys = _get_keys(d)
    allvars = Series(coarse)
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["PM_COURSE"] = add_multiple_lazy(d, newkeys)
        d["PM_COURSE"] = d["PM_COURSE"].assign_attrs(
            {
                "units": r"$\mu g m^{-3}$",
                "name": "PM_COURSE",
                "long_name": "Course Mode Particulate Matter",
            }
        )
    return d


def add_lazy_clf(d):
    keys = _get_keys(d)
    allvars = Series(["ACLI", "ACLJ", "ACLK"])
    weights = Series([1, 1, 0.2])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["CLf"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["CLf"] = d["CLf"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "CLf", "long_name": "Fine Mode particulate Cl"}
        )
    return d


def add_lazy_caf(d):
    keys = _get_keys(d)
    allvars = Series(["ACAI", "ACAJ", "ASEACAT", "ASOIL", "ACORS"])
    weights = Series([1, 1, 0.2 * 32.0 / 1000.0, 0.2 * 83.8 / 1000.0, 0.2 * 56.2 / 1000.0])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["CAf"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["CAf"] = d["CAf"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "CAf", "long_name": "Fine Mode particulate CA"}
        )
    return d


def add_lazy_naf(d):
    keys = _get_keys(d)
    allvars = Series(["ANAI", "ANAJ", "ASEACAT", "ASOIL", "ACORS"])
    weights = Series([1, 1, 0.2 * 837.3 / 1000.0, 0.2 * 62.6 / 1000.0, 0.2 * 2.3 / 1000.0])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["NAf"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["NAf"] = d["NAf"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "NAf", "long_name": "NAf"}
        )
    return d


def add_lazy_so4f(d):
    keys = _get_keys(d)
    allvars = Series(["ASO4I", "ASO4J", "ASO4K"])
    weights = Series([1.0, 1.0, 0.2])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["SO4f"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["SO4f"] = d["SO4f"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "SO4f", "long_name": "SO4f"}
        )
    return d


def add_lazy_nh4f(d):
    keys = _get_keys(d)
    allvars = Series(["ANH4I", "ANH4J", "ANH4K"])
    weights = Series([1.0, 1.0, 0.2])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["NH4f"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["NH4f"] = d["NH4f"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "NH4f", "long_name": "NH4f"}
        )
    return d


def add_lazy_no3f(d):
    keys = _get_keys(d)
    allvars = Series(["ANO3I", "ANO3J", "ANO3K"])
    weights = Series([1.0, 1.0, 0.2])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        neww = weights.loc[index]
        d["NO3f"] = add_multiple_lazy(d, newkeys, weights=neww)
        d["NO3f"] = d["NO3f"].assign_attrs(
            {"units": r"$\mu g m^{-3}$", "name": "NO3f", "long_name": "NO3f"}
        )
    return d


def add_lazy_noy(d):
    keys = _get_keys(d)
    allvars = Series(noy_gas)
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["NOy"] = add_multiple_lazy(d, newkeys)
        d["NOy"] = d["NOy"].assign_attrs({"name": "NOy", "long_name": "NOy"})
    return d


def add_lazy_nox(d):
    keys = _get_keys(d)
    allvars = Series(["NO", "NO2"])
    index = allvars.isin(keys)
    if can_do(index):
        newkeys = allvars.loc[index]
        d["NOx"] = add_multiple_lazy(d, newkeys)
        d["NOx"] = d["NOx"].assign_attrs({"name": "NOx", "long_name": "NOx"})
    return d


def add_lazy_rh(d):
    # Placeholder as in original code
    return d
