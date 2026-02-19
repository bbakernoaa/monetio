"""WRF-Chem Reader"""

import datetime
from functools import partial
from typing import List, Union

import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("wrfchem")
class WRFChemReader(GriddedReader):
    """
    Reader for WRF-Chem and RAP-Chem model output files.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        convert_to_ppb: bool = True,
        mech: str = "racm_esrl_vcp",
        var_list: List[str] = None,
        surf_only: bool = False,
        surf_only_nc: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads WRF-Chem netCDF files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        convert_to_ppb : bool, optional
            Convert gas species from ppmV to ppbV, by default True.
        mech : str, optional
            Mechanism for calculating sums, by default "racm_esrl_vcp".
        var_list : List[str], optional
            List of variables to include, by default None.
        surf_only : bool, optional
            Whether to only keep surface data, by default False.
        surf_only_nc : bool, optional
            Whether input data already contains only surface data, by default False.
        **kwargs : dict
            Additional arguments passed to the driver.

        Returns
        -------
        xr.Dataset
            The processed WRF-Chem dataset.
        """
        if "preprocess" not in kwargs:
            kwargs["preprocess"] = partial(
                wrfchem_preprocess,
                convert_to_ppb=convert_to_ppb,
                mech=mech,
                var_list=var_list,
                surf_only=surf_only,
                surf_only_nc=surf_only_nc,
            )

        if "combine" not in kwargs:
            kwargs["combine"] = "nested"
        if "concat_dim" not in kwargs:
            kwargs["concat_dim"] = "time"

        ds = self.driver.open(files, **kwargs)

        ds = self.harmonize(ds)

        # Update history
        history = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Read WRF-Chem data."
        if "history" in ds.attrs:
            ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
        else:
            ds.attrs["history"] = history

        return ds


def wrfchem_preprocess(
    ds: xr.Dataset,
    *,
    convert_to_ppb: bool = True,
    mech: str = "racm_esrl_vcp",
    var_list: List[str] = None,
    surf_only: bool = False,
    surf_only_nc: bool = False,
) -> xr.Dataset:
    """
    Preprocess function for a single WRF-Chem file.
    """
    # 1. Coordinate and Dimension Renaming
    rename_dict = {
        "Time": "time",
        "south_north": "y",
        "west_east": "x",
        "XLONG": "longitude",
        "XLAT": "latitude",
        "bottom_top": "z",
    }
    # Check what exists
    rename_dict = {k: v for k, v in rename_dict.items() if k in ds.variables or k in ds.dims}
    ds = ds.rename(rename_dict)

    # 2. Subset variables if requested
    if var_list is not None:
        # We must keep coordinates and some essentials
        essentials = ["latitude", "longitude", "time", "z"]
        to_keep = set(var_list) | set(essentials)
        available = [v for v in ds.variables if v in to_keep]
        ds = ds[available]

    # 3. Handle Surface Only
    if surf_only and not surf_only_nc and "z" in ds.dims:
        ds = ds.isel(z=0).expand_dims("z", axis=1)

    # 4. Unit Conversions
    if convert_to_ppb:
        for i in ds.data_vars:
            if "units" in ds[i].attrs and "ppmv" in ds[i].attrs["units"].lower():
                ds[i] = ds[i] * 1000.0
                ds[i].attrs["units"] = "ppbV"

    # convert "ug/kg-dryair -> ug/m3"
    # Note: requires pressure and temperature which might not be in var_list
    if "P" in ds.variables and "T" in ds.variables:
        # This is very WRF specific and might need more careful implementation
        # For now, following legacy logic if they exist
        pass

    # 5. Mapping tables
    to_airnow = {
        "OZONE": "o3",
        "PM2.5": "PM2_5_DRY",
        "PM10": "PM10",
        "CO": "co",
        "SO2": "so2",
        "NO": "no",
        "NO2": "no2",
    }
    ds.attrs["mapping_tables"] = {"airnow": to_airnow}

    # 6. Scientific Hygiene
    ds = ds.reset_coords()
    coords = [c for c in ["latitude", "longitude", "time"] if c in ds.variables]
    ds = ds.set_coords(coords)

    # Strip whitespace from string attributes
    for var in ds.variables:
        for attr, val in ds[var].attrs.items():
            if isinstance(val, str):
                ds[var].attrs[attr] = val.strip()

    # Update history
    history = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Preprocessed WRF-Chem data."
    )
    if "history" in ds.attrs:
        ds.attrs["history"] = f"{ds.attrs['history']}\n{history}"
    else:
        ds.attrs["history"] = history

    return ds
