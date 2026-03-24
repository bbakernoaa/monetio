"""MERRA-2 Reader"""

from typing import List, Optional, Union

import xarray as xr

from .base import GriddedReader, register_reader
from .nasa_utils import setup_netrc
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("merra2")
class MERRA2Reader(GriddedReader):
    """
    Reader for MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2) data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads MERRA-2 data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        username : str, optional
            NASA Earthdata username. If provided, will setup .netrc.
        password : str, optional
            NASA Earthdata password. If provided, will setup .netrc.
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The MERRA-2 dataset.
        """
        if username and password:
            setup_netrc(username, password)

        if "preprocess" not in kwargs:
            kwargs["preprocess"] = merra2_preprocess

        ds = super().open_dataset(files, **kwargs)

        # Update history
        ds = update_history(ds, "Read MERRA-2 data.")

        return ds


def merra2_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess MERRA-2 dataset: standardize coordinates and metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Pre-standardize coordinates to avoid losing them during dimension rename
    # if they share the same name.
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.assign_coords(latitude=ds.lat)
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.assign_coords(longitude=ds.lon)

    # 2. Standardize dimensions and coordinates
    # MERRA-2 typically uses 'lat', 'lon', 'time', 'lev'.
    ds = standardize_satellite_coords(
        ds,
        lat_name="latitude",
        lon_name="longitude",
        y_dim=["lat", "nlat", "y"],
        x_dim=["lon", "nlon", "x"],
        z_dim=["lev", "level", "layer"],
    )

    # 3. Expand 1D coords to 2D for UGRID/CF compliance in MONETIO if needed
    if "latitude" in ds.coords and ds["latitude"].ndim == 1:
        if "longitude" in ds.coords and ds["longitude"].ndim == 1:
            # Use lazy broadcasting
            lons, lats = xr.broadcast(ds.longitude, ds.latitude)
            # Ensure (y, x) order which is standard for gridded data in MONETIO
            if "y" in lons.dims and "x" in lons.dims:
                lons = lons.transpose("y", "x")
                lats = lats.transpose("y", "x")
            # Re-assign as 2D coordinates
            ds = ds.assign_coords(longitude=lons, latitude=lats)

    # 3. Variable renaming to standard names if they exist
    mapping = {
        "PS": "surface_pressure",
        "T": "temperature",
        "QV": "specific_humidity",
        "U": "u_wind",
        "V": "v_wind",
    }
    rename_dict = {
        old: new for old, new in mapping.items() if old in ds.variables and new not in ds.variables
    }
    if rename_dict:
        ds = ds.rename(rename_dict)

    # 4. Calculate Pressure (Lazy)
    ds = _add_merra2_pressure(ds)

    # Update history
    ds = update_history(ds, "Preprocessed MERRA-2 data via Aero Protocol.")

    return ds


def _add_merra2_pressure(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate pressure levels lazily for MERRA-2 using ak and bk coefficients.
    p = ak + bk * surface_pressure

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with calculated pressure.
    """
    # Look for coefficients. Common names: ak, bk or ap, bp
    ak = ds.get("ak") if "ak" in ds.variables or "ak" in ds.coords else ds.get("ap")
    bk = ds.get("bk") if "bk" in ds.variables or "bk" in ds.coords else ds.get("bp")
    ps = ds.get("surface_pressure") if "surface_pressure" in ds.variables else ds.get("PS")

    if ak is not None and bk is not None and ps is not None:
        # p = ak + bk * ps
        # The calculation is fully lazy and backend-agnostic
        pres = ak + bk * ps

        ds["pres_pa_mid"] = pres.assign_attrs(
            {
                "units": "Pa",
                "long_name": "pressure",
                "standard_name": "air_pressure",
                "description": "Pressure calculated as ak + bk * surface_pressure",
            }
        )
        ds = update_history(ds, "Calculated 3D pressure lazily using ak and bk.")

    return ds
