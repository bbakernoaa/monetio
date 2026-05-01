"""EARLINET Reader"""

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("earlinet")
class EARLINETReader(GriddedReader):
    """
    Reader for EARLINET (European Aerosol Research Lidar Network) NetCDF data.
    """

    def open_dataset(
        self,
        files: str | list[str],
        use_virtualizarr: bool = False,
        virtualizarr_file: str | None = None,
        use_icechunk: bool = False,
        icechunk_url: str | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Retrieve and load EARLINET data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path, list of paths, or glob pattern.
        **kwargs : dict
            Additional arguments passed to xarray.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The loaded EARLINET data.
        """
        # Default to combined dimensions if not specified
        kwargs.setdefault("combine", "by_coords")

        # Use XarrayDriver (via GriddedReader) to open
        ds = super().open_dataset(
            files,
            preprocess=earlinet_preprocess,
            use_virtualizarr=use_virtualizarr,
            virtualizarr_file=virtualizarr_file,
            use_icechunk=use_icechunk,
            icechunk_url=icechunk_url,
            **kwargs,
        )

        return ds


def earlinet_preprocess(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess EARLINET dataset: standardize coordinates and dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Input EARLINET dataset.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset.
    """
    # 1. Standardize Time
    # EARLINET NetCDF files are usually CF compliant and xarray handles them.

    # 2. Rename Dimensions/Coordinates
    # altitude is usually a coordinate/dimension already named 'altitude' or 'height'.
    # If it's something else, we can rename it.
    if "height" in ds.dims and "altitude" not in ds.dims:
        ds = ds.rename({"height": "altitude"})

    # 3. Coordinate handling
    # Ensure latitude and longitude are coordinates
    coord_vars = ["latitude", "longitude", "altitude", "time", "wavelength"]
    actual_coords = [v for v in coord_vars if v in ds.variables]
    if actual_coords:
        ds = ds.set_coords(actual_coords)

    # 4. Standard attributes
    if "altitude" in ds.coords:
        ds["altitude"].attrs.update({"units": "m", "standard_name": "altitude", "positive": "up"})

    if "latitude" in ds.coords:
        ds["latitude"].attrs.update({"units": "degrees_north", "standard_name": "latitude"})

    if "longitude" in ds.coords:
        ds["longitude"].attrs.update({"units": "degrees_east", "standard_name": "longitude"})

    # Update history
    ds = update_history(ds, "Preprocessed EARLINET data.")

    return ds
