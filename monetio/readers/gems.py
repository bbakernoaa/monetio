"""
GEMS (Geostationary Environment Monitoring Spectrometer) Reader.
GEMS is on the GEO-KOMPSAT-2B (GK-2B) satellite.
"""

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import standardize_satellite_coords, update_history


@register_reader("gems")
class GEMSReader(GriddedReader):
    """
    Reader for GEMS (Geostationary Environment Monitoring Spectrometer) L2 data.
    """

    def open_dataset(
        self,
        files: str | list[str],
        group: str | list[str] | None = None,
        variable_dict: dict | None = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads GEMS data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        group : str or list of str, optional
            The NetCDF group(s) to open. If a list is provided, groups will be merged.
            If None, common GEMS groups will be opened:
            - "Data Fields"
            - "Geolocation Fields"
            - "Metadata"
        variable_dict : dict, optional
            Dictionary mapping variable names to processing options (scale, minimum, maximum).
        **kwargs : dict
            Additional arguments passed to XarrayDriver.open.

        Returns
        -------
        xr.Dataset
            The GEMS dataset.
        """
        if group is None:
            groups = ["Data Fields", "Geolocation Fields", "Metadata"]
        elif isinstance(group, str):
            groups = [group]
        else:
            groups = group

        user_preprocess = kwargs.pop("preprocess", None)

        if "engine" not in kwargs:
            kwargs["engine"] = "h5netcdf"

        dsets = []
        for g in groups:
            g_kwargs = kwargs.copy()
            g_kwargs["group"] = g
            try:
                # Open without the preprocessor at this stage
                ds_g = super().open_dataset(files, **g_kwargs)
                dsets.append(ds_g)
            except Exception:
                # Not all groups may be present in all files
                continue

        if not dsets:
            # Fallback: try opening without group if groups failed
            try:
                ds = super().open_dataset(files, **kwargs)
            except Exception:
                raise RuntimeError("No GEMS groups could be opened.")
        else:
            # Merge groups
            ds = xr.merge(dsets, compat="no_conflicts")

        # Now apply GEMS preprocessing
        ds = gems_preprocess(ds, variable_dict=variable_dict)

        if user_preprocess:
            ds = user_preprocess(ds)

        # Update history
        ds = update_history(ds, "Read GEMS L2 data.")

        return ds


def gems_preprocess(ds: xr.Dataset, variable_dict: dict | None = None) -> xr.Dataset:
    """
    Preprocess GEMS dataset: standardize coordinates, handle units,
    and apply variable-specific transformations.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with merged groups.
    variable_dict : dict, optional
        Dictionary mapping variable names to processing options (scale, minimum, maximum).

    Returns
    -------
    xr.Dataset
        Processed dataset.
    """
    # 1. Standardize dimensions and coordinates
    # GEMS often uses 'nscans' and 'npixels' or similar.
    # We use sat_utils to handle common variations.
    ds = standardize_satellite_coords(
        ds,
        lat_name="Latitude",
        lon_name="Longitude",
        y_dim=["nscans", "rows", "Rows", "nTimes"],
        x_dim=["npixels", "columns", "Columns", "nIFOV"],
    )

    # 2. Handle Time if available in attributes or variables
    if "time" not in ds.coords and "time" not in ds.data_vars:
        # Check for common time variables
        for t_var in ["Time", "Time_at_Scan_Start"]:
            if t_var in ds.variables:
                ds = ds.rename({t_var: "time"})
                if ds["time"].ndim == 1:
                    ds = ds.set_coords("time")
                break

    # 3. Apply variable-specific processing from variable_dict
    if variable_dict:
        for var, options in variable_dict.items():
            if var in ds.variables:
                # Scale
                if "scale" in options:
                    ds[var] = ds[var] * options["scale"]

                # Clipping
                if "minimum" in options:
                    ds[var] = ds[var].where(ds[var] >= options["minimum"])
                if "maximum" in options:
                    ds[var] = ds[var].where(ds[var] <= options["maximum"])

    # Update history
    ds = update_history(ds, "Preprocessed GEMS data.")

    return ds
