import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseReader, register_reader
from .drivers import FileUtility
from .sat_utils import update_history


@register_reader("ioda")
class IODAReader(BaseReader):
    """
    Reader for JCSDA IODA NetCDF4 files.
    """

    def open_dataset(self, files, **kwargs):
        """
        Reads IODA format files and merges groups into a flat dataset.

        Parameters
        ----------
        files : str or list[str]
            File path, list of paths, or glob pattern.
        **kwargs : dict
            Additional arguments passed to xr.open_dataset.

        Returns
        -------
        xr.Dataset
            Flattened IODA dataset.
        """
        file_list = FileUtility.expand_paths(files)

        dsets = []
        for f in file_list:
            dsets.append(self._read_single_file(f, **kwargs))

        if not dsets:
            return xr.Dataset()

        if len(dsets) == 1:
            ds = dsets[0]
        else:
            # Determine concat dimension from the first dataset
            concat_dim = "Location"
            if "nlocs" in dsets[0].dims:
                concat_dim = "nlocs"
            elif "Location" not in dsets[0].dims:
                if dsets[0].dims:
                    concat_dim = list(dsets[0].dims)[0]

            ds = xr.concat(dsets, dim=concat_dim)

        ds = update_history(ds, "Read IODA data.")
        return ds

    def _read_single_file(self, filepath, **kwargs):
        import os
        import tempfile

        # Handle remote files for netCDF4 group discovery
        fs = FileUtility.get_fs(filepath)
        if filepath.startswith(("http", "s3", "ftp", "reference://")):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                fs.get(filepath, tmp.name)
                tmp_path = tmp.name
            try:
                available_groups = self._get_groups(tmp_path)
                ds = self._load_groups(tmp_path, available_groups, **kwargs)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            available_groups = self._get_groups(filepath)
            ds = self._load_groups(filepath, available_groups, **kwargs)

        return ds

    def _get_groups(self, filepath):
        import netCDF4

        try:
            with netCDF4.Dataset(filepath, "r") as nc:
                return list(nc.groups.keys())
        except Exception:
            try:
                import h5netcdf

                with h5netcdf.File(filepath, "r") as f:
                    return list(f.groups.keys())
            except Exception:
                return []

    def _load_groups(self, filepath, available_groups, **kwargs):
        target_groups = {
            "MetaData": "",
            "ObsValue": "",
            "ObsError": "_error",
            "EffectiveObsError": "_effective_error",
            "PreQC": "_qc",
            "HofX": "_sim",
        }

        group_datasets = []
        for group in available_groups:
            if group in target_groups:
                suffix = target_groups[group]
                try:
                    ds_group = xr.open_dataset(filepath, group=group, **kwargs)
                except Exception:
                    continue

                rename_dict = {}
                if group == "MetaData":
                    for v in ds_group.data_vars:
                        if v == "latitude":
                            rename_dict[v] = "latitude"
                        elif v == "longitude":
                            rename_dict[v] = "longitude"
                        elif v in ["datetime", "dateTime", "dateTimeString"]:
                            rename_dict[v] = "time"
                else:
                    for v in ds_group.data_vars:
                        rename_dict[v] = f"{v}{suffix}"

                if rename_dict:
                    # Filter rename_dict to only include variables that actually exist in ds_group
                    rename_dict = {k: v for k, v in rename_dict.items() if k in ds_group.variables}
                    ds_group = ds_group.rename(rename_dict)

                # Convert time coordinate/variable if present
                if "time" in ds_group.variables:
                    tv = "time"
                    if ds_group[tv].dtype == object or ds_group[tv].dtype.kind in ["S", "U"]:
                        try:
                            # Try ISO format first
                            new_time = pd.to_datetime(
                                ds_group[tv].values.astype(str), errors="coerce"
                            )
                            if hasattr(new_time, "tz"):
                                if new_time.tz is not None:
                                    new_time = new_time.tz_localize(None)
                            ds_group[tv] = (ds_group[tv].dims, new_time)
                        except Exception:
                            pass

                group_datasets.append(ds_group)

        if not group_datasets:
            return xr.Dataset()

        # Merge groups into a single flat dataset.
        # override compat to handle potential coordinate mismatch in non-core dims
        ds = xr.merge(group_datasets, compat="override")
        return ds


def export_to_ioda(ds_monet, variable_mapping, output_path):
    """
    Exports a monetio dataset to JCSDA IODA NetCDF4 format.

    Parameters
    ----------
    ds_monet : xr.Dataset
        Input dataset.
    variable_mapping : dict
        Mapping of monet variables to (IODA_Group, IODA_Var).
        Example: {'ozone': ('ObsValue', 'ozone_conc'), 'latitude': ('MetaData', 'latitude')}
    output_path : str
        Path to save the IODA file.
    """
    # 1. Stacking multi-dimensional datasets into a 1-D 'Location' index
    ds = ds_monet.copy()
    if len(ds.dims) > 1:
        stack_dims = list(ds.dims)
        ds = ds.stack(Location=stack_dims).reset_index("Location")
    elif len(ds.dims) == 1:
        curr_dim = list(ds.dims)[0]
        if curr_dim != "Location" and curr_dim != "nlocs":
            ds = ds.rename({curr_dim: "Location"})

    # 2. Build groups based on mapping
    groups = {}
    for monet_var, (ioda_group, ioda_var) in variable_mapping.items():
        if monet_var not in ds.variables:
            continue

        if ioda_group not in groups:
            groups[ioda_group] = xr.Dataset()

        # Drop coordinates to avoid MergeError when building groups
        da = ds[monet_var].drop_vars(ds.coords, errors="ignore")

        # Handle time conversion in MetaData to ISO 8601 strings
        if ioda_group == "MetaData" and ioda_var in [
            "datetime",
            "dateTime",
            "dateTimeString",
            "time",
        ]:
            if np.issubdtype(da.dtype, np.datetime64) or da.dtype.kind == "M":
                # Ensure UTC 'Z' suffix
                times = pd.to_datetime(da.values).strftime("%Y-%m-%dT%H:%M:%SZ")
                da = xr.DataArray(times, dims=da.dims, name=ioda_var)
            else:
                da.name = ioda_var
        else:
            da.name = ioda_var

        # Use update instead of __setitem__ to have more control if needed,
        # but here we just want to ensure it's a clean addition.
        groups[ioda_group][ioda_var] = da

    # Populate missing error/QC data with defaults if ObsValue is present
    if "ObsValue" in groups:
        if "ObsError" not in groups:
            groups["ObsError"] = xr.Dataset()
        if "PreQC" not in groups:
            groups["PreQC"] = xr.Dataset()

        for var in groups["ObsValue"].data_vars:
            if var not in groups["ObsError"].data_vars:
                groups["ObsError"][var] = xr.full_like(groups["ObsValue"][var], 0.0)
            if var not in groups["PreQC"].data_vars:
                groups["PreQC"][var] = xr.full_like(groups["ObsValue"][var], 0, dtype=np.int32)

    # 3. Write groups sequentially to NetCDF4
    if not groups:
        return

    # Start with MetaData if present, otherwise first available group
    primary_group = "MetaData" if "MetaData" in groups else list(groups.keys())[0]
    groups[primary_group].to_netcdf(output_path, group=primary_group, mode="w")

    for group_name, ds_group in groups.items():
        if group_name == primary_group:
            continue
        ds_group.to_netcdf(output_path, group=group_name, mode="a")
