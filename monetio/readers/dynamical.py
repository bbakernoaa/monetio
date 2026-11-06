"""Reader for gridded datasets in the dynamical.org catalog."""

import xarray as xr

from .base import GriddedReader, register_reader
from .sat_utils import update_history


@register_reader("dynamical")
class DynamicalReader(GriddedReader):
    """Open gridded catalog products by dynamical.org dataset ID.

    Parameters
    ----------
    files : None
        Catalog access uses ``dataset`` rather than a file path.
    dataset : str
        Catalog identifier, e.g. ``"noaa-gfs-analysis"``.
    variables : str or list of str, optional
        Variables to retain.
    sel : dict, optional
        Coordinate labels passed to :meth:`xarray.Dataset.sel`.
    **kwargs
        Options passed to ``dynamical_catalog.open`` / ``xarray.open_zarr``.
    """

    def open_dataset(self, files=None, *, dataset, variables=None, sel=None, **kwargs) -> xr.Dataset:
        """Open a catalog product and preserve its native coordinates."""
        if files is not None:
            raise ValueError("dynamical reader accepts a catalog dataset ID; files must be omitted")
        if not isinstance(dataset, str) or not dataset.strip():
            raise ValueError("dataset must be a non-empty dynamical.org catalog ID")

        try:
            import dynamical_catalog
        except ImportError as exc:
            raise ImportError(
                "The dynamical reader requires the optional 'dynamical-catalog' package; "
                "install it with `pip install monetio[dynamical]`."
            ) from exc

        ds = dynamical_catalog.open(dataset.strip(), **kwargs)
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"dynamical-catalog returned {type(ds).__name__}, expected xarray.Dataset")

        if variables is not None:
            requested = [variables] if isinstance(variables, str) else list(variables)
            missing = sorted(set(requested).difference(ds.data_vars))
            if missing:
                raise KeyError(f"Variables not found in dataset {dataset!r}: {missing}")
            ds = ds[requested]

        if sel is not None:
            if not isinstance(sel, dict):
                raise TypeError("sel must be a mapping of coordinate names to xarray selectors")
            ds = ds.sel(sel)

        return update_history(ds, f"Read {dataset} from the dynamical.org catalog.")
