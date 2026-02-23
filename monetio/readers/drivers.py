from typing import Any, List, Union

import xarray as xr


class XarrayDriver:
    """Driver for Xarray-based I/O."""

    @staticmethod
    def open_dataset(files: Union[str, List[str]], **kwargs: Any) -> xr.Dataset:
        """Open a single or multiple datasets.

        Parameters
        ----------
        files : str or list
            Path(s) to the file(s). Supports wildcard strings.
        **kwargs : Any
            Additional arguments passed to xr.open_dataset or xr.open_mfdataset.

        Returns
        -------
        xr.Dataset
            The opened dataset.
        """
        if isinstance(files, str):
            if "*" in files or "?" in files:
                # Handle wildcard string
                return xr.open_mfdataset(files, **kwargs)
            return xr.open_dataset(files, **kwargs)

        if isinstance(files, list):
            if len(files) > 1:
                return xr.open_mfdataset(files, **kwargs)
            return xr.open_dataset(files[0], **kwargs)

        raise TypeError(f"Unsupported type for files: {type(files)}")
