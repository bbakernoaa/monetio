"""MODIS L2 and Gridded EOS Reader"""

from typing import List, Union

import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("modis_l2")
class MODISL2Reader(GriddedReader):
    """
    Reader for MODIS L2 swath and Gridded EOS data.
    """

    def open_dataset(
        self,
        files: Union[str, List[str]],
        variable_dict: dict = None,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads MODIS L2 swath or Gridded EOS data.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s) or URL(s).
        variable_dict : dict, optional
            Dictionary of variables to read with metadata.
        **kwargs : dict
            Additional arguments passed to the reader.

        Returns
        -------
        xr.Dataset
            The processed MODIS dataset.
        """
        # Custom loading logic for HDF4 if needed, otherwise use XarrayDriver
        # Note: MODIS L2 files are often HDF4, which might need specialized handling.

        ds = self.driver.open(files, **kwargs)

        # Standardize coordinates
        # ... (Ported logic from monetio/sat/modis_l2.py)

        return ds
