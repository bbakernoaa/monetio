"""PREP-CHEM-SOURCES Reader"""

import xarray as xr

from .base import GriddedReader, register_reader


@register_reader("prepchem")
class PrepChemReader(GriddedReader):
    """
    Reader for PREP-CHEM-SOURCES binary data.
    """

    def open_dataset(
        self,
        files,
        dtype="f4",
        res="C384",
        tile=1,
        **kwargs,
    ) -> xr.Dataset:
        """
        Reads PREP-CHEM-SOURCES binary files.

        Parameters
        ----------
        files : Union[str, List[str]]
            File path(s).
        dtype : str, optional
            Data type, by default 'f4'.
        res : str, optional
            Resolution (e.g. 'C384'), by default 'C384'.
        tile : int, optional
            Tile number (1-6), by default 1.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        xr.Dataset
            The processed dataset.
        """
        from scipy.io import FortranFile

        # Handle file expansion
        if isinstance(files, str):
            fname = files
        else:
            fname = files[0]

        w = FortranFile(fname)
        a = w.read_reals(dtype=dtype)
        r = int(res[1:])
        s = a.reshape((r, r), order="F")

        # Basic DataArray for now
        da = xr.DataArray(s, dims=("x", "y"), name="emission")
        ds = da.to_dataset()

        return ds
