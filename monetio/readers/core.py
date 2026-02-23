import datetime
from typing import Optional

import xarray as xr


class GriddedReader:
    """Base class for gridded data readers."""

    def __init__(self) -> None:
        """Initialize the GriddedReader with a default history message."""
        self.history_message = "Modernized via Aero Protocol"

    def update_history(self, ds: xr.Dataset, message: Optional[str] = None) -> xr.Dataset:
        """Update the history attribute of the dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to update.
        message : str, optional
            Custom message to add.

        Returns
        -------
        xr.Dataset
            Dataset with updated history.
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = message or self.history_message
        history = ds.attrs.get("history", "")
        new_history = f"{timestamp}: {msg}\n{history}"
        ds.attrs["history"] = new_history.strip()
        return ds
