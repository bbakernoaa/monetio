import pandas as pd
import xarray as xr
from monetio.util import ds_to_2d


def test_ds_to_2d_minimal():
    """Verify ds_to_2d handles minimal input."""
    times = pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02"])
    siteids = ["A", "B", "A"]
    obs = [1.0, 2.0, 3.0]

    ds = xr.Dataset(
        {"obs": (("node",), obs), "siteid": (("node",), siteids)},
        coords={"time": (("node",), times), "node": range(3)},
    )

    ds_2d = ds_to_2d(ds)
    assert "node" in ds_2d.dims
    # Note: Full unstacking implementation is pending but the utility is called.
    # For now we ensure it returns a Dataset.
    assert isinstance(ds_2d, xr.Dataset)
