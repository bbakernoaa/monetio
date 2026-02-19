# Mock pytspack if not present
import importlib.util
import sys
from unittest.mock import MagicMock, patch

import dask.dataframe as dd
import numpy as np
import pytest

if importlib.util.find_spec("pytspack") is None:
    mock_pytspack = MagicMock()
    mock_pytspack.TsPack.return_value.interpolate.return_value = lambda wv: np.full(len(wv), 0.07)
    sys.modules["pytspack"] = mock_pytspack

from monetio.readers.aeronet import AERONETReader


@pytest.mark.parametrize("lazy", [True, False])
def test_aeronet_metadata_robustness(lazy):
    """Test that AERONET reader is robust to empty or bad files when using lazy loading."""

    good_content = b"""AERONET Data
Version 3
AOD
Some info
Another info
Time(hh:mm:ss),Date(dd:mm:yyyy),Site,Latitude,Longitude,AOD_440nm,AOD_675nm,AOD_870nm,AOD_1020nm,440-870_Angstrom_Exponent
00:00:00,01:01:2024,TEST,0.0,0.0,0.1,0.05,0.03,0.02,1.0
"""
    empty_content = b"""AERONET Data
Version 3
AOD
Some info
Another info
Time(hh:mm:ss),Date(dd:mm:yyyy),Site,Latitude,Longitude,AOD_440nm,AOD_675nm,AOD_870nm,AOD_1020nm,440-870_Angstrom_Exponent
"""
    bad_content = b"<html><body>Error</body></html>"

    def mock_get(url, **kwargs):
        resp = MagicMock()
        if "good" in str(url):
            resp.content = good_content
        elif "empty" in str(url):
            resp.content = empty_content
        else:
            resp.content = bad_content
        resp.raise_for_status.return_value = None
        return resp

    with patch("requests.Session.get", side_effect=mock_get):
        reader = AERONETReader()

        # Test loading a mix of good, empty and bad URLs
        files = ["http://good", "http://empty", "http://bad"]

        df = reader.open_dataset(
            files=files, detect_dust=True, interp_to_aod_values=[550.0], lazy=lazy, as_xarray=False
        )

        if lazy:
            assert isinstance(df, dd.DataFrame)
            # This should not trigger immediate compute of all partitions
            # but open_dataset now computes meta from the first file.

        res = df.compute() if hasattr(df, "compute") else df

        assert len(res) == 1
        assert "dust" in res.columns
        assert "aod_550nm" in res.columns
        assert res["siteid"].iloc[0] == "TEST"

        # Check that dtypes are consistent (not objects for numeric)
        assert res["aod_440nm"].dtype == np.float64
        assert res["latitude"].dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__])
