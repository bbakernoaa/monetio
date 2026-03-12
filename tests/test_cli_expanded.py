from unittest.mock import patch

import pandas as pd
import xarray as xr
from click.testing import CliRunner

from monetio.cli import cli


def test_cli_load_generic():
    runner = CliRunner()
    with patch("monetio.load") as mock_load:
        mock_load.return_value = xr.Dataset()
        result = runner.invoke(
            cli, ["load", "cmaq", "-d", "2023-01-01", "-k", "mech=cb6r3", "-k", "surf_only=True"]
        )
        assert result.exit_code == 0
        assert "Loading cmaq data" in result.output

        mock_load.assert_called_once()
        args, kwargs = mock_load.call_args
        assert args[0] == "cmaq"
        assert kwargs["dates"] == ["2023-01-01"]
        assert kwargs["mech"] == "cb6r3"
        assert kwargs["surf_only"] is True


def test_cli_load_observation():
    runner = CliRunner()
    with patch("monetio.load") as mock_load:
        mock_load.return_value = pd.DataFrame({"a": [1]})
        result = runner.invoke(cli, ["load", "ish", "-d", "2023-01-01", "--as-pandas"])
        assert result.exit_code == 0
        assert "Loading ish data" in result.output

        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == "ish"
        assert mock_load.call_args[1]["as_xarray"] is False
