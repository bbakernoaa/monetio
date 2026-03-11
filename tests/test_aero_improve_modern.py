import pytest
import xarray as xr

from monetio.readers.improve import IMPROVEReader


def create_mock_improve_file(path):
    """Creates a mock IMPROVE data file."""
    content = """Header info
Data
SiteCode\tEPACode\tDate\tParamCode\tVal\tUnit\tState
S1\t060371103\t2023-01-01\tOZONE\t1.5\tppb\tCA
S1\t060371103\t2023-01-02\tOZONE\t2.0\tppb\tCA
"""
    with open(path, "w") as f:
        f.write(content)


def test_improve_protocol_compliance(tmp_path):
    """Verify IMPROVE processing is backend-agnostic and lazy-friendly."""
    txt_path = tmp_path / "test_improve.txt"
    create_mock_improve_file(txt_path)

    reader = IMPROVEReader()

    # Test Eager
    res_eager = reader.open_dataset(files=str(txt_path), lazy=False, pivot=True)

    # Test Lazy
    res_lazy = reader.open_dataset(files=str(txt_path), lazy=True, pivot=True)

    # Check consistency
    xr.testing.assert_allclose(res_eager, res_lazy.compute())

    # Check history
    assert "history" in res_eager.attrs
    assert "Read IMPROVE data." in res_eager.attrs["history"]

    # Check variable names (case-insensitive search in mock)
    # The reader might keep original case or lowercase depending on harmonize/to_xarray
    assert any("OZONE" in v.upper() for v in res_eager.data_vars)


if __name__ == "__main__":
    pytest.main([__file__])
