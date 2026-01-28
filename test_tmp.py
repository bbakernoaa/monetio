import warnings
import pytest

def test_warning_caught():
    with pytest.warns(UserWarning, match="test"):
        warnings.warn("test", UserWarning)

def test_warning_not_emitted():
    with pytest.warns(UserWarning, match="test"):
        pass
