"""Tests for the deprecation warning decorator in monetio/readers/_deprecation.py."""

import warnings

import pytest

from monetio.readers._deprecation import deprecated_wrapper


class TestDeprecatedWrapper:
    """Tests for the deprecated_wrapper decorator."""

    def test_emits_deprecation_warning(self):
        """Decorated function emits a DeprecationWarning when called."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_warning_message_contains_legacy_name(self):
        """Warning message includes the legacy function name."""
        legacy_name = "monetio.models.cmaq.open_dataset"

        @deprecated_wrapper(legacy_name, 'monetio.load("cmaq", files=...)')
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert legacy_name in str(w[0].message)

    def test_warning_message_contains_load_equivalent(self):
        """Warning message includes the recommended monetio.load() call."""
        load_equiv = 'monetio.load("cmaq", files=...)'

        @deprecated_wrapper("monetio.models.cmaq.open_dataset", load_equiv)
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert load_equiv in str(w[0].message)

    def test_warning_message_contains_removal_version(self):
        """Warning message includes the target removal version."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
            removal_version="0.5.0",
        )
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert "v0.5.0" in str(w[0].message)

    def test_default_removal_version_is_0_4_0(self):
        """Default removal version is 0.4.0 when not specified."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert "v0.4.0" in str(w[0].message)

    def test_warning_message_format(self):
        """Warning message matches the expected format exactly."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
            removal_version="0.4.0",
        )
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            expected = (
                "monetio.models.cmaq.open_dataset is deprecated and will be removed in v0.4.0. "
                'Use monetio.load("cmaq", files=...) instead.'
            )
            assert str(w[0].message) == expected

    def test_decorated_function_returns_original_value(self):
        """Decorated function still returns the original function's return value."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def dummy():
            return 42

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = dummy()
            assert result == 42

    def test_decorated_function_passes_args_and_kwargs(self):
        """Decorated function forwards positional and keyword arguments."""

        @deprecated_wrapper(
            "monetio.obs.airnow.add_data",
            'monetio.load("airnow", files=...)',
        )
        def dummy(a, b, key=None):
            return (a, b, key)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = dummy(1, 2, key="val")
            assert result == (1, 2, "val")

    def test_preserves_function_name(self):
        """Decorated function preserves the original function's __name__."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def open_dataset():
            pass

        assert open_dataset.__name__ == "open_dataset"

    def test_preserves_function_docstring(self):
        """Decorated function preserves the original function's __doc__."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def open_dataset():
            """Original docstring."""
            pass

        assert open_dataset.__doc__ == "Original docstring."

    def test_warning_uses_deprecation_warning_category(self):
        """Warning uses the DeprecationWarning category (Requirement 9.3)."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def dummy():
            return "result"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dummy()
            assert w[0].category is DeprecationWarning

    def test_decorated_function_propagates_exceptions(self):
        """Decorated function propagates exceptions from the original function."""

        @deprecated_wrapper(
            "monetio.models.cmaq.open_dataset",
            'monetio.load("cmaq", files=...)',
        )
        def dummy():
            raise ValueError("test error")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="test error"):
                dummy()
