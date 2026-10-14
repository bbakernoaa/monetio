"""Property 5: Reader Structure Uniformity

For each reader module in monetio/readers/, inspect structure:
- Verify single @register_reader class (or known multi-class module)
- Correct inheritance from GriddedReader or PointReader (directly or transitively)
- Required methods: open_dataset() and harmonize()
- Non-empty module-level docstring

**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.7**

Since we are checking a fixed set of modules, we use pytest.mark.parametrize
over all reader module names rather than Hypothesis.
"""

import importlib
import inspect

import pytest

from monetio._manifest import SOURCE_MANIFEST
from monetio.readers.base import READER_REGISTRY, GriddedReader, PointReader

# Reader modules come from the manifest (the single source of truth): only
# modules that declare a source are audited, so the infrastructure modules
# (base, drivers, *_specs, *_utils, _deprecation) are excluded by
# construction rather than by a hand-maintained list.
MANIFEST_MODULE_NAMES = sorted({s.module.rsplit(".", 1)[-1] for s in SOURCE_MANIFEST})

# Modules that legitimately contain multiple distinct registered classes are
# exactly the modules whose manifest spec declares aliases — the aliases are
# what signal additional registrations (e.g. gfs also registers gefs/gdas).
MULTI_CLASS_MODULES = {s.module.rsplit(".", 1)[-1] for s in SOURCE_MANIFEST if s.aliases}


def _get_reader_module_names():
    """All audited reader module names, derived from the manifest."""
    return MANIFEST_MODULE_NAMES


READER_MODULE_NAMES = _get_reader_module_names()


def _import_reader_module(module_name):
    """Import a reader module and return it."""
    return importlib.import_module(f"monetio.readers.{module_name}")


def _get_registered_classes(mod, module_name):
    """Get unique classes from this module that are in READER_REGISTRY."""
    full_module = f"monetio.readers.{module_name}"
    registered = []
    for _name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ != full_module:
            continue
        for _reg_name, reg_cls in READER_REGISTRY.items():
            if reg_cls is obj and obj not in registered:
                registered.append(obj)
    return registered


# Feature: virtualizarr-reader-refactor, Property 5: Reader Structure Uniformity


@pytest.mark.parametrize("module_name", READER_MODULE_NAMES)
class TestReaderStructureUniformity:
    """
    **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.7**

    Property 5: For any reader module in monetio/readers/ (excluding infrastructure
    and utility modules), the module SHALL contain exactly one class decorated with
    @register_reader, that class SHALL inherit from GriddedReader or PointReader,
    SHALL implement open_dataset() and harmonize() methods, and the module SHALL
    have a non-empty module-level docstring.
    """

    def test_module_has_docstring(self, module_name):
        """Requirement 10.7: Reader module SHALL include a module-level docstring."""
        mod = _import_reader_module(module_name)
        docstring = mod.__doc__
        assert docstring is not None, f"Module {module_name} has no docstring"
        assert docstring.strip(), f"Module {module_name} has empty docstring"

    def test_has_registered_reader_class(self, module_name):
        """Requirement 10.1: Reader module SHALL contain exactly one @register_reader class."""
        mod = _import_reader_module(module_name)
        registered = _get_registered_classes(mod, module_name)

        assert len(registered) >= 1, f"Module {module_name} has no @register_reader decorated class"

        if module_name not in MULTI_CLASS_MODULES:
            assert len(registered) == 1, (
                f"Module {module_name} has {len(registered)} distinct registered classes "
                f"({[c.__name__ for c in registered]}), expected exactly 1"
            )

    def test_inherits_from_base_reader(self, module_name):
        """Requirement 10.2: Reader class SHALL inherit from GriddedReader or PointReader."""
        mod = _import_reader_module(module_name)
        registered = _get_registered_classes(mod, module_name)

        for cls in registered:
            assert issubclass(cls, GriddedReader | PointReader), (
                f"Class {cls.__name__} in {module_name} does not inherit from "
                f"GriddedReader or PointReader. MRO: {[c.__name__ for c in cls.__mro__]}"
            )

    def test_implements_open_dataset(self, module_name):
        """Requirement 10.3: Reader class SHALL implement open_dataset()."""
        mod = _import_reader_module(module_name)
        registered = _get_registered_classes(mod, module_name)

        for cls in registered:
            assert hasattr(cls, "open_dataset"), (
                f"Class {cls.__name__} in {module_name} missing open_dataset()"
            )
            assert callable(getattr(cls, "open_dataset")), (
                f"Class {cls.__name__} in {module_name}: open_dataset is not callable"
            )

    def test_implements_harmonize(self, module_name):
        """Requirement 10.4: Reader class SHALL implement harmonize()."""
        mod = _import_reader_module(module_name)
        registered = _get_registered_classes(mod, module_name)

        for cls in registered:
            assert hasattr(cls, "harmonize"), (
                f"Class {cls.__name__} in {module_name} missing harmonize()"
            )
            assert callable(getattr(cls, "harmonize")), (
                f"Class {cls.__name__} in {module_name}: harmonize is not callable"
            )
