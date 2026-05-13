# API Reference

This page provides a high-level overview of the most commonly used MONETIO functions. For a complete reference, see the [Code Reference](reference/monetio/index.md).

## Core API

The primary way to load data in MONETIO is the `load` function.

::: monetio.load

## Virtualization

Virtualization is supported via `monetio.load` with `use_virtualizarr=True` or via the `virtualize` function.

::: monetio.virtualize

## Utility Functions

::: monetio.rename_latlon

::: monetio.rename_to_monet_latlon

::: monetio.dataset_to_monet

::: monetio.coards_to_netcdf

## Readers Base Classes

::: monetio.readers.base.BaseReader
    options:
      show_bases: false

::: monetio.readers.base.GriddedReader
    options:
      show_bases: false

::: monetio.readers.base.PointReader
    options:
      show_bases: false

::: monetio.readers.drivers.FileUtility
