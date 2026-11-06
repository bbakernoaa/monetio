# dynamical.org Reader Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution. Steps use checkbox syntax for tracking.

**Goal:** Add a catalog-ID based gridded dynamical.org reader to MONETIO.

**Architecture:** Register one gridded reader in the manifest. Delegate catalog resolution and store opening to the optional provider package, apply user selection before materialization, and retain native scientific metadata.

**Tech Stack:** Python, xarray, dynamical-catalog, pytest, optional Dask.

**Spec:** `docs/superpowers/specs/2026-11-04-dynamical-reader-design.md`

## Global constraints

- Preserve native dimensions and metadata; never use a product allowlist or obsolete URLs.
- Respect MONETIO eager defaults and capability guards; do not claim unverified scale capabilities.
- Station GeoParquet products are outside this gridded reader.

## Review focus

- Large remote store: selection precedes any eager materialization.
- Forecast and ensemble axes: retain all dimensions and metadata.
- Missing dependency, dataset ID, variables or selection: informative errors.
- Regional projected grids: retain projection and two-dimensional coordinates.
- Provider API/version changes: verify actual supported API and optional dependency constraints.

### Task 1: Verify provider and establish failing behavior tests

**Files:** Create `tests/test_dynamical.py`; inspect `monetio/readers/base.py`, `monetio/readers/drivers.py`, `monetio/__init__.py`, provider API, and `tests/test_era5.py` (or equivalent existing tests).

- [ ] Inspect `dynamical_catalog.open` signature, supported IDs, storage and chunk behavior in the venv; record any version constraint required by the design.
- [ ] Create small offline xarray fixtures for analysis, forecast with lead time, ensemble, and a projected regional grid. Stub only remote catalog opening; assert returned values, coordinates, units and projection metadata.
- [ ] Add tests for `monetio.load("dynamical", dataset=..., variables=..., sel=...)`, unknown variables, bad selection, missing/empty ID, conflicting `files`, and missing optional dependency. Verify selection occurs before eager loading using an instrumented fixture.
- [ ] Run `.venv/bin/python -m pytest tests/test_dynamical.py -q` and confirm failures reflect missing reader behavior.

### Task 2: Implement and register reader

**Files:** Create `monetio/readers/dynamical.py`; modify `monetio/_manifest.py`, `pyproject.toml`.

- [ ] Implement `DynamicalReader(GriddedReader).open_dataset(self, files=None, *, dataset, variables=None, sel=None, **kwargs) -> xr.Dataset`, with optional dependency import at use time, selection and error handling as specified.
- [ ] Register `dynamical` in the manifest with unvalidated scale capabilities. Add an optional `dynamical` dependency extra with the verified provider version floor.
- [ ] Run `.venv/bin/python -m pytest tests/test_dynamical.py -q`; make all tests pass. Add and verify focused tests before introducing any further supported behavior.
- [ ] If Dask is supported, test eager and lazy equivalents and capability boundaries; do not advertise capabilities without the repository's required audit evidence.

### Task 3: Document and verify integration

**Files:** Modify `docs/models.md` or add `docs/dynamical.md` and link it from the appropriate index; update `tests/test_dynamical.py` if a network smoke test is feasible.

- [ ] Document installation, examples for analysis and forecast, selection, coordinate preservation, attribution, access costs and supported modes.
- [ ] Add a network-marked smoke test selecting a single variable and a tiny region/time from a current catalog product; run it when network access permits.
- [ ] Run `.venv/bin/python -m pytest tests/test_dynamical.py -q`, the project's full test command, and relevant lint checks. Report named failures and unavailable network checks without claiming unverified support.
- [ ] Inspect `git diff` and `git status`; review the implementation against the design and the source manifest contract.
