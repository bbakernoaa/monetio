# Offline Reader Samples and S3 CI Bundle

## Intent

Give every registered MONETIO reader at least one small, loadable offline example so parser and reader changes can be checked without depending on the upstream data service. Prefer a small real source file or captured response body wherever it can be retrieved and redistributed, including for readers whose normal entry point fetches data from an API. Save those bytes locally and add a local parsing path so tests do not repeat the retrieval. Use a deterministic synthetic file in the reader's accepted input format only when real bytes cannot be obtained or shared. Publish a versioned sample set to S3 and have CI fetch and verify it before the offline sample suite runs.

The reader outputs remain governed by the reader APIs. This work adds test inputs and local paths where needed; it does not change scientific calculations or claim that synthetic examples validate the real product.

## Current state and constraints

- The source manifest declares 84 registered source names.
- The sample-spec catalog currently accounts for 49 names: 21 captured payloads, 5 synthetic samples, and 23 reasoned exclusions.
- The conformance map currently contains 40 fixture entries and 44 exclusions. Some fixture entries use legacy test-local files instead of persistent per-source sample directories.
- There are 28 persistent sample directories under tests/data/samples/ and tests/data/synthetic/, totaling about 15 MiB. Tests resolve these directories through SampleRegistry; CI currently uses checked-out files and has no S3 fixture step. Several tests are marked network, but CI runs the full suite without excluding that marker, so live upstream retrievals can occur during ordinary CI.
- Per-source manifest.json files already record provenance and a SHA-256 hash. test_sample_coverage.py checks that every registered source is either sampled or explicitly excluded.
- Several readers need investigation before they can have a loadable local example. The exclusion notes identify remote-only paths, the retired MODIS ORNL endpoint, HDF4 decoder gaps, and known SOLRAD/TROPOMI parsing issues.
- No bucket URI or AWS role is configured in this repository. CI must receive the bucket prefix and short-lived AWS credentials through repository/environment configuration; credentials must not be committed.

## Goals

1. Every name in monetio.available_sources() resolves to exactly one persistent offline sample and manifest.
2. Each sample is accepted by the registered reader's local input path and runs without contacting its upstream service.
3. Captured samples use redistributable real payloads. Synthetic samples are deterministic, minimal, and identify their producer and synthetic status.
4. Every sample has a stable SHA-256, source/load arguments, and any optional decoder requirement recorded in its manifest.
5. A versioned S3 sample set can be downloaded into the existing test-data paths and verified before pytest runs.
6. Local developers can continue running the sample suite from repository files without AWS credentials.
7. CI can exercise the same sample paths after fetching the pinned S3 set, without contacting source data services.
8. The standard pytest run makes no live source-service requests. Network checks are opt-in and limited to a small smoke set.

## Non-goals

- Replacing the source-specific reader parsers with one generic parser.
- Claiming synthetic bytes are scientifically representative or validating source-service URL construction.
- Recording user data, credentials, or gated data without redistribution permission.
- Removing the checked-in samples in the first delivery.
- Automatically refreshing source samples on every CI run.

## Approaches considered

1. **Hybrid real and synthetic samples, with a versioned S3 bundle (selected).** Reuse the existing capture and synthetic specification tables, add samples for all current exclusions, and provide a common prepare/publish/fetch workflow. This meets the one-file-per-reader target while retaining provenance. Synthetic samples test the file and parser contract but not real-source quirks.
2. **Real payloads only.** This gives the strongest source fidelity but cannot cover retired services, gated sources, or sources with no redistributable example. It does not meet the one-file-per-reader target under current access conditions.
3. **S3 sync only.** Upload the existing sample directories and make CI download them. This is smaller but preserves the current exclusions and untracked fixture gaps, distributing an incomplete set as if it were complete.

The user selected the hybrid route and approved synthetic examples where a real, accessible, redistributable payload is unavailable.

## Design

### Sample preparation and coverage

- Build on record_samples.py, make_synthetic_sample.py, sample_specs/, and the current manifest schema rather than creating a second source inventory.
- Add a single preparation entry point for maintainers. It processes each registered source through its capture or synthetic specification, leaves existing samples unchanged unless refresh is requested, writes manifests, and validates that no registered source is missing a sample.
- Convert the current exclusions into one of: a public capture spec, or a deterministic native-format synthetic generator. For a reader that normally retrieves data from an API or remote catalog, first try to capture one minimal real response/file and preserve the bytes as a local sample. Synthetic generation is the fallback for retired, inaccessible, credential-gated, or non-redistributable sources. The existing exclusion list remains useful as an audit during migration, but the completed coverage gate must fail if any registered source remains excluded.
- Where a reader only fetches remote data today, add a local fixture route or isolate the response parser so the saved bytes can be loaded without a source-service call. Store the source URL and request parameters that are safe to publish in the manifest; never store credentials. Use source-specific load_kwargs in the manifest where required.
- Keep examples minimal while retaining valid file structure. Record optional parser/decoder packages in requires; the CI environment installs supported required decoders where practical. A sample file and hash remain mandatory even when a platform-specific optional decoder cannot run on every CI OS. Each sample must load successfully in at least one designated supported CI job; skips are allowed only on other matrix legs where the decoder is unavailable.
- Change the coverage tests to require a sample directory, payload, valid manifest, and local load path for every registered source. Preserve explicit checks that network-facing real samples name their origin and synthetic samples state why they are synthetic.

### S3 bundle

- Add a CLI for publishing and fetching the complete sample tree. It accepts an explicit S3 URI and immutable sample-set version; it does not hard-code a bucket.
- Store one root index with the reader names, relative paths, lengths, hashes, and source manifest references. Keep per-reader manifests next to their payloads.
- Publish to an immutable prefix keyed by a sample-set identifier (initially the repository commit that produced the set). Do not overwrite a previously published version. Upload and download operations verify hashes and report missing, extra, or mismatched files.
- Use the AWS default credential chain through the supported S3 filesystem package. CI obtains short-lived credentials with GitHub Actions OIDC and an AWS role; no static access keys enter repository files.
- Fetch into the existing tests/data/samples/ and tests/data/synthetic/ paths so the sample registry and conformance loaders continue to work.

### CI and local use

- Add an opt-in CI restore step before pytest. It uses repository/environment variables for the S3 URI, pinned sample-set version, and AWS role. If the bucket configuration is not present, CI continues using checked-in sample files and reports that S3 restore was skipped.
- Fetch and validate the pinned S3 bundle once per workflow, then distribute it to matrix jobs as a workflow artifact or version-keyed cache so each Python/OS job does not repeat the S3 retrieval.
- Run the regular pytest command with the network marker excluded. Audit existing unmarked network calls, convert reader-behavior tests to fixtures or mocked transports, and keep only a small allowlist of genuine source-connectivity smoke tests in a scheduled or manually dispatched job.
- Once configured, CI verifies the root index and all payload hashes before the local suite. The source services are not contacted by pytest.
- Document how maintainers run preparation, publish a new version, point CI to it, and restore a version locally. Keep checked-in samples as the local no-AWS fallback in this delivery.

## Acceptance criteria

- There is one sample directory with at least one non-empty payload and a valid manifest for every registered reader; there are no remaining sample-coverage exclusions.
- Each source's conformance entry loads that file through the registered reader without contacting the upstream service, in at least one supported CI job.
- Captured samples remain redistributable and have provenance. Synthetic samples are reproducible and clearly marked.
- The preparation command is deterministic for synthetic payloads and does not overwrite existing samples without an explicit refresh request.
- S3 publish/fetch is versioned, validates SHA-256 before reporting success, and never requires embedded credentials.
- CI fetches and validates a pinned sample set once, distributes it to matrix jobs, and runs the default pytest suite with no live source-service calls; local testing continues to work from checked-in samples without AWS.
- The remaining live retrieval tests are explicitly marked and limited to an opt-in scheduled/manual smoke set.
- Documentation describes the preparation, publish, local restore, and CI configuration steps.

## Implementation calls left to the builder

- Choose the smallest fixture that exercises each reader's actual local parser; do not use a generic placeholder that only tests NetCDF opening.
- Keep network access in sample preparation separate from the offline CI load.
- If a reader cannot currently accept a local file, add the narrowest local input route consistent with its parser and document that route in its sample manifest.
- Preserve the current checked-in files unless their replacement by a verified S3 copy has been confirmed.
- The bucket name, prefix, and AWS role ARN are deployment configuration to be supplied by the repository owner; the CLI and workflow must work with those values without code changes.
