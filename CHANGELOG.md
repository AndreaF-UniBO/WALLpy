# Changelog

All notable changes to WALLpy will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project intends to use [Semantic Versioning](https://semver.org/) for public releases.

## [0.1.0] - Unreleased

### Added

- Installable Python metadata and a `wallpy` console entry point.
- English project documentation, citation metadata, release notes, and model-acquisition guidance.
- Deterministic unit and smoke tests plus a GitHub Actions workflow.
- Configurable `WALLPY_LEGACY_ROOT` for the optional historical deep-learning integration.

### Changed

- Separated the core runtime dependencies from optional SAM 2 dependencies.
- Made temporary-directory cleanup reliable when the legacy inference process fails.

### Security

- Excluded model weights, local caches, generated output, local backups, and provenance-uncertain example artifacts from version control.

