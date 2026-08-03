# Changelog

All notable changes to PyWALL are documented in this file. The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use [Semantic Versioning](https://semver.org/) for public releases.

## [Unreleased]

### Documentation

- Linked the 2026 *Archeologia e Calcolatori* paper and clarified that it documents the 2025 DynUNet/TopoMortar implementation, whereas PyWALL v13 uses Meta's official SAM 2 backend.

### Added

- Three author-owned masonry photographs (`071.png`, `080.png`, and `input_04.png`) as reproducible sample inputs.

### Security

- Removed embedded EXIF, location, device, date, and colour-profile metadata from the published sample copies while preserving their pixels.

## [0.13.0] - 2026-08-03

### Added

- New PyWALL v13 application identity and `pywall` command.
- Official Meta SAM 2 backend with explicit local checkpoint discovery.
- Windows scripts for isolated installation, verified checkpoint download, testing and startup.
- Local-only `samples` directory for pre-publication testing.
- Optional real-model SAM 2 integration smoke test.

### Changed

- K-Means is identified explicitly as the self-contained baseline workflow.
- SAM 2 generator caching now includes the resolved checkpoint path.
- Public documentation now distinguishes source distribution from local test assets.

### Removed

- Historical **Segmentazione DL** button and TopoMortar/DynUNet inference pipeline.
- Ultralytics backend, dependency and automatic weight download.

### Security

- Model downloads require an explicit script invocation and SHA-256 verification.
- Checkpoints, virtual environments, generated outputs and test photographs remain excluded from Git.

## [0.1.0] - 2026-08-03

### Added

- Initial WALLpy public source release under Apache-2.0.
