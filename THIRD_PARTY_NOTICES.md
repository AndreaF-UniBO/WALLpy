# Third-party software and model notices

WALLpy does not vendor third-party source code, datasets, or model weights. Python packages are installed separately by `pip`, and SAM checkpoints are obtained separately by the user. Each component remains subject to its own upstream terms.

This notice is informational and is not a substitute for reviewing the exact license shipped with the version being installed.

## Core runtime dependencies

The declared core dependencies use permissive licenses, with some binary distributions containing additional notices:

| Component | Upstream license information |
|---|---|
| NumPy | BSD-3-Clause and notices for bundled or vendored components |
| Pillow | HPND / MIT-CMU family |
| OpenCV Python packages | Apache-2.0 |
| scikit-learn | BSD-3-Clause |
| scikit-image | BSD licenses and notices for individual components |
| SciPy | BSD-3-Clause and notices for bundled numerical libraries |
| ttkbootstrap | MIT, with separately licensed included themes or assets |
| ezdxf | MIT |

When distributing a bundled executable or offline environment in the future, collect and include the license texts and notices from the exact dependency versions in that distribution.

## Optional SAM 2 backends

- [Meta SAM 2](https://github.com/facebookresearch/sam2) is provided under Apache-2.0. Its checkpoints are downloaded separately and must be reviewed under the applicable upstream model terms.
- [Ultralytics](https://github.com/ultralytics/ultralytics) is provided under AGPL-3.0, with other licensing options described by its publisher. Installing WALLpy's `sam` extra installs Ultralytics as a separate runtime dependency; users and redistributors must evaluate and comply with those terms for their intended use.

No SAM checkpoint or Ultralytics source file is tracked in the WALLpy repository.

