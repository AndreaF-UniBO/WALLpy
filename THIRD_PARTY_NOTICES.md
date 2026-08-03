# Third-party software and model notices

PyWALL does not vendor third-party source code in its public source distribution. Python packages are installed separately by `pip`. A Meta SAM 2 checkpoint may be downloaded into a local test bundle through an explicit, checksum-verifying script. Each component remains subject to its own upstream terms.

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

## Optional Meta SAM 2 backend

- [Meta SAM 2](https://github.com/facebookresearch/sam2) is provided under Apache-2.0. PyWALL pins the optional source installation to commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` for local reproducibility.
- The default `sam2.1_hiera_base_plus.pt` checkpoint is obtained from Meta's official host and verified against the checksum documented in `checkpoints/SHA256SUMS`.

No SAM checkpoint is tracked in the PyWALL repository.
