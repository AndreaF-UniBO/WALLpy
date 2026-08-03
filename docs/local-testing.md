# Release verification record

Verified on 3 August 2026 in the local Windows environment at `C:\WALLpy\PyWALL_v13`.

## Environment

- Python: 3.11
- PyWALL: 0.13.0 (display name: PyWALL v13)
- PyTorch: 2.5.1+cu121
- Meta SAM 2: official package at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB
- CUDA available through PyTorch: yes
- Installed local-bundle size: 5.16 GiB
- Virtual-environment size: 4.84 GiB

## Core checks

- [x] Clean Python 3.11 virtual environment created
- [x] Core dependencies installed
- [x] Python sources compile
- [x] Unit and smoke tests pass
- [x] Hidden-window GUI smoke test passes
- [x] PyWALL v13 title and action buttons checked
- [x] Historical Segmentazione DL control absent
- [x] K-Means processes all six local samples
- [x] PNG mask export checked programmatically
- [x] DXF contour export checked programmatically

K-Means timings in the final environment:

| Sample | Dimensions | Result | Time |
|---|---:|---|---:|
| `071.png` | 512 × 512 | 3 clusters | 21.23 s |
| `080.png` | 1024 × 1024 | 3 clusters | 83.75 s |
| `1.jpg` | 1024 × 1024 | 3 clusters | 29.49 s |
| `input_02.png` | 1024 × 1024 | 3 clusters | 30.13 s |
| `input_03.png` | 1024 × 1024 | 3 clusters | 28.15 s |
| `input_04.png` | 1024 × 1024 | 3 clusters | 29.56 s |

An earlier diagnostic batch was stopped by its five-minute harness limit after parallelism had deliberately been restricted to one logical core. It was repeated with normal CPU scheduling and all six files completed successfully in 225.4 seconds.

## Meta SAM 2 checks

- [x] Official Meta package installed
- [x] Official checkpoint downloaded from Meta's host
- [x] Checkpoint SHA-256 verified
- [x] CUDA device detected
- [x] Reduced real-model smoke test passes with the official checkpoint
- [x] Returned mask dimensions, data type and values validated
- [x] User confirmed that PyWALL v13 functions correctly in local interactive use

Checkpoint:

```text
checkpoints/sam2.1_hiera_base_plus.pt
323,606,802 bytes
SHA-256 A2345AEDE8715AB1D5D31B4A509FB160C5A4AF1970F199D9054CCFB746C004C5
```

The current sample-publication candidate passed 12 tests with the optional SAM 2 test skipped. With the official checkpoint enabled, all 13 tests passed in 17.95 seconds. The suite verifies that the three published photographs exist, remain readable, and contain no EXIF metadata; pixel equality with the protected originals was checked separately before staging. The real SAM 2 test uses a generated synthetic image.

## Manual validation

The user launched the application locally and confirmed the interactive workflow. Scientific interpretation of every new dataset still requires case-specific expert review.

## Publication controls

- [x] `.venv` excluded
- [x] Meta checkpoint excluded and reproducible download documented
- [x] Three author-owned photographs published after explicit authorization and metadata removal
- [x] Three remaining local photographs excluded because publication has not been authorized
- [x] Repository URLs and public screenshot updated for PyWALL v13
- [x] Final test suite rerun on the publication candidate
