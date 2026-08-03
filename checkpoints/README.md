# SAM 2 checkpoints

Model weights are not stored in this repository. They are large third-party artifacts and remain subject to their upstream terms.

The simplest supported option is the Ultralytics backend:

```powershell
python -m pip install ".[sam]"
```

WALLpy then requests the selected `sam2.1_*.pt` weight through Ultralytics when it is first needed.

For Meta's official SAM 2 backend, install the upstream package and obtain one matching SAM 2 or SAM 2.1 checkpoint from the official project. Place the file in this directory using one of the names recognized in `sam2_segmentation.py`, such as `sam2.1_hiera_base_plus.pt` for the default model size.

Before using or redistributing any checkpoint, consult the license and model-card information published by its provider. Do not commit downloaded weights to this repository.

