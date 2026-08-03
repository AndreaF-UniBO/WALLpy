# Legacy supervised segmentation

WALLpy v12 retains an integration point for an earlier supervised segmentation pipeline. The public application repository does not contain that pipeline's `pred.py`, training dataset, trained model, or generated `output/` tree because their redistribution status has not been established.

The integration expects this structure under a local root:

```text
legacy-wallpy/
├── pred.py
└── output/
    └── supervised/accurate/large/clDiceLoss/1/config.yaml
```

On Windows PowerShell, configure the root for the current session:

```powershell
$env:WALLPY_LEGACY_ROOT = "C:\path\to\legacy-wallpy"
wallpy
```

The configured program is launched with the active Python interpreter, a temporary RGB input padded to multiples of 128, and a temporary output directory. WALLpy removes both temporary directories after success or failure. The five-minute subprocess timeout remains in place.

This adapter has been preserved for compatibility; it is not covered by the standalone public-install smoke tests because the required external scientific model is not distributed with WALLpy.

