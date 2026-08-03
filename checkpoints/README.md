# Official Meta SAM 2 checkpoint

PyWALL v13 supports only the official [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) backend.

The default model requires this local file:

```text
checkpoints/sam2.1_hiera_base_plus.pt
```

Download and verify it explicitly with:

```powershell
.\scripts\download_sam2_checkpoint.ps1
```

Official source:

```text
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```

Expected SHA-256:

```text
a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5
```

The checkpoint is a large third-party artifact. Keep it available for local testing, but do not commit it to Git. Before any redistribution, review the current terms and attribution published by Meta.
