"""Optional real-model smoke test for a PyWALL v13 source checkout."""

import os
from pathlib import Path

import numpy as np
import pytest
import sam2_segmentation


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("PYWALL_RUN_SAM2_TEST") != "1",
    reason="set PYWALL_RUN_SAM2_TEST=1 to run the real SAM 2 inference test",
)
def test_meta_sam2_segments_a_synthetic_image():
    checkpoint = ROOT / "checkpoints" / "sam2.1_hiera_base_plus.pt"
    if not checkpoint.is_file():
        pytest.fail(f"Official Meta checkpoint not found: {checkpoint}")

    image = np.full((256, 256, 3), (188, 174, 151), dtype=np.uint8)
    image[24:112, 20:120] = (128, 72, 48)
    image[132:232, 34:146] = (154, 92, 58)
    image[48:190, 164:238] = (102, 65, 49)
    mask, info = sam2_segmentation.segment_wall_sam2(
        image,
        model_size="base_plus",
        points_per_side=8,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.92,
        min_area_ratio=0.0005,
        max_area_ratio=0.35,
        max_side=256,
        checkpoint_dirs=[checkpoint.parent],
    )

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert info["backend"] == "Meta SAM 2"
    assert info["masks_total"] >= info["masks_kept"] >= 0
