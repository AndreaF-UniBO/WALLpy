from pathlib import Path
import tomllib

import numpy as np
import pytest
import yaml
from PIL import Image

import sam2_segmentation
import WALLpy_v12


def test_main_modules_import_and_version_is_consistent():
    assert WALLpy_v12.__version__ == "0.1.0"
    assert callable(WALLpy_v12.main)
    assert callable(sam2_segmentation.segment_wall_sam2)


def test_mask_fusion_filters_candidates_by_relative_area():
    small = np.zeros((10, 10), dtype=bool)
    small[1:3, 1:3] = True
    background = np.zeros((10, 10), dtype=bool)
    background[:8, :] = True

    fused, kept = sam2_segmentation._fuse_masks(
        [small, background],
        h=10,
        w=10,
        min_area_ratio=0.02,
        max_area_ratio=0.50,
    )

    assert kept == 1
    assert fused.dtype == np.uint8
    assert np.array_equal(fused == 255, small)


def test_kmeans_pipeline_processes_a_small_rgb_image(tmp_path):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :8] = (60, 55, 50)
    image[:, 8:16] = (145, 95, 65)
    image[:, 16:] = (220, 205, 180)
    image_path = tmp_path / "synthetic-masonry.png"
    Image.fromarray(image).save(image_path)

    original, labels, mortar_index, lab, mortar_lab, brick_lab = (
        WALLpy_v12.process_image_initial_segmentation(image_path)
    )

    assert original.size == (24, 24)
    assert labels.shape == (24, 24)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert mortar_index in {0, 1, 2}
    assert lab.shape == (24, 24, 3)
    assert mortar_lab.shape == brick_lab.shape == (3,)


def test_sam_input_validation_does_not_load_a_model():
    with pytest.raises(ValueError, match="RGB"):
        sam2_segmentation.segment_wall_sam2(np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="model_size"):
        sam2_segmentation.segment_wall_sam2(
            np.zeros((8, 8, 3), dtype=np.uint8), model_size="unknown"
        )


def test_legacy_root_can_be_configured(monkeypatch, tmp_path):
    (tmp_path / "pred.py").write_text("# test marker\n", encoding="utf-8")
    monkeypatch.setenv("WALLPY_LEGACY_ROOT", str(tmp_path))

    assert WALLpy_v12._find_project_root() == tmp_path.resolve()


def test_readme_relative_links_resolve():
    root = Path(__file__).resolve().parents[1]
    expected = [
        "docs/assets/wallpy-interface.png",
        "checkpoints/README.md",
        "docs/legacy-deep-learning.md",
        "CITATION.cff",
        "THIRD_PARTY_NOTICES.md",
    ]

    readme = (root / "README.md").read_text(encoding="utf-8")
    for relative_path in expected:
        assert f"]({relative_path})" in readme
        assert (root / relative_path).is_file(), relative_path


def test_project_configuration_files_parse():
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    workflow = yaml.safe_load(
        (root / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["version"] == WALLpy_v12.__version__
    assert citation["cff-version"] == "1.2.0"
    assert workflow["jobs"]["test"]["runs-on"] == "windows-latest"
