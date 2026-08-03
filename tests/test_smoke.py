from pathlib import Path
import tkinter as tk
import tomllib

import numpy as np
import pytest
import yaml
from PIL import Image

import PyWALL_v13
import sam2_segmentation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gui_app():
    root = tk.Tk()
    root.withdraw()
    app = PyWALL_v13.PyWALLApp(root)
    root.update_idletasks()
    try:
        yield root, app
    finally:
        root.destroy()


def test_main_modules_import_and_version_is_consistent():
    assert PyWALL_v13.__version__ == "0.13.0"
    assert callable(PyWALL_v13.main)
    assert callable(sam2_segmentation.segment_wall_sam2)


def test_gui_branding_and_public_segmentation_actions(gui_app):
    root, app = gui_app
    assert root.title() == "PyWALL v13 – Segmentazione murature"
    assert app.process_button.cget("text") == "🧮 K-Means"
    assert app.process_sam2_button.cget("text") == "🪄 SAM 2"
    assert not hasattr(app, "process_dl_button")
    assert not hasattr(app, "run_dl_segmentation")


def test_png_and_dxf_exports_create_readable_files(gui_app, monkeypatch, tmp_path):
    _, app = gui_app
    errors = []
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = PyWALL_v13.BRICK_MASK_VALUE
    app.cleaned_mask_auto = mask
    app.processed_pil_image = Image.new("RGB", (32, 32), "white")

    monkeypatch.setattr(PyWALL_v13.messagebox, "showinfo", lambda *args, **kwargs: None)
    monkeypatch.setattr(PyWALL_v13.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))

    assert app._find_and_prepare_contour_images()
    assert app.brick_contours

    dxf_path = tmp_path / "contours.dxf"
    monkeypatch.setattr(
        PyWALL_v13.filedialog,
        "asksaveasfilename",
        lambda **kwargs: str(dxf_path),
    )
    app._export_dxf()
    assert dxf_path.is_file() and dxf_path.stat().st_size > 0
    document = PyWALL_v13.ezdxf.readfile(dxf_path)
    assert len(document.modelspace().query("LWPOLYLINE")) >= 1

    png_path = tmp_path / "mask.png"
    monkeypatch.setattr(
        PyWALL_v13.filedialog,
        "asksaveasfilename",
        lambda **kwargs: str(png_path),
    )
    app._export_binary_mask_png()
    with Image.open(png_path) as exported:
        assert exported.mode == "L"
        assert exported.size == (32, 32)

    assert not errors


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


def test_official_checkpoint_discovery(tmp_path):
    checkpoint = tmp_path / "sam2.1_hiera_base_plus.pt"
    checkpoint.touch()

    found, config = sam2_segmentation._find_official_checkpoint(
        "base_plus", [tmp_path]
    )

    assert found == checkpoint
    assert config == "configs/sam2.1/sam2.1_hiera_b+.yaml"


def test_kmeans_pipeline_processes_a_small_rgb_image(tmp_path):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :8] = (60, 55, 50)
    image[:, 8:16] = (145, 95, 65)
    image[:, 16:] = (220, 205, 180)
    image_path = tmp_path / "synthetic-masonry.png"
    Image.fromarray(image).save(image_path)

    original, labels, mortar_index, lab, mortar_lab, brick_lab = (
        PyWALL_v13.process_image_initial_segmentation(image_path)
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


def test_local_samples_are_readable_when_present():
    samples = ROOT / "samples"
    if not samples.is_dir():
        pytest.skip("Local test samples are not included in this checkout")

    files = sorted(path for path in samples.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        pytest.skip("No local test photographs are present in this checkout")
    for path in files:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            assert rgb.width > 0 and rgb.height > 0


def test_public_runtime_has_no_legacy_dl_or_ultralytics_backend():
    main_source = (ROOT / "PyWALL_v13.py").read_text(encoding="utf-8")
    sam_source = (ROOT / "sam2_segmentation.py").read_text(encoding="utf-8")
    pyproject_source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    for forbidden in (
        "run_dl_segmentation",
        "process_dl_button",
        "WALLPY_LEGACY_ROOT",
        "model-12000",
        "pred.py",
    ):
        assert forbidden not in main_source

    assert "ultralytics" not in sam_source.lower()
    assert "ultralytics" not in pyproject_source.lower()


def test_readme_relative_links_resolve():
    expected = [
        "README_PyWALL_v13.md",
        "checkpoints/README.md",
        "samples/README.md",
        "CITATION.cff",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE",
        "NOTICE",
    ]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in expected:
        assert f"]({relative_path})" in readme
        assert (ROOT / relative_path).is_file(), relative_path


def test_project_configuration_files_parse():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["name"] == "pywall"
    assert pyproject["project"]["version"] == PyWALL_v13.__version__
    assert pyproject["project"]["license"] == "Apache-2.0"
    assert pyproject["project"]["scripts"]["pywall"] == "PyWALL_v13:main"
    assert citation["cff-version"] == "1.2.0"
    assert citation["license"] == "Apache-2.0"
    assert workflow["jobs"]["test"]["runs-on"] == "windows-latest"
