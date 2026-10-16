"""Exercise the real static-world consumer on a moving, still-present surface."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_static_world.py"


def fixture(root):
    directory = root / "grass"
    for stage in ("final", "sr", "geometry"):
        (directory / "stages" / stage).mkdir(parents=True)
    (directory / "frames_rgb").mkdir()
    h, w = 24, 32
    K = np.array([[40, 0, 15.5], [0, 40, 11.5], [0, 0, 1]], dtype=np.float32)
    records = []
    for i in range(3):
        depth = np.full((h, w), 2, np.float32)
        valid = np.ones((h, w), bool)
        depth[0, 0] = np.nan
        valid[0, 0] = False
        flow = np.zeros((h, w, 3), np.float32)
        flow[..., 0] = 0.05  # exactly one pixel/frame; object exists and moves
        flow_valid = valid.copy() if i < 2 else np.zeros_like(valid)
        np.savez(directory / f"stages/final/{i:06d}.npz", final_depth=depth, final_valid=valid)
        np.savez(directory / f"stages/sr/{i:06d}.npz", sr_flow=flow, sr_valid=flow_valid)
        np.savez(directory / f"stages/geometry/{i:06d}.npz", arrow_K=K,
                 arrow_camera_to_metric=np.eye(4), arrow_K_valid=True, arrow_pose_valid=True)
        image = np.zeros((h, w, 3), np.uint8)
        image[:] = [120 + i * 10, 200, 40]
        path = directory / f"frames_rgb/{i:06d}.png"
        Image.fromarray(image).save(path)
        records.append({"frame_index": i, "source_index": i, "rgb_path": str(path)})
    (directory / "frames.json").write_text(json.dumps({"dataset": "grass", "frame_count": 3,
                    "original_size": [h, w], "frames": records}))
    return directory


def test_cli_keeps_moving_surface_and_freezes_model(tmp_path):
    fixture(tmp_path / "input")
    out = tmp_path / "output"
    process = subprocess.run([sys.executable, str(SCRIPT), "--input-dir", str(tmp_path / "input"),
                 "--output-dir", str(out), "--dataset", "grass", "--model-stride", "1",
                 "--splat-radius", "0", "--min-free-gib", "0"],
                 capture_output=True, text=True, timeout=120)
    assert process.returncode == 0, process.stdout + process.stderr
    target = out / "grass" / "static_world"
    report = json.loads((target / "manifest.json").read_text())
    assert report["frame_count"] == 3 and report["pair_count"] == 2
    assert report["metric_calibrated"] is False
    with np.load(target / "model.npz") as model:
        origin = model["birth_frame"] == 0
        assert origin.sum() == 767
        assert np.all(model["points"][origin, 2] == 2)
        assert np.all(model["colors"][origin] == [120, 200, 40])
        assert model["observation_count"][origin].max() >= 2
    with np.load(target / "motion/000000.npz") as motion:
        support = motion["valid"]
        assert np.allclose(motion["rigid_flow"][support], 0)
        assert np.allclose(motion["nonrigid_flow"][support, 0], 1)
    for i in range(1, 3):
        with np.load(target / "rgbd/000000.npz") as a, np.load(target / f"rgbd/{i:06d}.npz") as b:
            assert np.array_equal(a["rgb"], b["rgb"])
            assert np.array_equal(a["valid"], b["valid"])
            np.testing.assert_array_equal(a["depth"], b["depth"])
            assert np.all(np.isnan(a["depth"][~a["valid"]]))
    assert (target / "grass_static_rgbd.mp4").stat().st_size > 0
    assert (target / "grass_static_world.rrd").stat().st_size > 0


def test_tiny_output_budget_refuses_before_build(tmp_path):
    fixture(tmp_path / "input")
    process = subprocess.run([sys.executable, str(SCRIPT), "--input-dir", str(tmp_path / "input"),
                 "--output-dir", str(tmp_path / "output"), "--dataset", "grass",
                 "--max-output-gib", "0.000001", "--min-free-gib", "0"],
                 capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert not list((tmp_path / "output").rglob("model.npz"))


def test_source_input_is_not_a_valid_output_directory(tmp_path):
    fixture(tmp_path / "input")
    process = subprocess.run([sys.executable, str(SCRIPT), "--input-dir", str(tmp_path / "input"),
                 "--output-dir", str(tmp_path / "input"), "--dataset", "grass"],
                 capture_output=True, text=True, timeout=30)
    assert process.returncode != 0


def test_importable_consumer():
    spec = importlib.util.spec_from_file_location("build_static_world", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_unmatched_surface_inherits_local_inverse_deformation():
    spec = importlib.util.spec_from_file_location("build_static_world", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    y, x = np.mgrid[:20, :20]
    original = np.stack([x * 0.01, y * 0.01, np.full_like(x, 2)], axis=-1).astype(np.float32)
    live = original + np.array([0.3, -0.1, 0], np.float32)
    labels = np.arange(400).reshape(20, 20)
    labels[10, 10] = -1
    delta, support = module.birth_corrections(live, labels, original.reshape(-1, 3),
        np.full((20, 20), 2, np.float32), np.full((20, 20, 3), 128, np.uint8), np.array([210]))
    assert support[0] >= 4
    np.testing.assert_allclose(delta[0], [0.3, -0.1, 0], atol=1e-6)
    np.testing.assert_allclose(live[10, 10] - delta[0], original[10, 10], atol=1e-6)
    # A different depth surface must not borrow its deformation.
    other_depth = np.full((20, 20), 3, np.float32)
    other_depth[10, 10] = 2
    delta, support = module.birth_corrections(live, labels, original.reshape(-1, 3),
        other_depth, np.full((20, 20, 3), 128, np.uint8), np.array([210]))
    assert support[0] == 0
    np.testing.assert_array_equal(delta, 0)


@pytest.mark.parametrize("value", [0, -1])
def test_invalid_stride_rejected(value):
    process = subprocess.run([sys.executable, str(SCRIPT), "--model-stride", str(value)],
                             capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
