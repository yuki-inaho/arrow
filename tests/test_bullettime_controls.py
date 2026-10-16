"""Numerical camera/ray/time contracts for the short BulletTime controls export."""

import hashlib
import importlib.util
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_bullettime_controls.py"
SPEC = importlib.util.spec_from_file_location("prepare_bullettime_controls", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


def camera(focal=30.0, center=15.5):
    return np.array([[focal, 0, center], [0, focal, center], [0, 0, 1]], dtype=float)


def test_identity_map_and_sampler_preserve_every_pixel():
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    K = np.array([[12, 0, 3.2], [0, 15, 2.6], [0, 0, 1]], dtype=float)
    x, y = controls.source_map(K, K, rgb.shape[:2])
    np.testing.assert_allclose(x, np.broadcast_to(np.arange(8), (6, 8)), atol=1e-13)
    np.testing.assert_allclose(y, np.broadcast_to(np.arange(6)[:, None], (6, 8)), atol=1e-13)
    np.testing.assert_array_equal(controls.bilinear_rgb(rgb, x, y), rgb)


def test_halfpixel_resize_has_known_sample_centers_and_colors():
    K = camera(10, 1.5)
    affine = controls.resize_affine((4, 4), (2, 2))
    np.testing.assert_array_equal(affine, [[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
    x, y = controls.source_map(K, affine @ K, (2, 2))
    np.testing.assert_allclose(x, [[0.5, 2.5], [0.5, 2.5]])
    np.testing.assert_allclose(y, [[0.5, 0.5], [2.5, 2.5]])
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4) * 10
    result = controls.bilinear_rgb(np.repeat(gray[..., None], 3, axis=2), x, y)
    np.testing.assert_array_equal(result[..., 0], [[25, 45], [105, 125]])


def test_varying_intrinsics_preserve_ray_directions():
    source = np.array([[8, 0, 3], [0, 10, 4], [0, 0, 1]], dtype=float)
    common = np.array([[4, 0, 1], [0, 5, 2], [0, 0, 1]], dtype=float)
    x, y = controls.source_map(source, common, (3, 3))
    np.testing.assert_array_equal(x, [[1, 3, 5]] * 3)
    np.testing.assert_array_equal(y, [[0] * 3, [2] * 3, [4] * 3])
    # Output center (2, 1) has normalized ray ((2-1)/4, (1-2)/5, 1).
    np.testing.assert_allclose(np.linalg.inv(source) @ [x[1, 2], y[1, 2], 1], [.25, -.2, 1])


def test_common_intrinsics_zoom_solves_bounds_and_enforces_cap():
    common, zoom = controls.choose_common_intrinsics(
        np.stack([camera(100), camera(103)]), (32, 32), (32, 32))
    assert zoom == pytest.approx(1.03, abs=1e-12)
    for K in [camera(100), camera(103)]:
        x, y = controls.source_map(K, common, (32, 32))
        assert x.min() >= -1e-12 and y.min() >= -1e-12
        assert x.max() <= 31 + 1e-12 and y.max() <= 31 + 1e-12
    with pytest.raises(ValueError, match="zoom above"):
        controls.choose_common_intrinsics(np.stack([camera(100), camera(120)]),
                                          (32, 32), (32, 32))


def test_normalized_model_time_preserves_frame_index_not_playback_seconds():
    source, target, actual = controls.time_controls(17, 5)
    assert source.dtype == target.dtype == np.float32
    np.testing.assert_allclose(source * 30, np.arange(17), atol=1e-6)
    np.testing.assert_array_equal(target, np.zeros(17, dtype=np.float32))
    assert actual[-1] == 3.2
    assert source[-1] == pytest.approx(16 / 30)
    np.testing.assert_allclose(source, actual * 5 / 30, rtol=1e-7)
    other_source, _, _ = controls.time_controls(17, 24)
    np.testing.assert_array_equal(source, other_source)


@pytest.mark.parametrize("count,fps", [(16, 5), (1, 5), (17, 0), (17, float("nan"))])
def test_invalid_time_controls_fail(count, fps):
    with pytest.raises(ValueError):
        controls.time_controls(count, fps)


def test_invalid_intrinsics_and_outside_samples_fail():
    with pytest.raises(ValueError):
        controls.validate_intrinsics(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        controls.validate_intrinsics(np.eye(4))
    with pytest.raises(ValueError, match="leaves source"):
        controls.bilinear_rgb(np.zeros((4, 4, 3), np.uint8), np.array([[-0.1]]), np.zeros((1, 1)))


def fixture_input(root):
    (root / "frames_rgb").mkdir(parents=True)
    (root / "stages" / "geometry").mkdir(parents=True)
    records, Ks, Ms = [], [], []
    for index in range(5):
        image_path = root / "frames_rgb" / f"{index:06d}.png"
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(32, dtype=np.uint8)[None] * 7
        rgb[..., 1] = index * 30
        Image.fromarray(rgb).save(image_path)
        K = camera(30 + index * .01).astype(np.float32)
        M = np.eye(4, dtype=np.float32)
        M[0, 3] = index * .125
        Ms.append(M)
        Ks.append(K)
        np.savez(root / "stages" / "geometry" / f"{index:06d}.npz", arrow_K=K,
                 arrow_camera_to_metric=M, arrow_K_valid=True, arrow_pose_valid=True)
        records.append({"frame_index": index, "source_index": index,
                        "rgb_path": f"frames_rgb/{index:06d}.png", "timestamp": None,
                        "rgb_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    (root / "frames.json").write_text(json.dumps({"dataset": "fixture", "frame_count": 5,
                                                "original_size": [32, 32], "frames": records}))
    return np.asarray(Ks, dtype=np.float64), np.asarray(Ms, dtype=np.float64)


def test_export_preserves_every_camera_and_original_intrinsic(tmp_path):
    source = tmp_path / "input"
    Ks, Ms = fixture_input(source)
    before = {p: controls.digest(p) for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "output"
    manifest = controls.prepare_controls(source, output, count=5, height=16, width=16, fps=5)
    with np.load(output / "controls.npz") as data:
        np.testing.assert_array_equal(data["camera_to_world"], Ms)
        np.testing.assert_array_equal(data["original_camera_to_world"], Ms)
        np.testing.assert_array_equal(data["original_K"], Ks)
        np.testing.assert_array_equal(data["K"], np.repeat(data["K"][:1], 5, axis=0))
        assert data["rgb"].shape == (5, 16, 16, 3)
        assert data["rgb"].dtype == np.uint8
        assert data["K"].dtype == data["camera_to_world"].dtype == np.float64
        assert data["fps"].item() == 5 and data["model_time_fps"].item() == 30
        np.testing.assert_allclose(data["source_time"], np.arange(5) / 30, rtol=1e-7)
    assert before == {p: controls.digest(p) for p in source.rglob("*") if p.is_file()}
    assert all(b["valid_fraction"] == 1 for b in manifest["boundary_validity"])
    with av.open(str(output / "source_rectified.mp4")) as video:
        average_rate = video.streams.video[0].average_rate
        assert average_rate is not None and float(average_rate) == 5
        assert len(list(video.decode(video=0))) == 5


def test_bad_source_hash_and_dimensions_fail_before_writing(tmp_path):
    source = tmp_path / "input"
    fixture_input(source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="multiples of 16"):
        controls.prepare_controls(source, output, count=5, height=17, width=16)
    image_path = source / "frames_rgb" / "000000.png"
    Image.new("RGB", (32, 32), "red").save(image_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        controls.prepare_controls(source, output, count=5, height=16, width=16)
    assert not output.exists()


def test_invalid_pose_flag_fails_before_writing(tmp_path):
    source = tmp_path / "input"
    fixture_input(source)
    np.savez(source / "stages" / "geometry" / "000000.npz", arrow_K=camera(),
             arrow_camera_to_metric=np.eye(4), arrow_K_valid=True, arrow_pose_valid=False)
    with pytest.raises(ValueError, match="Invalid source calibration"):
        controls.prepare_controls(source, tmp_path / "output", count=5, height=16, width=16)
