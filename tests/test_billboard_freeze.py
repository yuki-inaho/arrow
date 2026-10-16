"""Plane-guide math, observed-input identity and explicit non-geometric output."""
import importlib.util
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_billboard_freeze.py"
SPEC = importlib.util.spec_from_file_location("prepare_billboard_freeze", PATH)
assert SPEC is not None and SPEC.loader is not None
billboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(billboard)


def test_identity_plane_preserves_original_pixels_and_integer_centers():
    rgb = np.arange(4*5*3, dtype=np.uint8).reshape(4, 5, 3)
    mask = np.ones((4, 5), bool)
    actual, valid, footprint, uv, H = billboard.render_plane(
        rgb, mask, mask, np.eye(3), np.eye(3), np.eye(4), np.eye(4), 2, (4, 5))
    assert valid.all() and footprint.all()
    np.testing.assert_array_equal(actual, rgb)
    np.testing.assert_allclose(H, np.eye(3))
    np.testing.assert_allclose(uv, np.moveaxis(np.indices((4, 5))[::-1], 0, -1))


def test_camera_translation_has_expected_sign_and_depth_scale():
    K = np.diag([10., 10., 1.])
    target = np.eye(4)
    target[0, 3] = .5
    H, relative = billboard.plane_homography(K, K, np.eye(4), target, 2)
    np.testing.assert_allclose(H, [[1, 0, -2.5], [0, 1, 0], [0, 0, 1]])
    assert relative[0, 3] == -.5


def test_homography_matches_independent_3d_projection_with_rotated_world_and_K():
    angle = .2
    M0 = np.eye(4)
    M0[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    M0[:3, 3] = [1, -.2, .3]
    Mt = np.eye(4)
    Mt[:3, 3] = [.8, -.1, .1]
    K0 = np.array([[30., .2, 12], [0, 28, 14], [0, 0, 1]])
    Kout = np.array([[16., 0, 7], [0, 15, 8], [0, 0, 1]])
    pixels = np.array([[1, 2, 1], [10, 8, 1], [22, 24, 1]], float)
    original = (pixels @ np.linalg.inv(K0).T)*2.7
    world = original @ M0[:3, :3].T + M0[:3, 3]
    target = (world-Mt[:3, 3]) @ Mt[:3, :3]
    expected = target @ Kout.T
    expected = expected[:, :2]/expected[:, 2, None]
    H, _ = billboard.plane_homography(K0, Kout, M0, Mt, 2.7)
    actual = pixels @ H.T
    np.testing.assert_allclose(actual[:, :2]/actual[:, 2, None], expected, atol=1e-12)


def test_bilinear_donor_trust_rejects_fourth_bad_corner_but_ignores_zero_weight():
    rgb = np.full((2, 2, 3), [0, 200, 0], np.uint8)
    rgb[1, 0] = [255, 0, 0]
    seed = np.array([[True, True], [False, True]])
    result, valid, _, uv, _ = billboard.render_plane(
        rgb, seed, seed, np.eye(3), np.diag([2., 2., 1.]), np.eye(4), np.eye(4), 2, (3, 3))
    assert not valid[1, 1] and np.isnan(uv[1, 1]).all()
    assert valid[0, 1]
    np.testing.assert_array_equal(result[0, 1], [0, 200, 0])


def test_behind_camera_plane_never_supplies_colors_or_footprint():
    rgb = np.full((3, 3, 3), 128, np.uint8)
    mask = np.ones((3, 3), bool)
    target = np.eye(4)
    target[2, 3] = 4
    _, valid, footprint, uv, _ = billboard.render_plane(
        rgb, mask, mask, np.eye(3), np.eye(3), np.eye(4), target, 2, (3, 3))
    assert not valid.any() and not footprint.any() and np.isnan(uv).all()


@pytest.mark.parametrize("distance", [0, -1, np.nan, np.inf])
def test_invalid_plane_distance_fails(distance):
    with pytest.raises(ValueError, match="distance"):
        billboard.plane_homography(np.eye(3), np.eye(3), np.eye(4), np.eye(4), distance)


def test_invalid_pose_or_masks_fail_before_rendering():
    bad = np.eye(4)
    bad[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        billboard.plane_homography(np.eye(3), np.eye(3), np.eye(4), bad, 1)
    with pytest.raises(ValueError, match="RGB, mask"):
        billboard.render_plane(np.ones((2, 2, 3), np.uint8), np.ones((2, 2)), np.ones((2, 2), bool),
                              np.eye(3), np.eye(3), np.eye(4), np.eye(4), 1, (2, 2))


def make_inputs(root):
    base = root / "input"
    for name in ("frames_rgb", "stages/final", "stages/geometry"):
        (base / name).mkdir(parents=True, exist_ok=True)
    size, count, fps = 64, 5, 7.
    common = np.array([[42., 0, 31.5], [0, 42, 31.5], [0, 0, 1]])
    Ks, Ms, frames, records, native = [], [], [], [], []
    depth = np.full((size, size), 2., np.float32)
    for i in range(count):
        K = common.copy()
        K[0, 0] = 40 + .2*i
        K[1, 1] = 40 + .1*i
        M = np.eye(4)
        M[:3, 3] = [i*.01, 0, 0]
        rgb = np.full((size, size, 3), [90, 90, 90], np.uint8)
        rgb[20:44, 24+i:40+i] = [30, 160, 40]
        # Fine texture makes accidental source indexing/rectification observable.
        rgb[20:44, 24+i:40+i, 1] += np.arange(24, dtype=np.uint8)[:, None]
        path = base / "frames_rgb" / f"{i:06d}.png"
        Image.fromarray(rgb).save(path)
        records.append({"frame_index": i, "source_index": i, "rgb_path": str(path.resolve()),
                        "rgb_sha256": billboard.digest(path), "timestamp": i/fps})
        np.savez_compressed(base / "stages/final" / f"{i:06d}.npz", final_depth=depth,
                            final_valid=np.ones((size, size), bool))
        np.savez_compressed(base / "stages/geometry" / f"{i:06d}.npz", arrow_K=K,
                            arrow_camera_to_metric=M, arrow_K_valid=True, arrow_pose_valid=True)
        x, y = billboard.source_map(K, common, (size, size))
        frames.append(billboard.bilinear_rgb(rgb, x, y))
        Ks.append(K)
        Ms.append(M)
        native.append(rgb)
    (base / "frames.json").write_text(json.dumps({"original_size": [size, size], "frames": records}))
    controls = root / "common_controls.npz"
    np.savez_compressed(controls, rgb=np.stack(frames), original_K=Ks,
                        K=np.repeat(common[None], count, 0), camera_to_world=Ms,
                        native_indices=np.arange(count), actual_timestamps=np.arange(count)/fps, fps=np.array(fps))
    reference = root / "observed_reference.npz"
    uv = np.moveaxis(np.indices((size, size))[::-1], 0, -1)
    points = billboard.transform_points(billboard.unproject_depth(depth, Ks[0]), Ms[0])
    np.savez_compressed(reference, points=points.reshape(-1, 3), colors=native[0].reshape(-1, 3),
                        native_uv=uv.reshape(-1, 2), reference_frame=np.array(0))
    return base, controls, reference


def test_build_preserves_inputs_and_RGB_but_never_claims_later_plane_depth(tmp_path):
    base, controls, reference = make_inputs(tmp_path)
    originals = {p: billboard.digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    out = tmp_path / "output"
    report = billboard.build(base, controls, reference, out)
    assert report["status"] == "2D_billboard_experiment_guide_not_3D_reconstruction"
    assert report["plane_depth_arrow_provisional"] == 2
    assert report["source_verified_after_build"]
    with np.load(out / "controls.npz") as data, np.load(controls) as common:
        np.testing.assert_array_equal(data["source_rgb"], common["rgb"])
        np.testing.assert_array_equal(data["draft_rgb"][0], common["rgb"][0])
        for key in ("K", "camera_to_world", "original_K", "actual_timestamps", "native_indices"):
            np.testing.assert_array_equal(data[key], common[key])
        assert data["reference_projected_mask"][1:].any() and data["edit_mask"][1:].any()
        preserved, known, edit = data["source_preserved_mask"], data["reference_projected_mask"], data["edit_mask"]
        assert not (preserved & known).any()
        np.testing.assert_array_equal(preserved | known, ~edit)
        np.testing.assert_array_equal(data["draft_rgb"][preserved], data["source_rgb"][preserved])
        assert np.all(data["draft_rgb"][edit] == 127)
        assert np.isnan(data["canonical_depth"][1:]).all() and not data["geometry_valid"][1:].any()
        assert data["geometry_valid"][0].all() and np.all(data["canonical_depth"][0] == 2)
        assert np.isfinite(data["observed_background_depth"][1:]).any()
        assert np.isnan(data["source_uv"][~known]).all()
        assert np.all(data["source_uv_frame_index"][known] == 0)
        assert not data["warped_known_mask"].any()
    assert billboard.digest(reference) == billboard.digest(out / "reference_model.npz")
    for path, before in originals.items():
        assert billboard.digest(path) == before
    for name, width in (("draft.mp4", 64), ("edit_mask.mp4", 64), ("comparison.mp4", 128)):
        with av.open(str(out / name)) as video:
            stream = video.streams.video[0]
            assert stream.average_rate is not None
            assert stream.width == width and stream.height == 64 and float(stream.average_rate) == 7
            assert sum(1 for _ in video.decode(video=0)) == 5


@pytest.mark.parametrize("mutation,expected", [("pose", "Original K/pose"), ("rgb", "Rectified RGB"),
                                               ("model", "model points"), ("timestamps", "timestamps")])
def test_incompatible_or_forged_inputs_fail(tmp_path, mutation, expected):
    base, controls, reference = make_inputs(tmp_path)
    path = reference if mutation == "model" else controls
    with np.load(path) as data:
        values = {k: data[k].copy() for k in data.files}
    if mutation == "pose":
        values["camera_to_world"][1, 0, 3] += .1
    elif mutation == "rgb":
        values["rgb"][1, 0, 0, 0] ^= 1
    elif mutation == "model":
        values["points"][0, 0] += 1
    else:
        values["actual_timestamps"][1] += .1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match=expected):
        billboard.build(base, controls, reference, tmp_path / "output")
