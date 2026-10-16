"""Actual RRD readback with nonidentity pose and unknown depth; CPU only."""
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest


SPEC = importlib.util.spec_from_file_location(
    "guided_rrd_export_test", Path(__file__).parents[1] / "scripts/export_guided_freeze_rerun.py")
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def fixture_inputs(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = np.arange(96, dtype=np.uint8).reshape(2, 4, 4, 3)
    edit = np.zeros((2, 4, 4), bool)
    edit[1, 1, 2] = True
    draft = source.copy()
    draft[edit] = 127
    raw = np.full_like(source, 200)
    composite = source.copy()
    composite[1, 1, 2] = [200, 200, 200]
    depth = np.full((2, 4, 4), 2, np.float32)
    depth[edit] = np.nan
    background = np.full_like(depth, np.nan)
    background[:, :, 0] = 1.5
    cameras = np.repeat(np.eye(4)[None], 2, axis=0)
    cameras[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    cameras[1, :3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    cameras[:, :3, 3] = [[1, 2, 3], [2, 3, 4]]
    intrinsics = np.repeat(np.array([[31., 0, 1.4], [0, 37, 1.1], [0, 0, 1]])[None], 2, axis=0)
    controls: dict[str, Any] = dict(source_rgb=source, draft_rgb=draft, reference_rgb=source[0], edit_mask=edit,
                    fps=np.array(5.), K=intrinsics, camera_to_world=cameras,
                    canonical_depth=depth, geometry_valid=~edit, observed_background_depth=background,
                    source_preserved_mask=~edit, warped_known_mask=np.zeros_like(edit))
    np.savez(directory / "controls.npz", **controls)
    model: dict[str, Any] = dict(points=np.array([[.1, .2, 2], [.5, .6, 3], [-.2, -.3, 1]], np.float32),
                 colors=np.array([[1, 23, 45], [67, 89, 111], [123, 145, 167]], np.uint8),
                 units=np.array("arrow_provisional"), metric_calibrated=np.array(False),
                 reference_frame=np.array(0))
    np.savez(directory / "reference_model.npz", **model)
    for name, frames in (("raw_frames", raw), ("composite_frames", composite)):
        target = directory / "generated" / name
        target.mkdir(parents=True)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(target / f"frame_{index:04d}.png")
    return controls, model, composite


def chunks(path):
    from rerun.chunk import RrdReader
    return [(chunk.entity_path.lstrip("/"), chunk.to_record_batch())
            for chunk in RrdReader(str(path)).stream()]


def test_generated_geometry_claim_rejected(tmp_path):
    controls, _, _ = fixture_inputs(tmp_path)
    controls["geometry_valid"][1, 1, 2] = True
    controls["canonical_depth"][1, 1, 2] = 2
    np.savez(tmp_path / "controls.npz", **controls)
    with pytest.raises(ValueError, match="Generated pixels"):
        exporter.load_inputs(tmp_path / "controls.npz", tmp_path / "reference_model.npz", tmp_path / "generated")


def test_unknown_depth_must_be_nan(tmp_path):
    controls, _, _ = fixture_inputs(tmp_path)
    controls["canonical_depth"][1, 1, 2] = 0
    np.savez(tmp_path / "controls.npz", **controls)
    with pytest.raises(ValueError, match="remain NaN"):
        exporter.load_inputs(tmp_path / "controls.npz", tmp_path / "reference_model.npz", tmp_path / "generated")


def test_actual_rrd_readback(tmp_path):
    controls, model, composite = fixture_inputs(tmp_path)
    output = tmp_path / "output"
    manifest = exporter.export(tmp_path / "controls.npz", tmp_path / "reference_model.npz",
                               tmp_path / "generated", output)
    assert manifest["generated_3d_points"] == 0
    assert manifest["frames"][1]["unknown_generated_pixels"] == 1
    assert all(item["exit_code"] == 0 for item in manifest["rrd_verify"].values())
    records = chunks(output / "guided_freeze.rrd")
    canonical = [(entity, batch) for entity, batch in records if entity == "world/canonical"]
    assert len(canonical) == 1 and "frame" not in canonical[0][1].column_names
    batch = canonical[0][1]
    points = np.array(batch.column("Points3D:positions").to_pylist()).reshape(-1, 3)
    np.testing.assert_array_equal(points, model["points"])
    packed = np.array(batch.column("Points3D:colors").to_pylist(), dtype=np.uint32).reshape(-1)
    colors = np.stack([(packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255], -1)
    np.testing.assert_array_equal(colors, model["colors"])
    assert (output / "guided_freeze.rbl").stat().st_size > 0
    ply_header, _, ply_data = (output / "canonicalpointcloud.ply").read_bytes().partition(b"end_header\n")
    assert b"element vertex 3\n" in ply_header
    vertices = np.frombuffer(ply_data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    np.testing.assert_array_equal(np.stack([vertices[name] for name in ("x", "y", "z")], -1), model["points"])
    np.testing.assert_array_equal(np.stack([vertices[name] for name in ("red", "green", "blue")], -1), model["colors"])
    camera_rows = {}
    image_frames = {entity: set() for entity in ("view/source", "view/composite", "view/edit_mask",
                   "world/camera/canonical_depth", "diagnostic/observed_background_depth")}
    for entity, batch in records:
        if entity == "world":
            assert np.array(batch.column("ViewCoordinates:xyz").to_pylist()).reshape(-1).tolist() == [3, 2, 5]
        if entity == "world/camera":
            for row, index in enumerate(batch.column("frame").to_pylist()):
                target = camera_rows.setdefault(index, {})
                for name in batch.column_names:
                    if name.startswith(("Transform3D:", "Pinhole:")):
                        target[name] = batch.column(name).to_pylist()[row]
        if entity in image_frames:
            image_frames[entity].update(batch.column("frame").to_pylist())
            expected = {"view/source": controls["source_rgb"], "view/composite": composite,
                        "view/edit_mask": controls["edit_mask"].astype(np.uint8) * 255,
                        "world/camera/canonical_depth": controls["canonical_depth"],
                        "diagnostic/observed_background_depth": controls["observed_background_depth"]}[entity]
            prefix = "DepthImage" if "depth" in entity else "Image"
            for row, index in enumerate(batch.column("frame").to_pylist()):
                byte_values = np.asarray(batch.column(f"{prefix}:buffer").to_pylist()[row], np.uint8)
                restored = byte_values.reshape(-1).view(expected.dtype).reshape(expected[index].shape)
                np.testing.assert_array_equal(restored, expected[index])
                if entity == "world/camera/canonical_depth" and index == 1:
                    assert np.isnan(restored[1, 2])
                fmt = batch.column(f"{prefix}:format").to_pylist()[row][0]
                assert (fmt["height"], fmt["width"]) == (4, 4)
    for index in range(2):
        row = camera_rows[index]
        np.testing.assert_allclose(np.array(row["Transform3D:mat3x3"]).reshape(3, 3).T,
                                   controls["camera_to_world"][index, :3, :3])
        np.testing.assert_allclose(np.array(row["Transform3D:translation"]).reshape(3),
                                   controls["camera_to_world"][index, :3, 3])
        np.testing.assert_allclose(np.array(row["Pinhole:image_from_camera"]).reshape(3, 3).T,
                                   controls["K"][index], atol=1e-6)
        assert np.array(row["Pinhole:resolution"]).reshape(-1).tolist() == [4., 4.]
    assert all(indices == {0, 1} for indices in image_frames.values())
