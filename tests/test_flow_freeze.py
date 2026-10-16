"""Hand-computable continuous transport, visibility and masked-composition contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_flow_freeze.py"
SPEC = importlib.util.spec_from_file_location("prepare_flow_freeze", PATH)
assert SPEC is not None and SPEC.loader is not None
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)


def grid(h, w):
    y, x = np.indices((h, w), dtype=np.float32)
    return np.stack([x, y], axis=-1)


def test_identity_transport_preserves_continuous_coordinates_and_confidence():
    uv = grid(4, 5)
    field = np.concatenate([uv, np.ones((4, 5, 1), np.float32)], axis=-1)
    ones = np.ones((4, 5), np.float32)
    actual, fg, confidence = flow.transport_canonical(field, ones, ones, uv, ones, ones.astype(bool), ones)
    np.testing.assert_array_equal(actual, field)
    np.testing.assert_array_equal(fg, ones)
    np.testing.assert_array_equal(confidence, ones)


def test_halfpixel_transport_interpolates_field_without_collapsing_shared_ids():
    uv = np.array([[.5, 0], [1.5, 0]], float)
    attributes = np.array([[0., 2., 1.], [2., 2., 1.]], np.float32)
    values, valid, mass, _ = flow.splat_front(attributes, uv, np.ones(2), np.ones(2, bool), (1, 3))
    np.testing.assert_array_equal(valid, [[True, True, True]])
    np.testing.assert_allclose(values[0, :, 0], [0, 1, 2])
    np.testing.assert_allclose(mass, [[.5, 1, .5]])


def test_fractional_footprint_does_not_become_exponential_tracking_probability():
    field = np.array([[[0., 0., 1.]]], np.float32)
    fg = np.ones((1, 1), np.float32)
    quality = np.ones((1, 1), np.float32)
    # Half of the bilinear footprint is outside the image each time. The same
    # valid donor coordinate remains supported; 0.5**16 is not its confidence.
    for _ in range(16):
        field, fg, quality = flow.transport_canonical(field, fg, quality,
            np.array([[[.5, 0.]]]), np.ones((1, 1)), np.ones((1, 1), bool), np.ones((1, 1)))
    np.testing.assert_array_equal(field, [[[0, 0, 1]]])
    assert quality.item() == 1


def test_known_motion_transport_then_render_undoes_displacement():
    h, w = 3, 6
    uv = grid(h, w)
    field = np.concatenate([uv, np.ones((h, w, 1), np.float32)], axis=-1)
    ones = np.ones((h, w), np.float32)
    residual = np.zeros((h, w, 3), np.float32)
    residual[..., 0] = 1
    endpoint, endpoint_z, good = flow.total_endpoint(ones, np.eye(3), np.eye(3), np.eye(4), np.eye(4), residual, ones.astype(bool))
    np.testing.assert_array_equal(endpoint, uv + [1, 0])
    transported, fg, conf = flow.transport_canonical(field, ones, ones, endpoint, endpoint_z, good, ones)
    assert np.isnan(transported[:, 0]).all()
    np.testing.assert_array_equal(transported[:, 1:], field[:, :-1])
    initial = np.zeros((h, w, 3), np.uint8)
    initial[..., 1] = np.arange(w, dtype=np.uint8) * 20 + 80
    observed = np.zeros_like(initial)
    observed[:, 1:] = initial[:, :-1]
    rendered, supported, _, _, _ = flow.render_foreground(observed, transported, fg, conf,
        ones.astype(bool), np.eye(4), np.eye(3), (h, w))
    assert supported[:, :-1].all() and not supported[:, -1].any()
    np.testing.assert_array_equal(rendered[:, :-1], initial[:, :-1])


def test_camera_motion_is_part_of_endpoint_and_not_subtracted_twice():
    depth = np.full((2, 2), 2, np.float32)
    K = np.diag([10., 10., 1.])
    M1 = np.eye(4)
    M1[0, 3] = .5
    residual = np.zeros((2, 2, 3), np.float32)
    residual[..., 0] = .2
    uv, _, valid = flow.total_endpoint(depth, K, K, np.eye(4), M1, residual, np.ones((2, 2), bool))
    np.testing.assert_allclose(uv, grid(2, 2) + [-1.5, 0], atol=1e-7)
    assert valid.all()


def test_occluded_far_surface_cannot_bleed_into_near_color_or_coordinates():
    attrs = np.array([[255, 0, 0], [0, 0, 255]], np.float32)
    values, valid, _, depth = flow.splat_front(attrs, np.zeros((2, 2)), np.array([1., 2.]),
        np.ones(2, bool), (1, 1), layer_tolerance=.002)
    assert valid.item() and depth.item() == 1
    np.testing.assert_array_equal(values[0, 0], [255, 0, 0])


def test_observed_depth_mismatch_leaves_unknown_not_new_birth():
    values, valid, mass, _ = flow.splat_front(np.ones((1, 3)), np.zeros((1, 2)), np.array([2.]),
        np.ones(1, bool), (1, 1), observed_depth=np.ones((1, 1)), depth_tolerance=.05)
    assert not valid.item() and mass.item() == 0 and np.isnan(values).all()


def test_reference_frame_exact_and_unknown_inside_union_is_gray_not_source_copy():
    source = np.full((24, 24, 3), [50, 120, 30], np.uint8)
    warped = np.full_like(source, [0, 90, 0])
    foreground = np.zeros((24, 24), bool)
    foreground[10:14, 10:14] = True
    valid = np.zeros((24, 24), bool)
    draft, edit, preserved, known, union = flow.compose_draft(source, warped, valid, foreground, foreground)
    assert edit[10:14, 10:14].all() and np.all(draft[edit] == 127)
    assert not known.any() and np.array_equal(preserved, ~union)
    np.testing.assert_array_equal(draft[preserved], source[preserved])
    ref, ref_edit, ref_preserved, ref_known, _ = flow.compose_draft(source, warped, valid, foreground, foreground, reference=True)
    np.testing.assert_array_equal(ref, source)
    assert not ref_edit.any() and ref_preserved.all() and not ref_known.any()


def test_projected_destination_footprint_does_not_depend_on_current_tracking():
    world = np.array([[[4., 4., 1.]]], np.float32)
    footprint = flow.projected_footprint(world, np.ones((1, 1), bool), np.eye(4), np.eye(3), (9, 9))
    assert footprint[4, 4]
    # No current confidence/visibility argument exists; disappearance stays in the edit union.
    assert footprint.sum() == 25


def test_reference_projection_retains_observed_foreground_without_live_tracks():
    rgb = np.full((3, 3, 3), [40, 120, 20], np.uint8)
    xyz = np.concatenate([grid(3, 3), np.ones((3, 3, 1), np.float32)], axis=-1)
    seed = np.zeros((3, 3), bool)
    seed[1, 1] = True
    # The reference path uses the original field, not a NaN live-transport field.
    rendered, known, _, _, depth = flow.render_foreground(rgb, xyz, seed.astype(np.float32),
        np.ones((3, 3), np.float32), seed, np.eye(4), np.eye(3), (3, 3))
    assert known.sum() == 1 and known[1, 1]
    np.testing.assert_array_equal(rendered[1, 1], rgb[1, 1])
    assert depth[1, 1] == 1


def test_foreground_seed_does_not_trust_dilated_road_halo():
    rgb = np.full((64, 64, 3), [100, 100, 100], np.uint8)
    rgb[30:34, 30:34] = [40, 120, 20]
    seed = flow.foreground_seed(rgb)
    candidate = flow.foreground_candidate(rgb)
    assert seed.sum() == 16 and candidate.sum() > seed.sum()
    assert not seed[28, 28] and candidate[28, 28]


def test_safe_depth_rejects_depth_edges():
    depth = np.array([[1, 2], [1, 2]], np.float32)
    common = np.array([[1, 0, -.5], [0, 1, 0], [0, 0, 1]], float)
    sampled = flow.sample_depth_safe(depth, np.eye(3), common, (1, 1))
    assert np.isnan(sampled.item())


def test_bad_shapes_fail():
    with pytest.raises(ValueError):
        flow.splat_front(np.ones((1, 3)), np.ones((2, 2)), np.ones(1), np.ones(1, bool), (1, 1))


def test_reference_triangle_identity_is_dense_and_uses_original_texture():
    h, w = 5, 6
    xyz = np.concatenate([grid(h, w), np.ones((h, w, 1), np.float32)], -1)
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    out, valid, _, uv, depth, metrics = flow.raster_reference(
        rgb, xyz, np.ones((h, w)), np.ones((h, w), bool), np.eye(4), np.eye(3), (h, w))
    assert valid.all() and metrics["visible_surface_fraction"] == 1
    np.testing.assert_array_equal(out, rgb)
    np.testing.assert_allclose(uv, grid(h, w), atol=1e-6)
    np.testing.assert_allclose(depth, 1)


def test_reference_triangle_preserves_foreground_holes_and_rejects_depth_jump():
    h = w = 5
    xyz = np.concatenate([grid(h, w), np.ones((h, w, 1), np.float32)], -1)
    rgb = np.full((h, w, 3), 100, np.uint8)
    seed = np.ones((h, w), bool)
    seed[2, 2] = False
    _, valid, _, _, depth, _ = flow.raster_reference(
        rgb, xyz, np.ones((h, w)), seed, np.eye(4), np.eye(3), (h, w))
    assert not valid[2, 2] and depth[2, 2] == 1
    discontinuous = np.ones((h, w))
    discontinuous[2, 2] = 2
    _, valid, _, _, depth, _ = flow.raster_reference(
        rgb, xyz, discontinuous, np.ones((h, w), bool), np.eye(4), np.eye(3), (h, w))
    assert not valid[2, 2] and np.isinf(depth[2, 2])


def test_reference_triangle_rejects_foldover_and_extreme_magnification():
    rgb = np.ones((2, 2, 3), np.uint8)
    xyz = np.concatenate([grid(2, 2), np.ones((2, 2, 1), np.float32)], -1)
    for scale in (-1, 100):
        K = np.diag([scale, 1, 1])
        _, valid, _, _, _, metrics = flow.raster_reference(
            rgb, xyz, np.ones((2, 2)), np.ones((2, 2), bool), np.eye(4), K, (3, 3))
        assert not valid.any() and metrics["triangles_accepted"] == 0


def test_reference_triangle_perspective_uv_matches_ray_plane_intersection():
    # A shallow slanted plane z=1+0.01*x. Source depth differences remain <5%.
    h = w = 5
    xy = grid(h, w)
    depth = 1 / (1 - .01 * xy[..., 0])
    xyz = np.concatenate([xy * depth[..., None], depth[..., None]], -1)
    rgb = np.full((h, w, 3), [40, 120, 30], np.uint8)
    M = np.eye(4)
    M[0, 3] = .2
    _, valid, _, uv, actual_depth, _ = flow.raster_reference(
        rgb, xyz, depth, np.ones((h, w), bool), M, np.eye(3), (h, w))
    # target q=(u*z+0.2,v*z,z), so z=(1+0.002)/(1-0.01*u).
    expected_depth = 1.002 / (1 - .01 * xy[..., 0])
    expected_uv = xy.copy()
    expected_uv[..., 0] += .2 / expected_depth
    assert valid[1:4, 1:4].all()
    np.testing.assert_allclose(actual_depth[valid], expected_depth[valid], atol=2e-7)
    np.testing.assert_allclose(uv[valid], expected_uv[valid], atol=5e-7)


def test_triangle_bilinear_texture_cannot_read_nongreen_fourth_corner():
    rgb = np.full((2, 2, 3), [0, 200, 0], np.uint8)
    rgb[1, 0] = [255, 0, 0]
    seed = np.array([[True, True], [False, True]])
    xyz = np.concatenate([grid(2, 2), np.ones((2, 2, 1), np.float32)], -1)
    out, valid, _, uv, depth, _ = flow.raster_reference(
        rgb, xyz, np.ones((2, 2)), seed, np.eye(4), np.diag([2, 2, 1]), (3, 3))
    # The TL/TR/BR triangle is green, but its center's bilinear texture also
    # reads red BL at 25%. It must remain unknown rather than inherit road color.
    assert not valid[1, 1] and np.isnan(uv[1, 1]).all() and depth[1, 1] == 1
    # An exactly zero-weight bad donor does not invalidate the top edge.
    assert valid[0, 1]
    np.testing.assert_array_equal(out[0, 1], [0, 200, 0])
