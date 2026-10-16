"""``src/arrow/static_world.py`` 共有API契約の合成テスト。

契約: temp/workdoc_Oct16-2026_sr_static_world.md §2「固定数値契約 (TR3/TR4)」と
temp/herdr_static_world/core.initial.md の API contract。numpy のみで完結する。

期待値は実装をミラーせず、integer pixel centre（画素 (i, j) の中心 = (u=j, v=i)）の
逆投影/投影式から独立に手計算した既知数を与える。相対深度誤差は
``|z_pred - z1| / max(z_pred, z1)`` と定義し、tolerance 境界は inclusive とする。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from arrow.static_world import (
    advect_labels,
    decompose_flow,
    render_points,
    transform_points,
    unproject_depth,
)

DECOMPOSE_KEYS = {
    "rigid_flow",
    "nonrigid_flow",
    "total_flow",
    "rigid_3d",
    "nonrigid_3d",
    "total_3d",
    "world_nonrigid",
    "valid",
    "target_uv",
    "target_depth",
}


def _K(fx=100.0, fy=100.0, cx=0.0, cy=0.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _c2w(rotation=None, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _rot_z90():
    return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _rot_y90():
    return np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def _valid(h, w):
    return np.ones((h, w), dtype=bool)


def _pixel_grid(h, w):
    rows, cols = np.mgrid[0:h, 0:w]
    return np.stack([cols, rows], axis=-1).astype(np.float64)


def _expect_value_error(fn, args, kwargs=None, label=""):
    try:
        fn(*args, **(kwargs or {}))
    except ValueError:
        return
    raise AssertionError(f"{label or fn.__name__} did not raise ValueError")


# ---------------------------------------------------------------------------
# decompose_flow
# ---------------------------------------------------------------------------


def test_decompose_returns_contract_keys_shapes_dtypes():
    h, w = 4, 5
    depth = np.full((h, w), 2.0)
    K = _K(fx=100.0, fy=100.0, cx=2.0, cy=2.0)
    flow = np.zeros((h, w, 3))
    result = decompose_flow(depth, K, K, np.eye(4), np.eye(4), flow, _valid(h, w))

    assert set(result) == DECOMPOSE_KEYS
    for key in ("rigid_flow", "nonrigid_flow", "total_flow"):
        assert result[key].shape == (h, w, 2)
        assert result[key].dtype == np.float64
    for key in ("rigid_3d", "nonrigid_3d", "total_3d", "world_nonrigid"):
        assert result[key].shape == (h, w, 3)
        assert result[key].dtype == np.float64
    assert result["target_uv"].shape == (h, w, 2)
    assert result["target_depth"].shape == (h, w)
    assert result["valid"].shape == (h, w)
    assert result["valid"].dtype == bool
    assert result["valid"].all()


def test_decompose_pure_camera_motion_has_zero_nonrigid_even_with_rigid_motion():
    h, w = 4, 4
    depth = np.full((h, w), 2.0)
    K = _K(fx=100.0, fy=100.0, cx=2.0, cy=2.0)
    M0 = np.eye(4)
    M1 = _c2w(translation=(0.5, 0.0, 0.0))
    flow = np.zeros((h, w, 3))

    result = decompose_flow(depth, K, K, M0, M1, flow, _valid(h, w))

    assert result["valid"].all()
    assert np.any(result["rigid_flow"] != 0.0)
    np.testing.assert_array_equal(result["nonrigid_flow"], np.zeros((h, w, 2)))
    np.testing.assert_array_equal(result["nonrigid_3d"], np.zeros((h, w, 3)))
    np.testing.assert_array_equal(result["world_nonrigid"], np.zeros((h, w, 3)))
    np.testing.assert_array_equal(result["total_flow"], result["rigid_flow"])
    np.testing.assert_array_equal(result["total_3d"], result["rigid_3d"])
    np.testing.assert_allclose(
        result["rigid_3d"], np.broadcast_to([-0.5, 0.0, 0.0], (h, w, 3)), atol=1e-12
    )


def test_decompose_identity_camera_moving_point_known_pixel_flow():
    # z=2, fx=100, Sx=0.1 -> 100 * 0.1 / 2 = 5 px。
    K = _K(fx=100.0, fy=100.0, cx=100.0, cy=100.0)
    depth = np.array([[2.0]])
    flow = np.array([[[0.1, 0.0, 0.0]]])

    result = decompose_flow(depth, K, K, np.eye(4), np.eye(4), flow, np.array([[True]]))

    assert result["valid"][0, 0]
    np.testing.assert_allclose(result["rigid_flow"], [[[0.0, 0.0]]], atol=1e-12)
    np.testing.assert_allclose(result["total_flow"], [[[5.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_flow"], [[[5.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_3d"], [[[0.1, 0.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["rigid_3d"], 0.0, atol=1e-12)
    np.testing.assert_allclose(result["total_3d"], [[[0.1, 0.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["world_nonrigid"], [[[0.1, 0.0, 0.0]]], rtol=1e-12)
    # cx=100 の (0,0) 画素は P=(-2,-2,2)、Qt=(-1.9,-2,2) -> u=5, v=0。
    np.testing.assert_allclose(result["target_uv"], [[[5.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["target_depth"], [[2.0]], rtol=1e-12)


def test_decompose_combined_camera_and_object_translation_hand_numbers():
    # P=(0,0,2)。camera +x 0.5 で Qr=(-0.5,0,2) -> u=-25。S=(0.2,0,0) で Qt=(-0.3,0,2) -> u=-15。
    # frustum 外でも有限の数学的 flow は valid のまま保持する。
    K = _K(fx=100.0, fy=100.0)
    depth = np.array([[2.0]])
    flow = np.array([[[0.2, 0.0, 0.0]]])
    M0 = np.eye(4)
    M1 = _c2w(translation=(0.5, 0.0, 0.0))

    result = decompose_flow(depth, K, K, M0, M1, flow, np.array([[True]]))

    assert result["valid"][0, 0]
    np.testing.assert_allclose(result["rigid_flow"], [[[-25.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["total_flow"], [[[-15.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_flow"], [[[10.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["rigid_3d"], [[[-0.5, 0.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_3d"], [[[0.2, 0.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["total_3d"], [[[-0.3, 0.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["target_uv"], [[[-15.0, 0.0]]], rtol=1e-12)
    np.testing.assert_allclose(result["target_depth"], [[2.0]], rtol=1e-12)


def test_decompose_rotation_about_camera_z_hand_numbers():
    # pixel (0,50): P=(1,0,2)。T = inv(M1) M0 = Rz90^T = Rz(-90) で Qr=(0,-1,2)
    # -> proj (0,-50) -> rigid (-50,-50)。S=(0,0,0.5) で Qt=(0,-1,2.5) -> proj (0,-40)
    # -> total (-50,-40), nonrigid (0,10)。
    K = _K(fx=100.0, fy=100.0)
    depth = np.full((1, 64), 2.0)
    flow = np.zeros((1, 64, 3))
    flow[0, 50] = (0.0, 0.0, 0.5)
    M0 = np.eye(4)
    M1 = _c2w(rotation=_rot_z90())

    result = decompose_flow(depth, K, K, M0, M1, flow, _valid(1, 64))

    assert result["valid"].all()
    np.testing.assert_allclose(result["rigid_flow"][0, 50], [-50.0, -50.0], rtol=1e-12)
    np.testing.assert_allclose(result["total_flow"][0, 50], [-50.0, -40.0], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_flow"][0, 50], [0.0, 10.0], rtol=1e-12)
    np.testing.assert_allclose(result["rigid_3d"][0, 50], [-1.0, -1.0, 0.0], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_3d"][0, 50], [0.0, 0.0, 0.5], rtol=1e-12)
    np.testing.assert_allclose(result["total_3d"][0, 50], [-1.0, -1.0, 0.5], rtol=1e-12)
    np.testing.assert_allclose(result["target_uv"][0, 50], [0.0, -40.0], rtol=1e-12)
    np.testing.assert_allclose(result["nonrigid_flow"][0, 0], [0.0, 0.0], atol=1e-12)


def test_decompose_additivity_and_world_frame_residual():
    h, w = 3, 4
    depth = 1.0 + 0.1 * np.arange(h * w, dtype=np.float64).reshape(h, w)
    K = _K(fx=80.0, fy=90.0, cx=1.5, cy=1.0)
    R0 = _rot_z90()
    T_rot = _rot_y90()
    M0 = _c2w(rotation=R0, translation=(0.3, -0.2, 0.1))
    T = _c2w(rotation=T_rot, translation=(0.05, -0.1, 0.2))
    M1 = M0 @ np.linalg.inv(T)
    flow = np.broadcast_to(np.array([0.1, 0.2, 0.3]), (h, w, 3)).copy()

    result = decompose_flow(depth, K, K, M0, M1, flow, _valid(h, w))

    assert result["valid"].all()
    np.testing.assert_allclose(
        result["total_flow"], result["rigid_flow"] + result["nonrigid_flow"], rtol=1e-9
    )
    np.testing.assert_allclose(
        result["total_3d"], result["rigid_3d"] + result["nonrigid_3d"], rtol=1e-9
    )
    # T_rot = Ry90 -> R S = (Sz, Sy, -Sx)。R0 = Rz90 -> world = (-Sy, Sx, Sz)。
    np.testing.assert_allclose(
        result["nonrigid_3d"], np.broadcast_to([0.3, 0.2, -0.1], (h, w, 3)), rtol=1e-12
    )
    np.testing.assert_allclose(
        result["world_nonrigid"], np.broadcast_to([-0.2, 0.1, 0.3], (h, w, 3)), rtol=1e-12
    )
    assert np.isfinite(result["target_uv"]).all()
    assert (result["target_depth"] > 0.0).all()


def test_decompose_invalid_and_behind_camera_are_nan():
    K = _K(fx=100.0, fy=100.0, cx=1.0, cy=1.0)
    depth = np.array(
        [
            [2.0, np.nan, 2.0],
            [1.0, 0.0, 2.0],
            [2.0, -1.0, 2.0],
        ]
    )
    flow = np.zeros((3, 3, 3))
    flow[0, 2] = np.nan
    valid = np.array(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
        ]
    )
    # camera +z 1.5 -> T の z 成分 -1.5。(1,0) の source 深度 1.0 -> z=-0.5 で behind。
    M0 = np.eye(4)
    M1 = _c2w(translation=(0.0, 0.0, 1.5))

    result = decompose_flow(depth, K, K, M0, M1, flow, valid)

    expected_valid = np.array(
        [
            [True, False, False],
            [False, False, True],
            [True, False, False],
        ]
    )
    np.testing.assert_array_equal(result["valid"], expected_valid)
    for key in ("rigid_flow", "nonrigid_flow", "total_flow", "target_uv"):
        assert np.isnan(result[key][~expected_valid]).all()
        assert np.isfinite(result[key][expected_valid]).all()
    for key in ("rigid_3d", "nonrigid_3d", "total_3d", "world_nonrigid"):
        assert np.isnan(result[key][~expected_valid]).all()
        assert np.isfinite(result[key][expected_valid]).all()
    assert np.isnan(result["target_depth"][~expected_valid]).all()
    assert np.isfinite(result["target_depth"][expected_valid]).all()


def test_decompose_rejects_bad_shapes_and_matrices():
    h, w = 3, 4
    depth = np.full((h, w), 2.0)
    K = _K()
    M = np.eye(4)
    flow = np.zeros((h, w, 3))
    valid = _valid(h, w)
    singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    scaled = np.eye(4)
    scaled[:3, :3] *= 1.01
    reflection = np.eye(4)
    reflection[2, 2] = -1.0
    last_row = np.eye(4)
    last_row[3, 0] = 1.0
    nonfinite = np.eye(4)
    nonfinite[0, 3] = np.nan

    bad = [
        ("depth 3-D", (np.zeros((h, w, 1)), K, K, M, M, flow, valid)),
        ("flow 2-D", (depth, K, K, M, M, np.zeros((h, w)), valid)),
        ("flow shape", (depth, K, K, M, M, np.zeros((h, w + 1, 3)), valid)),
        ("valid shape", (depth, K, K, M, M, flow, _valid(h + 1, w))),
        ("K shape", (depth, (2, 2), K, M, M, flow, valid)),
        ("K singular", (depth, singular, K, M, M, flow, valid)),
        ("K nonfinite", (depth, np.full((3, 3), np.nan), K, M, M, flow, valid)),
        ("K focal", (depth, np.array([[0.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]), K, M, M, flow, valid)),
        ("M shape", (depth, K, K, np.eye(3), M, flow, valid)),
        ("M last row", (depth, K, K, last_row, M, flow, valid)),
        ("M scale", (depth, K, K, scaled, M, flow, valid)),
        ("M reflection", (depth, K, K, reflection, M, flow, valid)),
        ("M nonfinite", (depth, K, K, nonfinite, M, flow, valid)),
    ]
    for name, args in bad:
        _expect_value_error(decompose_flow, args, label=name)
    # 良い入力は ValueError を出さない
    decompose_flow(depth, K, K, M, M, flow, valid)


# ---------------------------------------------------------------------------
# advect_labels
# ---------------------------------------------------------------------------


def test_advect_identity_keeps_labels_and_indices():
    labels0 = np.array([[0, 1, -1], [2, 3, 4], [-1, 5, 6]], dtype=np.int64)
    grid = _pixel_grid(3, 3)

    result = advect_labels(
        labels0, grid, np.ones((3, 3)), np.ones((3, 3)), _valid(3, 3), max_pixel_distance=0.0
    )

    expected_index = np.arange(9, dtype=np.int64).reshape(3, 3)
    expected_index[labels0 < 0] = -1
    assert result["labels"].dtype == np.int64
    assert result["source_index"].dtype == np.int64
    assert result["valid"].dtype == bool
    np.testing.assert_array_equal(result["labels"], labels0)
    np.testing.assert_array_equal(result["source_index"], expected_index)
    np.testing.assert_array_equal(result["valid"], labels0 >= 0)
    assert result["stats"]["n_accepted_targets"] == 7
    assert result["stats"]["n_positive_candidates"] == 7


def test_advect_known_translation_keeps_anchor_ids():
    labels0 = np.arange(9, dtype=np.int64).reshape(3, 3)
    target_uv = _pixel_grid(3, 3) + np.array([1.0, 0.0])  # 1 px 右シフト

    result = advect_labels(
        labels0,
        target_uv,
        np.ones((3, 3)),
        np.ones((3, 3)),
        _valid(3, 3),
        max_pixel_distance=0.0,
    )

    expected = np.full((3, 3), -1, dtype=np.int64)
    expected[:, 1:] = labels0[:, :-1]
    np.testing.assert_array_equal(result["labels"], expected)
    np.testing.assert_array_equal(result["source_index"], expected)


def test_advect_z_buffer_prefers_nearest_predicted_depth():
    for near_is_a in (True, False):
        labels0 = np.full((5, 5), -1, dtype=np.int64)
        labels0[2, 1] = 70
        labels0[2, 3] = 80
        target_uv = np.full((5, 5, 2), np.nan)
        target_uv[2, 1] = (2.0, 2.0)
        target_uv[2, 3] = (2.4, 2.0)
        target_depth = np.full((5, 5), np.nan)
        target_depth[2, 1] = 1.0 if near_is_a else 1.04
        target_depth[2, 3] = 1.04 if near_is_a else 1.0

        result = advect_labels(labels0, target_uv, target_depth, np.ones((5, 5)), _valid(5, 5))

        # 相対誤差は max(z_pred, z1) 基準: 0.04/1.04 = 0.0385 <= 0.05 で両方通過、近い z が勝つ。
        assert result["labels"][2, 2] == (70 if near_is_a else 80)
        assert result["source_index"][2, 2] == (2 * 5 + 1 if near_is_a else 2 * 5 + 3)


def test_advect_relative_depth_tolerance_boundary_is_inclusive():
    labels0 = np.array([[7]], dtype=np.int64)
    target_uv = np.array([[[0.0, 0.0]]])
    target_depth = np.array([[2.0]])

    accepted = advect_labels(
        labels0,
        target_uv,
        target_depth,
        np.array([[1.9]]),
        np.array([[True]]),
        relative_depth_tolerance=0.05,
    )
    # |2.0 - 1.9| / max(2.0, 1.9) = 0.05 -> 境界は inclusive。
    assert accepted["labels"][0, 0] == 7
    assert accepted["valid"][0, 0]

    rejected = advect_labels(
        labels0,
        target_uv,
        target_depth,
        np.array([[1.899]]),
        np.array([[True]]),
        relative_depth_tolerance=0.05,
    )
    assert rejected["labels"][0, 0] == -1
    assert not rejected["valid"][0, 0]
    assert rejected["stats"]["rejected"]["depth_mismatch"] == 1

    exact = advect_labels(
        labels0,
        target_uv,
        target_depth,
        np.array([[2.0]]),
        np.array([[True]]),
        relative_depth_tolerance=0.0,
    )
    assert exact["labels"][0, 0] == 7

    off = advect_labels(
        labels0,
        target_uv,
        target_depth,
        np.array([[2.0000001]]),
        np.array([[True]]),
        relative_depth_tolerance=0.0,
    )
    assert off["labels"][0, 0] == -1


def test_advect_rejects_out_of_bounds_and_invalid_sources():
    labels0 = np.array([[10, -1, 11, 12, 13]], dtype=np.int64)
    target_uv = np.full((1, 5, 2), np.nan)
    target_uv[0, 0] = (-5.0, -5.0)  # 候補は out of bounds
    target_uv[0, 1] = (0.0, 0.0)  # label -1 は対象外
    target_uv[0, 2] = (0.5, 0.0)  # valid=False で対象外
    target_uv[0, 3] = (1.0, 0.0)  # 採用される
    target_uv[0, 4] = (np.inf, 0.0)  # 非有限 uv は対象外
    target_depth = np.full((1, 5), 1.0)
    valid = np.ones((1, 5), dtype=bool)
    valid[0, 2] = False

    result = advect_labels(labels0, target_uv, target_depth, np.ones((1, 2)), valid)

    assert result["labels"][0, 1] == 12
    assert result["source_index"][0, 1] == 3
    assert result["stats"]["n_positive_candidates"] == 2
    assert result["stats"]["rejected"]["out_of_bounds"] >= 1


@pytest.mark.parametrize("bad", [np.nan, 0.0, -2.0, np.inf])
def test_advect_rejects_invalid_depth1(bad):
    result = advect_labels(
        np.array([[5]], dtype=np.int64),
        np.array([[[0.0, 0.0]]]),
        np.array([[1.0]]),
        np.array([[bad]]),
        np.array([[True]]),
    )
    assert result["labels"][0, 0] == -1
    assert result["source_index"][0, 0] == -1
    assert not result["valid"][0, 0]
    assert result["stats"]["n_accepted_targets"] == 0


def test_advect_rejected_label_never_appears():
    result = advect_labels(
        np.array([[7]], dtype=np.int64),
        np.array([[[0.0, 0.0]]]),
        np.array([[2.0]]),
        np.array([[1.0]]),  # |2-1|/2 = 0.5 > 0.05
        np.array([[True]]),
    )
    assert result["labels"][0, 0] == -1
    assert result["source_index"][0, 0] == -1
    assert not result["valid"][0, 0]
    assert result["stats"]["rejected"]["depth_mismatch"] == 1
    assert result["stats"]["n_accepted_targets"] == 0


def test_advect_tie_break_by_distance_then_source_index():
    labels0 = np.array([[10, 11, -1, 13, 14]], dtype=np.int64)
    target_uv = np.full((1, 5, 2), np.nan)
    target_uv[0, 0] = (1.0, 0.0)  # A: idx 0
    target_uv[0, 1] = (2.4, 0.0)  # B: idx 1
    target_uv[0, 3] = (1.0, 0.0)  # D: idx 3 (A と同じ距離 0/1)
    target_uv[0, 4] = (2.0, 0.0)  # C: idx 4
    target_depth = np.full((1, 5), 1.0)

    result = advect_labels(labels0, target_uv, target_depth, np.ones((1, 5)), _valid(1, 5))

    # (0,1): A と D が距離 0 の tie -> 小さい source index の A
    assert result["labels"][0, 1] == 10
    assert result["source_index"][0, 1] == 0
    # (0,2): C が距離 0 で最小 -> 距離優先（index は A/B より大きい）
    assert result["labels"][0, 2] == 14
    assert result["source_index"][0, 2] == 4
    # (0,0): A と D が距離 1.0 の tie -> A
    assert result["labels"][0, 0] == 10
    # (0,4): どの source も 1.5 以内に無く -1 のまま（勝手に埋めない）
    assert result["labels"][0, 4] == -1


def test_advect_multiple_target_pixels_share_anchor_id_explicitly():
    result = advect_labels(
        np.array([[42]], dtype=np.int64),
        np.array([[[0.0, 0.0]]]),
        np.array([[1.0]]),
        np.ones((1, 3)),
        np.array([[True]]),
    )
    np.testing.assert_array_equal(result["labels"], [[42, 42, -1]])
    np.testing.assert_array_equal(result["source_index"], [[0, 0, -1]])
    np.testing.assert_array_equal(result["valid"], [[True, True, False]])
    assert result["stats"]["n_accepted_targets"] == 2


def test_advect_output_grid_can_differ_from_source_grid():
    labels0 = np.array([[0, 1], [2, 3]], dtype=np.int64)
    target_uv = _pixel_grid(2, 2) * 2.0

    result = advect_labels(
        labels0,
        target_uv,
        np.ones((2, 2)),
        np.ones((4, 4)),
        _valid(2, 2),
        max_pixel_distance=0.0,
    )

    assert result["labels"].shape == (4, 4)
    assert result["source_index"].shape == (4, 4)
    assert result["valid"].shape == (4, 4)
    assert result["labels"][0, 0] == 0
    assert result["labels"][0, 2] == 1
    assert result["labels"][2, 0] == 2
    assert result["labels"][2, 2] == 3
    assert np.count_nonzero(result["labels"] >= 0) == 4


def test_advect_stats_are_json_safe():
    result = advect_labels(
        np.array([[7, -1]], dtype=np.int64),
        np.array([[[0.0, 0.0], [np.nan, np.nan]]]),
        np.array([[2.0, np.nan]]),
        np.ones((1, 1)),
        np.array([[True, False]]),
    )
    stats = result["stats"]
    json.dumps(stats)
    for key in (
        "n_positive_candidates",
        "n_candidate_pairs",
        "n_accepted_targets",
        "n_accepted_pairs",
        "n_rejected_pairs",
    ):
        assert isinstance(stats[key], int)
    assert set(stats["rejected"]) == {
        "out_of_bounds",
        "beyond_max_pixel_distance",
        "invalid_depth1",
        "depth_mismatch",
    }
    assert all(isinstance(value, int) for value in stats["rejected"].values())
    assert stats["n_candidate_pairs"] == (
        stats["n_rejected_pairs"] + stats["n_accepted_pairs"]
    )


def test_advect_rejects_bad_inputs():
    labels0 = np.array([[7]], dtype=np.int64)
    target_uv = np.array([[[0.0, 0.0]]])
    target_depth = np.ones((1, 1))
    depth1 = np.ones((1, 1))
    valid = np.ones((1, 1), dtype=bool)

    bad = {
        "labels 1-D": (labels0.ravel(), target_uv, target_depth, depth1, valid),
        "target_uv shape": (labels0, np.zeros((1, 2, 2)), target_depth, depth1, valid),
        "target_depth shape": (labels0, target_uv, np.ones((1, 2)), depth1, valid),
        "valid shape": (labels0, target_uv, target_depth, depth1, np.ones((2, 1), bool)),
        "depth1 3-D": (labels0, target_uv, target_depth, np.ones((1, 1, 1)), valid),
    }
    for name, args in bad.items():
        _expect_value_error(advect_labels, args, label=name)

    for name, kwargs in {
        "tol negative": {"relative_depth_tolerance": -0.1},
        "tol nan": {"relative_depth_tolerance": float("nan")},
        "tol inf": {"relative_depth_tolerance": float("inf")},
        "distance negative": {"max_pixel_distance": -1.0},
        "distance nan": {"max_pixel_distance": float("nan")},
    }.items():
        _expect_value_error(
            advect_labels,
            (labels0, target_uv, target_depth, depth1, valid),
            kwargs=kwargs,
            label=name,
        )


# ---------------------------------------------------------------------------
# render_points
# ---------------------------------------------------------------------------


def test_render_single_point_exact_pixel_and_dtypes():
    # K=eye: u = x/z。x=2, z=2 -> u=1 で画素 (0,1)。
    points = np.array([[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])
    colors = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    result = render_points(points, colors, np.eye(3), np.eye(4), (2, 3), radius=0)

    assert result["rgb"].dtype == np.uint8
    assert result["depth"].dtype == np.float32
    assert result["valid"].dtype == bool
    assert result["point_index"].dtype == np.int64
    np.testing.assert_array_equal(result["rgb"][0, 0], colors[0])
    np.testing.assert_array_equal(result["rgb"][0, 1], colors[1])
    assert result["depth"][0, 0] == np.float32(2.0)
    assert result["depth"][0, 1] == np.float32(2.0)
    assert result["point_index"][0, 0] == 0
    assert result["point_index"][0, 1] == 1
    assert not result["valid"][1, 2]
    assert np.isnan(result["depth"][1, 2])
    np.testing.assert_array_equal(result["rgb"][1, 2], [0, 0, 0])
    assert result["point_index"][1, 2] == -1


def test_render_camera_translation_and_repeatability():
    points = np.array([[0.0, 0.0, 2.0]])
    colors = np.array([[1, 2, 3]], dtype=np.uint8)
    K = _K(fx=10.0, fy=10.0)

    first = render_points(points, colors, K, np.eye(4), (3, 5), radius=0)
    assert first["point_index"][0, 0] == 0

    # camera を -x 0.4 動かすと点は +u へ: u = 10*0.4/2 = 2。
    moved = render_points(points, colors, K, _c2w(translation=(-0.4, 0.0, 0.0)), (3, 5), radius=0)
    assert not moved["valid"][0, 0]
    assert moved["point_index"][0, 2] == 0
    assert moved["depth"][0, 2] == np.float32(2.0)
    np.testing.assert_array_equal(moved["rgb"][0, 2], colors[0])

    # 同一入力の再描画は byte-identical（静止 camera は時刻に依らない）。
    again = render_points(points, colors, K, np.eye(4), (3, 5), radius=0)
    for key in ("rgb", "depth", "valid", "point_index"):
        np.testing.assert_array_equal(again[key], first[key])


def test_render_near_point_occludes_far_and_ties_by_index():
    K = np.eye(3)
    far = np.array([0.0, 0.0, 5.0])
    near = np.array([0.0, 0.0, 2.0])
    colors = np.array([[9, 9, 9], [1, 2, 3]], dtype=np.uint8)

    result = render_points(np.stack([far, near]), colors, K, np.eye(4), (2, 2), radius=0)
    assert result["point_index"][0, 0] == 1
    assert result["depth"][0, 0] == np.float32(2.0)
    np.testing.assert_array_equal(result["rgb"][0, 0], colors[1])

    tie = render_points(np.stack([near, near]), colors, K, np.eye(4), (2, 2), radius=0)
    assert tie["point_index"][0, 0] == 0
    np.testing.assert_array_equal(tie["rgb"][0, 0], colors[0])


def test_render_radius_is_documented_disc():
    K = np.eye(3)
    point = np.array([[2.0, 2.0, 1.0]])
    colors = np.array([[7, 8, 9]], dtype=np.uint8)

    r1 = render_points(point, colors, K, np.eye(4), (5, 5), radius=1)
    filled = r1["valid"]
    assert r1["point_index"][2, 2] == 0
    np.testing.assert_array_equal(filled.sum(), 5)  # 中心 + 4近傍（disc であり正方形ではない）
    assert filled[2, 1] and filled[1, 2] and filled[3, 2] and filled[2, 3]
    assert not filled[1, 1] and not filled[3, 3]  # 対角は dist^2=2 > 1

    r0 = render_points(point, colors, K, np.eye(4), (5, 5), radius=0)
    assert r0["valid"].sum() == 1

    r2 = render_points(point, colors, K, np.eye(4), (5, 5), radius=2)
    assert r2["valid"].sum() == 13
    assert not r2["valid"][0, 0]  # dist^2=8 > 4

    r3 = render_points(point, colors, K, np.eye(4), (5, 5), radius=3)
    assert r3["valid"].sum() == 25  # 5x5 内は最大 dist^2=8 <= 9


def test_render_void_pixels_are_black_nan_and_minus_one():
    K = np.eye(3)

    empty = render_points(
        np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), K, np.eye(4), (2, 2), radius=0
    )
    assert not empty["valid"].any()
    assert np.isnan(empty["depth"]).all()
    assert (empty["rgb"] == 0).all()
    assert (empty["point_index"] == -1).all()

    points = np.array([[10.0, 0.0, 2.0], [0.0, 0.0, -1.0], [np.nan, 0.0, 1.0]])
    colors = np.zeros((3, 3), dtype=np.uint8)
    result = render_points(points, colors, K, np.eye(4), (2, 2), radius=0)
    assert not result["valid"].any()  # 画面外 / behind / 非有限は reject
    assert (result["point_index"] == -1).all()
    assert np.isnan(result["depth"]).all()


def test_render_rejects_bad_inputs():
    K = np.eye(3)
    M = np.eye(4)
    points = np.array([[0.0, 0.0, 2.0]])
    colors = np.zeros((1, 3), dtype=np.uint8)

    singular = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    scaled = np.eye(4)
    scaled[:3, :3] *= 1.01
    last_row = np.eye(4)
    last_row[3, 1] = 1.0

    bad = {
        "points 1-D": (points.ravel(), colors, K, M, (2, 2)),
        "points Nx2": (points[:, :2], colors, K, M, (2, 2)),
        "colors dtype": (points, np.zeros((1, 3), dtype=np.float32), K, M, (2, 2)),
        "colors shape": (points, np.zeros((2, 3), dtype=np.uint8), K, M, (2, 2)),
        "K singular": (points, colors, singular, M, (2, 2)),
        "pose scale": (points, colors, K, scaled, (2, 2)),
        "pose last row": (points, colors, K, last_row, (2, 2)),
        "shape": (points, colors, K, M, (0, 2)),
    }
    for name, args in bad.items():
        _expect_value_error(render_points, args, label=name)

    for bad_radius in (-1, 1.5, 4, True):
        _expect_value_error(
            render_points,
            (points, colors, K, M, (2, 2)),
            kwargs={"radius": bad_radius},
            label=f"radius={bad_radius!r}",
        )
    render_points(points, colors, K, M, (2, 2), radius=3)


# ---------------------------------------------------------------------------
# unproject_depth / transform_points
# ---------------------------------------------------------------------------


def test_unproject_depth_grid_centres_and_invalid_nan():
    K = np.array([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
    depth = np.array([[2.0, 2.0], [2.0, np.nan]])

    points = unproject_depth(depth, K)

    assert points.shape == (2, 2, 3)
    assert points.dtype == np.float64
    np.testing.assert_allclose(points[0, 0], [-1.0, -1.0, 2.0])
    np.testing.assert_allclose(points[0, 1], [0.0, -1.0, 2.0])
    np.testing.assert_allclose(points[1, 0], [-1.0, 0.0, 2.0])
    assert np.isnan(points[1, 1]).all()

    assert unproject_depth(depth.astype(np.float32), K).dtype == np.float32
    assert np.isnan(unproject_depth(np.array([[-1.0, 0.0]]), K)).all()


def test_transform_points_applies_affine_matrix_and_preserves_shape():
    matrix = _c2w(rotation=_rot_z90(), translation=(1.0, 2.0, 3.0))
    points = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]])

    moved = transform_points(points, matrix)

    assert moved.shape == (2, 2, 3)
    assert moved.dtype == np.float64
    np.testing.assert_allclose(moved[0, 0], [1.0, 3.0, 3.0])  # Rz90(1,0,0)=(0,1,0) + t
    np.testing.assert_allclose(moved[1, 1], [0.0, 3.0, 4.0])  # Rz90(1,1,1)=(-1,1,1) + t
    assert transform_points(points.astype(np.float32), matrix).dtype == np.float32


def test_helpers_reject_bad_inputs():
    K = np.eye(3)
    for name, args in {
        "depth 3-D": (np.zeros((2, 2, 2)), K),
        "K shape": (np.zeros((2, 2)), np.eye(2)),
        "K nonfinite": (np.zeros((2, 2)), np.full((3, 3), np.nan)),
    }.items():
        _expect_value_error(unproject_depth, args, label=name)

    points = np.zeros((2, 3))
    for name, matrix in {
        "matrix shape": np.eye(3),
        "matrix nonfinite": np.full((4, 4), np.nan),
        "matrix batch": np.eye(4)[None],
    }.items():
        _expect_value_error(transform_points, (points, matrix), label=name)
    _expect_value_error(transform_points, (np.zeros((2, 2)), np.eye(4)), label="points Nx2")
