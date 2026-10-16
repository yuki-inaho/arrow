"""``src/arrow/depth_detail.py`` 共有API契約の合成テスト。

固定契約は temp/workdoc_Oct16-2026_depth_workflow.md の
「固定する共有コード契約」に従う。numpy / Pillow だけで完結する。

テスト対象（6関数）:
- fit_scale: 正・有限のスケール推定、outlier 除去、対応不足の拒否
- regularize_scales: log 領域の時系列平滑化、正値保持
- gray_edges: [0,1] の境界マップ、uint8/float の厳密検証
- blend_depth: trust 内のみ detail を blend、invalid は NaN、穴を埋めない
- reproject_depth: integer pixel centre、z-buffer、occlusion、invalid 検証
- temporal_fuse: 境界/支持/trust ゲート付き時間融合
"""

from __future__ import annotations

import numpy as np
import pytest

from arrow.depth_detail import (
    blend_depth,
    fit_scale,
    gray_edges,
    regularize_scales,
    reproject_depth,
    superresolve_scene_flow,
    temporal_fuse,
)

RNG = np.random.default_rng(20261016)


# ---------------------------------------------------------------------------
# fit_scale
# ---------------------------------------------------------------------------


def test_fit_scale_recovers_exact_positive_scale():
    source = RNG.uniform(0.5, 5.0, size=(40, 50))
    target = 2.5 * source
    result = fit_scale(source, target)
    assert set(result) == {"scale", "inliers", "median_abs_log_residual"}
    assert result["scale"] == pytest.approx(2.5, rel=1e-9)
    assert np.isfinite(result["scale"]) and result["scale"] > 0.0
    assert result["inliers"] == source.size
    assert result["median_abs_log_residual"] < 1e-9


def test_fit_scale_is_robust_to_outliers():
    n = 64 * 64
    source = RNG.uniform(1.0, 10.0, size=(64, 64))
    target = 3.0 * source
    n_outliers = n // 20
    flat_idx = RNG.choice(n, size=n_outliers, replace=False)
    target.ravel()[flat_idx] = source.ravel()[flat_idx] * 300.0
    result = fit_scale(source, target)
    assert result["scale"] == pytest.approx(3.0, rel=2e-2)
    assert result["inliers"] == n - n_outliers
    assert result["median_abs_log_residual"] < 1e-6


def test_fit_scale_tolerates_moderate_noise():
    source = RNG.uniform(0.5, 4.0, size=(32, 32))
    noise = np.exp(RNG.normal(scale=0.01, size=source.shape))
    target = 1.5 * source * noise
    result = fit_scale(source, target)
    assert result["scale"] == pytest.approx(1.5, rel=0.01)
    assert result["inliers"] >= int(0.9 * source.size)
    assert result["median_abs_log_residual"] < 0.05


def test_fit_scale_rejects_insufficient_correspondences():
    source = np.ones((8, 8))
    target = np.ones((8, 8))
    target.ravel()[10:] = 0.0  # 有効対応は 10 点のみ
    with pytest.raises(ValueError):
        fit_scale(source, target, min_points=32)
    result = fit_scale(source, target, min_points=8)
    assert result["scale"] == pytest.approx(1.0)


def test_fit_scale_respects_mask_and_ignores_invalid_values():
    source = np.full((60, 60), 2.0)
    target = np.full((60, 60), 2.0)
    target[:30] *= 2.0  # 上半分は scale 2、下半分は scale 1
    mask_top = np.zeros((60, 60), dtype=bool)
    mask_top[:30] = True
    top = fit_scale(source, target, mask=mask_top)
    bottom = fit_scale(source, target, mask=~mask_top)
    assert top["scale"] == pytest.approx(2.0)
    assert bottom["scale"] == pytest.approx(1.0)

    source2 = RNG.uniform(1.0, 3.0, size=(20, 10))
    target2 = 4.0 * source2
    source2[0, 0] = np.nan
    source2[0, 1] = 0.0
    source2[0, 2] = -1.0
    target2[0, 3] = np.inf
    target2[0, 4] = 0.0
    target2[0, 5] = -2.0
    result = fit_scale(source2, target2)
    assert result["scale"] == pytest.approx(4.0)
    assert result["inliers"] == 20 * 10 - 6


def test_fit_scale_rejects_shape_mismatch_and_empty_mask():
    with pytest.raises(ValueError):
        fit_scale(np.ones((4, 4)), np.ones((4, 5)))
    with pytest.raises(ValueError):
        fit_scale(np.ones((4, 4)), np.ones((4, 4)), mask=np.zeros((4, 4), dtype=bool))


# ---------------------------------------------------------------------------
# regularize_scales
# ---------------------------------------------------------------------------


def test_regularize_strength_zero_is_exact_identity():
    scales = np.array([1.0, 2.0, 0.5, 3.0])
    weights = np.array([1.0, 0.5, 2.0, 1.0])
    out = regularize_scales(scales, weights, strength=0.0)
    assert np.array_equal(out, scales)


def test_regularize_single_and_empty_arrays():
    assert np.array_equal(regularize_scales(np.array([2.0]), np.array([1.0])), np.array([2.0]))
    empty = regularize_scales(np.array([]), np.array([]))
    assert empty.shape == (0,)


def test_regularize_reduces_low_weight_spike():
    scales = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
    weights = np.array([1.0, 1.0, 0.05, 1.0, 1.0])
    out = regularize_scales(scales, weights, strength=2.0)
    assert out.shape == scales.shape
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)
    assert out[2] < 5.0
    assert out[2] > 1.0
    # 隣接フレームはもとの 1.0 近傍に留まる
    assert out[0] == pytest.approx(1.0, abs=0.2)
    assert out[4] == pytest.approx(1.0, abs=0.2)


def test_regularize_high_weight_spike_is_less_smoothed():
    scales = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
    low = regularize_scales(scales, np.array([1.0, 1.0, 0.01, 1.0, 1.0]))
    high = regularize_scales(scales, np.array([1.0, 1.0, 100.0, 1.0, 1.0]))
    assert low[2] < high[2]


def test_regularize_reduces_log_variance_and_keeps_positivity():
    n = 107
    log_scales = np.cumsum(RNG.normal(scale=0.1, size=n)) + np.log(2.0)
    scales = np.exp(log_scales)
    weights = RNG.uniform(0.2, 2.0, size=n)
    out = regularize_scales(scales, weights, strength=2.0)
    assert out.shape == (n,)
    assert np.all(np.isfinite(out)) and np.all(out > 0.0)
    assert np.std(np.log(out)) < np.std(log_scales)


def test_regularize_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, -2.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, 2.0]), np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        regularize_scales(np.array([1.0, 2.0]), np.array([1.0, 1.0]), strength=-1.0)


# ---------------------------------------------------------------------------
# gray_edges
# ---------------------------------------------------------------------------


def _step_image_uint8(h: int = 8, w: int = 10) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :, :] = 255
    return img


def test_gray_edges_vertical_step_is_localized_and_normalized():
    img = _step_image_uint8()
    edges = gray_edges(img)
    assert edges.shape == img.shape[:2]
    assert edges.dtype == np.float64
    assert edges.min() >= 0.0
    assert edges.max() <= 1.0
    # 最大応答は境界の両側列のみ
    assert np.unravel_index(np.argmax(edges), edges.shape)[1] in (4, 5)
    assert np.all(edges[:, 4] > 0.9)
    assert np.all(edges[:, 5] > 0.9)
    assert np.all(edges[:, :4] == 0.0)
    assert np.all(edges[:, 6:] == 0.0)


def test_gray_edges_uint8_and_float01_agree():
    img = _step_image_uint8()
    edges_u8 = gray_edges(img)
    edges_f = gray_edges(img.astype(np.float32) / 255.0)
    assert np.allclose(edges_u8, edges_f, atol=1e-6)


def test_gray_edges_uniform_image_is_zero():
    img = np.full((6, 7, 3), 128, dtype=np.uint8)
    edges = gray_edges(img)
    assert np.all(edges == 0.0)


def test_gray_edges_strictly_validates_input():
    img = _step_image_uint8()
    with pytest.raises(ValueError):
        gray_edges(img.astype(np.float32))  # 0..255 の float は不許可
    with pytest.raises(ValueError):
        gray_edges(img.astype(np.int32))
    with pytest.raises(ValueError):
        gray_edges(img[:, :, :2])
    with pytest.raises(ValueError):
        gray_edges(img[:, :, 0])
    bad = img.astype(np.float32) / 255.0
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        gray_edges(bad)


# ---------------------------------------------------------------------------
# blend_depth
# ---------------------------------------------------------------------------


def test_blend_depth_blends_within_trust():
    anchor = np.full((4, 4), 2.0)
    detail = np.full((4, 4), 2.05)
    out = blend_depth(anchor, detail, detail_weight=0.5, trust_abs=0.2, trust_rel=0.01)
    assert np.allclose(out, 2.025)
    assert np.all(np.isfinite(out))


def test_blend_depth_keeps_anchor_outside_trust():
    anchor = np.full((4, 4), 2.0)
    detail = np.full((4, 4), 2.5)  # 差 0.5 > trust 0.2
    out = blend_depth(anchor, detail, detail_weight=0.5, trust_abs=0.2, trust_rel=0.01)
    assert np.all(out == anchor)


def test_blend_depth_relative_trust_dominates_for_large_depths():
    anchor = np.full((4, 4), 100.0)
    detail_close = np.full((4, 4), 104.0)
    detail_far = np.full((4, 4), 110.0)
    out_close = blend_depth(anchor, detail_close, detail_weight=0.25, trust_abs=0.05, trust_rel=0.05)
    out_far = blend_depth(anchor, detail_far, detail_weight=0.25, trust_abs=0.05, trust_rel=0.05)
    assert np.allclose(out_close, 101.0)  # trust 帯 = max(0.05, 5.0) = 5.0
    assert np.all(out_far == anchor)  # 差 10.0 > 5.0 なので anchor 維持


def test_blend_depth_invalid_stays_nan_and_holes_are_not_filled():
    anchor = np.full((3, 4), 1.0)
    detail = np.full((3, 4), 1.0)
    valid = np.ones((3, 4), dtype=bool)
    valid[0, 0] = False
    anchor[0, 1] = np.nan
    anchor[0, 2] = 0.0
    anchor[0, 3] = -1.0
    detail[1, 0] = np.nan  # detail 側の穴
    out = blend_depth(anchor, detail, valid=valid, detail_weight=0.7)
    assert np.isnan(out[0, 0])  # valid=False
    assert np.isnan(out[0, 1])  # anchor の穴は detail があっても埋めない
    assert np.isnan(out[0, 2])  # 非正 anchor は invalid
    assert np.isnan(out[0, 3])
    assert out[1, 0] == 1.0  # detail の穴は anchor 維持（NaN にしない）
    assert out[2, 2] == 1.0


def test_blend_depth_weight_limits():
    anchor = np.full((3, 3), 2.0)
    detail = np.full((3, 3), 2.01)
    zero = blend_depth(anchor, detail, detail_weight=0.0, trust_abs=0.1, trust_rel=0.01)
    one = blend_depth(anchor, detail, detail_weight=1.0, trust_abs=0.1, trust_rel=0.01)
    assert np.all(zero == anchor)
    assert np.allclose(one, detail)


def test_blend_depth_rejects_invalid_arguments():
    anchor = np.ones((3, 3))
    with pytest.raises(ValueError):
        blend_depth(anchor, np.ones((3, 4)))
    with pytest.raises(ValueError):
        blend_depth(anchor, anchor, detail_weight=1.5)
    with pytest.raises(ValueError):
        blend_depth(anchor, anchor, detail_weight=-0.1)
    with pytest.raises(ValueError):
        blend_depth(anchor, anchor, trust_abs=-1.0)
    with pytest.raises(ValueError):
        blend_depth(anchor, anchor, trust_rel=-0.1)
    with pytest.raises(ValueError):
        blend_depth(anchor, anchor, valid=np.ones((2, 2), dtype=bool))


# ---------------------------------------------------------------------------
# reproject_depth
# ---------------------------------------------------------------------------

K_5 = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])


def _T(rotation: np.ndarray | None = None, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def test_reproject_identity_roundtrip_exact():
    depth = RNG.uniform(0.5, 3.0, size=(6, 7))
    depth[0, 0] = 0.0
    depth[0, 1] = -1.0
    depth[0, 2] = np.nan
    depth[0, 3] = np.inf
    rgb = RNG.integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
    K = np.array([[4.2, 0.0, 2.9], [0.0, 3.7, 3.1], [0.0, 0.0, 1.0]])
    out = reproject_depth(depth, rgb, K, K, np.eye(4), (6, 7))
    assert set(out) == {"depth", "rgb", "valid"}
    valid = out["valid"]
    expected_valid = np.isfinite(depth) & (depth > 0.0)
    assert valid.dtype == np.bool_
    assert np.array_equal(valid, expected_valid)
    assert np.allclose(out["depth"][valid], depth[valid], atol=1e-12)
    assert np.all(np.isnan(out["depth"][~valid]))
    assert np.array_equal(out["rgb"][valid], rgb[valid])
    assert np.all(out["rgb"][~valid] == 0)
    assert out["rgb"].dtype == np.uint8


def test_reproject_identity_preserves_float_rgb_dtype():
    depth = np.ones((4, 4))
    rgb = RNG.random((4, 4, 3)).astype(np.float32)
    out = reproject_depth(depth, rgb, K_5[:3, :3], K_5, np.eye(4), (4, 4))
    assert out["rgb"].dtype == np.float32
    assert np.allclose(out["rgb"], rgb)


def test_reproject_pure_translation_shifts_plane():
    depth = np.full((5, 5), 2.0)
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    for col in range(5):
        rgb[:, col] = (col * 10, 0, 0)
    out = reproject_depth(depth, rgb, K_5, K_5, _T(translation=(2.0, 0.0, 0.0)), (5, 5))
    # fx=fy=1, cx=cy=2, t=(+2,0,0), z=2 → u_t = u + 1 (行は不変)
    assert out["valid"].sum() == 5 * 4
    assert np.all(out["valid"][:, 1:]) and not out["valid"][:, 0].any()
    assert np.allclose(out["depth"][:, 1:], 2.0)
    assert np.all(np.isnan(out["depth"][:, 0]))
    assert np.array_equal(out["rgb"][:, 1:], rgb[:, :-1])
    assert np.all(out["rgb"][:, 0] == 0)


def test_reproject_rotation_about_y_axis():
    depth = np.full((5, 5), np.nan)
    depth[2, 3] = 1.0  # X=(1,0,1)
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[2, 3] = (1, 2, 3)
    theta = -np.pi / 4.0
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    out = reproject_depth(depth, rgb, K_5, K_5, _T(rotation=R), (5, 5))
    assert out["valid"].sum() == 1
    assert out["valid"][2, 2]
    assert out["depth"][2, 2] == pytest.approx(np.sqrt(2.0), rel=1e-12)
    assert np.array_equal(out["rgb"][2, 2], np.array([1, 2, 3], dtype=np.uint8))
    assert np.isnan(out["depth"][2, 3])


def test_reproject_z_buffer_keeps_nearest_and_occludes():
    depth = np.full((5, 6), np.nan)
    depth[2, 2] = 1.0  # far: z_t = 1.0 → u_t = 0
    depth[2, 3] = 2.0 / 3.0  # near: z_t = 2/3 → u_t = 0
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[2, 2] = (255, 0, 0)  # 先に走査される far（naive first-writer では残ってしまう）
    rgb[2, 3] = (0, 255, 0)
    out = reproject_depth(depth, rgb, K_5, K_5, _T(translation=(-2.0, 0.0, 0.0)), (5, 6))
    assert out["valid"].sum() == 1
    assert out["valid"][2, 0]
    assert out["depth"][2, 0] == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert np.array_equal(out["rgb"][2, 0], np.array([0, 255, 0], dtype=np.uint8))


def test_reproject_rejects_out_of_bounds_and_behind_camera():
    depth = np.full((5, 5), np.nan)
    depth[2, 2] = 1.0
    rgb = np.full((5, 5, 3), 7, dtype=np.uint8)
    oob = reproject_depth(depth, rgb, K_5, K_5, _T(translation=(-4.0, 0.0, 0.0)), (5, 5))
    assert oob["valid"].sum() == 0
    assert np.all(np.isnan(oob["depth"]))
    assert np.all(oob["rgb"] == 0)
    behind = reproject_depth(depth, rgb, K_5, K_5, _T(translation=(0.0, 0.0, -2.0)), (5, 5))
    assert behind["valid"].sum() == 0


def test_reproject_strictly_validates_arguments():
    depth = np.ones((4, 4))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb[:, :, :2], K_5, K_5, np.eye(4), (4, 4))
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb[:3], K_5, K_5, np.eye(4), (4, 4))
    bad_Kt = K_5.copy()
    bad_Kt[1, 1] = np.inf
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, K_5, bad_Kt, np.eye(4), (4, 4))
    bad_K = K_5.copy()
    bad_K[0, 0] = np.nan
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, bad_K, K_5, np.eye(4), (4, 4))
    bad_K2 = K_5.copy()
    bad_K2[0, 0] = 0.0
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, bad_K2, K_5, np.eye(4), (4, 4))
    bad_T = np.eye(4)
    bad_T[3, 3] = 2.0  # 最終行が [0,0,0,1] でない
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, K_5, K_5, bad_T, (4, 4))
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, K_5, K_5, np.eye(3), (4, 4))
    with pytest.raises(ValueError):
        reproject_depth(depth, rgb, K_5, K_5, np.eye(4), (0, 4))
    with pytest.raises(ValueError):
        reproject_depth(np.ones(4), rgb, K_5, K_5, np.eye(4), (4, 4))


# ---------------------------------------------------------------------------
# temporal_fuse
# ---------------------------------------------------------------------------


def test_temporal_fuse_blends_supported_pixels():
    current = np.full((4, 4), 1.0)
    candidate = np.full((4, 4), 1.02)
    support = np.ones((4, 4), dtype=bool)
    edges = np.zeros((4, 4))
    out = temporal_fuse(current, candidate, support, edges, weight=0.5, trust_abs=0.1)
    assert np.allclose(out, 1.01)


def test_temporal_fuse_edge_gate_blocks_boundary_diffusion():
    current = np.full((4, 4), 1.0)
    candidate = np.full((4, 4), 1.02)
    support = np.ones((4, 4), dtype=bool)
    edges = np.zeros((4, 4))
    edges[:, 0] = 1.0  # 境界列は融合禁止
    out = temporal_fuse(current, candidate, support, edges, weight=0.5, trust_abs=0.1)
    assert np.all(out[:, 0] == 1.0)
    assert np.allclose(out[:, 1:], 1.01)


def test_temporal_fuse_partial_edge_attenuates_weight():
    current = np.full((2, 2), 1.0)
    candidate = np.full((2, 2), 1.02)
    edges = np.full((2, 2), 0.5)
    out = temporal_fuse(current, candidate, np.ones((2, 2), dtype=bool), edges,
                        weight=0.4, trust_abs=0.1)
    assert np.allclose(out, 1.0 + 0.2 * 0.02)


def test_temporal_fuse_trust_gate_keeps_current():
    current = np.full((3, 3), 1.0)
    candidate = np.full((3, 3), 1.5)  # trust 0.1 を大きく超える
    out = temporal_fuse(current, candidate, np.ones((3, 3), dtype=bool), np.zeros((3, 3)))
    assert np.all(out == current)


def test_temporal_fuse_support_gate_keeps_current():
    current = np.full((3, 3), 1.0)
    candidate = np.full((3, 3), 1.02)
    support = np.ones((3, 3), dtype=bool)
    support[1] = False  # 支持のない行は融合しない
    out = temporal_fuse(current, candidate, support, np.zeros((3, 3)), weight=0.5, trust_abs=0.1)
    assert np.all(out[1] == 1.0)
    assert np.allclose(out[0], 1.01)


def test_temporal_fuse_holes_are_preserved():
    current = np.full((2, 2), 1.0)
    current[0, 0] = np.nan
    candidate = np.full((2, 2), 1.02)
    candidate[1, 1] = np.nan
    out = temporal_fuse(current, candidate, np.ones((2, 2), dtype=bool), np.zeros((2, 2)),
                        weight=0.5, trust_abs=0.1)
    assert np.isnan(out[0, 0])  # 現フレームの穴は候補で埋めない
    assert out[1, 1] == 1.0  # 候補の穴は current 維持


def test_temporal_fuse_rejects_invalid_arguments():
    current = np.ones((3, 3))
    candidate = np.ones((3, 3)) * 1.01
    support = np.ones((3, 3), dtype=bool)
    edges = np.zeros((3, 3))
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate[:2], support, edges)
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support[:2], edges)
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support, edges[:2])
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support, np.full((3, 3), 1.2))
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support, np.full((3, 3), -0.1))
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support, edges, weight=2.0)
    with pytest.raises(ValueError):
        temporal_fuse(current, candidate, support, edges, trust_abs=-0.5)


# ---------------------------------------------------------------------------
# superresolve_scene_flow
# ---------------------------------------------------------------------------


def _bilinear_reference(low_flow: np.ndarray, high_h: int, high_w: int) -> np.ndarray:
    """テスト独立の単純 bilinear 参照（pixel centre 対応、3D 変位もそのまま補間）。"""
    hl, wl = low_flow.shape[:2]
    ys = np.clip((np.arange(high_h) + 0.5) * hl / high_h - 0.5, 0.0, hl - 1.0)
    xs = np.clip((np.arange(high_w) + 0.5) * wl / high_w - 0.5, 0.0, wl - 1.0)
    out = np.empty((high_h, high_w, low_flow.shape[2]), dtype=np.float64)
    for i in range(high_h):
        r0 = int(np.floor(ys[i]))
        r1 = min(r0 + 1, hl - 1)
        fv = float(ys[i] - r0)
        for j in range(high_w):
            c0 = int(np.floor(xs[j]))
            c1 = min(c0 + 1, wl - 1)
            fu = float(xs[j] - c0)
            w = ((1.0 - fv) * (1.0 - fu), (1.0 - fv) * fu, fv * (1.0 - fu), fv * fu)
            out[i, j] = (
                w[0] * low_flow[r0, c0]
                + w[1] * low_flow[r0, c1]
                + w[2] * low_flow[r1, c0]
                + w[3] * low_flow[r1, c1]
            )
    return out


def test_superresolve_constant_flow_is_not_rescaled_by_upsampling():
    low = np.zeros((4, 4, 3)) + np.array([0.3, -0.2, 0.05])
    depth = np.full((8, 8), 2.0)
    rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
    out = superresolve_scene_flow(low, depth, rgb)
    assert set(out) == {"flow", "valid"}
    assert out["flow"].shape == (8, 8, 3)
    assert out["valid"].shape == (8, 8)
    assert out["valid"].dtype == np.bool_
    assert out["valid"].all()
    # 3D 変位は解像度比 (8/4=2) を掛けない
    assert np.allclose(out["flow"], np.array([0.3, -0.2, 0.05]), atol=1e-12)

    low2 = np.zeros((3, 5, 3)) + np.array([0.4, -0.3, 0.2])  # 非整数倍率でも同様
    depth2 = np.full((7, 9), 1.5)
    rgb2 = np.full((7, 9, 3), 0.5, dtype=np.float32)
    out2 = superresolve_scene_flow(low2, depth2, rgb2)
    assert out2["flow"].shape == (7, 9, 3)
    assert out2["valid"].all()
    assert np.allclose(out2["flow"], np.array([0.4, -0.3, 0.2]), atol=1e-12)


def test_superresolve_step_motion_beats_bilinear_at_boundary():
    low = np.zeros((4, 4, 3))
    low[:, :2, 0] = 1.0  # 左半分は +1 のステップ運動
    depth = np.ones((8, 8))
    depth[:, 4:] = 2.0  # 深度境界
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, 4:, :] = 255  # 同じ位置の色境界
    out = superresolve_scene_flow(low, depth, rgb)
    gt = np.zeros((8, 8, 3))
    gt[:, :4, 0] = 1.0
    ref = _bilinear_reference(low, 8, 8)
    epe_sr = np.abs(out["flow"] - gt).sum(axis=2)
    epe_ref = np.abs(ref - gt).sum(axis=2)
    # bilinear は境界列で漏れるが SR は深度/色境界で運動辺を選ぶ
    assert epe_ref[:, 3].mean() > 0.2
    assert epe_ref[:, 4].mean() > 0.2
    assert epe_sr.max() < 1e-6
    assert epe_sr.mean() < epe_ref.mean()
    assert np.allclose(out["flow"][:, :4], [1.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(out["flow"][:, 4:], [0.0, 0.0, 0.0], atol=1e-9)


def test_superresolve_moving_camera_occlusion_uses_depth_boundaries():
    h = w = 8
    depth = np.full((h, w), 3.0)
    depth[2:6, 2:6] = 1.0  # 手前の矩形（occlusion boundary）
    fx = fy = 4.0
    cx = cy = 3.5
    theta = 0.05
    c, s = np.cos(theta), np.sin(theta)
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    x = (jj - cx) * depth / fx
    y = (ii - cy) * depth / fy
    points = np.stack([x, y, depth], axis=-1)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    gt = points @ (rotation - np.eye(3)).T  # GT 3D flow = R X - X
    low = gt.reshape(4, 2, 4, 2, 3).mean(axis=(1, 3))  # 低解像度は境界で混合
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[2:6, 2:6] = 255
    out = superresolve_scene_flow(low, depth, rgb)
    ref = _bilinear_reference(low, h, w)
    epe_sr = np.abs(out["flow"] - gt).sum(axis=2)
    epe_ref = np.abs(ref - gt).sum(axis=2)
    band = slice(2, 6)
    assert epe_ref[band, 2].mean() > epe_sr[band, 2].mean()
    assert epe_ref[band, 5].mean() > epe_sr[band, 5].mean()
    assert epe_sr[band, 2].mean() < 0.01  # 背景 flow の漏れなし
    assert epe_sr[band, 5].mean() < 0.01
    assert epe_sr.mean() < epe_ref.mean()
    assert np.allclose(out["flow"][..., 1], 0.0, atol=1e-9)  # y 変位は 0 のまま


def test_superresolve_invalid_support_becomes_invalid_nan():
    low = np.zeros((4, 4, 3)) + np.array([0.1, 0.0, 0.0])
    depth = np.full((8, 8), 1.0)
    rgb = np.full((8, 8, 3), 64, dtype=np.uint8)
    # 全 low 無効 → 全 invalid
    out = superresolve_scene_flow(low, depth, rgb, low_valid=np.zeros((4, 4), dtype=bool))
    assert not out["valid"].any()
    assert np.all(np.isnan(out["flow"]))
    # 下端 low 行のみ無効 → 高解像度最終行が invalid（他は有効）
    valid = np.ones((4, 4), dtype=bool)
    valid[3] = False
    out2 = superresolve_scene_flow(low, depth, rgb, low_valid=valid)
    assert not out2["valid"][7].any()
    assert np.all(np.isnan(out2["flow"][7]))
    assert out2["valid"][:7].any()
    # 角の 1 点だけ無効 → そこだけ invalid、隣接は有効
    valid3 = np.ones((4, 4), dtype=bool)
    valid3[0, 0] = False
    out3 = superresolve_scene_flow(low, depth, rgb, low_valid=valid3)
    assert not out3["valid"][0, 0]
    assert out3["valid"][0, 1] and out3["valid"][1, 0]
    # low_flow の NaN も無効サンプル扱い
    low_nan = low.copy()
    low_nan[0, 0] = np.nan
    out4 = superresolve_scene_flow(low_nan, depth, rgb)
    assert not out4["valid"][0, 0]
    assert out4["valid"][0, 1]
    assert np.all(np.isfinite(out4["flow"][out4["valid"]]))


def test_superresolve_all_invalid_depth_rejects_without_fallback():
    # 統括独立反例: all-NaN depth + finite low_flow → unsupported は全 NaN/false
    low = np.ones((2, 2, 3))
    depth = np.full((4, 4), np.nan)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    out = superresolve_scene_flow(low, depth, rgb)
    assert set(out) == {"flow", "valid"}
    assert int(out["valid"].sum()) == 0  # 期待 valid_count = 0
    assert not out["valid"].any()
    assert np.all(np.isnan(out["flow"]))


def test_superresolve_mixed_depth_holes_are_unsupported_only_there():
    low = np.ones((4, 4, 3)) * 0.5
    depth = np.full((8, 8), 2.0)
    depth[0:2, :] = np.nan  # NaN 行
    depth[:, 6:] = np.inf  # inf 列
    depth[3, 3] = 0.0  # 非正画素
    invalid = ~(np.isfinite(depth) & (depth > 0.0))
    rgb = np.full((8, 8, 3), 90, dtype=np.uint8)
    out = superresolve_scene_flow(low, depth, rgb)
    # 有効深度の画素は（少なくとも1つの深度支持 candidate があるため）valid、
    # invalid 画素だけが NaN/false になる
    assert np.array_equal(out["valid"], ~invalid)
    assert np.all(np.isnan(out["flow"][invalid]))
    assert np.all(np.isfinite(out["flow"][~invalid]))
    assert int((~out["valid"]).sum()) == int(invalid.sum())  # 診断: 失敗数


def test_superresolve_depth_weight_underflow_is_unsupported():
    low = np.ones((4, 4, 3))
    depth = np.full((8, 8), 1e30)
    depth[3, 3] = 1.0  # guide 位置（偶数格子）には現れない target のみ有効
    rgb = np.full((8, 8, 3), 50, dtype=np.uint8)
    out = superresolve_scene_flow(low, depth, rgb)
    # 候補深度 1e30 対 target 1.0 → log 比 ~69 → 重みが完全 underflow。
    # 無条件 bilinear fallback は行わず unsupported = NaN/false。
    assert not out["valid"][3, 3]
    assert np.all(np.isnan(out["flow"][3, 3]))
    assert int((~out["valid"]).sum()) == 1  # 診断: 失敗数 1
    assert int(out["valid"].sum()) == 63
    assert np.all(np.isfinite(out["flow"][out["valid"]]))


def test_superresolve_strictly_validates_arguments():
    low = np.zeros((4, 4, 3))
    depth = np.ones((8, 8))
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        superresolve_scene_flow(np.zeros((4, 4, 2)), depth, rgb)
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, np.ones((8, 8, 1)), rgb)
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, rgb[:, :, :2])
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, np.zeros((8, 7, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, rgb.astype(np.float32) + 1.5)
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, rgb.astype(np.int32))
    with pytest.raises(ValueError):
        superresolve_scene_flow(low, depth, rgb, low_valid=np.ones((3, 3), dtype=bool))
