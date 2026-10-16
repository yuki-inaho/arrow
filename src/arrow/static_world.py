"""canonical 静止世界の数値プリミティブ（rigid/non-rigid 分解・label 移流・点群 render）。

固定契約: temp/workdoc_Oct16-2026_sr_static_world.md §2「固定数値契約 (TR3/TR4)」。
numpy のみで import できる（torch・GPU 依存を持ち込まない）。

共通規約:
- integer pixel centre: 画素 (i, j) の中心 = 連続座標 (u=j, v=i)。投影の整数化は ``np.rint``。
- 3D 変位は camera 間の差として定義する（異時刻の camera 座標の差）。SR の S は camera0 で
  補償済みの 3D residual であり、ここで再度 ego 補償しない。
- invalid（valid=False / 非有限 / depth<=0 / behind camera）の出力数値は NaN、mask は False。
  穴を勝手に補間・valid 化しない。frustum 外でも有限に定義できる flow/3D は保持し、
  track 側（advect_labels）が bounds で拒否する。
- decompose_flow は float64 で計算・返す。unproject_depth / transform_points は
  入力 dtype が float32 のときだけ float32、それ以外は float64 を返す。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "decompose_flow",
    "advect_labels",
    "render_points",
    "unproject_depth",
    "transform_points",
]

# rigid pose 検証の許容幅（float32 入力の丸めより大きく、実害のある shear/scale より小さい）
_RIGID_ATOL = 1e-5
# 距離ゲートの境界 inclusive 用の僅かな許容（浮動小数の丸め対策）
_DIST_EPS = 1e-9
# 相対深度ゲートの境界 inclusive 用の僅かな許容（丸めで 0.05 + 極小になる境界を救う）
_RELATIVE_EPS = 1e-12
# render の splat 半径上限（メモリ量を制御する）
_MAX_RENDER_RADIUS = 3
# render の 1 chunk あたり候補ペア数（offset 数を掛けて点数へ換算する）
_RENDER_CHUNK_PAIRS = 4_000_000


def _as_float_array(array, name):
    values = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite array")
    return values


def _check_intrinsics(K, name="K"):
    """3x3 有限・正 focal・非特異の intrinsics を検証して返す。"""
    K = _as_float_array(K, name)
    if K.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
        raise ValueError(f"{name} must have positive focal lengths")
    if abs(float(np.linalg.det(K))) < 1e-12:
        raise ValueError(f"{name} must be non-singular")
    return K


def _check_rigid(matrix, name):
    """4x4 rigid c2w（最終行 [0,0,0,1]、R は直交かつ det=+1）を検証して返す。"""
    M = _as_float_array(matrix, name)
    if M.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.allclose(M[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    R = M[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=_RIGID_ATOL):
        raise ValueError(f"{name} rotation part must be orthonormal")
    if abs(float(np.linalg.det(R)) - 1.0) > _RIGID_ATOL:
        raise ValueError(f"{name} rotation determinant must be +1")
    return M


def _validate_shape(shape, name="shape"):
    """(H, W) の正の整数 2 つへ正規化する。"""
    try:
        values = tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be two positive integers") from None
    if len(values) != 2 or values[0] < 1 or values[1] < 1:
        raise ValueError(f"{name} must be two positive integers")
    return values


def _valid_mask(valid, shape, name="valid"):
    """任意 dtype の valid を bool mask へ変換する（非有限・0 は False）。"""
    values = np.asarray(valid)
    if values.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    numbers = values.astype(np.float64)
    return np.isfinite(numbers) & (numbers != 0.0)


def _pixel_grid(h, w):
    """画素中心 (u=j, v=i) の (H, W, 2) float64 グリッド。"""
    u = np.broadcast_to(np.arange(w, dtype=np.float64), (h, w))
    v = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    return np.stack([u, v], axis=-1)


def _homogeneous_grid(h, w):
    """画素中心の同次座標 (H, W, 3) = [u, v, 1]。"""
    uv = _pixel_grid(h, w)
    ones = np.ones((h, w, 1), dtype=np.float64)
    return np.concatenate([uv, ones], axis=-1)


def _project(K, points):
    """camera 座標 (..., 3) を K で投影し (uv (..., 2), z (...)) を返す。

    z<=0 でも計算だけは行う（呼び出し側の mask で扱う）。
    """
    z = points[..., 2]
    u_num = K[0, 0] * points[..., 0] + K[0, 1] * points[..., 1] + K[0, 2] * z
    v_num = K[1, 0] * points[..., 0] + K[1, 1] * points[..., 1] + K[1, 2] * z
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([u_num / z, v_num / z], axis=-1)
    return uv, z


def unproject_depth(depth, K):
    """depth (H, W) を K で camera 座標 (H, W, 3) へ逆投影する。

    P = depth * K^-1 [u, v, 1]。非有限・非正 depth の画素は 3 成分すべて NaN。
    入力 dtype が float32 のときだけ float32、それ以外は float64 を返す。
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError("depth must be 2-D")
    K = _check_intrinsics(K, "K")
    dtype = np.float32 if depth.dtype == np.float32 else np.float64
    values = depth.astype(np.float64)
    ray = _homogeneous_grid(*values.shape) @ np.linalg.inv(K).T
    z = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)
    return (ray * z[..., None]).astype(dtype)


def transform_points(points, matrix):
    """points (..., 3) を affine 4x4 matrix で変換する。

    matrix は有限 4x4 で最終行 [0, 0, 0, 1] を要求する（rigid に限らない）。
    入力 dtype が float32 のときだけ float32、それ以外は float64 を返す。
    """
    points = np.asarray(points)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        raise ValueError("matrix must be a finite 4x4 matrix")
    if not np.allclose(M[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("matrix last row must be [0, 0, 0, 1]")
    dtype = np.float32 if points.dtype == np.float32 else np.float64
    values = points.astype(np.float64)
    return (values @ M[:3, :3].T + M[:3, 3]).astype(dtype)


def decompose_flow(depth, K0, K1, M0, M1, flow, valid):
    """camera 補償済み SR flow を rigid / non-rigid / total へ分解する。

    P = depth * K0^-1 [u, v, 1]（camera0）、T = inv(M1) @ M0 = (R, t)、
    Qr = R P + t、Qt = R (P + S) + t とする。S は camera0 で補償済みの 3D residual で、
    ここで再度 ego 補償しない。

    2D:
      rigid_flow    = project(K1, Qr) - [u, v]
      total_flow    = project(K1, Qt) - [u, v]
      nonrigid_flow = project(K1, Qt) - project(K1, Qr)
    3D（camera 間の差。座標系が異なる時刻の差であることに注意）:
      rigid_3d      = Qr - P
      nonrigid_3d   = R S（camera1 の向きの局所変位）
      total_3d      = Qt - P = rigid_3d + nonrigid_3d
      world_nonrigid= R0 S（R0 は M0 の回転。world 向きの residual）

    valid は入力 valid・有限 depth>0・有限 S・Qr/Qt が camera 前方（z>0）・投影有限の
    すべてを満たす画素のみ True。frustum 外は有限値を保持し、behind camera は False。
    返り値の数値配列はすべて float64、invalid 画素は NaN。
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth must be 2-D")
    h, w = depth.shape
    flow = np.asarray(flow, dtype=np.float64)
    if flow.shape != (h, w, 3):
        raise ValueError("flow must have shape (H, W, 3) matching depth")
    K0 = _check_intrinsics(K0, "K0")
    K1 = _check_intrinsics(K1, "K1")
    M0 = _check_rigid(M0, "M0")
    M1 = _check_rigid(M1, "M1")
    mask = _valid_mask(valid, (h, w))

    source_ok = mask & np.isfinite(depth) & (depth > 0.0) & np.all(np.isfinite(flow), axis=-1)

    grid = _pixel_grid(h, w)
    z0 = np.where(source_ok, depth, 0.0)
    points = (_homogeneous_grid(h, w) @ np.linalg.inv(K0).T) * z0[..., None]
    residual = np.where(source_ok[..., None], flow, 0.0)

    # M は rigid なので逆行列は必ず存在する。
    transform = np.linalg.inv(M1) @ M0
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    rigid_camera = points @ rotation.T + translation
    total_camera = (points + residual) @ rotation.T + translation

    rigid_uv, rigid_z = _project(K1, rigid_camera)
    total_uv, total_z = _project(K1, total_camera)
    ok = (
        source_ok
        & (rigid_z > 0.0)
        & (total_z > 0.0)
        & np.isfinite(rigid_z)
        & np.isfinite(total_z)
        & np.isfinite(rigid_uv).all(axis=-1)
        & np.isfinite(total_uv).all(axis=-1)
    )

    nan2 = np.full((h, w, 2), np.nan)
    nan3 = np.full((h, w, 3), np.nan)
    ok2 = ok[..., None]
    return {
        "rigid_flow": np.where(ok2, rigid_uv - grid, nan2),
        "nonrigid_flow": np.where(ok2, total_uv - rigid_uv, nan2),
        "total_flow": np.where(ok2, total_uv - grid, nan2),
        "rigid_3d": np.where(ok2, rigid_camera - points, nan3),
        "nonrigid_3d": np.where(ok2, residual @ rotation.T, nan3),
        "total_3d": np.where(ok2, total_camera - points, nan3),
        "world_nonrigid": np.where(ok2, residual @ M0[:3, :3].T, nan3),
        "valid": ok,
        "target_uv": np.where(ok2, total_uv, nan2),
        "target_depth": np.where(ok, total_z, np.nan),
    }


def advect_labels(
    labels0,
    target_uv,
    target_depth,
    depth1,
    valid,
    *,
    relative_depth_tolerance=0.05,
    max_pixel_distance=1.5,
):
    """frame0 の anchor label を frame1 の画素へ forward splat する。

    labels0 は int (H, W) で -1 は invalid。target_uv (H, W, 2) は camera0 の各画素が
    frame1 で予測される連続座標 (u=col, v=row)、target_depth (H, W) はその camera-z。
    depth1 (H1, W1) は frame1 の実測 depth（source と異なる解像度でもよい）。

    各 source から max_pixel_distance 以内の整数画素を候補とし、
      - finite な target_uv / finite かつ正の target_depth
      - 候補画素が depth1 の bounds 内
      - depth1 が finite かつ正
      - 相対深度誤差 |target_depth - depth1| / max(target_depth, depth1) <= tol
        （境界は inclusive。浮動小数の丸め分のみ 1e-12 を許容する）
    を満たす候補だけを採用する。target 画素ごとに最近傍の target_depth（小さい z）を残し、
    z が同値の tie は pixel 再投影距離 → source flattened index の順で決める（決定的）。
    拒否された label が後から現れることはない。複数 target 画素が同じ label を持つのは明示的に許容する
    （同一 anchor ID の維持）。

    Returns:
        dict: labels (H1, W1) int64（未支持は -1）、source_index (H1, W1) int64（source の
              flattened index、未支持は -1）、valid (H1, W1) bool、stats（JSON-safe な dict）。
    labels0 は変更しない。出力 label は入力 label のコピーのみで、既定値で上書きしない。
    """
    labels0 = np.asarray(labels0)
    if labels0.ndim != 2:
        raise ValueError("labels0 must be 2-D")
    h, w = labels0.shape
    labels_i = labels0.astype(np.int64)
    target_uv = np.asarray(target_uv, dtype=np.float64)
    if target_uv.shape != (h, w, 2):
        raise ValueError("target_uv must have shape (H, W, 2) matching labels0")
    target_depth = np.asarray(target_depth, dtype=np.float64)
    if target_depth.shape != (h, w):
        raise ValueError("target_depth must have shape (H, W) matching labels0")
    mask = _valid_mask(valid, (h, w))
    depth1 = np.asarray(depth1, dtype=np.float64)
    if depth1.ndim != 2:
        raise ValueError("depth1 must be 2-D")
    h1, w1 = depth1.shape
    tolerance = float(relative_depth_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_depth_tolerance must be finite and >= 0")
    distance_limit = float(max_pixel_distance)
    if not np.isfinite(distance_limit) or distance_limit < 0.0:
        raise ValueError("max_pixel_distance must be finite and >= 0")

    source_ok = (
        mask
        & (labels_i >= 0)
        & np.all(np.isfinite(target_uv), axis=-1)
        & np.isfinite(target_depth)
        & (target_depth > 0.0)
    )
    n_positive = int(np.count_nonzero(source_ok))
    src_flat = np.flatnonzero(source_ok.ravel())
    source_u = target_uv[..., 0].ravel()[src_flat]
    source_v = target_uv[..., 1].ravel()[src_flat]
    source_z = target_depth.ravel()[src_flat]
    base_u = np.floor(source_u).astype(np.int64)
    base_v = np.floor(source_v).astype(np.int64)

    pair_src = []
    pair_tgt = []
    pair_z = []
    pair_dist = []
    n_out_of_bounds = 0
    n_too_far = 0
    n_bad_depth1 = 0
    n_mismatch = 0
    n_gate_accepted = 0
    rint = int(np.ceil(distance_limit))
    if src_flat.size:
        for dv in range(-rint, rint + 2):
            for du in range(-rint, rint + 2):
                rows = base_v + dv
                cols = base_u + du
                inside = (rows >= 0) & (rows < h1) & (cols >= 0) & (cols < w1)
                n_out_of_bounds += int(np.count_nonzero(~inside))
                candidates = np.hypot(source_u - cols, source_v - rows)
                far = candidates > distance_limit + _DIST_EPS
                n_too_far += int(np.count_nonzero(inside & far))
                near = inside & ~far
                if not np.any(near):
                    continue
                depth1_at = np.full(source_u.shape, np.nan)
                depth1_at[near] = depth1[rows[near], cols[near]]
                depth_ok = np.isfinite(depth1_at) & (depth1_at > 0.0)
                n_bad_depth1 += int(np.count_nonzero(near & ~depth_ok))
                with np.errstate(invalid="ignore"):
                    relative = np.abs(source_z - depth1_at) / np.maximum(source_z, depth1_at)
                matched = near & depth_ok & (relative <= tolerance + _RELATIVE_EPS)
                n_mismatch += int(
                    np.count_nonzero(near & depth_ok & ~(relative <= tolerance + _RELATIVE_EPS))
                )
                n_gate_accepted += int(np.count_nonzero(matched))
                if np.any(matched):
                    pair_src.append(src_flat[matched])
                    pair_tgt.append((rows[matched] * w1 + cols[matched]).astype(np.int64))
                    pair_z.append(source_z[matched])
                    pair_dist.append(candidates[matched])

    labels_out = np.full((h1, w1), -1, dtype=np.int64)
    index_out = np.full((h1, w1), -1, dtype=np.int64)
    valid_out = np.zeros((h1, w1), dtype=bool)
    if pair_src:
        flat_src = np.concatenate(pair_src)
        flat_tgt = np.concatenate(pair_tgt)
        predicted_z = np.concatenate(pair_z)
        reprojection = np.concatenate(pair_dist)
        # z 昇順（最近傍）→ 再投影距離 → source index。last key が最優先。
        order = np.lexsort((flat_src, reprojection, predicted_z))
        # z 昇順に並べた先頭出現 = 各 target の勝者（tie は上記順序で決定済み）。
        _, first = np.unique(flat_tgt[order], return_index=True)
        winners = order[first]
        labels_out.ravel()[flat_tgt[winners]] = labels_i.ravel()[flat_src[winners]]
        index_out.ravel()[flat_tgt[winners]] = flat_src[winners]
        valid_out.ravel()[flat_tgt[winners]] = True

    n_accepted_targets = int(np.count_nonzero(valid_out))
    rejected = {
        "out_of_bounds": n_out_of_bounds,
        "beyond_max_pixel_distance": n_too_far,
        "invalid_depth1": n_bad_depth1,
        "depth_mismatch": n_mismatch,
    }
    n_rejected = sum(rejected.values())
    stats = {
        "n_positive_candidates": n_positive,
        "n_candidate_pairs": n_rejected + n_gate_accepted,
        "n_accepted_targets": n_accepted_targets,
        "n_accepted_pairs": n_gate_accepted,
        "n_rejected_pairs": n_rejected,
        "rejected": rejected,
    }
    return {
        "labels": labels_out,
        "source_index": index_out,
        "valid": valid_out,
        "stats": stats,
    }


def render_points(points, colors, K, camera_to_world, shape, *, radius=1):
    """静的 world 点群を camera から z-buffer 描画する。

    points (N, 3) は world 座標の静的モデル、colors (N, 3) uint8。camera_to_world は
    rigid c2w で、points を camera 座標へ移して K で投影する。非有限・behind camera
    (z<=0)・画面外は reject する。camera-z の小さい点が勝ち、z が同値の tie は
    point index の小さい方が勝つ（決定的）。補間や新しい面の生成はしない。

    radius は 0..3 の整数で、投影中心の integer pixel から Euclidean disc
    (du^2 + dv^2 <= radius^2) の画素を覆う（radius=1 は 5 画素。正方形ではない）。

    Returns:
        dict: rgb (H, W, 3) uint8（void は 0）、depth (H, W) float32 camera-z（void は NaN）、
              valid (H, W) bool、point_index (H, W) int64（void は -1）。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    colors = np.asarray(colors)
    if colors.shape != points.shape or colors.dtype != np.uint8:
        raise ValueError("colors must be a uint8 array with shape (N, 3) matching points")
    K = _check_intrinsics(K, "K")
    camera_to_world = _check_rigid(camera_to_world, "camera_to_world")
    h, w = _validate_shape(shape)
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)):
        raise ValueError("radius must be an integer in [0, 3]")
    radius = int(radius)
    if radius < 0 or radius > _MAX_RENDER_RADIUS:
        raise ValueError("radius must be an integer in [0, 3]")

    depth_out = np.full((h, w), np.nan, dtype=np.float32)
    rgb_out = np.zeros((h, w, 3), dtype=np.uint8)
    valid_out = np.zeros((h, w), dtype=bool)
    index_out = np.full((h, w), -1, dtype=np.int64)
    n_points = points.shape[0]
    if n_points == 0:
        return {"rgb": rgb_out, "depth": depth_out, "valid": valid_out, "point_index": index_out}

    offsets = [
        (du, dv)
        for dv in range(-radius, radius + 1)
        for du in range(-radius, radius + 1)
        if du * du + dv * dv <= radius * radius
    ]
    world_to_camera = np.linalg.inv(camera_to_world)
    chunk = max(1, _RENDER_CHUNK_PAIRS // max(1, len(offsets)))
    best_pixel = np.empty(0, dtype=np.int64)
    best_z = np.empty(0, dtype=np.float64)
    best_index = np.empty(0, dtype=np.int64)
    for start in range(0, n_points, chunk):
        block = points[start : start + chunk]
        cam = block @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        z_cam = cam[:, 2]
        projected = np.all(np.isfinite(cam), axis=1) & (z_cam > 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (K[0, 0] * cam[:, 0] + K[0, 1] * cam[:, 1] + K[0, 2] * z_cam) / z_cam
            v = (K[1, 0] * cam[:, 0] + K[1, 1] * cam[:, 1] + K[1, 2] * z_cam) / z_cam
        projected &= np.isfinite(u) & np.isfinite(v)
        local = np.flatnonzero(projected)
        if local.size == 0:
            continue
        rows = np.rint(v[local]).astype(np.int64)
        cols = np.rint(u[local]).astype(np.int64)
        z_sel = z_cam[local]
        chunk_pixel = []
        chunk_z = []
        chunk_index = []
        for du, dv in offsets:
            rr = rows + dv
            cc = cols + du
            inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            if not np.any(inside):
                continue
            chunk_pixel.append(rr[inside] * w + cc[inside])
            chunk_z.append(z_sel[inside])
            chunk_index.append((start + local[inside]).astype(np.int64))
        if not chunk_pixel:
            continue
        all_pixel = np.concatenate([best_pixel, *chunk_pixel])
        all_z = np.concatenate([best_z, *chunk_z])
        all_index = np.concatenate([best_index, *chunk_index])
        # z 昇順（最近傍）→ point index。last key が最優先。
        order = np.lexsort((all_index, all_z))
        # z 昇順に並べた先頭出現 = 各画素の勝者。
        _, first = np.unique(all_pixel[order], return_index=True)
        winners = order[first]
        best_pixel = all_pixel[winners]
        best_z = all_z[winners]
        best_index = all_index[winners]

    rows = best_pixel // w
    cols = best_pixel % w
    depth_out[rows, cols] = best_z.astype(np.float32)
    rgb_out[rows, cols] = colors[best_index]
    valid_out[rows, cols] = True
    index_out[rows, cols] = best_index
    return {"rgb": rgb_out, "depth": depth_out, "valid": valid_out, "point_index": index_out}
