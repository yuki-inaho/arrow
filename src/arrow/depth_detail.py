"""深度詳細化の共有数値プリミティブ。

固定契約: temp/workdoc_Oct16-2026_depth_workflow.md「固定する共有コード契約」。
numpy / Pillow だけで import できる（torch・GPU 依存を持ち込まない）。

共通の前提:
- depth は正の有限値だけを有効とみなす（メートルはモデル推定値であり実寸保証ではない）。
- invalid（valid=False / 非有限 / 非正）画素の出力は NaN。穴を勝手に valid 化しない。
- reproject_depth の画素中心は integer pixel centre。画素 (row i, col j) の中心を
  連続座標 (u=j, v=i) とみなす。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fit_scale",
    "regularize_scales",
    "gray_edges",
    "blend_depth",
    "reproject_depth",
    "temporal_fuse",
    "superresolve_scene_flow",
]

# MAD → 標準偏差相当への換算係数（正規分布仮定）
_MAD_SCALE = 1.4826
# log 領域での最小許容幅（丸め誤差より十分大きく、粗外れより十分小さい）
_LOG_TOLERANCE_FLOOR = 1e-6
# scene-flow SR のガイド重み幅（log 深度の相対差、[0,1] 色差）
_SR_DEPTH_SIGMA = 0.05
_SR_COLOR_SIGMA = 0.1


def fit_scale(source, target, mask=None, min_points=32):
    """target ≈ scale * source となる正の頑健スケールを推定する。

    Args:
        source: 推定元の非負値配列（log 推定のため正値のみ使用）。
        target: 推定先の配列。source と同形状。
        mask: True の画素のみを対応として使う任意の bool 配列。
        min_points: 有効対応の最小数。不足時は ValueError。

    Returns:
        dict: {"scale": 正の有限 float, "inliers": 採用点数,
               "median_abs_log_residual": inlier の log 残差中央値}
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")
    if mask is None:
        allowed = np.ones(source.shape, dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape != source.shape:
            raise ValueError("mask must have the same shape as source")
        allowed = mask.astype(bool)
    if int(min_points) < 1:
        raise ValueError("min_points must be >= 1")

    usable = (
        allowed
        & np.isfinite(source)
        & np.isfinite(target)
        & (source > 0.0)
        & (target > 0.0)
    )
    n_usable = int(np.count_nonzero(usable))
    if n_usable < int(min_points):
        raise ValueError(
            f"fit_scale: insufficient usable correspondences "
            f"{n_usable} < min_points={int(min_points)}"
        )

    log_ratio = np.log(target[usable]) - np.log(source[usable])
    center = float(np.median(log_ratio))
    mad = float(np.median(np.abs(log_ratio - center)))
    tolerance = 3.0 * _MAD_SCALE * mad + _LOG_TOLERANCE_FLOOR
    inlier_flags = np.abs(log_ratio - center) <= tolerance
    inlier_log = log_ratio[inlier_flags]

    log_scale = float(np.median(inlier_log))
    scale = float(np.exp(log_scale))
    residual = float(np.median(np.abs(inlier_log - log_scale)))
    return {
        "scale": scale,
        "inliers": int(inlier_log.size),
        "median_abs_log_residual": residual,
    }


def _solve_tridiagonal(sub, diag, sup, rhs):
    """三重対角線形系を Thomas アルゴリズムで解く（破壊的でない）。"""
    n = int(diag.size)
    upper = sup.astype(np.float64, copy=True)
    middle = diag.astype(np.float64, copy=True)
    right = rhs.astype(np.float64, copy=True)
    for i in range(1, n):
        factor = sub[i - 1] / middle[i - 1]
        middle[i] -= factor * upper[i - 1]
        right[i] -= factor * right[i - 1]
    x = np.empty(n, dtype=np.float64)
    x[-1] = right[-1] / middle[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (right[i] - upper[i] * x[i + 1]) / middle[i]
    return x


def regularize_scales(scales, weights, strength=2.0):
    """log 尺度の時系列平滑化。正値と個数を保持する。

    Σ_i w_i (y_i - log s_i)^2 + strength Σ_i (y_i - y_{i+1})^2
    を最小化する y を厳密に解き、exp(y) を返す。strength=0 は恒等。

    Args:
        scales: 正の有限スケール列（1-D）。
        weights: 各フレームの信頼度（非負、1つ以上正）。
        strength: 平滑化強度（0 以上）。大きいほど前後フレームへ寄る。

    Returns:
        ndarray: scales と同じ形状の正値配列。
    """
    scales = np.asarray(scales, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scales.ndim != 1:
        raise ValueError("scales must be 1-D")
    if weights.shape != scales.shape:
        raise ValueError("weights must have the same shape as scales")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("scales must be finite and positive")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and non-negative")

    n = int(scales.size)
    if n <= 1 or strength == 0.0:
        return scales.copy()
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("weights must contain at least one positive value")

    x = np.log(scales)
    # 正規方程式: (diag(w) + strength * L_chain) y = w * x
    sub = np.full(n - 1, -strength, dtype=np.float64)
    sup = np.full(n - 1, -strength, dtype=np.float64)
    diag = weights.copy()
    diag[0] += strength
    diag[-1] += strength
    diag[1:-1] += 2.0 * strength
    y = _solve_tridiagonal(sub, diag, sup, weights * x)
    result = np.exp(y)
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("regularize_scales: non-positive smoothing result")
    return result


def gray_edges(rgb):
    """RGB 画像から [0,1] の境界（勾配強度）マップを作る。

    Args:
        rgb: uint8 (0..255) もしくは float (0..1) の (H, W, 3) 配列。
             dtype と値域は厳密に検証し、範囲外・非有限は ValueError。

    Returns:
        ndarray[H, W]: float64 の勾配強度。最大値で正規化した値域 [0,1]。
        一様画像はすべて 0。
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h < 1 or w < 1:
        raise ValueError("rgb must be non-empty")

    if rgb.dtype == np.uint8:
        gray = rgb.astype(np.float64)
    elif np.issubdtype(rgb.dtype, np.floating):
        values = rgb.astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("float rgb must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("float rgb must be in [0, 1]")
        gray = values * 255.0
    else:
        raise ValueError(f"unsupported rgb dtype: {rgb.dtype}")

    gray = 0.299 * gray[..., 0] + 0.587 * gray[..., 1] + 0.114 * gray[..., 2]

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    if w >= 2:
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
        gx[:, 0] = gray[:, 1] - gray[:, 0]
        gx[:, -1] = gray[:, -1] - gray[:, -2]
    if h >= 2:
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
        gy[0, :] = gray[1, :] - gray[0, :]
        gy[-1, :] = gray[-1, :] - gray[-2, :]

    magnitude = np.hypot(gx, gy)
    peak = float(np.max(magnitude)) if magnitude.size else 0.0
    if peak > 0.0:
        magnitude = magnitude / peak
    return magnitude


def blend_depth(
    anchor,
    detail,
    valid=None,
    detail_weight=0.7,
    trust_abs=0.05,
    trust_rel=0.05,
):
    """trust 領域内だけ detail を blend した深度を返す。

    out = (1-w)*anchor + w*detail （|detail-anchor| <= max(trust_abs, trust_rel*|anchor|) の画素のみ）。
    detail が trust 外・invalid の画素は anchor を保持する。
    valid=False / anchor 非有限・非正の画素は NaN（穴を勝手に valid 化しない）。

    Args:
        anchor: 基準深度（正値、メートル）。
        detail: 詳細深度。anchor と同形状。
        valid: 任意の bool 共有マスク。
        detail_weight: blend 重み w ∈ [0, 1]。
        trust_abs: 絶対 trust 幅（メートル）。0 以上。
        trust_rel: 相対 trust 幅（|anchor| 比）。0 以上。

    Returns:
        ndarray: anchor と同形状の float64。invalid は NaN。
    """
    anchor = np.asarray(anchor, dtype=np.float64)
    detail = np.asarray(detail, dtype=np.float64)
    if anchor.shape != detail.shape:
        raise ValueError("anchor and detail must have the same shape")
    detail_weight = float(detail_weight)
    trust_abs = float(trust_abs)
    trust_rel = float(trust_rel)
    if not np.isfinite(detail_weight) or not 0.0 <= detail_weight <= 1.0:
        raise ValueError("detail_weight must be within [0, 1]")
    if not np.isfinite(trust_abs) or trust_abs < 0.0:
        raise ValueError("trust_abs must be finite and non-negative")
    if not np.isfinite(trust_rel) or trust_rel < 0.0:
        raise ValueError("trust_rel must be finite and non-negative")

    if valid is None:
        valid_mask = np.ones(anchor.shape, dtype=bool)
    else:
        valid = np.asarray(valid)
        if valid.shape != anchor.shape:
            raise ValueError("valid must have the same shape as anchor")
        valid_mask = valid.astype(bool)

    anchor_ok = valid_mask & np.isfinite(anchor) & (anchor > 0.0)
    detail_ok = np.isfinite(detail) & (detail > 0.0)
    trust = np.maximum(trust_abs, trust_rel * np.abs(anchor))
    within = anchor_ok & detail_ok & (np.abs(detail - anchor) <= trust)

    out = np.full(anchor.shape, np.nan, dtype=np.float64)
    keep = anchor_ok & ~within
    out[keep] = anchor[keep]
    out[within] = (1.0 - detail_weight) * anchor[within] + detail_weight * detail[within]
    return out


def reproject_depth(depth, rgb, K_source, K_target, target_from_source, output_shape):
    """source 深度/RGB を target カメラへ z-buffer 付きで再投影する。

    integer pixel centre 規約（画素 (i, j) の中心 = 連続座標 (u=j, v=i)）。
    depth<=0 や非有限画素は invalid として扱う。target 画素に複数点が落ちる場合は
    最小の target 深度（最近傍）を残し、他の点は occlusion として捨てる。

    Args:
        depth: source 深度 (H, W)（メートル、正値のみ有効）。
        rgb: source RGB (H, W, 3)。
        K_source / K_target: 3x3 intrinsics（fx, fy > 0）。
        target_from_source: 4x4 の target←source 変換（最終行 [0,0,0,1]）。
        output_shape: target の (H, W)。正の整数。

    Returns:
        dict: {"depth": (Ht, Wt) float64（invalid は NaN）,
               "rgb": (Ht, Wt, 3) 入力 dtype（invalid は 0）,
               "valid": (Ht, Wt) bool}
    """
    depth = np.asarray(depth, dtype=np.float64)
    rgb = np.asarray(rgb)
    K_source = np.asarray(K_source, dtype=np.float64)
    K_target = np.asarray(K_target, dtype=np.float64)
    transform = np.asarray(target_from_source, dtype=np.float64)

    if depth.ndim != 2:
        raise ValueError("depth must be 2-D")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or tuple(rgb.shape[:2]) != tuple(depth.shape):
        raise ValueError("rgb must have shape (H, W, 3) matching depth")
    for name, K in (("K_source", K_source), ("K_target", K_target)):
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"{name} must be a finite 3x3 matrix")
        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise ValueError(f"{name} must have positive focal lengths")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("target_from_source must be a finite 4x4 matrix")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("target_from_source last row must be [0, 0, 0, 1]")

    output_shape = tuple(int(v) for v in output_shape)
    if len(output_shape) != 2 or output_shape[0] < 1 or output_shape[1] < 1:
        raise ValueError("output_shape must be two positive integers")
    out_h, out_w = output_shape

    rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0.0))
    z_src = depth[rows, cols]
    x = (cols - K_source[0, 2]) * z_src / K_source[0, 0]
    y = (rows - K_source[1, 2]) * z_src / K_source[1, 1]
    points = np.stack((x, y, z_src, np.ones_like(z_src)), axis=0)
    moved = transform @ points
    z_t = moved[2]
    u = K_target[0, 0] * moved[0] / z_t + K_target[0, 2]
    v = K_target[1, 1] * moved[1] / z_t + K_target[1, 2]

    front = (z_t > 0.0) & np.isfinite(u) & np.isfinite(v) & (np.abs(u) < 1e15) & (np.abs(v) < 1e15)
    idx = np.nonzero(front)[0]
    col_i = np.rint(u[idx]).astype(np.int64)
    row_i = np.rint(v[idx]).astype(np.int64)
    inside = (col_i >= 0) & (col_i < out_w) & (row_i >= 0) & (row_i < out_h)
    sel = idx[inside]
    row_i = row_i[inside]
    col_i = col_i[inside]
    z_sel = z_t[sel]

    depth_out = np.full((out_h, out_w), np.nan, dtype=np.float64)
    rgb_out = np.zeros((out_h, out_w, 3), dtype=rgb.dtype)
    valid_out = np.zeros((out_h, out_w), dtype=bool)
    if sel.size:
        order = np.argsort(z_sel, kind="stable")
        rows_o = row_i[order]
        cols_o = col_i[order]
        flat = rows_o * out_w + cols_o
        # z 昇順に並べた先頭出現 = 最近傍（z-buffer）
        _, first = np.unique(flat, return_index=True)
        win_rows = rows_o[first]
        win_cols = cols_o[first]
        winners = sel[order][first]
        depth_out[win_rows, win_cols] = z_t[winners]
        rgb_out[win_rows, win_cols] = rgb[rows[winners], cols[winners]]
        valid_out[win_rows, win_cols] = True

    return {"depth": depth_out, "rgb": rgb_out, "valid": valid_out}


def temporal_fuse(
    current,
    candidate,
    support,
    edges,
    weight=0.25,
    trust_abs=0.05,
    trust_rel=0.05,
):
    """境界/支持/trust ゲート付きの保守的な時間融合。

    有効画素では out = current + w_eff * (candidate - current)、
    w_eff = weight * (1 - edges)。support=False、edges=1、trust 外（
    |candidate-current| > max(trust_abs, trust_rel*|current|)）、
    candidate invalid の画素は current を保持する。
    current が invalid の画素は NaN（穴は候補で埋めない）。

    Args:
        current: 現フレーム深度。
        candidate: 時間対応付け済み候補深度（同形状）。
        support: 対応の支持マスク（bool 同形状）。
        edges: [0,1] の境界マップ（1 が境界）。同形状。
        weight: 最大融合重み ∈ [0, 1]。
        trust_abs / trust_rel: blend_depth と同じ trust 幅。

    Returns:
        ndarray: current と同形状の float64。invalid は NaN。
    """
    current = np.asarray(current, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    support = np.asarray(support)
    edges = np.asarray(edges, dtype=np.float64)
    if current.shape != candidate.shape:
        raise ValueError("current and candidate must have the same shape")
    if support.shape != current.shape:
        raise ValueError("support must have the same shape as current")
    if edges.shape != current.shape:
        raise ValueError("edges must have the same shape as current")
    if not np.all(np.isfinite(edges)):
        raise ValueError("edges must be finite")
    if np.any(edges < 0.0) or np.any(edges > 1.0):
        raise ValueError("edges must be within [0, 1]")
    weight = float(weight)
    trust_abs = float(trust_abs)
    trust_rel = float(trust_rel)
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be within [0, 1]")
    if not np.isfinite(trust_abs) or trust_abs < 0.0:
        raise ValueError("trust_abs must be finite and non-negative")
    if not np.isfinite(trust_rel) or trust_rel < 0.0:
        raise ValueError("trust_rel must be finite and non-negative")

    support_mask = support.astype(bool)
    cur_ok = np.isfinite(current) & (current > 0.0)
    cand_ok = np.isfinite(candidate) & (candidate > 0.0)
    trust = np.maximum(trust_abs, trust_rel * np.abs(current))
    w_eff = weight * (1.0 - np.clip(edges, 0.0, 1.0))
    fuse = cur_ok & support_mask & cand_ok & (np.abs(candidate - current) <= trust)

    out = np.full(current.shape, np.nan, dtype=np.float64)
    out[cur_ok] = current[cur_ok]
    out[fuse] = current[fuse] + w_eff[fuse] * (candidate[fuse] - current[fuse])
    return out


def superresolve_scene_flow(low_flow, high_depth, high_rgb, low_valid=None):
    """低解像度 3D scene flow を高解像度 depth/RGB ガイドで元解像度へ上げる。

    depth/RGB guided な joint-bilateral 補間（2x2 bilinear 候補への重み付け）。
    運動境界を跨ぐ混色（leakage）を抑え、境界では各辺側のサンプルを選ぶ。
    3D 変位は物理量のまま扱い、解像度比による倍率変更はしない
    （2D pixel flow の SR とは区別する）。

    深度支持は必須: target 画素の high_depth が正有限で、かつ重みを持つ
    candidate の深度も正有限でなければ out は unsupported（NaN / valid=False）。
    weight が完全 underflow した場合も暗黙の bilinear fallback は行わず
    unsupported として NaN/false を返す。valid = low support AND depth support。

    Args:
        low_flow: (Hl, Wl, 3) の 3D 変位。NaN/Inf のサンプルは無効扱い。
        high_depth: (H, W) の高解像度深度ガイド。非有限/非正は支持なし。
        high_rgb: (H, W, 3)。uint8 (0..255) か float (0..1)。厳密検証。
        low_valid: 任意の (Hl, Wl) bool。False のサンプルは無効扱い。

    Returns:
        dict: {"flow": (H, W, 3) float64（unsupported は NaN）,
               "valid": (H, W) bool（low と深度の両支持がある画素のみ True）}
    """
    low_flow = np.asarray(low_flow, dtype=np.float64)
    high_depth = np.asarray(high_depth, dtype=np.float64)
    high_rgb = np.asarray(high_rgb)
    if low_flow.ndim != 3 or low_flow.shape[2] != 3:
        raise ValueError("low_flow must have shape (Hl, Wl, 3)")
    if high_depth.ndim != 2:
        raise ValueError("high_depth must be 2-D")
    h, w = int(high_depth.shape[0]), int(high_depth.shape[1])
    if h < 1 or w < 1:
        raise ValueError("high_depth must be non-empty")
    if high_rgb.ndim != 3 or high_rgb.shape[2] != 3 or tuple(high_rgb.shape[:2]) != (h, w):
        raise ValueError("high_rgb must have shape (H, W, 3) matching high_depth")
    if high_rgb.dtype == np.uint8:
        rgb = high_rgb.astype(np.float64) / 255.0
    elif np.issubdtype(high_rgb.dtype, np.floating):
        rgb = high_rgb.astype(np.float64)
        if not np.all(np.isfinite(rgb)):
            raise ValueError("float high_rgb must be finite")
        if np.any(rgb < 0.0) or np.any(rgb > 1.0):
            raise ValueError("float high_rgb must be in [0, 1]")
    else:
        raise ValueError(f"unsupported high_rgb dtype: {high_rgb.dtype}")
    hl, wl = int(low_flow.shape[0]), int(low_flow.shape[1])
    if hl < 1 or wl < 1:
        raise ValueError("low_flow must be non-empty")
    if low_valid is None:
        valid_low = np.ones((hl, wl), dtype=bool)
    else:
        low_valid = np.asarray(low_valid)
        if low_valid.shape != (hl, wl):
            raise ValueError("low_valid must have shape (Hl, Wl)")
        valid_low = low_valid.astype(bool)
    usable_low = valid_low & np.all(np.isfinite(low_flow), axis=2)

    rows = np.arange(h, dtype=np.float64)[:, None]
    cols = np.arange(w, dtype=np.float64)[None, :]
    v_low = np.clip((rows + 0.5) * hl / h - 0.5, 0.0, hl - 1.0)
    u_low = np.clip((cols + 0.5) * wl / w - 0.5, 0.0, wl - 1.0)
    r0 = np.floor(v_low).astype(np.int64)
    c0 = np.floor(u_low).astype(np.int64)
    fv = v_low - r0
    fu = u_low - c0
    r1 = np.minimum(r0 + 1, hl - 1)
    c1 = np.minimum(c0 + 1, wl - 1)

    cand_r = np.stack(np.broadcast_arrays(r0, r0, r1, r1), axis=-1)
    cand_c = np.stack(np.broadcast_arrays(c0, c1, c0, c1), axis=-1)
    cand_w = np.stack(
        np.broadcast_arrays(
            (1.0 - fv) * (1.0 - fu),
            (1.0 - fv) * fu,
            fv * (1.0 - fu),
            fv * fu,
        ),
        axis=-1,
    )

    flow_c = low_flow[cand_r, cand_c]
    usable_c = usable_low[cand_r, cand_c]
    flow_c = np.where(usable_c[..., None], flow_c, 0.0)

    # 低解像度サンプルに対応する高解像度ガイド位置
    guide_r = np.clip(np.rint((np.arange(hl) + 0.5) * h / hl - 0.5), 0, h - 1).astype(np.int64)
    guide_c = np.clip(np.rint((np.arange(wl) + 0.5) * w / wl - 0.5), 0, w - 1).astype(np.int64)
    cand_gr = guide_r[cand_r]
    cand_gc = guide_c[cand_c]

    target_ok = np.isfinite(high_depth) & (high_depth > 0.0)
    depth_c = high_depth[cand_gr, cand_gc]
    cand_ok = np.isfinite(depth_c) & (depth_c > 0.0)
    supported_c = cand_ok & target_ok[..., None]
    depth_weight = np.zeros_like(depth_c)  # 深度非支持 candidate は重み 0（除外）
    if np.any(supported_c):
        depth_safe = np.where(cand_ok, depth_c, 1.0)
        target_safe = np.where(target_ok, high_depth, 1.0)
        log_ratio = np.log(depth_safe) - np.log(target_safe)[..., None]
        depth_weight = np.where(
            supported_c,
            np.exp(-(log_ratio * log_ratio) / (2.0 * _SR_DEPTH_SIGMA**2)),
            0.0,
        )

    rgb_c = rgb[cand_gr, cand_gc]
    color_diff = np.max(np.abs(rgb_c - rgb[..., None, :]), axis=-1)
    color_weight = np.exp(-(color_diff * color_diff) / (2.0 * _SR_COLOR_SIGMA**2))

    # 暗黙 bilinear fallback は行わない: 深度支持 candidate の重み和が 0
    # （target/候補の深度非支持、low 無効、重み完全 underflow）なら
    # unsupported = NaN / valid False を明示する。
    weight = cand_w * depth_weight * color_weight * usable_c
    weight_sum = weight.sum(axis=-1)

    flow_out = np.full((h, w, 3), np.nan, dtype=np.float64)
    valid_out = target_ok & (weight_sum > 0.0)
    if np.any(valid_out):
        weighted = (weight[..., None] * flow_c).sum(axis=2)
        flow_out[valid_out] = weighted[valid_out] / weight_sum[valid_out][:, None]
    return {"flow": flow_out, "valid": valid_out}
