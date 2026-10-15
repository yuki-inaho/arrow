#!/usr/bin/env python3
"""重複区間からクリップ間の座標系接続（相似変換）を推定し、世界座標への変換表を保存する。

背景: ARROW の各クリップは先頭フレームのカメラを基準にした独立の参照座標系を持ち、
3D 点群・カメラ軌跡・深度はクリップ内の相対値になる。クリップを結合すると
カメラ軌跡がクリップごとに原点付近へリセットされ、深度のスケールも揃わない。

本ツールは process_sequence.py --clip-overlap で設けた重複フレームの**再構成点群**を
使って、各クリップ → クリップ 0 基準の世界座標への相似変換 (s, R, t) を連鎖推定する:

- 重複フレームの再構成は同じ画像格子から得られるため、「同じ格子点 = 同じ物理点」の
  密な対応が使える。これを Umeyama 法（回転 + 並進 + スケールの最小二乗）で解く。
- カメラ変位が小さい定点シーンでも安定する（カメラ姿勢だけでは退化する）。
- 深度: 各クリップの深度範囲（5–95 パーセンタイル）をスケール補正して統合した
  世界共通表示範囲 depth_range_world も計算する。

出力: alignments.json（render_multipanel.py / merge_clip_outputs.py が読む）
"""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from render_dashboard import default_cache_dir, ensure_cache, utc_now  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


CAMERA_WEIGHT = 400  # カメラ位置 1 点あたりの重み（点群 ~8800 点に対して ~15% になる複製数）
STATIC_TRACK_THRESHOLD = 0.02  # 静止トラック点の判定（重複区間内の 3D 位置変動。シーン ~16 に対して）
DEPTH_WORLD_PERCENTILES = (10, 90)  # 世界共通の深度表示範囲（中央を重視して色のコントラストを確保）


def load_scene(npz_path: Path, pick: int) -> dict[str, Any]:
    """キャッシュ/推論 NPZ から、カメラ姿勢と再構成点群グリッドを読み込む。"""
    cache_dir = default_cache_dir(npz_path)
    loaded = ensure_cache(npz_path, cache_dir, pick, None, rebuild=False)
    result: Any = loaded[0]

    from arrow.tasks.pose import estimate_poses

    pose_entry = next(g for g in result.metadata["groups"] if g["task"] == "pose")
    pose_start, pose_count = pose_entry["start"], pose_entry["count"]
    pose_predictions = type(result.predictions)(
        **{
            f.name: getattr(result.predictions, f.name)[:, pose_start : pose_start + pose_count]
            for f in fields(type(result.predictions))
        }
    )
    xy = result.queries.xy[:, pose_start : pose_start + pose_count]
    frames = len(result.metadata["frame_info"])
    poses = estimate_poses(
        pose_predictions, xy, num_frames=frames,
        reference_frame=pose_entry["reference_frame"], **pose_entry["pose"],
    )
    cameras = {
        "camera_to_reference": poses.camera_to_reference[0].numpy().astype(np.float32),
        "valid": poses.valid[0].numpy().astype(bool),
    }

    reconstruct_entry = next(g for g in result.metadata["groups"] if g["task"] == "reconstruct")
    r_start, r_count = reconstruct_entry["start"], reconstruct_entry["count"]
    points = result.predictions.points[0, r_start : r_start + r_count].numpy()
    per_frame = r_count // frames
    # 再構成はフレームごとに同じ画像格子から得られる（「同じ格子点 = 同じ物理点」）
    grid = points.reshape(frames, per_frame, 3)

    # トラック点（全追跡対象）: target 主導の並び（フレームごとに seeds 個）。
    # 同じ seed は追跡済みの同一物理点なので、クリップ間で対応が取れる。
    track_entry = next(g for g in result.metadata["groups"] if g["task"] == "track")
    t_start, t_count = track_entry["start"], track_entry["count"]
    seeds = t_count // frames
    track_points = result.predictions.points[0, t_start : t_start + t_count].numpy()
    tracks_grid = track_points.reshape(frames, seeds, 3)

    # 深度（参照カメラ座標の z）: 世界共通の表示範囲の計算に使う
    depth_entry = next(g for g in result.metadata["groups"] if g["task"] == "depth")
    d_start, d_count = depth_entry["start"], depth_entry["count"]
    depth_points = result.predictions.points[0, d_start : d_start + d_count].numpy()
    depth_z = depth_points.reshape(frames, d_count // frames, 3)[..., 2]

    return {"cache_dir": cache_dir, "cameras": cameras, "points": grid,
            "tracks": tracks_grid, "depth": depth_z}


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """相似変換 target ≈ scale * rotation @ source + translation を最小二乗で推定する。"""
    mean_s = source.mean(axis=0)
    mean_t = target.mean(axis=0)
    src = source - mean_s
    dst = target - mean_t
    covariance = (dst.T @ src) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = float((src**2).sum()) / len(source)
    scale = float(np.trace(np.diag(singular) @ sign) / variance) if variance > 1e-12 else 1.0
    translation = mean_t - scale * (rotation @ mean_s)
    return rotation, translation, scale


def ransac_umeyama(source: np.ndarray, target: np.ndarray, *, iterations: int = 300,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, float, int]:
    """RANSAC + coarse-to-fine（ICP 的）のロバスト反復で相似変換を推定する。

    1. RANSAC: 広めの閾値で外れ値の少ない初期モデルを選ぶ（動点・誤対応に強い）
    2. coarse-to-fine: 閾値を段階的に絞りながら再推定する（収束を安定させる）
    戻り値: (rotation, translation, scale, インライア数)
    """
    rng = np.random.default_rng(seed)
    count = len(source)
    if count < 12:
        rotation, translation, scale = umeyama(source, target)
        return rotation, translation, scale, count

    spread = float(np.median(np.linalg.norm(target - target.mean(axis=0), axis=1)))
    best_inliers = None
    for _ in range(iterations):
        indices = rng.choice(count, 6, replace=False)
        try:
            rotation, translation, scale = umeyama(source[indices], target[indices])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(
            target - (scale * (source @ rotation.T) + translation), axis=1)
        inliers = residual < max(0.08 * spread, 1e-6)
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers
    if best_inliers is None:
        rotation, translation, scale = umeyama(source, target)
        return rotation, translation, scale, count

    inliers = best_inliers
    rotation, translation, scale = umeyama(source[inliers], target[inliers])
    for ratio in (0.05, 0.03, 0.018):
        residual = np.linalg.norm(
            target - (scale * (source @ rotation.T) + translation), axis=1)
        new_inliers = residual < max(ratio * spread, 1e-6)
        if int(new_inliers.sum()) < 12:
            break
        rotation, translation, scale = umeyama(source[new_inliers], target[new_inliers])
        inliers = new_inliers
    return rotation, translation, scale, int(inliers.sum())


def estimate_chain(clips: list[dict]) -> list[dict]:
    """クリップごとの世界座標への相似変換 (scale, rotation, translation) を連鎖推定する。"""
    transforms: list[dict] = [
        {"scale": 1.0, "rotation": np.eye(3), "translation": np.zeros(3)}
    ]
    clips[0]["note"] = "基準クリップ（恒等変換）"
    for index in range(1, len(clips)):
        prev, cur = clips[index - 1], clips[index]
        previous = transforms[-1]
        overlap = list(range(cur["start"], min(prev["end"], cur["end"])))

        sources: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for frame in overlap:
            # 再構成点の対応（同じ画像格子 = 同じ物理点）
            points_prev = prev["points"][frame - prev["start"]]
            points_cur = cur["points"][frame - cur["start"]]
            if points_prev.shape != points_cur.shape:
                continue
            keep = np.isfinite(points_prev).all(-1) & np.isfinite(points_cur).all(-1)
            if int(keep.sum()) < 8:
                continue
            world_points = (
                previous["scale"] * (points_prev[keep] @ previous["rotation"].T)
                + previous["translation"]
            )
            sources.append(points_cur[keep])
            targets.append(world_points)

        if not sources:
            transforms.append(
                {"scale": previous["scale"], "rotation": previous["rotation"].copy(),
                 "translation": previous["translation"].copy()}
            )
            cur["note"] = "重複区間の対応点が不足（前クリップの変換をコピー）"
            continue

        source = np.concatenate(sources)
        target = np.concatenate(targets)

        # カメラ位置の対応（カメラ軌跡の連続性に直結するため、複製で重み付けして併用）
        camera_source = []
        camera_target = []
        for frame in overlap:
            matrix_prev = prev["cameras"]["camera_to_reference"][frame - prev["start"]]
            matrix_cur = cur["cameras"]["camera_to_reference"][frame - cur["start"]]
            camera_target.append(
                previous["scale"] * (previous["rotation"] @ matrix_prev[:3, 3])
                + previous["translation"]
            )
            camera_source.append(matrix_cur[:3, 3])
        camera_source = np.asarray(camera_source)
        camera_target = np.asarray(camera_target)

        # RANSAC（外れ値に強い初期化）→ パーセンタイル反復（カメラ位置を重み付けした精緻化）
        combined_source = np.concatenate(
            [source, np.repeat(camera_source, CAMERA_WEIGHT, axis=0)]
        )
        combined_target = np.concatenate(
            [target, np.repeat(camera_target, CAMERA_WEIGHT, axis=0)]
        )
        rotation, translation, scale, inlier_count = ransac_umeyama(
            combined_source, combined_target
        )
        for _ in range(3):
            residual = np.linalg.norm(
                target - (scale * (source @ rotation.T) + translation), axis=1
            )
            keep = residual <= np.percentile(residual, 85)
            if int(keep.sum()) < 100:
                break
            source, target = source[keep], target[keep]
            combined_source = np.concatenate(
                [source, np.repeat(camera_source, CAMERA_WEIGHT, axis=0)]
            )
            combined_target = np.concatenate(
                [target, np.repeat(camera_target, CAMERA_WEIGHT, axis=0)]
            )
            rotation, translation, scale = umeyama(combined_source, combined_target)
            inlier_count = int(keep.sum())

        residual = target - (scale * (source @ rotation.T) + translation)
        rms = float(np.sqrt((residual**2).sum(axis=1).mean()))
        # 検証: 変換後のカメラ位置と前クリップ（世界）のカメラ位置の一致度
        world_cur_cameras = scale * (camera_source @ rotation.T) + translation
        camera_errors = np.linalg.norm(world_cur_cameras - camera_target, axis=1)
        camera_rms = float(np.sqrt((camera_errors**2).mean()))
        note = (f"scale {scale:.4f}（対応点 {len(source)} 点、RANSAC インライア {inlier_count}、"
                f"重複 {len(overlap)} フレーム、点群残差 {rms:.4f}、カメラ位置残差 {camera_rms:.4f}）")
        transforms.append({"scale": scale, "rotation": rotation, "translation": translation})
        cur["note"] = note
    return transforms


def world_depth_range(clips: list[dict], transforms: list[dict]) -> list[float] | None:
    """全クリップの深度値をスケール補正して統合した世界共通の表示範囲。

    各クリップの範囲を機械的に統合すると端（外れ値）で範囲が広がり色の
    コントラストが弱まるため、全深度値の分布からパーセンタイルを取る。
    """
    values = []
    for clip, transform in zip(clips, transforms):
        z = np.asarray(clip["depth"])
        finite = z[np.isfinite(z)]
        if len(finite):
            values.append(finite * transform["scale"])
    if not values:
        return None
    merged = np.concatenate(values)
    low, high = np.percentile(merged, list(DEPTH_WORLD_PERCENTILES))
    return [float(low), float(high)]


def world_extents(clips: list[dict], transforms: list[dict]) -> dict | None:
    """世界座標での範囲を「点群」「カメラ位置」「統合」に分けて返す（パネル別視点用）。"""

    def to_world(samples: np.ndarray, transform: dict) -> np.ndarray | None:
        finite = samples[np.isfinite(samples).all(-1)]
        if not len(finite):
            return None
        return (transform["scale"] * (finite @ transform["rotation"].T)
                + transform["translation"])

    point_lows, point_highs = [], []
    camera_lows, camera_highs = [], []
    for clip, transform in zip(clips, transforms):
        world_points = to_world(clip["points"].reshape(-1, 3), transform)
        if world_points is not None:
            point_lows.append(world_points.min(axis=0))
            point_highs.append(world_points.max(axis=0))
        world_cameras = to_world(clip["cameras"]["camera_to_reference"][:, :3, 3], transform)
        if world_cameras is not None:
            camera_lows.append(world_cameras.min(axis=0))
            camera_highs.append(world_cameras.max(axis=0))

    def merge(lows: list, highs: list) -> dict | None:
        if not lows:
            return None
        low = np.min(np.stack(lows), axis=0)
        high = np.max(np.stack(highs), axis=0)
        return {"low": low.round(4).tolist(), "high": high.round(4).tolist()}

    points = merge(point_lows, point_highs)
    cameras = merge(camera_lows, camera_highs)
    overall = merge(point_lows + camera_lows, point_highs + camera_highs)
    return {"points": points or overall, "cameras": cameras or points, "overall": overall}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="重複区間からクリップ間の座標系接続（相似変換）を推定する")
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="クリップ NPZ と manifest.json の場所")
    parser.add_argument("--output", type=Path,
                        help="alignments.json の出力先（既定: clips-dir/alignments.json）")
    parser.add_argument("--pick", type=int, default=56,
                        help="軌跡の本数（キャッシュと一致させる。既定 56）")
    args = parser.parse_args(argv)
    args.clips_dir = args.clips_dir.expanduser().resolve()
    args.output = (args.output or args.clips_dir / "alignments.json").expanduser().resolve()
    if not args.clips_dir.is_dir():
        parser.error(f"クリップディレクトリが見つかりません: {args.clips_dir}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.clips_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest.json が見つかりません: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    overlap = int(manifest.get("signature", {}).get("clip_overlap", 0))
    if overlap < 1:
        raise SystemExit("manifest の clip_overlap が 0 です。--clip-overlap 付きで処理したクリップが必要です")

    entries = sorted(
        (entry for entry in manifest["clips"] if entry.get("status") == "done"),
        key=lambda entry: entry["start"],
    )
    clips = []
    for entry in entries:
        npz_path = args.clips_dir / entry["name"]
        scene = load_scene(npz_path, args.pick)
        valid = scene["cameras"]["valid"].astype(bool)
        clips.append({
            "name": entry["name"], "start": int(entry["start"]), "end": int(entry["end"]),
            **scene,
        })
        log(f"クリップ {entry['name']}: カメラ {int(valid.sum())}/{len(valid)} 有効、"
            f"再構成点 1 フレームあたり {scene['points'].shape[1]} 点")

    transforms = estimate_chain(clips)
    depth_range = world_depth_range(clips, transforms)
    output = {
        "created_at": utc_now(),
        "overlap_frames": overlap,
        "clips": [
            {
                "name": clip["name"], "start": clip["start"], "end": clip["end"],
                "scale": transform["scale"],
                "rotation": np.asarray(transform["rotation"]).round(8).tolist(),
                "translation": np.asarray(transform["translation"]).round(8).tolist(),
                "note": clip.get("note", ""),
            }
            for clip, transform in zip(clips, transforms)
        ],
        "depth_range_world": depth_range,
        "world_bounds": world_extents(clips, transforms),
        "notes": "相似変換は重複フレームの再構成点群の密対応から Umeyama 法で推定。",
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    log(f"変換表を保存: {args.output}")
    for entry in output["clips"]:
        log(f"  {entry['name']}: scale {entry['scale']:.4f} — {entry['note']}")
    if depth_range:
        log(f"  世界共通の深度範囲: {depth_range[0]:.3f} – {depth_range[1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
