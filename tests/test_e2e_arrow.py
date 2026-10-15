"""バンドル済み railway アセットに対する ARROW 推論のエンドツーエンドテスト。

リポジトリ同梱の実アセット（assets/videos/railway.mp4）に対して、公開済みの
ifradlin/arrow の実重みを動かし、次の経路を検証する:

- 動画ローダーの幾何（フレーム選択と 14 の倍数へのリサイズ・パディング）
- 1 回のエンコードを共有した reconstruct / depth / track / pose のマルチタスク推論
- エンコードを再利用した選択点トラッキング
- 画像フォルダ入力
- マルチビュー入力（同期した 2 ストリーム）
- NPZ の保存と再読み込みのラウンドトリップ
- 保存済み結果を配信するビューアの起動

実行方法（環境は ``uv sync`` で作成済みとする。dev グループにビューア依存も含まれる）::

    uv sync
    uv run pytest tests/test_e2e_arrow.py -v -s

初回の実行では ifradlin/arrow の重み（約 5.3 GB）をダウンロードする。
デバイスは CUDA・MPS・CPU のうち利用可能なものが自動で選ばれる。

システムの cuDNN（例: /usr/lib/x86_64-linux-gnu）が torch wheel 同梱の cuDNN と
異なるマシンでは、wheel 側の cuDNN をローダー探索パスの先頭に置くこと。
置かないと cuDNN が CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED で失敗する::

    export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
    uv run pytest tests/test_e2e_arrow.py -v -s
"""

from dataclasses import fields
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest
import torch

from arrow import ARROW
from arrow.infer import infer, load_input, load_multiview_input
from arrow.tasks.pose import estimate_poses
from arrow.tasks.tracking import tracking_predictions
from arrow.types import Predictions
from arrow.utils.results_io import load_result, save_result

pytestmark = pytest.mark.e2e  # 実重み・実アセットを使う E2E テスト

REPO = Path(__file__).resolve().parents[1]
ASSET = REPO / "assets" / "videos" / "railway.mp4"

FRAMES = 12
FRAME_STRIDE = 6  # 80 フレームの動画から 0, 6, ..., 66 を選択する


def finite_fraction(tensor):
    """NaN でない要素の割合。"""
    return float(torch.isfinite(tensor).float().mean())


def group_fields(result, task):
    """指定タスクのグループの予測（スライス済み Predictions）とメタデータを返す。"""
    group = next(entry for entry in result.metadata["groups"] if entry["task"] == task)
    start, count = group["start"], group["count"]
    sliced = Predictions(
        **{
            f.name: getattr(result.predictions, f.name)[:, start : start + count]
            for f in fields(Predictions)
        }
    )
    return sliced, group


def extract_frames(directory, *, count=6, step=5, mirror=False):
    """step フレームおきに count 枚の PNG をアセットから書き出す（mirror=True で左右反転）。"""
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(ASSET),
            "-vf", f"select='not(mod(n,{step}))'", "-frames:v", str(count), "-vsync", "0",
            str(directory / "frame_%03d.png"),
        ],
        check=True,
    )
    if mirror:
        from PIL import Image, ImageOps

        for path in sorted(directory.glob("*.png")):
            with Image.open(path) as image:
                ImageOps.mirror(image).convert("RGB").save(path)
    return directory


@pytest.fixture(scope="module")
def model():
    """ifradlin/arrow の実重みを読み込んだモデル（モジュール内で共有）。"""
    return ARROW.from_pretrained("ifradlin/arrow", device="auto")


@pytest.fixture(scope="module")
def video_observations():
    """動画アセットから準備済みフレームを読み込む。"""
    return load_input(ASSET, stride=FRAME_STRIDE, max_frames=FRAMES)


@pytest.fixture(scope="module")
def baseline(model, video_observations):
    """タスク・選択点・ラウンドトリップのテストで共有する 1 回のエンコード結果。"""
    return infer(
        model,
        video_observations,
        tasks=["reconstruct", "depth", "track", "pose"],
        track_stride=16,
        reconstruction_stride=16,
        depth_stride=16,
    )


def test_video_loader_geometry(video_observations):
    """動画入力のフレーム選択・リサイズ・パディングが期待どおりであること。"""
    observations = video_observations
    assert observations.num_frames == FRAMES
    assert observations.num_views == 1
    assert [frame["index"] for frame in observations.frame_info] == [
        FRAME_STRIDE * index for index in range(FRAMES)
    ]
    height, width = observations.images.shape[-2:]
    assert height % 14 == 0 and width % 14 == 0  # パッチサイズ 14 の倍数に整列される
    for frame in observations.frame_info:
        assert frame["original_size"] == [284, 504]
        assert len(frame["original_to_input"]) == 3


def test_multitask_inference(baseline):
    """4 タスクを 1 回のエンコードで実行し、各出力の健全性を確認する。"""
    result, _ = baseline
    metadata = result.metadata
    assert metadata["format_version"] == 1
    assert metadata["coordinate_frame"].startswith("reference_camera")
    assert metadata["num_views"] == 1 and metadata["num_timesteps"] == FRAMES
    assert [group["task"] for group in metadata["groups"]] == [
        "reconstruct", "depth", "track", "pose",
    ]
    assert sum(group["count"] for group in metadata["groups"]) == result.queries.num_queries
    runtime = metadata["runtime"]
    assert runtime["seconds"] > 0 and runtime["encoded_reused"] is False
    if torch.cuda.is_available():
        assert runtime["device"].startswith("cuda")
        assert runtime["peak_allocated_bytes"] > 0
    assert result.images.shape[:2] == (1, FRAMES)
    assert "sky" in metadata and 0.0 <= metadata["sky"]["fraction"] <= 1.0

    # reconstruct: 各フレームの点群。有限値が過半であること（天空画素は NaN になる）
    reconstruct, _ = group_fields(result, "reconstruct")
    assert reconstruct.points.ndim == 3 and reconstruct.points.shape[0] == 1
    assert reconstruct.points.shape[1] % FRAMES == 0
    assert finite_fraction(reconstruct.points) > 0.5

    # depth: 各フレーム自身のカメラでの点群。前方を向くカメラなので Z はほぼ正
    depth, _ = group_fields(result, "depth")
    z = depth.points[..., 2]
    finite_z = z[torch.isfinite(z)]
    assert finite_z.numel() > 0
    assert float((finite_z > 0).float().mean()) > 0.9

    # track: 正規化ターゲット座標と信頼度の範囲
    tracks, track_group = group_fields(result, "track")
    assert finite_fraction(tracks.points) > 0.5
    uv = tracks.uv[torch.isfinite(tracks.uv)]
    assert uv.numel() > 0
    # 画面外へ出た点のターゲット予測は [0, 1] を少し超えてよい。
    # この境界はスケール誤り（例: 画素座標の混入）を検出するためのもの
    assert uv.min() >= -0.5 and uv.max() <= 1.5
    inside = float(((uv >= 0) & (uv <= 1)).float().mean())
    assert inside > 0.75
    confidence = tracks.point_confidence[torch.isfinite(tracks.point_confidence)]
    assert confidence.numel() > 0 and bool((confidence > 0).all())

    # pose: ペア点群からのカメラ姿勢推定
    pose_predictions, pose_group = group_fields(result, "pose")
    reference = pose_group["reference_frame"]
    assert reference == FRAMES // 2  # 既定の参照フレームは中央
    poses = estimate_poses(
        pose_predictions,
        result.queries.xy[:, pose_group["start"] : pose_group["start"] + pose_group["count"]],
        num_frames=FRAMES,
        reference_frame=reference,
        **pose_group["pose"],
    )
    assert poses.camera_to_reference.shape == (1, FRAMES, 4, 4)
    assert bool(poses.valid[0, reference])
    assert float(poses.valid.float().mean()) >= 0.5
    for frame in range(FRAMES):
        if bool(poses.valid[0, frame]) and frame != reference:
            # 有効な姿勢の回転成分は直交行列であること
            rotation = poses.camera_to_reference[0, frame, :3, :3].double()
            identity = rotation @ rotation.T
            assert torch.allclose(identity, torch.eye(3).double(), atol=1e-3)
    finite_intrinsics = torch.isfinite(poses.intrinsics).all(-1).all(-1)
    assert finite_intrinsics.float().mean() >= 0.5
    focal = poses.intrinsics[finite_intrinsics][:, 0, 0]
    assert bool((focal > 0).all())


def test_selected_point_tracking(model, video_observations, baseline):
    """任意の選択点を、共有済みエンコードを再利用して追跡できること。"""
    _, encoded = baseline
    source = torch.tensor([0, FRAMES // 2])
    original = torch.tensor([[250.0, 200.0], [120.0, 240.0]])
    xy = video_observations.transform_points(original, source)
    result, _ = infer(
        model,
        video_observations,
        tasks=["track"],
        xy=xy,
        source_frame=source,
        encoded=encoded,
    )
    assert result.metadata["runtime"]["encoded_reused"] is True
    tracks = tracking_predictions(result)
    assert tracks.points.shape == (1, 2, FRAMES, 3)  # [バッチ, シード, ターゲット, xyz]
    assert finite_fraction(tracks.points) > 0.8
    assert finite_fraction(tracks.uv) > 0.8
    confidence = tracks.point_confidence[torch.isfinite(tracks.point_confidence)]
    assert confidence.numel() > 0 and bool((confidence > 0).all())


def test_image_folder_input(model, tmp_path):
    """画像フォルダを入力として推論できること。"""
    folder = extract_frames(tmp_path / "frames")
    observations = load_input(folder, max_frames=6)
    assert observations.num_frames == 6
    # フォルダ内はファイル名の自然順（frame_001, frame_002, ...）で読まれる
    assert [frame["name"] for frame in observations.frame_info][0] == "frame_001.png"
    result, _ = infer(model, observations, tasks=["track"], track_stride=16)
    assert result.metadata["num_timesteps"] == 6
    assert finite_fraction(result.predictions.points) > 0.5


def test_multiview_input(model, tmp_path):
    """同期した 2 カメラのマルチビュー入力で追跡できること。"""
    first = extract_frames(tmp_path / "cam_0", count=6)
    second = extract_frames(tmp_path / "cam_1", count=6, mirror=True)
    rig = load_multiview_input({"cam_0": first, "cam_1": second}, max_frames=4)
    assert rig.num_views == 2 and rig.num_timesteps == 4 and rig.num_frames == 8
    assert rig.frame_info[0]["view_name"] == "cam_0" and rig.frame_info[1]["timestep"] == 0
    result, _ = infer(model, rig, tasks=["track"], track_stride=16)
    assert result.metadata["num_views"] == 2 and result.metadata["num_timesteps"] == 4
    tracks = tracking_predictions(result)
    xyz = tracks.points.unflatten(2, (4, 2))  # [バッチ, シード, タイムステップ, カメラ, xyz]
    assert xyz.shape[0] == 1 and xyz.shape[2:] == (4, 2, 3)
    assert finite_fraction(tracks.points) > 0.3


def test_save_load_roundtrip(baseline, tmp_path):
    """NPZ へ保存して読み戻しても、数値とメタデータが完全に一致すること。"""
    result, _ = baseline
    path = tmp_path / "railway_e2e.npz"
    save_result(result, path)
    restored = load_result(path)
    assert restored.metadata == result.metadata
    assert torch.allclose(restored.images, result.images, atol=0, rtol=0)
    for record, other in [
        (result.queries, restored.queries),
        (result.predictions, restored.predictions),
    ]:
        for f in fields(record):
            assert torch.allclose(
                getattr(record, f.name), getattr(other, f.name), atol=0, rtol=0, equal_nan=True
            )


def test_viewer_serves_saved_scene(baseline, tmp_path):
    """保存済み結果を指定してビューアを起動し、API が応答すること。"""
    pytest.importorskip("fastapi", reason="viewer エクストラが未導入")
    result, _ = baseline
    scene_path = tmp_path / "scene.npz"
    save_result(result, scene_path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [
            sys.executable, "-m", "arrow.viewer.server",
            "--scene", str(scene_path), "--host", "127.0.0.1", "--port", str(port),
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 120
        while True:
            # ビューアが起動前に落ちていないかを先に確認する
            if process.poll() is not None:
                output = (process.stdout.read() if process.stdout else b"").decode(errors="replace")
                pytest.fail(f"ビューアが早期終了した（コード {process.returncode}）: {output[-2000:]}")
            try:
                if httpx.get(f"{url}/api/status", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("ビューアが 120 秒以内に起動しなかった")
            time.sleep(0.5)
        status = httpx.get(f"{url}/api/status", timeout=10).json()
        assert status["has_result"] is True
        scene = httpx.get(f"{url}/api/scene", timeout=30)
        assert scene.status_code == 200
        payload = scene.json()
        assert payload  # 保存済みシーンが空でないオブジェクトとして配信される
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
