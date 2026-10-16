# 連番画像 → 中間成果物 → 可視化ワークフロー

> 目的: 連番画像（`frame_0001.png` …）を入力として、**可視化に使える中間成果物**（推論結果 NPZ）を
> 出力し、**処理（推論）と可視化を別工程・別マシンで進められる**ようにする。
> 本ドキュメントは現状の整理、推奨手順、追加提案（何をどうしたらよいか）を示す。
> 実測値はこのマシン（RTX A5000 24GB / torch 2.14.1+cu130）での検証結果（2026-10-15）。

## 1. 現状でできること（検証済み）

| 段階 | 手段 | 状態 |
|------|------|------|
| 入力（連番） | `arrow infer <フォルダ>`（ファイル名の自然順で読む。`frame_0001.png` … の連番を含む） | ✅ `tests/test_e2e_arrow.py` で検証 |
| 入力（その他） | 動画ファイル、画像パスのリスト、マルチビュー（`--multi-view`） | ✅ |
| フレーム選択 | `--start` / `--stride` / `--max-frames` / `--end`（排他終端。指定時は start/stride/end で選択） | ✅ |
| 解像度 | `--max-side`（既定 504。14 の倍数に整列し白でパディング） | ✅ |
| **中間成果物** | 推論結果 **NPZ**（画像・クエリ・予測・メタデータが自己完結） | ✅ |
| 検査 | `arrow inspect result.npz` / `load_result()`（Python） | ✅ |
| 可視化（対話） | `arrow-viewer --scene result.npz`、または **Open → From server**（開始フォルダは `<repo>/results`） | ✅ |
| 可視化（ヘッドレス） | `scripts/viewer_screenshot.py --scene result.npz --output scene.png`（PNG 保存） | ✅ |
| サーバで処理 → 手元で可視化 | ビューアの `POST /api/run`（処理）→ `GET /api/result`（NPZ ダウンロード） | ✅ `tests/viewer_run_e2e.py` |

つまり「**処理 = CLI / API で NPZ を出す**」「**可視化 = NPZ を開くだけ**」という分離は、すでに成立している。
あとは運用手順（連番のクリップ分割・量産・命名・保持）を整えるのが次の一歩（§5）。

## 2. 中間成果物の設計（何を残すか）

### レベル 1: 推論結果 NPZ（推奨・現行形式）
- 内容: `images`（前処理済み 24 枚）、`queries.*`（xy / source / target / reference）、
  `predictions.*`（points / confidence / uv / visibility / normals / displacement）、`metadata`（来歴・実行時間・グループ構成・frame_info）。
- これ 1 つでビューアの全レイヤー（3D 点群・トラック・カメラ姿勢・深度・法線・可視性）を再現できる。
- 実測（24 フレーム・357,744 クエリ・4 タスク）: **48MB（圧縮後）/ 67.4MB（非圧縮）**。
  内訳: `images` 38.8MB、`predictions.*` 約 18MB、`queries.*` 約 11MB。**画像が最大**。
- 保存は `save_result()`（ビューアの「Save predictions」と同じ形式）。再読込は `load_result()`。

### レベル 2: 入力スナップショット（軽量・要オプション整備）
- 用途: 前処理・フレーム選択の再現確認、フレーム表示専用の軽量配布。
- 現状: `save_result` は常に `images` を含む。軽量化には `--images {full,small,none}` のような
  オプションとビューア側の「画像なし時」の表示確認が必要（§5 Phase 2）。
- 目安: 画像を除けば 1 シーンの成果物は数 MB（クエリと予測のみ）。

### レベル 3: エンコード特徴（`context_tokens` / `id_tokens`）（非推奨・必要時のみ）
- 用途: 同一入力への**追加クエリ**（選択点の追跡など）を再エンコードなしで実行する。
- 現状: 保存 API なし。プロセスをまたぐ追加実行は「再エンコード」が現実的
  （本機で 24 枚エンコード + 4 タスク = 7.8 秒のため、コストは小さい）。
- 実装する場合は大容量（GB 級になり得る）かつ内部形式の安定性が前提になるため、必要になった時点で設計する。

### 補助成果物
- スクリーンショット PNG（可視化の証跡。`scripts/viewer_screenshot.py`）。
- `manifest.json`（量産時の来歴: 入力・選択・タスク・実行時刻。§5 Phase 1 で追加予定）。
  ※ 現状でも NPZ の `metadata` に同じ情報（`frame_info` / `groups` / `runtime`）が入っている。

## 3. 推奨手順（コマンド例）

### 3.1 連番画像を用意する
既存の連番フォルダをそのまま使える。動画から作る場合の例（3 フレームおきに 24 枚）:

```bash
mkdir -p results/demo_seq/frames
ffmpeg -v error -i assets/videos/railway.mp4 -vf "select='not(mod(n,3))'" -vsync 0 \
  -frames:v 24 results/demo_seq/frames/frame_%04d.png
```

### 3.2 クリップ単位で推論し、NPZ を出力する

```bash
uv run arrow infer results/demo_seq/frames --tasks track pose reconstruct depth \
  --track-stride 8 --reconstruction-stride 8 --depth-stride 8 \
  --output results/demo_seq/railway_seq.npz
```

- 長い連番は **24〜48 フレーム程度のクリップに分割**して複数 NPZ にする（ビューアはタブごとに 1 シーン）。
  既定 `--max-frames 24` でも同じ。分割の例:

```bash
uv run arrow infer frames/ --start 0  --end 48 --stride 2 --tasks track pose --output results/seq/clip_0000.npz
uv run arrow infer frames/ --start 48 --end 96 --stride 2 --tasks track pose --output results/seq/clip_0001.npz
```

- **まとめて量産する場合は `scripts/process_sequence.py` を使う（推奨・P1）**。クリップ分割・実行・
  検査・`manifest.json` への記録まで行い、中断後は同じコマンドの再実行で完了済みクリップをスキップする
  （壊れた出力は自動で実行し直す）:

```bash
# 実行計画の確認（推論はしない）
uv run python scripts/process_sequence.py frames/ --dry-run

# 実行（例: 24 フレームずつ、4 タスク）
uv run python scripts/process_sequence.py frames/ --output-dir results/seq_clips \
  --clip-frames 24 --frame-stride 1 --tasks track pose reconstruct depth \
  --track-stride 8 --reconstruction-stride 8 --depth-stride 8
```

  出力は `clip_<start>-<end>_s<stride>.npz` と `manifest.json`（入力・設定・各クリップの
  実行時間・クエリ数・サイズを記録）。

- メモリの目安: 24 フレーム・密トラック + 姿勢でピーク 15〜16GB（24GB GPU の 2/3）。
  クエリ数を増やすときは `--decode-chunk-size` を下げる。

### 3.3 中間成果物を検査する

```bash
uv run arrow inspect results/demo_seq/railway_seq.npz   # メタデータ（フレーム・グループ・実行時間）
```

Python で中身を確認する場合: `load_result(path)` → `result.predictions.points` などの形状確認。

### 3.4 可視化する（処理とは別の工程）

```bash
# 対話ビューア（http://127.0.0.1:7860）
uv run arrow-viewer --scene results/demo_seq/railway_seq.npz
# 実行済みビューアなら: Open → From server → results/… を選ぶ（開始フォルダが results になっている）

# ヘッドレスで PNG 化（CI・証跡向け）
uv run python scripts/viewer_screenshot.py --scene results/demo_seq/railway_seq.npz \
  --output results/demo_seq/scene.png --canvas-output results/demo_seq/scene_canvas.png
```

### 3.5 別マシン・別プロセスで可視化する
- NPZ をコピーして `arrow-viewer --scene`（ビューアは推論をしない。ファイルを開くだけ）。
- またはビューアの UI から: `Open → From device` に NPZ をアップロード。
- サーバ側で処理した場合は `GET /api/result` で NPZ をダウンロードして手元で開ける。

## 4. 実演（本機での実測）

| 項目 | 値 |
|------|-----|
| 入力 | `results/demo_seq/frames/frame_0001.png` … `frame_0024.png`（504x284 の連番 24 枚、7.9MB） |
| 実行 | `arrow infer … --tasks track pose reconstruct depth`（strides 8） |
| 所要時間 | **7.8 秒**（cuda:0、ピーク 16.5GB） |
| 中間成果物 | `results/demo_seq/railway_seq.npz` **48MB**（357,744 クエリ） |
| 可視化 | 同一 NPZ を `--scene` で開き、点群 52,489・トラック 2,302・カメラ姿勢 24 を描画 |
| ヘッドレス証跡 | `results/demo_seq/scene.png` / `scene_canvas.png`（Depth ビュー: 深度・法線・可視性も表示） |

このとおり、**推論は 1 回だけ・可視化は NPZ を開くだけ**で、両者を分けて作業できる。

### grass 連番（107 枚、5fps）の量産（実データ・`process_sequence.py` 使用）

`scripts/process_sequence.py` で 5 クリップに分割し、4 タスク（track / pose / reconstruct / depth）を
実行した実測:

| 項目 | 値 |
|------|-----|
| 入力 | 連番 107 枚（720x1280 の縦位置、5fps 動画からの切り出し） |
| クリップ | 24 フレーム × 4 + 11 フレーム × 1（`results/grass_clips/`） |
| 所要時間 | 合計 **108.9 秒**（クリップあたり 22〜26 秒、11 枚クリップは 12.8 秒） |
| 成果物 | 各クリップ 357,744 クエリ・**約 50MB**（11 枚クリップ 22.8MB）、`manifest.json` 付き |
| 再開 | 同コマンド再実行は **0.14 秒**で全スキップ。壊れた NPZ は検出して**該当クリップだけ再実行**（24 秒で復旧） |
| 可視化 | `--scene` で点群 52,855・トラック 2,302・カメラ姿勢 24 を描画（`clip_0000_scene.png`） |

## 5. 足りないものと提案（フェーズ）

| Phase | 内容 | 目的・補足 |
|-------|------|-----------|
| **P1** | `scripts/process_sequence.py`（**実装済み**） | 連番フォルダ → クリップ分割 → NPZ 量産。`manifest.json`（来歴・実行時間・クエリ数）、再開可能（完了済みはスキップ、壊れた出力は自動再実行）、`--dry-run` 対応。 |
| **P2** | 成果物の軽量化オプション `--images {full,small,none}` | 配布・蓄積用。ビューアの「画像なし」動作確認とセットで（中） |
| **P3** | 命名・配置の規約（`results/<sequence>/clip_<start>-<end>_<stride>.npz` など）を docs に固定 | 量産時の混乱防止。タブ名もクリップ名に合わせる運用 |
| **P4** | E2E テストに「連番フォルダ → NPZ → `--scene` でスクショ」を追加 | 分離ワークフローの回帰検知（既存の `e2e` / `ui` マーカーに追加） |
| **P5**（任意） | クリップ横断の選択点トラッキング等、encoded 再利用を前提にした設計 | 必要になったら（再エンコード 7.8 秒/24 枚なので急ぎではない） |

## 6. 決めごと（TBD・要相談）

- 成果物の**命名規約**とディレクトリ構成（現行: `<output-dir>/clip_<start>-<end>_s<stride>.npz`、
  P1 が採用。複数シーケンスの親ディレクトリ構成は TBD）
- クリップ長の既定（24 / 48 / 96 フレーム）と `max_side`（504 / 720 …）
- タスクの既定セット（軽い `track` のみ / `track + pose` / 4 タスク全部）
- 成果物の保持期間・削除ポリシー（48MB/24 枚 が目安）
- 複数クリップ運用（1 タブ = 1 クリップ）で困らないかの確認

## 関連ドキュメント

- [深度詳細化](DEPTH_DETAIL.md) — ARROW暫定scale → LingBot-Depth → NDR → scene flow SR → temporal
- [SR版の可視化と静止世界](SR_STATIC_WORLD.md) — 従来4列動画/Rerun、camera/物体側flow分離、動体を残す固定点群・仮想RGB-D
- `docs/INFERENCE.md` — タスク・クエリ・CLI の詳細
- `docs/VIEWER.md` — ビューアの入力選択・実行・表示・保存
- `docs/ONBOARDING.md` — 環境構築と検証手順のまとめ
