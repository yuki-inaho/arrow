# LLMオンボーディングサマリー

> 新任LLMエージェントが `arrow`（yuki-inaho フォーク）で作業を始める際の初期資料です。
> 数値・コマンドはこのマシン（RTX A5000 24GB / ドライバ 580.178.04 / CUDA 13.0 / cu130 ビルド）での
> 検証実測に基づきます。雛形は関連リポジトリの `docs/ONBOARDING.md`
> （[`yuki-inaho/vggt-omega`](https://github.com/yuki-inaho/vggt-omega),
> [`yuki-inaho/Open-d4rt`](https://github.com/yuki-inaho/Open-d4rt)）を参照しました。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** ARROW（`arrow`、upstream は `ifradlin/arrow`）。RWTH Aachen の
  **Arbitrary Reconstruction and Tracking of 4D Observations in the Wild**。任意の画像集合
  （動画・多視点カメラ列・非順序の写真コレクション）からの**フィードフォワード型 3D 再構成と
  3D 点追跡**（[arXiv:2610.01314](https://arxiv.org/abs/2610.01314)）。
- **最終成果物:** 推論 CLI（`arrow infer`）、Python API（`ARROW.from_pretrained` / `infer`）、
  対話ビューア（`arrow-viewer`）、保存済み予測（`.npz`）の保存・再読込・配信、E2E テスト、
  Playwright による UI 実行・スクリーンショット導線。
- **ビジネス背景・価値:** D4RT の point-query デコードを任意観測（視点・時刻）へ拡張したモデル。
  WorldTrack / TAPVid-3D で SOTA、RGB-only MVTracker では専用マルチビュートラッカーを上回る（README より）。
  ロボティクス・4D シーン理解の再構成／追跡コンポーネントとして利用する。
- **現時点の進捗サマリ:** 推論コード・学習済み重み・ビューアは 2026-10-12 に公開済み
  （学習・評価コードは「Coming next」で未公開）。本フォークでは uv 環境構築 →
  実重みでの E2E テスト（CLI / Python API / ビューア API / Playwright UI）→ スクリーンショット
  可視化まで **RTX A5000 で検証完了**し、その手順を本ドキュメントに記録した。

## 2. クリティカルな要求・制約

> 「壊してはいけない」品質・仕様ライン。

- **Python は 3.11 以上**（`pyproject.toml`。`uv` が `.python-version` = 3.13 を使用）。
  環境は各リポジトリ独立の `<repo>/.venv`。`uv sync`（既定で dev グループを導入）で作成する。
- **torch は `>=2.10,<2.15`**。Linux / Windows では **cu130 wheel**
  （`[tool.uv.sources]` が `https://download.pytorch.org/whl/cu130` に固定）。CUDA 13 対応ドライバが必要
  （本機: 580.178.04 / CUDA 13.0）。
- **システム cuDNN と wheel 内蔵 cuDNN の競合に注意。** システム側（例
  `/usr/lib/x86_64-linux-gnu/libcudnn*` = 9.27）が wheel 内蔵（9.24）より先に解決されると
  `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` で落ちる。wheel 側を `LD_LIBRARY_PATH` の**先頭**に置くこと
  （§8 の 0) を参照）。
- **チェックポイントは Hugging Face `ifradlin/arrow`**（公開・gated ではない、約 5.3GB、
  検証時 revision `fd2b2f74c49990f739264a6aee786029227a93d4`）。初回利用時に自動ダウンロードされる。
  `HF_TOKEN` は環境変数のみで扱い、**値はログ・コミット・チャット出力に残さない**。
- **重みのライセンスは非商用（non-commercial）**。Waymo terms も適用される（README の License 参照）。
  商用利用を前提にした記述・提案をしない。
- **大型生成物・機密はコミットしない。** `.venv/`、`results/`、`/hf`、重みは対象外。
  `tests/` と `results/` は `.gitignore` 済みなので、E2E テスト本体を追加するときは `git add -f` を使う。
- **GPU メモリは 24GB（A5000）で 24 フレーム・密グリッド（track stride 2 + pose）実行のピークは約 15GB。**
  フレーム数・クエリ数を増やすと逼迫する。`decode_chunk_size`（既定 100,000）を下げると decoder の
  メモリを抑えられる。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| プロジェクト概要 | `README.md` | モデル概要・インストール・CLI・Python API・引用 |
| 推論仕様 | `docs/INFERENCE.md` | タスク、クエリ構築、選択点、マルチビュー、CLI/設定一覧 |
| ビューア仕様 | `docs/VIEWER.md` | 入力選択、実行、レイヤー、表示、シーン保存 |
| 設定 | `configs/arrow.py`, `configs/inference.py`, `configs/viewer.py` | モデル構築・推論既定値・ビューア（推論を継承） |
| 推論 E2E テスト | `tests/test_e2e_arrow.py` | 実重み・実アセットで動く 7 テスト（マーカー `e2e`） |
| UI E2E テスト | `tests/test_viewer_ui.py` | Playwright でビューアを操作・スクショ（マーカー `ui`） |
| スクリーンショットツール | `scripts/viewer_screenshot.py` | ビューア起動 → UI 実行 → 描画待ち → PNG 保存 |
| ビューア実行 E2E | `tests/viewer_run_e2e.py` | `POST /api/plan` → `/api/run` → `complete` → シーン取得の確認 |
| ワークフロー整理 | `docs/WORKFLOW.md` | 連番画像 → 中間成果物 NPZ → 可視化（処理と可視化の分離）の手順・実測・提案 |
| 本ドキュメント | `docs/ONBOARDING.md` | 本資料（機械固有検証値の記録場所） |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク
- uv 環境構築、CUDA / cuDNN の実演算確認、依存の更新。
- 推論 CLI・Python API・ビューアの実行と拡張、E2E テストの追加・修正。
- Playwright による UI 実行とスクリーンショット、`docs/` の整備。

### 任せないタスク
- モデル本体の学習・重み改変（学習・評価コードは未公開）。
- 重みの再配布、商用利用の前提での作業・提案。
- Hugging Face のアクセス権限操作、`HF_TOKEN` を含むあらゆる外部送信。
- `results/` 等の大型生成物のコミット、ユーザーの未コミット作業の削除・reset・revert。

## 5. インタラクション方針

- **回答スタイル:** 日本語、見出し＋箇条書き。確認値・実測値は表で提示する。
- **回答手順:** 環境確認 → 実行（実測コマンド）→ 結果と根拠 → 残課題 の順に報告する。
- **禁止事項・注意:** 未確認事項を断定しない（不明は「TBD」）。`HF_TOKEN` の値を出力しない
  （環境へは `export HF_TOKEN=$(sed -n 's/^export HF_TOKEN="\(.*\)"$/\1/p' ~/.bashrc)` のように展開する）。
- **Git 操作:** 対象ファイルを `git status` で確認してから add / commit / push する。

## 6. 試行タスク（オンボーディング演習）

1. `uv sync` の後、`uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` で
   `2.14.1+cu130` / `True` を確認する（cuDNN 競合時は §8 の 0) を先に設定）。
2. `uv run arrow infer assets/videos/railway.mp4 --stride 3 --output results/railway.npz` を実行し、
   24 フレーム・約 856,000 クエリが 20 秒未満で完了することを確認する（初回は重み約 5.3GB をダウンロード）。
3. `uv run pytest tests/test_e2e_arrow.py -v` を実行し、7 テストが通ることを確認する。
4. `uv run python scripts/viewer_screenshot.py --run-sample --output results/ui/sample.png` を実行し、
   UI からサンプル推論が走り、3D 点群・トラック・カメラフラスタが描画された PNG が保存されることを確認する。

## 7. 運用ルール・変更管理

- **ドキュメント更新時の記載ルール:** 数値は実測コマンドの出力に基づき、マシン・日付・revision を明記する。
  未確認は TBD とし、確認コマンドと判断基準を添える。
- **commit 前チェック:** `uv run ruff check`、`uv run pyright`、該当テスト（`-m e2e` / `-m ui`）を通す。
- **`uv sync` の注意:** uv の sync は完全同期のため、`--extra viewer` を付けずに sync すると
  viewer 依存が**削除される**。本フォークでは viewer 実行時依存を dev グループにも含め、
  `uv sync` 一発でモデル・ビューア・テスト環境が揃うようにしてある。
- **アップストリーム:** `ifradlin/arrow` への還元は別途 PR。本フォークの `main` は検証済み変更の反映先。

## 8. サンプルデータで推論 → ビューア可視化 → スクリーンショット

リポジトリ同梱アセット `assets/videos/railway.mp4`（504x284、80 フレーム、24 fps）を一貫して使う。

```bash
# 0) 環境（初回のみ）
uv sync                                     # モデル・ビューア・テスト依存（dev グループに viewer 依存を含む）
uv run playwright install chromium          # UI スクリーンショット用ブラウザ
# このマシンでは cuDNN 競合回避のため毎回前置する（§2 参照）
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

# 1) CLI 推論（24 フレーム、3 フレームおき）
uv run arrow infer assets/videos/railway.mp4 --stride 3 --output results/railway.npz

# 2) 保存済み結果をビューアで可視化（http://127.0.0.1:7860）
uv run arrow-viewer --scene results/railway.npz

# 3) ヘッドレスブラウザで UI を実際に操作してスクリーンショット
uv run python scripts/viewer_screenshot.py --run-sample --output results/ui/railway_sample.png
uv run python scripts/viewer_screenshot.py --scene results/railway.npz --output results/ui/railway_scene.png

# 4) テスト（重み＋GPU を使用）
uv run pytest -m e2e -v                     # 推論 E2E（7 テスト）
uv run pytest -m ui  -v                     # Playwright UI E2E（2 テスト）
```

- **表示パネル:** 3D ビュー（RGB 色付け点群・トラック、カメラフラスタ、時間スライダー）、
  入力フレーム（Frame n/N）、Sky mask（青 = 除外された天空画素）、Results レイヤー管理。
- **UI 実行フロー:** `From server` → `railway.mp4`（事前選択）→ `Add selected` → `Run 2 tasks`
  （既定: 3D tracking + Camera poses）。`Scene ready` 表示後に描画完了。
- **実測値（RTX A5000）:**

  | 実行 | クエリ数 | 所要時間 | GPU ピーク |
  |------|----------|----------|------------|
  | CLI `arrow infer`（track, stride 3, 24 フレーム） | 856,152 | 16.6 s | 15.4 GiB |
  | ビューア UI（track + pose, 24 フレーム） | 1,052,760 | 42.8 s | 14.9 GB |
  | 再構成結果 | 35,667 tracks + 24 camera poses | — | — |

- スクリーンショットは `results/ui/` に保存される（`.gitignore` 済み）。
- トラブルシューティング: `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` は 0) の `LD_LIBRARY_PATH` 前置で解消。
  Playwright の Chromium は SwiftShader（ソフトウェア WebGL）で描画している。

---

### 付録: 参考情報

- **主要リポジトリ/ディレクトリ:** 本フォーク `yuki-inaho/arrow`、upstream `ifradlin/arrow`、
  モデル配布 `ifradlin/arrow`（Hugging Face）。実装 `src/arrow/`、フロントエンド `frontend/`（ビルド済みは
  `src/arrow/viewer/static/`）、設定 `configs/`、アセット `assets/videos/railway.mp4`。
- **代表的なコマンド:**
  - `uv run arrow infer assets/videos/railway.mp4 --stride 3 --output results/railway.npz`
  - `uv run arrow inspect results/railway.npz`
  - `uv run arrow-viewer --scene results/railway.npz`（`--port` / `--host` / `--root` 可）
  - `uv run python scripts/viewer_screenshot.py --run-sample`
  - `uv run python scripts/process_sequence.py <連番フォルダ> --output-dir results/<name>_clips --clip-frames 24 --tasks track pose reconstruct depth`（クリップ分割の量産。再開可）
  - `uv run pytest tests/test_e2e_arrow.py tests/test_viewer_ui.py -v -s`
- **依存ライブラリ（本機検証値）:** `torch 2.14.1+cu130`, `torchvision 0.29.1`, `timm 1.0.29`,
  `onnxruntime 1.31.0`（skyseg、CPU 実行）, `numpy 2.x`, `av`, `safetensors`, `huggingface-hub`,
  `playwright`, `fastapi` / `uvicorn`, `pytest`（マーカー `e2e` / `ui` は `pyproject.toml` で登録済み）。
- **参考ドキュメント:** `yuki-inaho/vggt-omega`・`yuki-inaho/Open-d4rt` の `docs/ONBOARDING.md`
  （本資料の雛形。可視化は Rerun / Gradio + Playwright スクリーンショット方式で、
  本フォークは ARROW ビューア + Playwright 方式に対応）。
- **連絡先/責任者:** リポジトリオーナー `yuki-inaho`（詳細 TBD）。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みドキュメントはバージョン管理の対象です。
