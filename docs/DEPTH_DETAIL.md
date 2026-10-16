# ARROW 深度詳細化（LingBot-Depth + NDR + scene flow SR）

追加成果物: [SR版の従来可視化・rigid/non-rigid分解・静止世界](SR_STATIC_WORLD.md)。
grassの入力からMP4/Rerunまでの詳しい手順と配布先: [GRASS_SR_WORKFLOW.md](GRASS_SR_WORKFLOW.md)。
従来4列動画/Rerun、疎ARROWの表示改善、動く物体を残した固定点群・仮想RGB-Dはこの独立consumerで生成する。

検証日: 2026-10-16。grass107枚・railway80枚を実処理し、独立監査と実Rerun表示を確認済み。

## 1. 目的と主経路

ARROW の連番画像/動画から、**元RGB解像度**でスケール・境界・時間的一貫性を保った深度を作る。

```
ARROW(暫定scale) → LingBot-Depth → depth由来normal + gray edge → NDR(backend=rust, iterations=40)
  → カメラ補償 native scene flow（長辺160） → RGB/depth-guided joint-bilateral 3D flow SR
  → SR motion correspondence temporal fusion → final
```

- **SR はニューラルモデルではない**: 低解像度3D変位を高解像度 depth/RGB で導く決定論的 joint-bilateral 補間。倍率は解像度比で変えない。
- LingBot-Depth は dense completion、NDR は法線誘導の精緻化、temporal は SR 3D変位による **motion correspondence** 融合（同一pixel比較はしない）。
- MoGe v3 は **`--with-moge` 指定時のみ optional 比較**（主経路外）。

## 2. 尺度・単位

- 単位は **`arrow_provisional`**、`metric_calibrated=false`。ARROW の低解像度深度を暫定基準として統一。
- 将来、**単一 scalar** を depth / camera translation / 3D scene flow へ同一適用して一括校正できる構造
  （`calibration.single_scalar`, `applied=false`）。現状の数値を「校正済みメートル」と呼ばない。

## 3. 出力（権威データと構成）

- 出力ルート: `results/depth_detail_oct16/{grass,railway}/`
- **権威データ**: `frames/<idx>.npz`（全stage統合pack、`metadata` JSON に stages/calibration/units）+ `stages/final.npz`。
  `stages/<stage>/<idx>.npz` は中間生成物。
- 元RGBと同じ `H,W` の float depth/valid mask、geometry（pixel中心K、world rigid pose + 単一scale translation）、
  各stageの metadata（model/params/input_sha256/argv/status/created_at）と provenance を保存。
- Kは元画像の整数画素中心。LingBot入力時のみ正規化し、両方を保存。camera poseはrigidなc2w、Sim3 scaleはdepthとtranslationに1回適用する。

## 4. 結果の見方（ユーザー入口）

`<dataset>/visuals/` に生成:

| ファイル | 内容 |
|---|---|
| `<dataset>_compare.mp4` | 7面パネル（RGB / ARROW / LingBot / NDR / Scene flow / Temporal / SR 3D flow）。共通 depth range・flow range・frame番号付き |
| `<dataset>_representative.png` | 代表フレームの同パネル |
| `<dataset>_sr_comparison.png` | 境界cropでの bilinear vs SR の3D flow比較 |
| `<dataset>_final_pointcloud.rrd` | final RGB点群（表示用stride4）+ SR 3D変位矢印（世界回転のみ適用、RDF） |
| `<dataset>_view.rbl` | 点群/SR・RGB・最終深度を並べるRerun表示設定 |
| `<dataset>_rerun_preview.png` | 実viewerの代表フレーム表示 |

表示上の注意:

- **ARROW パネルは sparse anchor**（入力そのもの）であり、縮小表示では暗く見える。dense 化した比較画像ではない。
- 最終フレームの SR パネルは **次ペアが無いため空**（正常）。

## 5. 再実行（実CLI）

```bash
cd /home/kasm-user/Desktop/arrow
# 長い実checkpointパスはliteralなshell変数で安全に扱う（HOME等は上書きしない）
MODEL_CHECKPOINT="/home/kasm-user/.cache/huggingface/hub/models--robbyant--lingbot-depth-pretrain-vitl-14-v0.5/snapshots/79204ed6b837f4fdd192cf563e59481fecfa0295/model.pt"

.venv/bin/python scripts/run_depth_detail.py --dataset all --stage all \
  --output-dir results/depth_detail_oct16 \
  --arrow-device auto --device cuda \
  --lingbot-model "$MODEL_CHECKPOINT" --lingbot-resolution-level 9 \
  --ndr-backend rust --ndr-iterations 40 --long-side 160 --sceneflow-workers 4
```

- 個別stageは `--stage ndr` 等。実行前に `--dry-run` でargvを確認。`--force` で resume を無視。
- 環境分離: ARROW `.venv`、LingBot、NDR、scene flow は**専用venv**で段階subprocess実行（MoGeはoptional）。
  - LingBot venv には **setuptools 84.0.0** を追加導入済み（Triton警告対応、実測事実）。
  - runner は段階ごとに **cuDNN wheel lib を環境別に前置**（ARROW側 `LD_LIBRARY_PATH_PREPEND`）。
- **ARROW native cache と signature 失効は別**:
  - `arrow_clips` は native cache として再利用可能。
  - helper本体の `code_hash()` は全stage署名に入るため、helper修正時は上流から **signature不一致→再実行**が正しい（署名改竄禁止）。

scene flow の範囲は補償後の全ペア・全有効値から決め、native solver開始前に全入力を検査する。
深度のclampや黙示的な除外で範囲外値を消さない。独立CPUペアは `--sceneflow-workers` で並列化する（既定1）。

Rerunは保存済み記録と表示設定を一緒に開く。推論用モデルは不要。

```bash
.venv/bin/python scripts/prepare_depth_detail_view.py --dataset all
.venv/bin/rerun results/depth_detail_oct16/grass/visuals/grass_final_pointcloud.rrd \
  results/depth_detail_oct16/grass/visuals/grass_view.rbl
# railwayも同じ2ファイルを指定する。
```

## 6. 検証（受入CLI）

```bash
uv run --no-sync pytest tests/test_depth_detail.py tests/test_depth_detail_pipeline.py \
    tests/test_depth_detail_runner.py tests/test_depth_detail_verifier.py \
    tests/test_depth_detail_execution.py -q    # 5 files / 166 tests
uv run --no-sync ruff check src/arrow/depth_detail.py scripts/depth_detail_models.py \
    scripts/run_depth_detail.py scripts/verify_depth_detail.py scripts/prepare_depth_detail_view.py
uv run --no-sync pyright

uv run --no-sync python scripts/verify_depth_detail.py --dataset all \
    --output-dir results/depth_detail_oct16 --evidence-dir temp/herdr_depth_workflow/evidence \
    --check-render --rerun-bin /home/kasm-user/Desktop/arrow/.venv/bin/rerun \
    --json-report temp/herdr_depth_workflow/evidence/audit/verify_report.json
```

- verifier は frames pack 必須（欠落は拒否）、`--partial` は明示的な途中監査のみ。
- RRD は `--rerun-bin` 指定時に `rerun rrd verify` を実行。

## 7. 結果（数値）

| dataset | frames | depth H×W | SR pairs | 共通支持の時間残差 before → after | 悪化 / 拒否 |
|---|---:|---|---:|---|---|
| grass | 107 | 1280×720 | 106 | 0.00959170 → 0.00750809 | 0 / 0 |
| railway | 80 | 284×504 | 79 | 0.00901256 → 0.00699223 | 0 / 0 |

- 全モデル段階の実終了0、再開controller6段階も全終了0（22分）。完了済みgrassのARROW/LingBot/NDRは、数値処理の不変と入力範囲を確認して再利用した。
- 独立検証: grass 11,208項目、railway 8,400項目が全合格。RRD全187framesの一意coverageと代表点の位置/色/姿勢/矢印を読み戻して一致。実画面も確認。
- 対象166 tests / ruff / pyright合格。既存GPU/UI E2Eは8 passed, 1 skipped（既存の保存sceneがないケース）。このPCのE2Eはtest docstring記載のcuDNN wheel lib前置が必要。
- native scene flowの独立4ペア測定: workers1=26.046秒、2=15.892秒、4=11.272秒。4並列は約2.31倍、比較した全深度/3D変位/支持maskの最大差0。全workflowの速度比ではない。
- 残差は次フレーム深度との整合性。frame0とmiddleの別実装による再投影も確認した。実3D GT精度を示す数値ではない。

## 8. 限界

- 絶対メートルは**未校正**（`metric_calibrated=false`）。実3D GT は無い。
- SR の境界改善は**合成GTテスト**での検証であり、実データの再投影残差検査（独立CLI）とは別物。
- NDR の normal は**depth由来**（専用法線センサではない）。
- 低解像度flowから真の細部運動を観測した保証はない。低解像度で未観測の細部運動は検証できない。

## 9. 重要な運用tips（短く）

- frame0 は bwd 更新が無く 中間更新の影響を受けにくい比較に最適。最初に実行して回帰検知する。
- 監査CLIは実行前後で入力NPZの sha256 を比較し、read-only（誤書き込みなし）を確認できる。
- 境界補間規約は analytic 2x2 fixture（-0.49 / -0.5 / W-0.5 / -0.51）で固定する。

> 詳細な findings / struggles / tips は `temp/workdoc_Oct16-2026_depth_workflow.md` §8と audit evidence に集約済み。
