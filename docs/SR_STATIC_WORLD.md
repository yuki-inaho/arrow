# SR生成版の可視化・運動分解・静止世界

> **2026-10-16 視覚品質の差戻し:** grassの固定世界はユーザー確認で不合格となった。
> 以下の数値・ファイル検証結果は履歴であり、元映像に近い外観の達成を意味しない。
> reference版でも基準frameの描画が色を崩し、後続frameでは葉形の大穴と背景投影ずれが残った。
> reference版を完成成果物として優先案内した判断を撤回。後続生成実験も画質未合格で、ユーザー指示により現状で停止した（[実験記録](GUIDED_STATIC_WORLD.md)）。

**grass SR動画・Rerunの作り方と日付付きprivate配布先は [GRASS_SR_WORKFLOW.md](GRASS_SR_WORKFLOW.md)。**

既存の [深度詳細化](DEPTH_DETAIL.md) で得た LingBot-Depth → NDR → scene flow SR → temporal の結果から、
従来の4列動画/Rerunと、動く物体も残した静止世界を生成する。モデル推論は再実行しない。
入力は `results/depth_detail_oct16/{grass,railway}`、追加出力は `results/sr_static_oct16/{grass,railway}`。

固定世界の比較対象 **`reference_world/`** は、基準frame0に見えていた草などを、その形と色で固定する設計だったが、上記のとおり視覚品質は不合格。
全フレームから面を追加する `static_world/` は実験比較版として残す。草では後続点の累積が強いにじみを生むため、
単に高いcoverageを画質の成功とみなさず、基準形を優先する版を追加した。

## 最初に開くもの

`<dataset>` は `grass` または `railway`。

| 相対パス | 内容 |
|---|---|
| `sr_visuals/<dataset>_sr_multipanel.mp4` | 従来の1920×1080・4列構成。元RGB、SR移動、最終深度、法線、支持、RGB点群、カメラ、SR軌跡 |
| `sr_visuals/<dataset>_sr_workflow.rrd` + `.rbl` | 同じ情報のRerun記録と初期レイアウト |
| `sr_visuals/<dataset>_sr_compare.mp4` | 従来7面比較のARROW表示を読みやすくした版 |
| `static_world/<dataset>_motion_decomposition.mp4` | 元RGB / total / camera由来rigid / non-camera residual |
| `reference_world/<dataset>_static_rgbd.mp4` | 観測RGB / 固定世界からの仮想RGB / 仮想depth |
| `reference_world/<dataset>_static_rgb.mp4` | 注釈なしの仮想RGB動画。元画像サイズ・元fps |
| `reference_world/<dataset>_static_depth.mp4` | 仮想depthの表示用動画。定量深度は下のNPZ |
| `reference_world/<dataset>_reference_world.rrd` + `<dataset>_reference_view.rbl` | 固定点群と元カメラ軌跡、観測/仮想RGB-D |
| `reference_world/model.ply` / `model.npz` | 全点の静止モデル。PLYはbinary little-endian |
| `reference_world/rgbd/NNNNNN.npz` | float32 camera-z depth、RGB、valid、K、camera_to_world、モデル点ID |
| `static_world/motion/NNNNNN.npz` | native解像度のrigid/nonrigid/total 2D・3Dフローとmask |
| `static_world/correspondence/NNNNNN.npz` | 各観測画素と固定anchor IDの対応。無対応=-1 |
| `reference_world/manifest.json` | 基準点の選択規則、元model SHA、各frameのcoverage、成果物hash |
| `static_world/model.npz` と `<dataset>_static_rgbd.mp4` | 後続面も統合した実験比較版。草のにじみが強く、基準形確認には上のreference版を使う |
| `static_world/manifest.json` | 全185pairの処理、対応数/新規点/局所補正/coverage、入力と出力hash |

動画のdepth色は全frame共通レンジ。黒い欠損をゼロ深度と解釈しない。
grassは107枚・5fps、railwayは80枚・24fps。最後のフレームには次フレームへのフローが無い。
生成物・重み・元画像・私的作業記録はGitに含めない。

## ARROWパネルが暗かった理由

以前の2番目のパネルは、低解像度ARROW深度をnative画像へ散布した**疎なanchor**だった。
無支持画素の黒と有効な色を一緒に縮小すると暗くなる。数値深度の明暗問題ではない。

新しい比較では、規則的なanchor格子を検出して元の粗いセルへまとめ、共通レンジで着色してnearest拡大する。
不規則な点配置では表示セルに重なる有効深度だけを面積加重平均する。
無支持セルは別の背景色、元の支持率（grass先頭では3.83%）も記載する。
これは表示する区画の変更で、dense深度への推定ではない。
元の `grass_compare.mp4`、`railway_compare.mp4` と数値anchorは保持している。

## rigid / non-rigidの定義と座標

`M_t` はcamera→worldのrigid 4×4 pose、`K_t` は元画像の整数pixel中心intrinsics。
深度 `z` は光軸方向。全3D値の単位は **arrow_provisional**、`metric_calibrated=false`。

保存済み `sr_flow=S` は、次のRGB-Dを現在のカメラへ補償して推定した**camera0座標の物体側3D変位**。
ここからさらにcamera flowを引くと二重補償になる。

```text
P  = z K0^-1 [u,v,1]
T  = inv(M1) M0 = (R,t)
Qr = R P + t                 # 世界が静止したとき次カメラに見える座標
Qt = R (P + S) + t           # 物体側の変位も含む次カメラ座標

rigid_flow    = project(K1,Qr) - [u,v]
total_flow    = project(K1,Qt) - [u,v]
nonrigid_flow = total_flow - rigid_flow

rigid_3d       = Qr - P
nonrigid_3d    = R S
total_3d       = Qt - P = rigid_3d + nonrigid_3d
world_nonrigid = rotation(M0) S
```

2D flowはnative pixel単位。3Dの`*_3d`は異時刻camera座標の成分差を分解したもの。
物理的な同一world座標での変位には `world_nonrigid` を使う。
3D flowに解像度倍率を掛けない。有限な画面外投影はフローとして保存し、対応追跡では境界外を拒否する。
behind camera / invalid depth / invalid SRはNaN＋false mask。

ここでnon-rigidと呼ぶ残差には、真の変形だけでなく、独立に剛体移動する物体、姿勢/深度/flow推定の誤差も含む。
材質の変形と独立物体の剛体運動を分類した結果ではない。

## 基準形を優先した固定世界

`build_reference_world.py` は全統合モデルから `birth_frame==0` の全anchorを順序を保って選ぶ。
動く物体も静止背景も同じ条件で残し、後続frameの形や色を混ぜない。`source_anchor_id` から統合モデルのIDへ戻れる。
点XYZ/RGBは全フレーム不変で、元のカメラだけを動かして描画する。観測されていない面は黒/NaNの欠損になる。
この版は基準時刻の可視面を固定するもので、後から初めて見えた物体や裏面は含まない。

## 全観測を使う実験的な固定世界

1. frame0の有効な点を固定anchorとして登録する。標準のmodel stride=2。voxel内の重複は最初の点で統合する。
2. `Qt` を次画像へ投影し、pixel距離≤1.5、相対depth差≤0.05、z-bufferで対応を取る。
   相対差の分母は `max(z_pred,z_observed)`。near優先、同値はpixel距離・source index順。
3. 対応できた点は元のanchor IDを引き継ぐ。anchorの位置・色は変更しないので、動く部分も最初の形で残る。
4. 未対応の新しい点には、追跡済み近傍の `live_world - canonical_anchor` を逆向きに適用する。
   半径1/2/4/8pixelの8方向、depth差≤5%、RGB L2差≤80、最低4支持のcomponent medianを使う。
   これにより対応切れのたびに変形後の形をそのまま重ねることを減らす。
5. 支持不足の面は初観測時の形に固定する。`birth_frame`、`birth_correction`、`correction_support` に区別を残す。

`model.npz` は `points, colors, birth_frame, birth_pixel, birth_correction, correction_support,
observation_count, last_seen` を含む。位置は出生時のworld座標からbirth_correctionを引いた値。
`correction_support=0` は近傍から逆変形を推定していない点、4以上は局所推定の支持数。
これを正解確率として解釈しない。観測回数は同じframeの複数pixelを重複して数えない。

基準時刻に見えなかった面の形は一意に決まらない。初観測固定・局所逆変形を含む**推定の静止モデル**であり、
基準時刻の完全な真値ではない。大変形、遮蔽、細い草、境界のdepth誤差、長い追跡のドリフトにより、
重複・にじみ・位置誤差が残り得る。無支持面の形を生成AIで補完してはいない。

この構成は [DynamicFusionのcanonical/live分離](https://grail.cs.washington.edu/projects/dynamicfusion/) と
[カメラ運動とresidual flowの分離](https://www.microsoft.com/en-us/research/publication/learning-residual-flow-as-dynamic-motion-from-stereo-videos/)
を参考にした、既存SR用の明示的な点対応方式。これらの学習モデルやvolumetric warp optimizerの再実装ではない。

## 仮想RGB-Dの意味

reference版と全統合版のいずれも、全frameで同じ各版の点群XYZ/RGBを使い、元の `M_t/K_t` だけを変える。camera-zのnear点が遮蔽する。
既定は半径2pixelの円盤splatで、欠損をinpaintしない。`rgbd/*.npz` はdepth NaN/valid=false/RGB黒で欠損を明示する。
`model_point_index` で描画元の固定点へ戻れる。`reference_support` はframe0出生点から描画されたpixelだけtrue。

MP4のdepthは色付きプレビュー。定量的なRGB-DにはNPZを使う。例えば:

```python
import numpy as np
with np.load("results/sr_static_oct16/grass/reference_world/rgbd/000053.npz") as frame:
    depth = frame["depth"]          # float32 camera-z, arrow_provisional
    valid = frame["valid"]          # bool; invalid depthはNaN
    rgb = frame["rgb"]              # uint8 H,W,3
    K = frame["K"]
    camera_to_world = frame["camera_to_world"]
```

## 再現コマンド

repo rootから、既存のdepth詳細化データがある状態で実行する。

```bash
uv run --no-sync python scripts/render_sr_workflow.py \
  --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 --dataset all

uv run --no-sync python scripts/build_static_world.py \
  --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 --dataset all \
  --max-points 10000000

uv run --no-sync python scripts/build_reference_world.py \
  --input-dir results/depth_detail_oct16 --source-dir results/sr_static_oct16 --dataset all

uv run --no-sync python scripts/render_motion_decomposition.py \
  --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 --dataset all

uv run --no-sync python scripts/verify_static_world.py \
  --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 --dataset all \
  --check-media --rerun-bin .venv/bin/rerun --report temp/static_world_verify.json
uv run --no-sync python scripts/verify_static_world.py \
  --input-dir results/depth_detail_oct16 --output-dir results/sr_static_oct16 --dataset all \
  --reference-only --check-media --rerun-bin .venv/bin/rerun --report temp/reference_world_verify.json

.venv/bin/rerun results/sr_static_oct16/grass/sr_visuals/grass_sr_workflow.rrd \
  results/sr_static_oct16/grass/sr_visuals/grass_sr_workflow.rbl
.venv/bin/rerun results/sr_static_oct16/grass/reference_world/grass_reference_world.rrd \
  results/sr_static_oct16/grass/reference_world/grass_reference_view.rbl
```

初回は新規の出力先を使用する。static consumerの既存出力を再生成するときは明示的に`--overwrite`。
`--stage build` / `--stage render` で固定モデル構築と再描画を分離できる。renderだけでも元入力hashとmodel hashを検査する。
最大出力30GiB・空き35GiBのreserve・最大点数のgateがある。異常時は失敗を残し、設定変更後に別attemptで再実行する。
grass全frameでは既定500万点に到達したため、上記コマンドは1000万点を明示している。
標準の深度パイプラインのsignatureを変えないため、これらは独立consumerにしている。

## 検証と作業上の知見

対象テスト:

```bash
uv run --no-sync pytest tests/test_static_world.py tests/test_static_world_pipeline.py \
  tests/test_static_world_verifier.py tests/test_sr_workflow.py tests/test_reference_world.py -q
uv run --no-sync ruff check
uv run --no-sync pyright
```

- findings: SRの定義を確認してから分離すること。すでに補償済みのflowへego motionを再び引かない。
- struggles: 追跡を失った点を常に初観測world位置へ追加すると、動いた形が重複する。近傍の逆変形と支持情報を追加した。
- struggles: canonical点とlive点は意図的に異なる。両者の距離に恣意的なhard上限を置く監査は誤り。
  分解式、出生逆変形、anchorの不変性、投影元点ID/色/depthを独立に検証する。
- findings: 全観測の融合は深度/姿勢/flow誤差を重ねる。草の中間frameでは全統合coverage約100%でも強いにじみがあり、reference版は83.68%で形を保った。欠損率と画質を別々に見る。
- tips: 数値モデル、表示用間引き、splat、色レンジを区別する。Rerunは初期レイアウトをRRDにも埋め込む。読込完了後の実frameとパネル名を確認する。
- tips: 元の草は全107frame、railwayは80frame。重複clip境界を再度数えず、native frame indexで全工程を揃える。

2026-10-16、このPCでの受入結果。coverageは仮想画像の有効画素率で、復元精度ではない。

| データ | frames / pairs | 基準形モデル点数 | reference coverage 平均 / 最小 | 全統合モデル点数 |
|---|---:|---:|---:|---:|
| grass | 107 / 106 | 230,395 | 74.41% / 59.78% | 5,701,822 |
| railway | 80 / 79 | 34,599 | 74.16% / 47.98% | 143,469 |

- 全185pairの分解式、全統合モデルの出生式、両版各187frameのRGB/深度/固定点ID/元pose/Kを独立検証。
- 実RRDを読み戻し、点XYZ・RGB・カメラ・画像frame被覆・最後のSR空フローを確認。
- grass #53 / railway #40でnative Rerunを停止・画面保存。RRDの数値読込成功とUI成功を別々に確認した。
- 対象テスト256件成功、ruff/pyright成功。生成物18動画を実デコードしフレーム数・fpsを確認、935 stageファイルとRGBの入力hashを照合した。
- 成果物は817ファイル、9,125,279,028 bytes（約8.50GiB、`deliverables.json`自身を除く）。
- 既存GPU/UI E2Eは8pass/1skip。skipは既存の `results/railway.npz` が無い再読込ケース。
- SR軌跡は48 seedからの実移流。末尾まで残った数はgrass 0、railway 7。対応を失った軌跡を捏造して延長しない。
- main 3Dの初期表示範囲は全観測点の1–99 percentileと全カメラから決める。少数の遠方外れ値で主点群が小さくなるのを防ぐ表示設定で、NPZ/PLY/RRDの点やflowは削除しない。
- `results/sr_static_oct16/HANDOFF.md` が成果物の入口、`deliverables.json` が実ファイルのbytes/SHA一覧。モデル、生成物、作業中の私的ログはGit対象外。


- struggles: Rerun 0.39のheadless起動はCJKパネル名でmissing-glyph panicになった。Rerunパネル名は英語、動画注釈は日本語にした。
- struggles: 別RBLの同時読込だけでは埋込layoutが優先されたため、reference版はRGB点群専用layoutを埋め込んだ専用RRDも生成する。depthの自動3D投影を固定RGB点群に重ねない。
- tips: 逆行列をfloat32で計算すると半pixel境界の丸めが変わる。独立検証もfloat64へ変換してから投影する。実反例を回帰テストに残した。
