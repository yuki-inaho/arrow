# 元動画を参照する静止化生成の再試行

2026-10-16、ユーザーの「現状まででcommit & push」の指示で停止した。grassの旧点群描画、BulletTime、VACE1.3Bの実生成はいずれも視覚品質で不合格。VACE14Bは重み13分割中7分割の取得・BF16変換・検証まで完了し、本生成は未実行。以下に研究、実装、失敗結果、再開地点を保存する。画質が承認された新しい静止世界RGB-Dはまだ得られていない。

## 前の方式で失敗したこと

`results/sr_static_oct16/grass/reference_world/grass_static_rgbd.mp4` の先頭・中盤・後半を元RGBと比較すると、葉形の黒い大穴、点状に崩れた路面、初期画像の外枠が現れる。全フレーム融合版は草の形が強くにじむ。

基準frame0は本来元RGBを再現できる条件だが、stride2点群の半径2pixel splatが隣のnear点の色で中心画素を上書きし、元RGBとの平均絶対差は18.51/255。全画素をradius0で描く対照では差0となった。ただしこれは後半の遮蔽解除や不足面を解決しない。reference版が後続観測面を全て捨てる構造自体も、本来の目標の十分な代替ではなかった。

また、静的路面の特徴対応でもframe0→53の53点で保存深度・姿勢からの再投影中央値が27.79pixel。これはpose/K/depthを合わせた誤差で、姿勢だけの誤りとは断定できない。短区間のholdout付きPnP補正も一部の時刻で不連続を増やしたため、全面置換しない。

従来の数値テストは「実装が保存モデルどおり動いた」ことを確かめた。元動画に近い見た目の十分さを証明していない。旧版の視覚受入れを撤回する。

## 研究比較と選択

比較軸は、元動画全体の外観を条件にできるか、カメラ運動と物体側の時間を分けられるか、公開実装・重みがあるか、このPCで試せるか。

| 方法 | 条件と制御 | この作業での位置づけ |
|---|---|---|
| [BulletTime / CVPR2026](https://github.com/19reborn/BulletTime) | 元動画全体＋別々のworld timeとcamera pose | 第一候補。物体の時刻を止めて同じカメラ軌跡を与える要求に直接対応 |
| [VACE / ICCV2025](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md) | RGB/control video、編集mask、参照画像 | 代案。補正済みRGB draftの不足部分を生成し、元背景を保持 |
| [Wan-Move / NeurIPS2025](https://github.com/ali-vilab/Wan-Move) | 初期画像＋点軌跡/visibility | 動き指定に向くが、後半の元動画外観を直接条件にしない。公式14B BF16例は40GB GPU |
| [MotionPro](https://github.com/HiDream-ai/MotionPro) / [MotionCtrl](https://github.com/TencentARC/MotionCtrl) | 初期画像等＋軌跡、camera/object motion | 独立運動制御は参考になるが、元動画全体を保つ条件が弱い |
| [Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow) | optical flowでnoiseをwarp、text/初期画像 | flow制御が可能だが、元動画の全RGBを拘束する編集とは異なる |
| [ReVideo](https://github.com/MC-E/ReVideo) | 元動画の領域外、初期編集画像、局所軌跡 | 直接V2Vの比較候補。多数の細い草への広域制御は別途適応が必要 |
| [Edit-by-Track](https://edit-by-track.github.io/) / [MotionV2V](https://github.com/RyannDaGreat/MotionV2V) | 元動画＋編集軌跡 | 概念は適合するが、調査時に実行用公式code/weights一式を確認できなかった |

[FlowEdit](https://github.com/fallenshock/FlowEdit)のflowはrectified-flow編集であり、任意optical-flow場を渡す制御と混同しない。VACEの公式flow条件もraw2Dベクトルではなく可視化RGB。単一APIに元RGBと別のdense flow入力が自動で存在するとは解釈しない。

BulletTimeの[論文](https://arxiv.org/abs/2512.05076)は、動画時刻と世界の時刻を区別してattention位置表現・adaptive normalizationへ注入する。今回の選択は入力仕様が目的に直接一致するという判断であり、草に対する実測品質の順位ではない。生成した未知面や葉の形は推定で、3D真値ではない。

## 採用候補の固定情報

- 公式code: `19reborn/BulletTime`, commit `e0331619d74d09ea149874cbde145a3a0a3faf7a`。
- BulletTime重み: `19reborn/bulletTime_ckpt`, revision `19e8f9aa3af3ffdd5101a911639c9ca0afec1d9e`。
- 基盤components: `THUDM/CogVideoX1.5-5B`, revision `fdc5267c90b5c06492985b966e43aae984e189e0`。基盤transformer重みは使わず、設定とtokenizer/T5/VAE/schedulerのみ取得。
- checkpointのフォルダ名は`fp32_ckpt`だが、実safetensors headerはBF16。名称からdtypeを推測しない。
- RTX A5000 24GiBでCPU offloadを使う。公式の1GPU例だけでは、この構成のVRAM成功を保証しない。

## 入力契約で避ける落とし穴

公開real-world loaderはsource cameraを仮置きし、intrinsicsのexampleパスも固定する。本件ではそのloaderを使わず、同じ公式推論pipelineへ実tensorを渡す。JSON parserはREADMEのc2w説明と異なるinverse処理を持つため、source/target双方のc2wを直接検証する。

sourceとtargetのcameraは共通の元軌跡、target world timeは基準時刻で一定。生成cameraを止めて静止画を複製する試験ではない。

元Kはframeごとに変動するが、公開pipelineは各streamで一定Kを要求する。元画像を共通Kへrayを保ってrectifyし、原K、共通K、微小な画角変更を記録する。Kを平均値に差し替えるだけでは元画像とrayが一致しない。

公式Time-RoPEはfps30を内部定数として持つため、モデル時刻はframe index/30に正規化する。実時刻frame index/5と再生fps5は別保存する。freezeではtargetモデル時刻を0とし、source時間は通常進行させる。

## 検証方針

まず短区間で同じ入力・seedの通常時間controlとfreezeを生成し、元動画・旧点群描画・生成を同一時刻で見比べる。元の背景の保持、草の輪郭と色、camera動作、カメラ補償後の草の動きを別々に判定する。単に穴がなくなったことやMP4が読めることを画質合格にしない。

生成RGBに旧depthを無条件に添えてRGB-Dの復元成功とは呼ばない。生成領域の深度・多視点整合は別途検証が必要。

作業書は`temp/workdoc_Oct16-2026_guided_freeze.md`、調査・失敗・実行ログは`temp/grass_static_review/`。重み・元動画・私的ログはGitへ含めない。


## 実行環境と再現

推論用依存は `requirements/bullettime.txt` に固定した。元repositoryの環境とは分離する。

```sh
uv venv --python 3.11 temp/grass_static_review/env/venv
uv pip install --python temp/grass_static_review/env/venv/bin/python -r requirements/bullettime.txt
git clone https://github.com/19reborn/BulletTime temp/grass_static_review/BulletTime
git -C temp/grass_static_review/BulletTime checkout e0331619d74d09ea149874cbde145a3a0a3faf7a
```

重みは上記revisionを指定してHugging Faceから取得する。基盤側の必要なpatternは `tokenizer/*`, `text_encoder/*`, `vae/*`, `scheduler/*`, `transformer/config.json`, `model_index.json`。BulletTime側は`fp32_ckpt/*`。それぞれ `temp/grass_static_review/models/cogvideo`, `temp/grass_static_review/models/bullettime` に置く。モデルlicenseは各公開repositoryの条件に従う。

入力準備はPyAVを含む既存repository環境で実行する。出力先が既に非空なら上書きせず別のディレクトリを使う。

```sh
.venv/bin/python scripts/prepare_bullettime_controls.py \
  --input-dir results/depth_detail_oct16/grass \
  --output-dir temp/grass_static_review/controls/pilot17_final \
  --frames 17 --height 640 --width 368 --fps 5 --max-zoom 1.05

temp/grass_static_review/env/venv/bin/python scripts/infer_bullettime.py \
  --upstream-dir temp/grass_static_review/BulletTime \
  --base-model temp/grass_static_review/models/cogvideo \
  --checkpoint temp/grass_static_review/models/bullettime \
  --controls temp/grass_static_review/controls/pilot17_final/controls.npz \
  --mode freeze --dry-run
```

本生成では`--dry-run`を外し、`--output-dir`と`--steps 50 --seed 42 --guidance 6`を加える。通常時間対照は`--mode source_time`で同じ入力を使う。正確な実行argvとfrozen source hashはローカルの`temp/grass_static_review/evidence/*.spec.json`に残す。MP4は5fps、raw RGB PNGも保存する。

Kの画像変換はinteger-centerで計算し、モデルへ渡すところでcx/cyに0.5を加える。公式Pluckerは画素中心をj+0.5で扱い、PRoPEもedge-center座標を使うため、両者を同時に合わせる。controlsの原Kは変更しない。17frame入力では必要な追加zoomは1.0000857139（約0.0086%）、全出力画素が元画像内に入る。

公式schedulerのnoise seedは内部で42固定されている。CLIのseedだけで全乱数が任意に切り替わるとは扱わず、この試行は全て42に揃える。時刻0の形を保つ目的では、長い動画を個別窓に分けて各窓の時刻0で止めると基準形が変わる。長尺化には共通基準を保持する入力設計が必要になる。

## findings / struggles / tips

- findings: 公式checkpointの1255keyを実ロードしてmissing/unexpected/shape mismatchが全て0でも、実推論の成功までは証明しない。
- struggles: 初回実行はtokenizerのprotobuf欠損でexit1。隔離環境にprotobuf5.29.5を追加し、初回receiptと旧lockを保存した。
- struggles: 次の実行では公式encode_videoがoffload中のVAE.device=CPUを参照し、encode hookがweightだけCUDAへ移したため失敗。diffusers0.34のhookは入力引数を受け取らない。adapterに限って先にpipeline execution deviceへ入力を移す互換処理を追加した。重み、sample、scaling、公式checkoutは変更しない。
- tips: 元画像の見た目、通常時間対照の再現性、時間固定の効果を分けて実動画で確認する。路面homographyからの草の残差には高さによるparallaxも含まれ、non-rigid motionだけの指標にはならない。


## BulletTimeの実生成結果

17frame・640×368・5fpsでsource_timeとfreezeを実生成した。source_timeは217.825秒、freezeは194.972秒でともにexit0。推論のGPU allocated peakは約12.24GiB（モデルロード後の計測）。ところがframe4以降の道路・縁石・白線が作り変わり、camera軌跡も外れたため両方不採用。比較動画はローカル `results/guided_static_oct16/grass/bullettime_comparison/comparison.mp4`。

同一camera/timeでもこのモデルはtarget側を独立noiseから生成し、sourceと一致させる反転処理を持たない。公開real_world経路の参照camera仮置き、訓練/evalの横長81frameと今回の短い縦動画との差もある。原因を一つに断定せず、source保持を明示的に制約するVACE方式へ進む。

CFG=1の通常時間対照も128.844秒で完走したが、背景の幻覚が残った。テキスト誘導を下げるだけではこの入力の外観保持を解決できなかった。

## VACEで使うガイドと生成部分の意味

[VACE論文](https://arxiv.org/abs/2503.07598)と[公式UserGuide](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md)を基に、元動画から作るRGB draft、編集mask、frame0参照画像を組み合わせる。これは本作業用の組合せで、VACE自体が3Dの剛体・非剛体分解を行うわけではない。

保存されたscene-flow SRはcamera0向きのego補償済み残差である。元camera-zから復元した点を`P_t`、camera-to-worldを`M_t`とすると、元動画の次フレームへの対応は次式で求める。

```text
Q_(t+1) = inverse(M_(t+1)) @ M_t @ (P_t + sr_flow_t)
uv_(t+1) = project(K_(t+1), Q_(t+1))
```

単にflowの符号を反転せず、この対応に沿って基準時刻のworld座標を運び、各時刻の元カメラへ再投影する。次深度との5%整合、前面判定、対応支持を確認し、不明部分はunknownとする。累積対応の支持が弱くなるため、frame0の観測RGBと形状の直接投影を優先し、current観測のwarpはその欠損を補助する。背景全面をframe0へ置き換える方式ではない。

元の植物領域と固定後の植物領域の和集合を編集候補とし、その外側は各時刻の元RGBを保持する。内部ではreference投影・current warp・unknownを別maskに保存する。緑色と画面ROIに基づく候補抽出のため、枯れ草・非緑色物体・ROI外の運動まで全て除去できる保証はない。camera/depthの誤差も投影に残る。

推論入力はunknownのRGBを127、maskを白にする。raw生成と`where(edit_mask, raw, draft)`の合成版を別保存する。後者の既知画素はPNGで厳密に保持されるが、圧縮MP4にbyte一致は要求しない。frame0は元RGBと一致させる。既知画素の一致は、生成された葉の形や時間的一貫性の合格判定とは別である。

- モデル: `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, revision `ec4d2cb062b548996b179d493fdd05340de702a1`。
- 環境: `requirements/vace.txt`。Diffusers0.34.0、BF16 transformer、FP32 VAE、CPU offload、VAE tiling。
- scheduler: モデル同梱UniPC、既定flow shift3.0。`--flow-shift`で明示した比較試験だけ上書きし、要求値と実効値をmanifestへ保存する。
- 出力契約: 17frame、640高×368幅、5fps。基準時刻はframe0、元camera軌跡は維持する。
- Rerun/PLY: 固定点群は観測されたframe0の形状のみ。生成領域のcanonical depthはNaN。各時刻の観測background depthは別診断として保存し、固定cloudへ融合しない。単位は暫定ARROWスケール。

接続時に、実`VideoProcessor.postprocess_video(output_type='np')`の出力がuint8ではなくfloat32[0,1]であることを確認し、明示的な丸めでPNGへ変換した。mask前処理・reference latent追加と除去・offload時の入力deviceも実インストール済みコードで確認した。

### VACEの再現手順

前述の共通K controlsを作った後、以下を実行する。VACEモデルは指定revisionの全ファイルを`temp/grass_static_review/models/vace13b`へ取得する。既存の非空出力先は上書きせず、再試行では新しい名前にする。

```sh
uv pip install --python temp/grass_static_review/env/venv/bin/python -r requirements/vace.txt
.venv/bin/python scripts/prepare_flow_freeze.py \
  --input-dir results/depth_detail_oct16/grass \
  --rectified-controls temp/grass_static_review/controls/pilot17_final/controls.npz \
  --output-dir temp/grass_static_review/vace_controls/pilot17_r5

temp/grass_static_review/env/venv/bin/python scripts/infer_vace_inpaint.py \
  --model temp/grass_static_review/models/vace13b \
  --controls temp/grass_static_review/vace_controls/pilot17_r5/controls.npz \
  --output-dir results/guided_static_oct16/grass/vace_pilot_r1 \
  --steps 50 --seed 42 --guidance 5 --conditioning-scale 1
```

`prepare_flow_freeze.py`は元RGB、LingBot/refiner後のfinal depth、ARROWのK/pose、scene-flow SRを読む。参照描画は隣接pixelの三角形を使い、source深度差5%・正の投影面積・最大辺8pixelで制限する。全参照面で前後関係を計算し、緑の3頂点に加え、色補間で実際に重みを持つ全画素も緑である場合だけ既知の草色として採用する。

検査・比較動画には`evaluate_vace_freeze.py`、観測形状を併記したRerun/PLYには`export_guided_freeze_rerun.py`を使う。評価側はNumPy/Pillow/OpenCVとffmpeg/ffprobe、export側は既存repo環境のRerun0.39を必要とする。

```sh
python scripts/evaluate_vace_freeze.py \
  --controls temp/grass_static_review/vace_controls/pilot17_r5/controls.npz \
  --generated-dir results/guided_static_oct16/grass/vace_pilot_r1 \
  --output-dir results/guided_static_oct16/grass/vace_comparison

.venv/bin/python scripts/export_guided_freeze_rerun.py \
  --controls temp/grass_static_review/vace_controls/pilot17_r5/controls.npz \
  --reference-model temp/grass_static_review/vace_controls/pilot17_r5/reference_model.npz \
  --generated-dir results/guided_static_oct16/grass/vace_pilot_r1 \
  --output-dir results/guided_static_oct16/grass/vace_review_rerun
```

Rerunは`rerun <output>/guided_freeze.rrd <output>/guided_freeze.rbl`で開く。観測した基準点群・元カメラ・合成RGB・生成mask・支持のあるcanonical depthを分けて表示する。出力manifestの`exported_not_quality_accepted`は構造確認済みという意味で、見た目の承認ではない。

### 実生成による切り分け

以下はいずれもGPUで実行した17frameの短区間試験。表の終了コード0は実行成功であり、映像品質の合格ではない。

| 試行 | 条件の変更 | 実測wall秒 | 実画像の評価 |
|---|---|---:|---|
| `vace_pilot_r1` | 3D参照triangle＋追跡補助、conditioning1、shift3 | 103.336 | rawにも草の周囲の灰色塊・縁石断片、合成版には投影の裂けも残り不合格 |
| `vace_pilot_cond05` | r1からconditioningだけ0.5 | 95.903 | frame8/16で同様の灰色残り。不合格 |
| `vace_pilot_shift16` | r1からshiftだけ16 | 93.792 | raw frame16の灰色残りを解消せず、不合格 |
| `vace_billboard_r1` | 単一面の参照RGB投影、conditioning1、shift3 | 93.590 | 葉輪郭は改善したが灰色haloと縁石ghostが残り、不合格 |

r1は2,804,033既知pixelの貼戻しと1,199,807生成pixelの整合を独立検証した。明示的なsource保持領域は1,981,936pixelで全一致。推論中のtorch allocated peakは12.316GiB。保持の数値検査に通っても、見た目の不良を覆さない。

shift3は[Diffusers公式例](https://huggingface.co/docs/diffusers/v0.34.0/en/api/pipelines/wan)に従う。一方、[著者CLI](https://github.com/ali-vilab/VACE/blob/main/vace/vace_wan_inference.py)の既定shiftは16である。両実装のsigma変換は独立に比較し、50個の整数timestepsは一致、sigma差は最大5.96e-8だった。既定3を誤実装とは断定しない。

追加の2D billboard試験では、初期の緑色画素で観測したcamera-z中央値2.86102355（暫定ARROW単位）に1枚の面を置き、元カメラからのhomographyで参照RGBを投影する。葉の細かな裂けは減るが、厚さ・parallax・遮蔽を単一面で近似するため、実三次元復元ではない。frame1以降のcanonical depthは全NaNとし、観測された基準点群は別に保持する。

struggles: gray127を128へ変えるだけの修正は採らない。[著者の回答](https://github.com/ali-vilab/VACE/issues/107)でもこの差は主要な感度ではないとされ、実mask分割・64channel化・時間縮小は公式コードと整合した。保存fpsだけを5から16へ変えてもモデル入力tensorは変わらない。原因不明のまま見た目だけの合格宣言はしない。


### 14B比較の準備と精度条件

1.3Bでconditioning、scheduler、参照投影を切り分けても未知領域の灰色残りが改善しなかったため、次の比較候補として同じbillboard入力を[公式VACE14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)でも試す準備をした。ユーザー指示による停止時点では実推論していない。revisionは`db79b90c60bbb45ceec9e41b9d5a4df934538ac4`。公開transformerはF32/BF16混在で約63.25GBあり、全floatをCPUでBF16へ変換した派生重みからNF4で読み込む。元LFS SHA、全tensorのkey/shape/dtype/有限性/明示castとの一致、派生SHAを保存する。元と同一のバイト列とは呼ばない。T5/VAE/tokenizer等は両モデルの公開blobが一致したものだけ、実ローカルhashを検証してhardlinkで共有する。

実行環境は`requirements/vace_nf4.txt`。bitsandbytes0.46.1では実小型WanVACE transformerの36 NF4層を読み込み、CPU→GPU→CPUの2回のforwardで有限値、同一packed bytes、出力差0を確認した。非量子化moduleのロードdtypeは明示BF16、モデルが精度保持を指定するmoduleはFP32のまま。VAEはFP32、T5はBF16、モデル全体のCPU offloadを使う。比較はモデル規模と量子化精度の両方が変わるため、改善・悪化を規模だけの効果とは断定できない。

背景補完の別試験では、他フレームの静的路面をhomographyで運ぶ方法も調べた。heldout再投影誤差が小さくても、草の取り違え・縁石混入を避けて2観測以上の一致を要求すると、unknown回復はframe4で5.36%、frame16で3.20%に留まった。閾値を緩めて穴を埋めると外観を誤るため、全動画への適用は採らなかった。


再現用の平面ガイドは次のコマンドで作る。`reference-model`は先の3D flow準備時に保存した、実frame0の観測点群を指定する。新しい出力先で実行した17frameの23共通配列と参照点群バイト列は、試験に使った凍結入力と完全一致した。

```sh
.venv/bin/python scripts/prepare_billboard_freeze.py \
  --input-dir results/depth_detail_oct16/grass \
  --rectified-controls temp/grass_static_review/controls/pilot17_final/controls.npz \
  --reference-model temp/grass_static_review/vace_controls/pilot17_r5/reference_model.npz \
  --output-dir temp/grass_static_review/vace_controls/billboard_reproduce

uv pip install --python temp/grass_static_review/env/venv/bin/python -r requirements/vace_nf4.txt
temp/grass_static_review/env/venv/bin/python scripts/prepare_vace14b_weights.py --execute \
  --shared-dir temp/grass_static_review/models/vace13b \
  --output-dir temp/grass_static_review/models/vace14b_bf16 \
  --raw-dir temp/grass_static_review/models/vace14b_raw_download
```

重み準備は初回に約80.2GBの空きを要求する（最大一時保持約42.62GBと35GiBの予約領域）。`--plan`でネットワークを使わず容量条件を確認できる。中断時は検証済みreceiptと派生ファイルが一致するものだけ再利用する。元入力の上書きや、未確認の中間ファイルの自動削除は行わない。sharedモデルは既存1.3Bの実ファイルが必要で、hardlink後に両モデルの共有重みを変更してはいけない。


取得時の注意: このPCでは7番目のHTTP転送が約1.33MiB/sに低下したため、最初の処理を実終了確認してから、検証済み6分割と旧連続partialを保存し、`hf-transfer==0.1.9`の8並列取得を別attemptで試した。追加依存は`requirements/vace_download_accel.txt`。この版のHugging Face Hubは高速転送時に既存partialを削除する。また高速転送の途中ファイルは順不同に書かれるため、通常HTTPへそのままサイズ基準で再開してはいけない。転送方式を戻す場合は途中ファイルを隔離し、該当する連続prefixが保全されている場合に限って復元する。今回停止した8番目には保全prefixがないため、その分割を先頭から再取得する。[転送ライブラリ](https://github.com/huggingface/hf_transfer)


## 停止時点の記録と再開地点

2026-10-16 12:34:11 UTC、ユーザーの停止指示に従って進行中の取得processだけを終了した。高速取得attemptの実終了コードはSIGTERMによる-15、wall 965.925秒、timeoutではない。検証済み7分割・各変換receipt・過去の失敗結果は保持した。順不同に書かれた8番目の途中ファイル4,267,704,320 bytesは `temp/grass_static_review/evidence/quarantined_accelerated_partials/` に隔離し、通常HTTPが誤って末尾から再開する状態を解消した。取得とGPU処理は停止済み。今回作成したHerdrの2ペインと表示bridgeを終了し、他の作業ペインを保持した。

| 保存内容 | ローカルパス / 状態 |
|---|---|
| 作業記録・現在地 | `temp/workdoc_Oct16-2026_guided_freeze.md`, `temp/grass_static_review/CURRENT_STATE.json` |
| BulletTime比較動画 | `results/guided_static_oct16/grass/bullettime_comparison/comparison.mp4`（不合格） |
| VACE1.3B平面ガイド試行 | `results/guided_static_oct16/grass/vace_billboard_r1/`（不合格、rawと合成版を分離） |
| 同時刻の比較PNG | `temp/grass_static_review/quality/vace/storyboard/prepared_r2/`。旧版はユーザーが指摘した `reference_world` を使用。14B欄は未生成と表示 |
| 14B用の固定入力 | `temp/grass_static_review/vace_controls/billboard17_r1/controls.npz`、SHA256 `37c4e56f71c3012f133addc483eb8c6a2d2e2852233300271c3e71dc7e05bc71` |
| 14B変換済み重み | `temp/grass_static_review/models/vace14b_bf16/`、7/13分割。全体の完成receiptは未生成 |
| 停止・保全・cleanup証拠 | `temp/grass_static_review/evidence/user_stop_partial_quarantine.json`, `user_stop_cleanup/report.json` |

再開する場合は上記の重み準備コマンドで検証済み7分割を再利用し、残りを取得する。隔離した高速転送partialをcacheへ戻してはいけない。全13分割と共有componentsの実hash・indexを再検証してから、同じbillboard入力で14B NF4の生成と見た目の比較を行う段階が残る。14B出力、最終比較動画、実grassの新しいRerun出力は未作成。Rerun exporterは小さな実fixtureの構造・数値読み戻しまで検証した。

停止前のコード検査は、主環境の59件、推論環境の79件、評価helperの8意味検査が成功。推論環境は初回78件の後にJSON出力の回帰検査を1件追加し、変更対象の38件を再実行した。対象scripts/testsのruffと、環境を分けたpyrightも成功した。これは入力契約・保存・再現処理の検証であり、生成画質の合格を意味しない。

- findings: 既知pixelの完全保持と生成部分の自然さは別々に確認する必要がある。今回の1.3Bは前者を満たしても草周囲の灰色残りを解消できなかった。
- struggles: 14Bの配布重みは約63.25GBあり、変換・取得速度・空き容量にも実測に基づく準備が必要だった。本生成に到達する前にユーザー指定の区切りで停止した。
- tips: bitsandbytesの実quantization設定はobjectの場合がある。公式 `to_dict()` でJSON化し、長い生成を始める前にmanifestの保存可能性を検証する。途中のモデルファイルや小型GPU fixtureの成功を、14Bの完成・実行成功と扱わない。
