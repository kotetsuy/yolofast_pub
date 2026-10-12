# yolofast: ONNX量子化・NPU評価ツール

640×640、batch=1の外部ONNXモデルに対する出力分離、AMD Quark A16W8量子化、
INT16入力変換、Ryzen AI NPU測定、COCO評価のスクリプトです。
モデル重み、Ultralyticsのソース、SDK、画像データセットは含みません。

## ライセンス

本リポジトリの独自コード・ドキュメントに **Apache License 2.0** を適用します。
[LICENSE](LICENSE)、[NOTICE](NOTICE)、[ANSWER.md](ANSWER.md)、
[依存関係の条件](THIRD_PARTY.md)を参照してください。
入力モデルと生成モデルのライセンスは元モデルの条件を引き継ぐ前提で扱ってください。
YOLO26の重みをApache-2.0に変更するものではありません。
Ultralyticsライブラリを呼び出すエクスポート・GPU測定コードは収録していません。

## 環境

CPU側の依存関係は `python -m pip install -r requirements.txt` で導入できます。
量子化にはAMD Quarkを別途導入してください（元実験では0.11.2）。
NPU測定ではRyzen AI SDKのVitisAI対応ONNX Runtimeを使用します。
SDK環境に通常のonnxruntimeを上書きインストールしないでください。
元測定環境: Ryzen AI MAX+ 395、Ryzen AI SDK 1.8、Python 3.12.13、ORT VitisAI 1.27.0。
ここでは依存関係の再インストールやNPU再測定は行っていません。

## 手順

利用条件を満たすFP32 ONNXと、独自に用意した校正画像を準備します。
画像は配布していません。検証画像3枚以上について、`examples/images-manifest.json` の
SHA256を実値に置換してください。画像パスはマニフェストからの相対パスにできます。
`sha256sum examples/images/*.jpg` でハッシュを取得できます。
校正画像とは別の画像を検証に使ってください。

```bash
python benchmarks/prepare_npu_split.py \
  --input models/fp32.onnx --output models/split-fp32.onnx \
  --images-manifest examples/images-manifest.json
python benchmarks/quantize_npu.py \
  --input models/split-fp32.onnx --output models/a16w8.onnx \
  --calib-dir calibration --calib-limit 60 --per-channel
python benchmarks/prepare_npu_int16.py \
  --input models/a16w8.onnx --output models/a16w8-int16.onnx \
  --images-manifest examples/images-manifest.json
python benchmarks/npu_split.py \
  --model models/a16w8-int16.onnx --quant-meta models/a16w8-int16.json \
  --reference-model models/a16w8.onnx --images-manifest examples/images-manifest.json \
  --output outputs/recheck --int16-input --execution run --postprocess stable \
  --quantization reference --opt-level 2 --disk-cache --skip-profile --seconds 300 --repeats 1
```

出力分離はYOLO26nの特定グラフ名・shapeを検査するため、別グラフには調整が必要です。
INT16変換では入力量子化ノードを外し、実画像・合成入力でCPU出力一致を確認します。
同名出力の上書きは拒否します。各ツールの全引数は `--help` で確認してください。
COCO評価は `benchmarks/evaluate_npu_coco.py`、演算配置の確認は
`benchmarks/audit_npu_compile.py` と `benchmarks/inspect_npu_runtime.py` を使用します。

## 保存済み測定値

匿名化した集計を [summaries/](summaries/) に収録しています。
元実験のNPU COCO val2017 mAP50–95は39.6676%、FP32比98.5196%、
5分連続測定は68.3919 FPSでした。モデル本体は別途入手が必要です。
FPSの境界は正規化済みCPU入力→外部INT16量子化→NPU推論→CPU Top-kで、
画像読込、前処理、表示、コンパイルは含みません。
新たに生成したモデルで同じ精度・速度になることを保証する値ではありません。

## 公開とプライバシー

元Git履歴、ローカル設定、認証情報、生ログ、モデル、画像はコピーしていません。
集計のホームパスは `$HOME` に置換しました。
新たな測定ログは絶対パス等を記録するため、`outputs/` はGit対象外です。
公開前の一覧は `PUBLICATION_MANIFEST.json` にあります。

```bash
git init -b main
git add .
git -c user.name="yolofast contributors" -c user.email="contributors@users.noreply.github.com" commit -m "Publish Apache-2.0 ONNX tools"
# GitHubで空の公開リポジトリを作成した後:
git remote add origin https://github.com/YOUR_ACCOUNT/yolofast_pub.git
git push -u origin main
```
