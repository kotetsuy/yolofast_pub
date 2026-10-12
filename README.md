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

## clone・コピー・独立したvenvの作成

Ubuntu 26.04、Git、導入済みの `uv` を使用します。
システムのPython 3.14とは別に、`uv` でこのプロジェクト用のPython 3.12を導入します。
初回のcloneとパッケージ導入にはネットワーク接続が必要です。
OSのドライバーを除き、Python本体と実行に必要なパッケージ・モデル・画像は
すべて `~/yolofast` 内に配置します。他プロジェクトのvenvやsite-packagesは参照しません。

### 1. 公開リポジトリをcloneして作業先にコピー

`~/yolofast_pub` と `~/yolofast` が存在しない状態から実行します。
既存のフォルダがある場合は、内容を確認して別名に移してから実行してください。
コピーはGit管理情報も含みます。コピー後の実行は `~/yolofast` 内だけで完結し、
clone元の `~/yolofast_pub` は実行時に使用しません。

```bash
cd ~
# 既存フォルダへの上書き・入れ子コピーを防止:
if [ -e yolofast_pub ] || [ -e yolofast ]; then
  echo 'yolofast_pub または yolofast が存在します。先に配置先を整理してください。'
else
  git clone https://github.com/kotetsuy/yolofast_pub.git yolofast_pub &&
    cp -a yolofast_pub yolofast
fi
```

cloneとコピーが成功したことを確認して、次へ進みます。

### 2. `~/yolofast/.venv` を新規作成

他のvenvが有効な場合は `deactivate` してから実行します。
Python 3.12の配置先とuvのキャッシュをプロジェクト内に指定します。
システムPythonの変更や、OSへのPython 3.12・venvパッケージの追加は不要です。

```bash
cd ~/yolofast
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv python install 3.12 --no-bin
uv venv --managed-python --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# 量子化も行う場合（元実験で使用した版）:
python -m pip install 'amd-quark==0.11.2'
python -m pip check
mkdir -p models calibration examples/images datasets outputs
```

`--system-site-packages` は指定せず、他フォルダへの参照を追加する `.pth` や
シンボリックリンクも作成しません。venv内のPythonはプロジェクト内の `.python/` を参照します。
`.python/` はvenvの実行に必要なので保持してください。
venvはコピーせず、コピー先で新規作成します。
この公開版のONNX処理にはPyTorch・Ultralytics・ROCm用venvは不要です。

### 3. 環境を確認

```bash
cd ~/yolofast
source .venv/bin/activate
python - <<'PYTHON'
from pathlib import Path
import sys
import numpy, cv2, onnx, onnxruntime

root = Path.cwd().resolve()
assert Path(sys.prefix).resolve() == root / '.venv', sys.prefix
assert sys.version_info[:2] == (3, 12), sys.version
assert Path(sys.base_prefix).resolve().is_relative_to(root / '.python'), sys.base_prefix
for module in (numpy, cv2, onnx, onnxruntime):
    location = Path(module.__file__).resolve()
    assert location.is_relative_to(root / '.venv'), location
print('Python:', sys.executable)
print('ONNX Runtime:', onnxruntime.__version__)
print('Providers:', onnxruntime.get_available_providers())
PYTHON
python benchmarks/prepare_npu_split.py --help
python benchmarks/quantize_npu.py --help
python benchmarks/prepare_npu_int16.py --help
```

量子化用パッケージも確認する場合は `python -c 'from quark.onnx import ModelQuantizer'`
を実行します。終了時は `deactivate`、次回は以下で再開できます。

```bash
cd ~/yolofast
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source .venv/bin/activate
```

### NPU測定を追加する場合

上記の通常版ONNX RuntimeはCPU処理用です。NPU測定には対応するNPUドライバーと
Ryzen AI SDKのVitisAI対応ONNX Runtimeが必要で、`requirements.txt` だけでは導入できません。
元測定環境はRyzen AI MAX+ 395、Ryzen AI SDK 1.8、Python 3.12.13、ORT VitisAI 1.27.0です。

SDKは配布物に含まれません。NPUも利用する場合は、使用許諾に従って取得したSDKの
ユーザー空間ライブラリ・設定・対応wheelを `~/yolofast/vendor/ryzenai/` 内に配置し、
そのSDKの導入手順に従って対応wheelを `~/yolofast/.venv` に導入してください。
SDKの導入手順が別のvenvを作る場合も、配置先は `~/yolofast` 内に限定してください。
SDKのライブラリ・設定を参照する環境変数もこの配置先に合わせます。
他フォルダにあるSDKの環境有効化スクリプトやvenvは使用しません。

VitisAI版導入後は `pip install -r requirements.txt` を再実行せず、
通常版ONNX Runtimeへの置き換えを防ぎます。SDKのwheel名や設定は配布版に依存するため、
上記CPU環境の作成だけでNPU測定まで可能になるわけではありません。
NPU対応環境で次の確認が通ってからNPU測定を実行してください。

```bash
python -m pip check
python -c 'import sys, onnxruntime as ort; print(sys.executable); print(ort.__version__); print(ort.get_available_providers()); assert "VitisAIExecutionProvider" in ort.get_available_providers()'
```

## 手順

以下は `~/yolofast` でvenvを有効化した状態で実行します。
利用条件を満たすFP32 ONNXを `models/fp32.onnx` に、校正画像を `calibration/` に配置します。
検証画像とCOCOデータも `examples/images/` と `datasets/` に配置し、他フォルダを参照しません。
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
```

NPU対応環境の確認後、次を実行します（CPU環境だけでは実行できません）。

```bash
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
