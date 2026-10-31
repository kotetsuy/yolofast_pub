# 2026-10-31 同期記録

`~/yolofast` の2026年10月31日時点の内容を確認し、公開版に収録する情報を選別した。
初回同期では環境記録と測定方法を追加し、追加同期では独立した実行コード・回帰テスト・測定結果も収録した。
既存のNPUツールは、ライセンスヘッダーや公開用の入力指定・パス処理を除いて
同期元と同じだった。公開版専用の量子化エントリーポイントも保持した。

## GPU実行環境の記録

同期元の記録から抽出したバージョンを
[summaries/migraphx-environment.json](summaries/migraphx-environment.json) に保存した。
ハードウェアはRyzen AI Max+ 395 / Radeon 8060S、Pythonは3.12.13、
PyTorchは2.13.0+rocm10.0.0、ONNX Runtimeは1.29.0、
MIGraphXは2.17.0+rocm10.0.0、MIGraphX EPプラグインは1.0.0+rocm10.0.0だった。
これは同期元の環境記録であり、この公開版でGPU実行を再検証した結果ではない。
既存のNPU用requirementsは変更していない。

## 同期元で使用した測定方法

GPU推論単体の測定では、GPUに入力バッファを事前確保し、固定shapeでウォームアップした。
各フレームのGPU処理完了を待ち、完了フレーム数を壁時計の経過時間で割った。
画像読込、前処理、入力コピー、結果転送、JSON生成、描画は測定範囲から除外した。
この方法による推論速度と、カメラ入力から表示までの速度は測定範囲が異なる。

演算配置は別セッションのORTプロファイルで確認し、その実行時間を速度測定に含めなかった。
長時間測定では区間ごとのFPS、ホスト側レイテンシ分布、温度、電力のセンサー値を記録した。
APUの電力センサー値をGPU単体の消費電力として扱わなかった。

同期元では `ORT_MIGRAPHX_HIP_GRAPH_ENABLE=1` を使用した構成も測定した。
グラフ再生の有無を比較する際は、測定時間や実行順の違いを考慮する必要がある。
これらは測定手順の記録であり、任意のモデルでの動作や性能を保証するものではない。

## 追加した実行コードとテスト

同期元から次の4本を追加した。公開版では外部ファイルを引数で指定する。

| ファイル | 用途 | 必要な入力引数 |
| --- | --- | --- |
| `benchmarks/benchmark_npu_host.py` | CPU前後処理のマイクロベンチマーク | `--saved-outputs`、`--quant-meta`、`--image`、`--output` |
| `benchmarks/evaluate_npu_cpu.py` | FP32と量子化モデル2種のCPU出力比較 | `--fp32`、`--v1`、`--v2`、`--images-manifest`、`--output` |
| `benchmarks/compare_npu_variants.py` | 別プロセスでのNPU構成比較・逆順再測定・長時間測定 | `--config`、`--images-manifest`、`--output` |
| `benchmarks/validate_npu_final.py` | 保存済み最終検出出力とCPU参照の比較 | `--results`、`--reference-model`、`--images-manifest` |

各コマンドの引数は `--help` で確認できる。CPUマイクロベンチマークはNPUの推論速度を測らない。
構成比較のJSON形式は同期元のままで、`baseline_reference`、`baseline`、`candidates` を使用する。
モデルと参照結果は利用者が準備する。元の作業フォルダに依存する設定は同梱していない。

`tests/test_npu_paths.py` を追加した。Top-kの同点順序、INT16境界値、整数後処理、
量子化、バッファ再利用、画像前処理、測定結果の採否、ウォームアップ除外を検査する。
画像テストは再現可能な合成画像を使い、境界値テストは固定の合成スケール値を使う。
保存済み出力3件の照合は `YOLOFAST_SAVED_OUTPUTS` に外部データのディレクトリを指定すると実行する。
指定しない場合、その1件だけをスキップする。

```bash
python -B -m unittest discover -s tests -v
# 外部の outputs-*.npz が3件ある場合:
YOLOFAST_SAVED_OUTPUTS=/path/to/saved-outputs python -B -m unittest discover -s tests -v
```

今回の確認では、同期元のMIGraphX用venvを検証用インタープリターとして使い、
外部データなしで14件成功・1件スキップ、同期元の保存済み出力を参照して15件すべて成功した。
このvenvを公開版の実行要件に追加したわけではなく、venv自体もコピーしていない。
追加した4本のCLI確認、全Pythonファイルの構文確認も実施した。GPU・NPUの推論測定は再実行していない。

## 追加した測定結果

[GPU測定結果](summaries/gpu-20261031/README.md)に同期元の10月31日の集計を収録した。
モデル名・ハッシュ・測定条件は結果を識別する情報として残した。
成功した測定だけでなく、FP16検出照合とPyTorchグラフ再生の失敗結果も含めた。
ユーザーのホームパスは `$HOME` に置換し、生の配置プロファイルへの参照は除いた。

## コピー対象から除外したもの

対象モデルの本体、重み、変換・量子化済みモデル、Ultralyticsのソースと
それを直接使用するGPU実行・エクスポートコード、出所確認が必要な元の量子化コードは追加していない。
元のREADME、REPORT、MIGRAPHX_REPORTは丸ごとコピーしていない。
データセット、画像、検出配列、生ログ、キャッシュ、venv、認証情報、Git履歴も対象外とした。
既存の公開版専用量子化エントリーポイント、NPUコード、集計、ライセンス説明は保持している。

## Ultralytics依存コードの作成手順

[Codexに渡す指示文](ULTRALYTICS_WORKFLOW.md)を追加した。
公開版に含まれないエクスポート・GPU測定・精度比較の作成要件、検証条件、
実行記録の保存方法を記載した。実行コード自体は追加していない。
