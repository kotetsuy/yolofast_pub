# YOLO26nの量子化とライセンスに関する整理

## 結論

- **YOLO26nの学習済み重みをFP16、INT8、INT4などに量子化しても、量子化だけで元のモデルのライセンス条件が消えるわけではありません。** 元モデルがUltralyticsのAGPL-3.0条件で提供されている場合、変換後の重みもその条件を前提に扱うのが安全です。
- **量子化手順を記載した独自ドキュメントや、独立して自作した量子化スクリプトは、原則としてApache License 2.0で公開できます。** ただし、第三者のAGPLコード等を複製・改変して含めた場合は別途条件が生じます。
- **Apache-2.0で公開したスクリプトを使っても、生成される量子化済みモデルやそれを組み込む製品が自動的にApache-2.0になるわけではありません。**

## 1. 量子化済みモデルの扱い

| 処理 | ライセンス上の基本的な考え方 |
| --- | --- |
| FP32 → FP16 | 元モデルの条件を前提に扱う |
| FP32 → INT8 / INT4 | 元モデルの条件を前提に扱う |
| PyTorch → ONNX | ファイル形式変更だけで元モデルの条件は消えない |
| ONNX → TensorRT / OpenVINO | 推論形式の変更だけで元モデルの条件は消えない |
| 量子化後に追加学習 | 元モデルの条件に加え、追加データ・コードの権利関係も確認する |

モデルの重みへの著作権やオープンソースライセンスの適用範囲は、ソースコードの場合と同じ程度に法的に確立しているとは限りません。上表は実務上の慎重な取り扱いであり、すべての法域での確定的な法的判断ではありません。

## 2. Apache-2.0で公開しやすいもの

| 公開物 | Apache-2.0での公開 |
| --- | --- |
| 自作の量子化アルゴリズム説明 | 原則可能 |
| 独自実装した量子化スクリプト | 原則可能 |
| 自作の設定ファイルや実行例 | 原則可能 |
| AGPL-3.0のコードを複製・改変したもの | Apache-2.0単独での再ライセンスは原則不可 |
| 元のYOLO26n学習済み重み | Apache-2.0への一方的な変更は不可 |
| 量子化済みYOLO26n重み | Apache-2.0への一方的な変更は避けるべき |
| 元モデルの入手先URL | 通常可能（配布元の条件に従う） |

独自スクリプトがUltralyticsのライブラリを**単に呼び出す**場合、スクリプト自身へのAGPLの波及範囲は、依存・結合方法、配布形態、権利関係に左右されます。単に「コードをコピーしていない」というだけで、すべてのケースでApache-2.0単独配布が保証されるわけではありません。可能なら独立したツールとして構成し、依存ライブラリのライセンスを明示してください。

## 3. 推奨リポジトリ構成

```text
yolo26n-quantizer/
├── LICENSE             # 自作部分: Apache-2.0
├── README.md           # 手順・依存関係・元モデルのライセンスを明記
├── requirements.txt    # 依存ライブラリとバージョン
├── quantize.py         # 独自に作成した変換スクリプト
└── examples/
    └── config.yaml     # 自作の設定例
```

元の重み、量子化済み重み、Ultralytics由来のAGPLコードは、Apache-2.0のみを適用するリポジトリに無条件で同梱しないことを推奨します。ライセンス表記は「リポジトリ内の自作コードとドキュメントにApache-2.0を適用する」と対象を明確にしてください。

READMEに記載する例：

> The independently developed scripts and documentation in this repository are licensed under Apache License 2.0. YOLO26 model weights are not included. Obtain model weights separately and comply with their applicable licenses. This repository's license does not change the license of any input or output model weights.

## 4. 商用利用・組み込み時の注意

AGPL-3.0は商用利用そのものを禁止しません。一方、AGPL対象プログラムの改変・配布、改変版をネットワーク経由で利用させる場合などには、条件に応じて対応するソースコードの提供義務が生じます。**単に社内でローカル実行する場合は通常、外部へのソース公開義務は生じません。**

量子化済みモデルを独自のC++アプリ、ONNX Runtime、TensorRTなどと組み合わせて販売・配布する場合、**アプリ全体にAGPL条件が及ぶかどうかは、モデルとコードの権利関係、結合・配布方法、適用法によって変わります。** Ultralyticsは学習済みモデルを含む利用に関して広いAGPL適用の見解を示しており、非公開の商用製品にはEnterpriseライセンスを案内しています。ただし、事業者の見解と裁判所で確定した法的結論は区別する必要があります。

## 5. 実務上の推奨

1. 元のYOLO26n重みの入手先、版、ライセンス表記を記録する。
2. 自作の量子化スクリプト・文書と、元モデルおよびその変換物を分けて管理する。
3. Apache-2.0の適用範囲を自作部分に限定して明記する。
4. 依存ライブラリのライセンスと配布条件を個別に確認する。
5. 量子化済み重みの再配布や非公開商用製品への搭載前には、Ultralyticsのライセンス条件を確認し、必要に応じて権利者または法務専門家に相談する。

## 参考資料

- Ultralytics Licensing: https://www.ultralytics.com/license
- GNU AGPL-3.0 本文: https://www.gnu.org/licenses/agpl-3.0.html
- GNU GPL FAQ: https://www.gnu.org/licenses/gpl-faq.html
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

*本書は一般的な技術・ライセンス情報の整理であり、個別案件に対する法律意見ではありません。*
