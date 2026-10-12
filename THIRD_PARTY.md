# External dependencies and model licenses

Apache-2.0 covers the original code and documentation in this repository only.
No third-party source, binaries, SDKs, dataset images, annotations, or weights are redistributed.

| Dependency | Upstream license / conditions | Source |
| --- | --- | --- |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| OpenCV 4.12+ | Apache-2.0; bundled components have separate notices | https://github.com/opencv/opencv/blob/4.x/LICENSE |
| ONNX | Apache-2.0 | https://github.com/onnx/onnx/blob/main/LICENSE |
| ONNX Runtime | MIT; provider/vendor components may differ | https://github.com/microsoft/onnxruntime/blob/main/LICENSE |
| pycocotools | BSD-2-Clause | https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/license.txt |
| AMD Quark | Check the license supplied with the installed release and bundled components | https://quark.docs.amd.com/latest/ |
| Ryzen AI SDK / VitisAI provider / XRT | Separate vendor and component terms; not distributed here | https://ryzenai.docs.amd.com/en/latest/ |
| YOLO26 / Ultralytics models | AGPL-3.0 or separately obtained Enterprise terms | https://www.ultralytics.com/license |
| COCO data | Separate dataset and image conditions; obtain independently | https://cocodataset.org/#termsofuse |

Ultralytics is not imported or installed by these tools. Obtain a compatible ONNX
input independently using a workflow for which you have the necessary rights.
Converting, splitting, or quantizing a model does not relicense its weights.
Check the exact installed dependency versions and their bundled notices when redistributing them.
